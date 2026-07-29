import cv2
import numpy as np
import onnxruntime as rt
import time
from pathlib import Path
from sparkybotmini import SparkyBotMini

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "best.onnx"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
INPUT_SIZE = 320
INPUT_NAME = "images"
OUTPUT_NAMES = ["output0"]

CLASS_NAMES = ["green", "stand", "yellow", "background"]

# Robot Control Parameters
SERIAL_PORT = "/dev/ttyUSB0"
DRIVE_SPEED = 15              # Speed for general forward driving (-100 to 100)
ADJUST_SPEED = 8             # Fine-tuning speed when centering
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_CENTER_Y = 240         # Ideal Y-coordinate center for swinging range
Y_TOLERANCE = 35              # Allowed deadband pixels (+/-) for Y-axis alignment

# Servo Arm Configuration
SERVO_PORT = 1                # Port S1
SERVO_IDLE_ANGLE = 0          # Rest position
SERVO_STRIKE_ANGLE = 90      # Full knock stroke angle
SWING_DELAY = 0.25            # Delay between swing phases (seconds)

# Robot state definitions
STATE_SEARCH = "SEARCHING (Driving Forward)"
STATE_CENTER = "CENTERING (Adjusting Position)"
STATE_SWING = "SWINGING (Knocking Target)"


def load_model(model_path):
    """Load ONNX model optimized for CPU inference."""
    try:
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        return rt.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None


def preprocess_image(frame, input_size=INPUT_SIZE):
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(rgb_frame, (input_size, input_size)).astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, 0), (h, w)


def postprocess_predictions(outputs, original_size, input_size=INPUT_SIZE, confidence_threshold=0.5, iou_threshold=0.45):
    detections = []
    orig_h, orig_w = original_size

    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 0:
            return detections
        outputs = outputs[0]

    predictions = np.squeeze(outputs)
    if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    if predictions.ndim != 2 or predictions.shape[1] < 5:
        return detections

    boxes = predictions[:, :4]
    scores = predictions[:, 4:]
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    mask = confidences > confidence_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if boxes.shape[0] == 0:
        return detections

    scale_x = orig_w / input_size
    scale_y = orig_h / input_size

    x_center = boxes[:, 0].astype(float) * scale_x
    y_center = boxes[:, 1].astype(float) * scale_y
    width = boxes[:, 2].astype(float) * scale_x
    height = boxes[:, 3].astype(float) * scale_y

    x1 = (x_center - width / 2).astype(int)
    y1 = (y_center - height / 2).astype(int)
    x2 = (x_center + width / 2).astype(int)
    y2 = (y_center + height / 2).astype(int)

    nms_boxes = np.stack([x1, y1, (x2 - x1), (y2 - y1)], axis=1).tolist()
    nms_confidences = confidences.astype(float).tolist()

    try:
        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_confidences, confidence_threshold, iou_threshold)
    except Exception:
        indices = np.arange(len(nms_boxes))

    sel = indices.flatten() if hasattr(indices, 'flatten') else indices

    for idx in sel:
        bx, by, bw, bh = nms_boxes[int(idx)]
        cid = int(class_ids[int(idx)])
        detections.append({
            'bbox': (int(bx), int(by), int(bx + bw), int(by + bh)),
            'center': (int(bx + bw // 2), int(by + bh // 2)),
            'confidence': float(confidences[int(idx)]),
            'class_id': cid,
            'class_name': CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"id_{cid}"
        })

    return detections


def get_target_yellow_corn(detections):
    """Finds the largest yellow corn detection in view."""
    yellow_detections = [d for d in detections if d['class_name'] == "yellow"]
    if not yellow_detections:
        return None
    return max(yellow_detections, key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]))


def draw_overlay(frame, detections, state, current_target):
    # Draw horizontal target alignment bounds
    cv2.line(frame, (0, TARGET_CENTER_Y - Y_TOLERANCE), (FRAME_WIDTH, TARGET_CENTER_Y - Y_TOLERANCE), (0, 255, 255), 1)
    cv2.line(frame, (0, TARGET_CENTER_Y + Y_TOLERANCE), (FRAME_WIDTH, TARGET_CENTER_Y + Y_TOLERANCE), (0, 255, 255), 1)

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        box_color = (0, 255, 255) if det['class_name'] == "yellow" else (255, 0, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.circle(frame, det['center'], 4, (0, 0, 255), -1)

    # Display robot status
    cv2.putText(frame, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if current_target:
        cy = current_target['center'][1]
        cv2.putText(frame, f"Target Y: {cy} (Goal: {TARGET_CENTER_Y})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    return frame


def main():
    robot = SparkyBotMini(port=SERIAL_PORT, debug=False)
    if not robot.connect():
        print("[ERROR] Could not connect to SparkyBotMini hardware.")
        return

    session = load_model(MODEL_PATH)
    if session is None:
        robot.disconnect()
        return

    try:
        model_input_name = session.get_inputs()[0].name
        model_output_name = session.get_outputs()[0].name
    except Exception:
        model_input_name = INPUT_NAME
        model_output_name = OUTPUT_NAMES[0]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera initialization failed.")
        robot.disconnect()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # Initial safety reset
    robot.set_motor(0, 0, 0, 0)
    robot.set_pwm_servo(SERVO_PORT, SERVO_IDLE_ANGLE)

    current_state = STATE_SEARCH
    servo_state = False  # Toggle switch for swinging arm

    print("[SYSTEM] Indefinite Autonomous Hunt Engine Active. Press 'q' or Ctrl+C to quit.")

    try:
        # INDEFINITE CONTINUOUS LOOP
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame drop detected. Retrying...")
                continue

            input_data, original_size = preprocess_image(frame, input_size=INPUT_SIZE)
            out_arr = session.run([model_output_name], {model_input_name: input_data})[0]
            detections = postprocess_predictions(out_arr, original_size, INPUT_SIZE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

            target = get_target_yellow_corn(detections)

            # ==========================================
            # INDEFINITE STATE MACHINE LOGIC
            # ==========================================
            if target is None:
                # Target absent or knocked out of view -> Keep driving forward endlessly searching
                current_state = STATE_SEARCH
                robot.set_led(1, 0, 0, 0)
                robot.set_pwm_servo(SERVO_PORT, SERVO_IDLE_ANGLE)
                robot.set_motor(DRIVE_SPEED, DRIVE_SPEED, DRIVE_SPEED, DRIVE_SPEED)

            else:
                center_y = target['center'][1]

                # Target present and centered -> Keep swinging arm until it disappears
                if abs(center_y - TARGET_CENTER_Y) <= Y_TOLERANCE:
                    current_state = STATE_SWING
                    robot.set_motor(0, 0, 0, 0)          # Hold position
                    robot.set_led(1, 0, 255, 0)          # Green LED

                    # Continuous alternating swing cycle
                    target_angle = SERVO_STRIKE_ANGLE if servo_state else SERVO_IDLE_ANGLE
                    robot.set_pwm_servo(SERVO_PORT, target_angle)
                    servo_state = not servo_state
                    time.sleep(SWING_DELAY)

                # Target present but not centered -> Adjust position
                elif center_y < (TARGET_CENTER_Y - Y_TOLERANCE):
                    current_state = STATE_CENTER
                    robot.set_led(1, 255, 255, 0)        # Yellow LED
                    robot.set_pwm_servo(SERVO_PORT, SERVO_IDLE_ANGLE)
                    robot.set_motor(ADJUST_SPEED, ADJUST_SPEED, ADJUST_SPEED, ADJUST_SPEED)

                else:
                    current_state = STATE_CENTER
                    robot.set_led(1, 255, 255, 0)        # Yellow LED
                    robot.set_pwm_servo(SERVO_PORT, SERVO_IDLE_ANGLE)
                    robot.set_motor(-ADJUST_SPEED, -ADJUST_SPEED, -ADJUST_SPEED, -ADJUST_SPEED)

            # Render view frame
            frame = draw_overlay(frame, detections, current_state, target)
            cv2.imshow("Autodrive Corn Classifier", frame)

            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[SYSTEM] User quit signal received ('q').")
                break

    except KeyboardInterrupt:
        print("[SYSTEM] Keyboard interrupt detected (Ctrl+C).")

    finally:
        print("[SYSTEM] Safely stopping motors and cleaning up hardware connections...")
        robot.set_motor(0, 0, 0, 0)
        robot.set_led(1, 0, 0, 0)
        robot.set_pwm_servo(SERVO_PORT, SERVO_IDLE_ANGLE)
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("[SYSTEM] Program closed cleanly.")


if __name__ == "__main__":
    main()
