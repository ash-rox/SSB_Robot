###WORKS

### STATIONARY CORN CLASSIFIER - YOLOV11 OPTIMIZED
import cv2
import numpy as np
import onnxruntime as rt
import time
from pathlib import Path
from sparkybotmini import SparkyBotMini

# Configuration
MODEL_PATH = "best.onnx"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45       # Cleans up duplicate overlapping bounding boxes
INPUT_SIZE = 320  
INPUT_NAME = "images"
OUTPUT_NAMES = ["output0"]

# Class maps matching your exact YOLOv11 model definitions
CLASS_NAMES = ["green", "stand", "yellow", "background"]

robot = None

def load_model(model_path):
    """Load ONNX model optimized for Raspberry Pi 5 CPU"""
    try:
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
        session = rt.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])
        return session
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(frame, input_size=INPUT_SIZE):
    """
    Preprocesses frame for YOLOv11 model inference.
    Converts OpenCV BGR frame to RGB (Critical for Yellow/Ripe Corn detection).
    """
    h, w = frame.shape[:2]
    
    # FIX: Convert BGR (OpenCV default) to RGB expected by YOLO models
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(rgb_frame, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # Convert HWC to CHW format
    img = np.expand_dims(img, 0)        # Add batch dimension [1, C, H, W]
    return img, (h, w)

def postprocess_predictions(outputs, original_size, input_size=INPUT_SIZE, confidence_threshold=0.5, iou_threshold=0.45):
    """
    Parses YOLOv11 ONNX outputs.
    YOLOv11 output structure: [1, 4 + num_classes, num_anchors]
      - Indices 0..3: Box coordinates (x_center, y_center, width, height)
      - Indices 4..N: Direct class probability scores
    """
    detections = []
    orig_h, orig_w = original_size

    if isinstance(outputs, (list, tuple)):
        if len(outputs) == 0:
            return detections
        outputs = outputs[0]

    predictions = np.squeeze(outputs)

    # YOLOv11 outputs shape (4 + num_classes, anchors). Transpose to (anchors, 4 + num_classes)
    if predictions.ndim == 2 and predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    if predictions.ndim != 2 or predictions.shape[1] < 5:
        print(f"Unexpected prediction shape: {predictions.shape}")
        return detections

    # Separate bounding box coordinates from class confidence scores
    boxes = predictions[:, :4]
    scores = predictions[:, 4:]  # Direct class confidence scores in YOLOv11

    if scores.ndim == 1:
        scores = scores[:, np.newaxis]

    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)

    # Filter out weak detections
    mask = confidences > confidence_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    if boxes.shape[0] == 0:
        return detections

    # Scale coordinates back to original camera resolution
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

    # Format bounding boxes for OpenCV Non-Maximum Suppression (NMS)
    nms_boxes = np.stack([x1, y1, (x2 - x1), (y2 - y1)], axis=1).tolist()
    nms_confidences = confidences.astype(float).tolist()

    try:
        indices = cv2.dnn.NMSBoxes(nms_boxes, nms_confidences, confidence_threshold, iou_threshold)
    except Exception:
        indices = np.arange(len(nms_boxes))

    if hasattr(indices, 'flatten'):
        sel = indices.flatten()
    else:
        sel = indices

    for idx in sel:
        bx, by, bw, bh = nms_boxes[int(idx)]
        cid = int(class_ids[int(idx)])
        conf = float(confidences[int(idx)])

        detections.append({
            'bbox': (int(bx), int(by), int(bx + bw), int(by + bh)),
            'center': (int(bx + bw // 2), int(by + bh // 2)),
            'confidence': conf,
            'class_id': cid,
            'class_name': CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"id_{cid}"
        })

    return detections

def process_corn_logic_stationary(detections):
    """Identifies detected targets and updates terminal output and status LEDs."""
    if not detections:
        try:
            robot.set_led(1, 0, 0, 0)  # Off
        except Exception:
            pass
        return "Scanning... No targets in view."

    found_types = set()

    for det in detections:
        cname = det.get('class_name')
        if cname == "yellow":
            found_types.add("RIPE CORN (yellow)")
        elif cname == "green":
            found_types.add("UNRIPE CORN (green)")
        elif cname == "stand":
            found_types.add("STRUCTURAL STAND")

    if found_types:
        print(f"Detected: {', '.join(found_types)}", flush=True)

    try:
        if "RIPE CORN (yellow)" in found_types:
            robot.set_led(1, 0, 255, 0)   # Solid GREEN LED for ripe corn
        elif "UNRIPE CORN (green)" in found_types:
            robot.set_led(1, 255, 0, 0)   # Solid RED LED for unripe corn
        elif "STRUCTURAL STAND" in found_types:
            robot.set_led(1, 0, 0, 255)   # Solid BLUE LED for stand
    except Exception:
        pass

    return f"In View: {', '.join(found_types)}"

def draw_detections(frame, detections):
    """Draws bounding boxes on frame (Uses OpenCV BGR color values)."""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']

        if det['class_name'] == "yellow":
            box_color = (0, 255, 255)  # Yellow
            status_lbl = "RIPE CORN"
        elif det['class_name'] == "green":
            box_color = (0, 255, 0)    # Green
            status_lbl = "UNRIPE CORN"
        elif det['class_name'] == "stand":
            box_color = (255, 255, 0)  # Cyan
            status_lbl = "STRUCTURAL STAND"
        else:
            box_color = (255, 0, 0)    # Blue
            status_lbl = det['class_name'].upper()

        label = f"{status_lbl} [{det.get('confidence', 0):.2f}]"
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    return frame

def main():
    global robot
    robot = SparkyBotMini(port="/dev/ttyUSB0", debug=False)
    if not robot.connect():
        print("Hardware serial interface failed to connect.")
        return

    try:
        robot.set_motor(0, 0, 0, 0)
    except Exception:
        pass

    session = load_model(MODEL_PATH)
    if session is None:
        try:
            robot.disconnect()
        except Exception:
            pass
        return

    try:
        model_input_name = session.get_inputs()[0].name
        model_output_name = session.get_outputs()[0].name
        print("Model input:", model_input_name, "output:", model_output_name)
    except Exception:
        model_input_name = INPUT_NAME
        model_output_name = OUTPUT_NAMES[0]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera device (index 0).")
        try:
            robot.disconnect()
        except Exception:
            pass
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Stationary Identification Engine Active. Press 'q' to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read failed; stopping.")
                break

            input_data, original_size = preprocess_image(frame, input_size=INPUT_SIZE)

            try:
                out_arr = session.run([model_output_name], {model_input_name: input_data})[0]
            except Exception as e:
                print(f"Model inference failed: {e}")
                break

            detections = postprocess_predictions(out_arr, original_size, INPUT_SIZE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)

            status_text = process_corn_logic_stationary(detections)

            frame = draw_detections(frame, detections)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            try:
                cv2.imshow("Corn AI Classifier Matrix", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except Exception:
                pass
    finally:
        try:
            robot.set_motor(0, 0, 0, 0)
            robot.set_led(1, 0, 0, 0)
        except Exception:
            pass
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass
        print("Program closed cleanly.")

if __name__ == "__main__":
    main()
