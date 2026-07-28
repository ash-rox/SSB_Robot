import cv2
import numpy as np
import onnxruntime as rt
import time
from pathlib import Path
from sparkybotmini import SparkyBotMini

# Configuration
MODEL_PATH = "best.onnx"
CONFIDENCE_THRESHOLD = 0.5
INPUT_SIZE = 320  
INPUT_NAME = "images"
OUTPUT_NAMES = ["output0"]

# Class maps matching your exact model definitions
CLASS_NAMES = ["yellow", "green", "stand", "background"]

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

def preprocess_image(frame, input_size=320):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  
    img = np.expand_dims(img, 0)        
    return img, (h, w)

def postprocess_predictions(outputs, original_size, input_size=320, confidence_threshold=0.5):
    detections = []
    orig_h, orig_w = original_size
    predictions = np.squeeze(outputs)
    
    if predictions.shape < predictions.shape:
        predictions = predictions.T

    boxes = predictions[:, :4]
    scores = predictions[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    
    mask = confidences > confidence_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0: return detections

    scale_x = orig_w / input_size
    scale_y = orig_h / input_size

    for i in range(len(boxes)):
        x_center, y_center, width, height = boxes[i]
        x_center *= scale_x
        y_center *= scale_y
        width *= scale_x
        height *= scale_y
        
        x1 = int(x_center - (width / 2))
        y1 = int(y_center - (height / 2))
        x2 = int(x_center + (width / 2))
        y2 = int(y_center + (height / 2))
        
        cid = int(class_ids[i])
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'center': (int(x_center), int(y_center)),
            'size': int(width * height), 
            'confidence': float(confidences[i]),
            'class_id': cid,
            'class_name': CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"id_{cid}"
        })
    return detections

def process_corn_logic_stationary(detections):
    """
    Identifies the detected types, prints them to the terminal, 
    and sets indicator lights without moving the robot motors.
    """
    if not detections:
        robot.set_led(1, 0, 0, 0) # Turn off LED if nothing is seen
        return "Scanning... No targets in view."

    # Keep track of unique types found in this specific frame
    found_types = set()
    
    for det in detections:
        if det['class_name'] == "yellow":
            found_types.add("RIPE CORN (yellow)")
        elif det['class_name'] == "green":
            found_types.add("UNRIPE CORN (green)")
        elif det['class_name'] == "stand":
            found_types.add("STRUCTURAL STAND")

    # Print the specific types found to the terminal screen
    if found_types:
        print(f"Detected: {', '.join(found_types)}")

    # Update onboard LED color based on priority of what is currently seen
    if "RIPE CORN (yellow)" in found_types:
        robot.set_led(1, 0, 255, 0)  # Solid GREEN light for ripe corn presence
    elif "UNRIPE CORN (green)" in found_types:
        robot.set_led(1, 255, 0, 0)  # Solid RED light for unripe corn presence
    elif "STRUCTURAL STAND" in found_types:
        robot.set_led(1, 0, 0, 255)  # Solid BLUE light if only a stand is visible
        
    return f"In View: {', '.join(found_types)}"

def draw_detections(frame, detections):
    """Draws custom colored bounding boxes based on class sorting labels"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        
        if det['class_name'] == "yellow":
            box_color = (0, 255, 255)  # Yellow BGR
            status_lbl = "RIPE CORN"
        elif det['class_name'] == "green":
            box_color = (0, 255, 0)    # Green BGR
            status_lbl = "UNRIPE CORN"
        elif det['class_name'] == "stand":
            box_color = (255, 255, 0)  # Cyan BGR
            status_lbl = "STRUCTURAL STAND"
        else:
            box_color = (255, 0, 0)    # Blue for alternative classes
            status_lbl = det['class_name'].upper()

        label = f"{status_lbl} [{det['confidence']:.2f}]"
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
    return frame

def main():
    global robot
    # Initialize connection to the SparkyBot hardware
    robot = SparkyBotMini(port="/dev/ttyUSB0", debug=False)
    if not robot.connect():
        print("Hardware serial interface failed to connect.")
        return

    # Ensure motors are completely stopped/idle at bootup
    robot.set_motor(0, 0, 0, 0)

    session = load_model(MODEL_PATH)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Stationary Identification Engine Active. Press 'q' to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            h, w = frame.shape[:2]
            
            # Run the AI detection frame extraction pipeline
            input_data, original_size = preprocess_image(frame, input_size=INPUT_SIZE)
            outputs = session.run(OUTPUT_NAMES, {INPUT_NAME: input_data})
            detections = postprocess_predictions(outputs, original_size, INPUT_SIZE, CONFIDENCE_THRESHOLD)
            
            # Run stationary print logic
            status_text = process_corn_logic_stationary(detections)
            
            # Render visual window feedback
            frame = draw_detections(frame, detections)
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("Corn AI Classifier Matrix", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        # Safety reset step
        robot.set_motor(0, 0, 0, 0)
        robot.set_led(1, 0, 0, 0)
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Program closed cleanly.")

if __name__ == "__main__":
    main()
