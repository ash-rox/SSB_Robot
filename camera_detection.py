import cv2
import numpy as np
import onnxruntime as rt
import time
from pathlib import Path
from sparky_bot import SparkyBotMini

# Configuration
MODEL_PATH = "best.onnx"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45       # Cleans up duplicate overlapping bounding boxes
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

def preprocess_image(frame, input_size=640):
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (input_size, input_size))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  
    img = np.expand_dims(img, 0)        
    return img, (h, w)

def postprocess_predictions(outputs, original_size, input_size=640, confidence_threshold=0.5, iou_threshold=0.45):
    """Parse standard raw YOLOv11 output tensors [1, 4 + classes, 8400]"""
    detections = []
    orig_h, orig_w = original_size
    
    # Remove batch dimension -> shape: [4 + classes, 8400]
    predictions = np.squeeze(outputs)
    
    # YOLOv11 outputs dimensions in rows. Transpose to shape: [8400, 4 + classes]
    predictions = predictions.T

    # YOLOv11 structure: [x_center, y_center, width, height, class0_score, class1_score...]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:]
    
    # Highest class score defines the confidence
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    
    # Filter arrays using confidence threshold mask
    mask = confidences > confidence_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0: 
        return detections

    scale_x = orig_w / input_size
    scale_y = orig_h / input_size

    # Convert center coordinates (xywh) to corner extremes (x1, y1, x2, y2)
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (x_center - width / 2) * scale_x
    y1 = (y_center - height / 2) * scale_y
    x2 = (x_center + width / 2) * scale_x
    y2 = (y_center + height / 2) * scale_y
    
    # Group coordinates into expected format for OpenCV Non-Maximum Suppression
    nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).astype(int).tolist()
    nms_confidences = confidences.astype(float).tolist()
    
    indices = cv2.dnn.NMSBoxes(nms_boxes, nms_confidences, confidence_threshold, iou_threshold)
    
    if len(indices) > 0:
        for idx in indices.flatten():
            bx, by, bw, bh = nms_boxes[idx]
            cid = int(class_ids[idx])
            
            detections.append({
                'bbox': (bx, by, bx + bw, by + bh),
                'center': (int(bx + bw // 2), int(by + bh // 2)),
                'confidence': float(confidences[idx]),
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

    # Print the specific types found directly to your terminal screen
    if found_types:
        print(f"Detected: {', '.join(found_types)}", flush=True)

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
        
        # Draw bounding rectangle limits
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

    # Keep motors completely stopped
    robot.set_motor(0, 0, 0, 0)

    session = load_model(MODEL_PATH)
    if session is None:
        robot.disconnect()
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("Stationary YOLOv11 Identification Engine Active. Press 'q' to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            h, w = frame.shape[:2]
            
            # Run the AI detection frame extraction pipeline
            input_data, original_size = preprocess_image(frame, input_size=INPUT_SIZE)
            outputs = session.run(OUTPUT_NAMES, {INPUT_NAME: input_data})
            detections = postprocess_predictions(outputs, original_size, INPUT_SIZE, CONFIDENCE_THRESHOLD, IOU_THRESHOLD)
            
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
