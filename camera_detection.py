import cv2
import numpy as np
import onnxruntime as rt
from pathlib import Path

# Configuration
MODEL_PATH = "best.onnx"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Model input/output names (common for YOLOv8-based models)
INPUT_NAME = "images"
OUTPUT_NAMES = ["output0"]  # Adjust based on your model's output layer

# Class names - UPDATE THIS with your model's actual class names
CLASS_NAMES = [
    "class_0", "class_1", "class_2", "class_3"  # Replace with actual class names
]

def load_model(model_path):
    """Load ONNX model"""
    try:
        session = rt.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        print(f"Model loaded successfully from {model_path}")
        return session
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def preprocess_image(frame, input_size=640):
    """Preprocess image for model input"""
    h, w = frame.shape[:2]
    
    # Resize to model input size (commonly 640x640 for YOLOv8)
    img = cv2.resize(frame, (input_size, input_size))
    
    # Normalize to 0-1 and convert to float32
    img = img.astype(np.float32) / 255.0
    
    # Add batch dimension and convert to NCHW format
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    
    return img, (h, w)

def postprocess_predictions(output, original_size, confidence_threshold=0.5):
    """Parse model output and extract detections"""
    detections = []
    
    # output shape is typically [1, num_detections, 85] for YOLOv8
    # Format: [x, y, w, h, confidence, class_scores...]
    predictions = output[0]
    
    if predictions.ndim == 3:
        predictions = predictions[0]
    
    # Filter by confidence
    conf_mask = predictions[:, 4] > confidence_threshold
    predictions = predictions[conf_mask]
    
    if len(predictions) == 0:
        return detections
    
    # Get class predictions
    class_scores = predictions[:, 5:]
    class_ids = np.argmax(class_scores, axis=1)
    class_confs = np.max(class_scores, axis=1)
    
    # Scale coordinates to original image size
    orig_h, orig_w = original_size
    scale_x = orig_w / 640
    scale_y = orig_h / 640
    
    for i, pred in enumerate(predictions):
        x_center, y_center, width, height = pred[:4] * [scale_x, scale_y, scale_x, scale_y]
        confidence = pred[4]
        class_id = class_ids[i]
        
        # Convert from center coordinates to top-left and bottom-right
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'confidence': float(confidence),
            'class_id': int(class_id),
            'class_name': CLASS_NAMES[int(class_id)] if int(class_id) < len(CLASS_NAMES) else f"class_{class_id}"
        })
    
    return detections

def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame"""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        confidence = det['confidence']
        class_name = det['class_name']
        
        # Draw bounding box
        color = (0, 255, 0)  # Green
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label with background
        label = f"{class_name} {confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 - 2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def main():
    """Main detection loop"""
    # Load model
    session = load_model(MODEL_PATH)
    if session is None:
        return
    
    # Open camera (0 is typically the default USB camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot open camera. Check if USB camera is connected.")
        return
    
    print("Camera opened successfully. Press 'q' to quit.")
    
    # Set camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to read frame")
            break
        
        # Preprocess image
        input_data, original_size = preprocess_image(frame, input_size=640)
        
        # Run inference
        try:
            outputs = session.run(OUTPUT_NAMES, {INPUT_NAME: input_data})
            
            # Post-process predictions
            detections = postprocess_predictions(outputs[0], original_size, CONFIDENCE_THRESHOLD)
            
            # Draw detections
            frame = draw_detections(frame, detections)
            
            # Add FPS and detection count
            frame_count += 1
            fps_text = f"Detections: {len(detections)}"
            cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
        except Exception as e:
            print(f"Error during inference: {e}")
        
        # Display frame
        cv2.imshow("Object Detection - Press 'q' to quit", frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera detection closed.")

if __name__ == "__main__":
    main()
