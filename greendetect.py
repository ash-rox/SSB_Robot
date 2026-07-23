#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import time

# --- CONFIGURATION ---
CAMERA_INDEX = 0          # 0 is usually the default USB front camera
MIN_AREA = 800             # Minimum pixel clump size to validate an object
COOLDOWN_SECONDS = 0.5     # Prevent console spamming

# --- RGB COLOR THRESHOLDS ---
# Note: OpenCV standard channels natively map to [Blue, Green, Red] (BGR).
# These limits are structured to require the Green channel to be significantly 
# stronger than Red or Blue, allowing it to catch dark, dull, or faint greens 
# while preventing white walls or gray floors from triggering false positives.
#
# Format: [Blue_Bound, Green_Bound, Red_Bound]
LOWER_BOUND = np.array([0, 50, 0])      
UPPER_BOUND = np.array([160, 255, 160]) 

def main():
    # Initialize the USB video capture feed
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Low resolution optimized for Pi 5
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {CAMERA_INDEX}.")
        sys.exit(1)

    print("\n=============================================")
    print("  SparkyBot Mini RGB Green Recognition Running")
    print("  (Click on the video window to keep it focused)")
    print("  ")
    print("  [ESC KEY] : Quit Program")
    print("=============================================\n")

    last_detection_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                time.sleep(0.1)
                continue

            # Directly generate a mask matching our targeted RGB/BGR profile
            mask = cv2.inRange(frame, LOWER_BOUND, UPPER_BOUND)

            # Clean up microscopic video grain and camera sensor artifacts
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # Isolate contours inside our isolated color field
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            green_found = False
            largest_contour = None

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) >= MIN_AREA:
                    green_found = True

            # Print feedback to the console with structured pacing
            now = time.time()
            if green_found and (now - last_detection_time) >= COOLDOWN_SECONDS:
                print("green detected")
                last_detection_time = now

            # If a valid green target exists, trace it live in the video view
            if green_found and largest_contour is not None:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "GREEN TARGET", (x, max(y - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Render display windows
            cv2.imshow("Live Video View", frame)
            cv2.imshow("RGB Isolated Mask", mask)

            # Check OpenCV's GUI environment loop for an ESC keypress (ASCII 27)
            if cv2.waitKey(30) & 0xFF == 27:
                print("\nExiting program cleanly.")
                break

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        # Release system resources safely
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
