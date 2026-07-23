#!/usr/bin/env python3
"""
Green Object Detection - USB Camera
=====================================

Watches the USB camera feed and prints "green detected" whenever a
green object appears in front of the robot, with a 0.5 second cooldown
between messages (so a continuously-visible green object doesn't spam
the console).

Detection approach:
  1. Grab a frame from the camera.
  2. Convert it to HSV color space (much easier to isolate a color
     than in RGB, since hue is separated from brightness).
  3. Threshold the frame to a mask of "is this pixel green".
  4. Clean up the mask (remove small noise specks).
  5. If the green area is bigger than MIN_AREA, count it as a
     detection.

Install dependencies:
    pip install opencv-python numpy --break-system-packages

Run with:  python3 green_detector.py
Stop with: Ctrl+C
"""

import time
import sys
from sparkybotmini import SparkyBotMini

try:
    import cv2
    import numpy as np
except ImportError:
    print("Missing dependency. Install with:")
    print("  pip install opencv-python numpy --break-system-packages")
    sys.exit(1)

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

CAMERA_INDEX = 0          # 0 is usually the first/only USB camera
FRAME_WIDTH = 320          # lower resolution = faster processing
FRAME_HEIGHT = 240

# HSV green range - tune these if detection is too strict/loose.
# OpenCV HSV ranges: H 0-179, S 0-255, V 0-255
GREEN_LOWER = np.array([40, 70, 70])
GREEN_UPPER = np.array([80, 255, 255])

MIN_AREA = 800             # minimum pixel area to count as "detected"
                           # (raise this if small green specs of noise
                           # falsely trigger detection; lower it to
                           # detect smaller/farther-away objects)

COOLDOWN_SECONDS = 0.5

SHOW_PREVIEW = False       # set True to display a debug window
                           # (only works if you have a display attached,
                           # not over a headless SSH session)


# ------------------------------------------------------------------
# DETECTION
# ------------------------------------------------------------------

def detect_green(frame):
    """Returns True if a green object above MIN_AREA is present."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    # Clean up noise: erode then dilate to remove small false positives
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, mask, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area >= MIN_AREA:
        return True, mask, largest

    return False, mask, None


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"Could not open camera at index {CAMERA_INDEX}.")
        print("Try a different CAMERA_INDEX (1, 2, ...) or check 'ls /dev/video*'.")
        sys.exit(1)

    print("Green object detection running. Ctrl+C to stop.")

    last_detection_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                time.sleep(0.1)
                continue

            found, mask, contour = detect_green(frame)
            now = time.time()

            if found and (now - last_detection_time) >= COOLDOWN_SECONDS:
                print("green detected")
                last_detection_time = now

            if SHOW_PREVIEW:
                display = frame.copy()
                if contour is not None:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Camera", display)
                cv2.imshow("Green Mask", mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
