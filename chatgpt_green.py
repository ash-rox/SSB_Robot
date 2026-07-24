#!/usr/bin/env python3

import cv2
import numpy as np
from sparkybotmini import SparkyBotMini

# Open the USB camera (0 is usually the first camera)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Looking for green... Press 'q' to quit.")

green_detected = False

while True:
    ret, frame = camera.read()

    if not ret:
        break

    # Convert image to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV range for green
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([85, 255, 255])

    # Create mask
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Remove small blobs
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Count green pixels
    green_pixels = cv2.countNonZero(mask)

    # Only print once until green disappears
    if green_pixels > 2000:
        if not green_detected:
            print("Green detected")
            green_detected = True
    else:
        green_detected = False

    # Optional display
    cv2.imshow("Camera", frame)
    cv2.imshow("Green Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
