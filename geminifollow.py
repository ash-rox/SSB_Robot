#!/usr/bin/env python3
import time
import cv2
import numpy as np
from sparkybotmini import SparkyBotMini  # Import the robot library


def main():
    # 1. Initialize Robot Connection
    # Adjust port for Raspberry Pi (usually /dev/ttyUSB0 or /dev/ttyACM0)
    robot = SparkyBotMini(port="/dev/ttyUSB0", debug=False)
    if not robot.connect():
        print("Error: Could not connect to SparkyBotMini.")
        return

    # 2. Initialize USB Camera
    # Index 0 is standard for the first USB camera on Raspberry Pi
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution for fast processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("Error: Could not open USB camera.")
        robot.disconnect()
        return

    # Steering Parameters
    BASE_SPEED = 30  # Forward speed (0 to 100)
    KP = 0.2         # Proportional control gain (adjust as needed)
    
    print("Line following started. Press 'q' on video window or Ctrl+C to stop.")
    robot.beep(200)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # --- Image Processing ---
            # 1. Convert to Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2. Apply Gaussian Blur to smooth out noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # 3. Thresholding: Detect a DARK line on a LIGHT background
            # Change cv2.THRESH_BINARY_INV to cv2.THRESH_BINARY if line is WHITE on dark floor
            _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

            # 4. Crop ROI (Region of Interest) - Focus on lower half of image
            height, width = thresh.shape
            roi = thresh[int(height * 0.5):, :]

            # --- Centroid & Steering Logic ---
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get the largest contour (assumed to be the line)
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)

                if M["m00"] > 0:
                    # Calculate center (cx) of the detected line contour
                    cx = int(M["m10"] / M["m00"])

                    # Frame center in ROI coordinates
                    frame_center = width // 2

                    # Calculate steering error (negative = line is left, positive = line is right)
                    error = cx - frame_center

                    # Calculate turn adjustment using Proportional (P) control
                    turn = int(KP * error)

                    # Compute motor speeds for left and right wheels
                    left_speed = BASE_SPEED + turn
                    right_speed = BASE_SPEED - turn

                    # Drive the motors (M1/M3 left side, M2/M4 right side)
                    robot.set_motor(left_speed, right_speed, left_speed, right_speed)

                    # --- Visual Feedback ---
                    cv2.circle(roi, (cx, roi.shape[0] // 2), 5, (255, 255, 255), -1)
                    cv2.line(roi, (frame_center, 0), (frame_center, roi.shape[0]), (128, 128, 128), 1)

            else:
                # Line lost: Stop motors or slow spin to search
                print("Line lost! Stopping...")
                robot.set_motor(0, 0, 0, 0)

            # --- Display Windows ---
            cv2.imshow("Camera View", frame)
            cv2.imshow("Binary Mask (ROI)", roi)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")

    finally:
        # --- Safe Cleanup ---
        print("Cleaning up and stopping robot...")
        robot.set_motor(0, 0, 0, 0)
        robot.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
