
#!/usr/bin/env python3
import time
import cv2
import numpy as np
from sparkybotmini import SparkyBotMini  # Import your robot library


def clamp(value, min_val=-100, max_val=100):
    """Utility function to ensure motor speeds stay within [-100, 100]."""
    return max(min_val, min(max_val, int(value)))


def main():
    # 1. Connect to the robot
    robot = SparkyBotMini(port="/dev/ttyUSB0", debug=False)
    if not robot.connect():
        print("Error: Could not connect to SparkyBotMini.")
        return

    # 2. Connect to the USB Camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Low resolution for fast processing
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("Error: Could not open USB camera.")
        robot.disconnect()
        return

    # --- Tuning Parameters ---
    BASE_SPEED = 25  # Base forward speed (-100 to 100)
    KP = 0.18        # Proportional gain for steering
    KD = 0.05        # Derivative gain (smooths out turns)
    
    last_error = 0
    
    print("Omni-wheel line following started. Press 'q' on video window to quit.")
    robot.beep(200)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image frame.")
                break

            # --- Image Processing ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Detect dark line on a light surface
            # (Swap cv2.THRESH_BINARY_INV with cv2.THRESH_BINARY if line is white)
            _, thresh = cv2.threshold(blurred, 80, 255, cv2.THRESH_BINARY_INV)

            # Focus on the lower half of the frame (Region of Interest)
            height, width = thresh.shape
            roi = thresh[int(height * 0.5):, :]

            # --- Centroid & Differential Calculation ---
            contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(largest_contour)

                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    frame_center = width // 2

                    error = cx - frame_center

                    # PD Controller output
                    turn_correction = (KP * error) + (KD * (error - last_error))
                    last_error = error

                    # Calculate differential drive wheel speeds
                    left_speed = clamp(BASE_SPEED + turn_correction)
                    right_speed = clamp(BASE_SPEED - turn_correction)

                    # Corrected Motor Order:
                    # m1 = Front Left  | m2 = Back Left
                    # m3 = Front Right | m4 = Back Right
                    robot.set_motor(left_speed, left_speed, right_speed, right_speed)

                    # --- Visual Annotations ---
                    cv2.circle(roi, (cx, roi.shape[0] // 2), 5, (255, 255, 255), -1)
                    cv2.line(roi, (frame_center, 0), (frame_center, roi.shape[0]), (128, 128, 128), 1)

            else:
                print("Line lost! Stopping motors...")
                robot.set_motor(0, 0, 0, 0)
                last_error = 0

            # Show feeds
            cv2.imshow("Camera Feed", frame)
            cv2.imshow("Line Mask (ROI)", roi)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nEmergency interrupt caught.")

    finally:
        print("Stopping robot and releasing resources...")
        robot.set_motor(0, 0, 0, 0)
        robot.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
