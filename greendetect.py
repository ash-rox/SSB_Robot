#!/usr/bin/env python3
import cv2
import numpy as np
import sys
import time
from sparky_bot import SparkyBotMini

# --- CONFIGURATION ---
SPEED = 60                 # Base driving speed magnitude (0 to 100)
MIN_AREA = 800             # Minimum pixel area to count as a detection
COOLDOWN_SECONDS = 0.5     # Cooldown to prevent console spamming

# Wide HSV range for general, dull, shadow-casting, or imperfect greens
GREEN_LOWER = np.array([35, 40, 40])
GREEN_UPPER = np.array([85, 255, 255])


def move_robot(robot, vx: float, vy: float):
    """Calculates and sets the X-formation omni wheel motor speeds."""
    m1 = vy + vx  # Front Left Motor
    m2 = vy - vx  # Front Right Motor
    m3 = vy - vx  # Rear Left Motor
    m4 = vy + vx  # Rear Right Motor

    # Proportional scaling to safety boundaries [-100, 100]
    max_val = max(abs(m1), abs(m2), abs(m3), abs(m4))
    if max_val > 100.0:
        scale = 100.0 / max_val
        m1 *= scale; m2 *= scale; m3 *= scale; m4 *= scale

    robot.set_motor(int(m1), int(m2), int(m3), int(m4))


def main():
    # Initialize SparkyBot Mini hardware connection
    bot = SparkyBotMini(port="/dev/ttyUSB0", debug=False)
    if not bot.connect():
        print("Error: Could not connect to SparkyBotMini hardware.")
        sys.exit(1)

    # Initialize USB Camera feed (index 0)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution for Pi performance
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    if not cap.isOpened():
        print("Error: Could not open USB camera.")
        bot.disconnect()
        sys.exit(1)

    print("\n=== SparkyBot Mini Live Vision Controller ===")
    print("  DRIVE CONTROLS (Keep video window focused):")
    print("  U (Diag UL)   W (Forward)   Y (Diag UR)")
    print("  A (Strafe L)  S (Backward)  D (Strafe R)")
    print("  J (Diag DL)                 H (Diag DR)")
    print("  ")
    print("  [SPACEBAR] : Take Screenshot")
    print("  [ESC KEY]  : Quit Program")
    print("=============================================\n")

    last_detection_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab camera frame.")
                break

            # --- GREEN DETECTION ENGINE ---
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

            # Clean up fine noise specks
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            # Find contours inside the thresholded binary mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            green_found = False
            largest_contour = None

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) >= MIN_AREA:
                    green_found = True

            # Print notice to console handling the cooldown window
            now = time.time()
            if green_found and (now - last_detection_time) >= COOLDOWN_SECONDS:
                print("green detected")
                last_detection_time = now

            # --- USER INTERFACE WINDOW ---
            # Visual Feedback: Draw bounding box around green targets
            if green_found and largest_contour is not None:
                x, y, w, h = cv2.boundingRect(largest_contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "GREEN DETECTED", (x, max(y - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display streams
            cv2.imshow("SparkyBot View", frame)
            cv2.imshow("Vision Target Mask", mask)

            # Capture key inputs safely (30ms wait cycle)
            key_code = cv2.waitKey(30) & 0xFF
            key = chr(key_code).lower() if key_code < 256 else ""

            # --- MOVEMENT MAPPING CONTROL ---
            if key == 'w':          # Forward
                move_robot(bot, vx=0, vy=SPEED)
            elif key == 's':        # Backward
                move_robot(bot, vx=0, vy=-SPEED)
            elif key == 'a':        # Strafe Left
                move_robot(bot, vx=-SPEED, vy=0)
            elif key == 'd':        # Strafe Right
                move_robot(bot, vx=SPEED, vy=0)
            elif key == 'u':        # Diagonal Up-Left
                move_robot(bot, vx=-SPEED, vy=SPEED)
            elif key == 'y':        # Diagonal Up-Right
                move_robot(bot, vx=SPEED, vy=SPEED)
            elif key == 'j':        # Diagonal Back-Left
                move_robot(bot, vx=-SPEED, vy=-SPEED)
            elif key == 'h':        # Diagonal Back-Right
                move_robot(bot, vx=SPEED, vy=-SPEED)
            
            # --- INTERACTION CONTROLS ---
            elif key_code == 32:    # Spacebar captures screenshots
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"sparky_snap_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved screenshot: {filename}")
            
            elif key_code == 27:    # ESC safely quits out
                print("\nShutting down controller cleanly.")
                break
                
            else:
                # Active deadman braking when no movement keys are depressed
                move_robot(bot, vx=0, vy=0)

    except KeyboardInterrupt:
        print("\nEmergency termination initiated.")
    finally:
        # Resource cleanup safety sequence
        move_robot(bot, vx=0, vy=0)
        bot.set_motor(0, 0, 0, 0)
        bot.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
