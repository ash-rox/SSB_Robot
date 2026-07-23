#!/usr/bin/env python3
import cv2
import sys
import time
from sparkybotmini import SparkyBotMini

# Base driving speed magnitude (0 to 100)
SPEED = 60

def move_robot(robot, vx: float, vy: float):
    """Calculates and sets the X-formation omni wheel motor speeds."""
    # X-formation inverse kinematics equations
    m1 = vy + vx  # Front Left Motor
    m2 = vy - vx  # Front Right Motor
    m3 = vy - vx  # Rear Left Motor
    m4 = vy + vx  # Rear Right Motor

    # Proportional scaling to ensure safety range limits inside [-100, 100]
    max_val = max(abs(m1), abs(m2), abs(m3), abs(m4))
    if max_val > 100.0:
        scale = 100.0 / max_val
        m1 *= scale; m2 *= scale; m3 *= scale; m4 *= scale

    robot.set_motor(int(m1), int(m2), int(m3), int(m4))

def main():
    # Initialize SparkyBot Mini hardware
    bot = SparkyBotMini(port="/dev/ttyUSB0", debug=False)
    if not bot.connect():
        print("Error: Could not connect to SparkyBotMini hardware.")
        sys.exit(1)

    # Initialize USB Camera index 0 (default front camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open USB camera.")
        bot.disconnect()
        sys.exit(1)

    print("\n=== SparkyBot Mini Live Camera Controller ===")
    print("  CONTROLS (Click on the Video Window to focus):")
    print("  U (Diag UL)   W (Forward)   Y (Diag UR)")
    print("  A (Strafe L)  S (Backward)  D (Strafe R)")
    print("  J (Diag DL)                 H (Diag DR)")
    print("  ")
    print("  [SPACEBAR] : Take Screenshot")
    print("  [ESC KEY]  : Quit Program")
    print("=============================================\n")

    try:
        while True:
            # Capture frame-by-frame from USB feed
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab camera frame.")
                break

            # Overlay control reminder on the actual video window feed
            cv2.putText(frame, "Space: Screenshot | ESC: Quit", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the video window stream
            cv2.imshow("SparkyBot Mini Front View", frame)

            # cv2.waitKey(30) waits 30ms for a keypress and returns its ASCII value
            key = cv2.waitKey(30) & 0xFF

            # --- STRAIGHT MOVEMENT CONTROLS ---
            if key == ord('w'):       # Forward
                move_robot(bot, vx=0, vy=SPEED)
            elif key == ord('s'):     # Backward
                move_robot(bot, vx=0, vy=-SPEED)
            elif key == ord('a'):     # Strafe Left
                move_robot(bot, vx=-SPEED, vy=0)
            elif key == ord('d'):     # Strafe Right
                move_robot(bot, vx=SPEED, vy=0)
            
            # --- DIAGONAL MOVEMENT CONTROLS ---
            elif key == ord('u'):     # Diagonal Up-Left
                move_robot(bot, vx=-SPEED, vy=SPEED)
            elif key == ord('y'):     # Diagonal Up-Right
                move_robot(bot, vx=SPEED, vy=SPEED)
            elif key == ord('j'):     # Diagonal Back-Left
                move_robot(bot, vx=-SPEED, vy=-SPEED)
            elif key == ord('h'):     # Diagonal Back-Right
                move_robot(bot, vx=SPEED, vy=-SPEED)
            
            # --- CAMERA ACTIONS ---
            elif key == 32:           # Spacebar code for screenshot
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"sparky_snap_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f" Saved screenshot: {filename}")
            
            # --- QUIT APPLICATION ---
            elif key == 27:           # ESC key to safely exit
                print("\nShutting down controller application.")
                break
                
            else:
                # Automatically brake the motors if no active direction key is held down
                move_robot(bot, vx=0, vy=0)

    except KeyboardInterrupt:
        print("\nEmergency stop triggered via terminal.")
    finally:
        # Clean up steps
        move_robot(bot, vx=0, vy=0)
        bot.set_motor(0, 0, 0, 0)
        bot.disconnect()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
