#!/usr/bin/env python3
# coding: utf-8
"""
SparkyBotMini - Dual-PID X-Omni Line Follower (Sharp 90-Degree Turn Support)
Features memory-based corner recovery and fast camera frame clearing.
"""

import sys
import time
import cv2
import numpy as np
from sparkybotmini import SparkyBotMini

# ===================== CONFIGURATION =====================
PORT = "/dev/ttyUSB0"
CAM_IDX = 0
WIDTH, HEIGHT = 320, 240

# Vision Settings
ROI_TOP = 0.50          # Slightly larger ROI (bottom 50%) to catch sharp turns early
THRESH_VAL = 70         # Set 0-255, or -1 for Otsu
INVERT_COLOR = True     # True = dark line on light floor
MIN_AREA = 250          

# Motion Settings
BASE_SPEED = 16         # Slightly lower base speed makes 90-degree corners far easier
INVERT_STEERING = False

# PID Gains
KP_TURN = 0.18
KI_TURN = 0.0005
KD_TURN = 0.006
MAX_TURN = 30.0

KP_STRAFE = 0.06
KI_STRAFE = 0.0002
KD_STRAFE = 0.003
MAX_STRAFE = 18.0

# Sharp Corner Recovery
PIVOT_SPEED = 25        # Spin speed when performing a corner recovery pivot
LOST_LINE_FRAMES = 45   # ~1.0 sec before full emergency stop
SHOW_DEBUG = True
# =========================================================


class PID:
    def __init__(self, kp, ki, kd, output_limit, integral_limit=150.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = max(now - self.prev_time, 1e-4)
        self.prev_time = now

        if self.prev_error is None:
            self.prev_error = error

        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral + error * dt))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        return max(-self.output_limit, min(self.output_limit, output))


class LineFollower:
    def __init__(self):
        self.pid_turn = PID(KP_TURN, KI_TURN, KD_TURN, MAX_TURN)
        self.pid_strafe = PID(KP_STRAFE, KI_STRAFE, KD_STRAFE, MAX_STRAFE)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.last_valid_x = WIDTH // 2  # Remembers last seen X coordinate

    def reset_pids(self):
        self.pid_turn.reset()
        self.pid_strafe.reset()
        self.last_valid_x = WIDTH // 2

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        roi_y0 = int(h * ROI_TOP)
        
        # 1. Crop ROI
        gray_roi = cv2.cvtColor(frame[roi_y0:h, :], cv2.COLOR_BGR2GRAY)

        # 2. Thresholding
        if THRESH_VAL < 0:
            flag = (cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) if INVERT_COLOR else (cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            _, mask = cv2.threshold(gray_roi, 0, 255, flag)
        else:
            flag = cv2.THRESH_BINARY_INV if INVERT_COLOR else cv2.THRESH_BINARY
            _, mask = cv2.threshold(gray_roi, THRESH_VAL, 255, flag)

        # 3. Clean up noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

        # 4. Find line centroid
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) >= MIN_AREA:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    self.last_valid_x = cx  # Save position memory
                    return cx, roi_y0, mask

        return None, roi_y0, mask


def main():
    robot = SparkyBotMini(port=PORT, debug=False)
    if not robot.connect():
        return 1

    cap = cv2.VideoCapture(CAM_IDX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    
    # Speed up camera frame rate to reduce motion blur
    cap.set(cv2.CAP_PROP_FPS, 60)

    follower = LineFollower()
    lost_count = 0

    time.sleep(0.5)

    try:
        while cap.isOpened():
            # Flush camera buffer to prevent stale video frames
            cap.grab()
            ret, frame = cap.retrieve()
            if not ret:
                break

            w = frame.shape[1]
            cx, roi_y0, mask = follower.process_frame(frame)

            if cx is not None:
                lost_count = 0
                error = cx - (w // 2)

                if INVERT_STEERING:
                    error = -error

                turn = follower.pid_turn.compute(error)
                strafe = follower.pid_strafe.compute(error)

                # X-Pattern Omni Kinematics
                m1 = BASE_SPEED - strafe - turn
                m2 = BASE_SPEED + strafe - turn
                m3 = BASE_SPEED + strafe + turn
                m4 = BASE_SPEED - strafe + turn

                robot.set_motor(
                    int(np.clip(m1, -100, 100)),
                    int(np.clip(m2, -100, 100)),
                    int(np.clip(m3, -100, 100)),
                    int(np.clip(m4, -100, 100))
                )
            else:
                lost_count += 1
                
                # --- RECOVERY LOGIC FOR SHARP 90-DEGREE CORNERS ---
                if lost_count < LOST_LINE_FRAMES:
                    # Determine which direction the line vanished towards
                    if follower.last_valid_x < (w // 2):
                        # Line vanished to the LEFT -> Spin Left in place
                        robot.set_motor(-PIVOT_SPEED, -PIVOT_SPEED, PIVOT_SPEED, PIVOT_SPEED)
                    else:
                        # Line vanished to the RIGHT -> Spin Right in place
                        robot.set_motor(PIVOT_SPEED, PIVOT_SPEED, -PIVOT_SPEED, -PIVOT_SPEED)
                else:
                    # Full stop after grace period expires
                    robot.set_motor(0, 0, 0, 0)
                    follower.reset_pids()

            if SHOW_DEBUG:
                debug_img = frame.copy()
                if cx is not None:
                    cv2.circle(debug_img, (cx, roi_y0 + 15), 5, (0, 0, 255), -1)
                else:
                    cv2.putText(debug_img, "CORNER SEARCH", (20, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                cv2.line(debug_img, (w // 2, 0), (w // 2, frame.shape[0]), (255, 0, 0), 1)
                cv2.rectangle(debug_img, (0, roi_y0), (w, frame.shape[0]), (0, 255, 0), 1)

                cv2.imshow("Camera View", debug_img)
                cv2.imshow("Mask ROI", mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        robot.set_motor(0, 0, 0, 0)
        robot.disconnect()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
