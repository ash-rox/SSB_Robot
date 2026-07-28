#!/usr/bin/env python3
# coding: utf-8
"""
SparkyBotMini - Refined X-Omni Line Follower
Features: Compact PID control, ROI-first vision processing, holonomic kinematics.
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
ROI_TOP = 0.60         # Look at bottom 40% of image
THRESH_VAL = 70        # Set 0-255, or -1 for Otsu Automatic Thresholding
INVERT = True          # True = dark line on light floor; False = light line
MIN_AREA = 200         # Min pixel contour area

# Motion & Kinematics (X-Pattern Omni)
BASE_SPEED = 20        # Forward speed (Vx)
KP, KI, KD = 0.20, 0.001, 0.005  # Turning PID Gains
KP_STRAFE = 0.08       # Proportional Lateral Strafe (Vy)
MAX_TURN = 30
MAX_STRAFE = 15

SHOW_DEBUG = True
# =========================================================


class LineFollower:
    def __init__(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    def pid_compute(self, error):
        now = time.time()
        dt = max(now - self.prev_time, 1e-4)
        self.prev_time = now

        # Integral with windup clamping
        self.integral = max(-200.0, min(200.0, self.integral + error * dt))
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = (KP * error) + (KI * self.integral) + (KD * derivative)
        return max(-MAX_TURN, min(MAX_TURN, output))

    def process_frame(self, frame):
        """Direct ROI cropping and noise filtering for maximum FPS."""
        h, w = frame.shape[:2]
        roi_y0 = int(h * ROI_TOP)
        
        # 1. Crop ROI first, then convert color
        gray_roi = cv2.cvtColor(frame[roi_y0:h, :], cv2.COLOR_BGR2GRAY)

        # 2. Fast Thresholding (Otsu Auto or Manual Cutoff)
        if THRESH_VAL < 0:
            flag = (cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) if INVERT else (cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            _, mask = cv2.threshold(gray_roi, 0, 255, flag)
        else:
            flag = cv2.THRESH_BINARY_INV if INVERT else cv2.THRESH_BINARY
            _, mask = cv2.threshold(gray_roi, THRESH_VAL, 255, flag)

        # 3. Morphological opening to remove small noise dots
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

        # 4. Find line centroid
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) >= MIN_AREA:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    return int(M["m10"] / M["m00"]), roi_y0, mask

        return None, roi_y0, mask


def main():
    robot = SparkyBotMini(port=PORT, debug=False)
    if not robot.connect():
        return 1

    cap = cv2.VideoCapture(CAM_IDX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    follower = LineFollower()
    lost_count = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            w = frame.shape[1]
            cx, roi_y0, mask = follower.process_frame(frame)

            if cx is not None:
                lost_count = 0
                error = cx - (w // 2)

                # Corrections: Rotation (PID) & Strafe (P)
                turn = follower.pid_compute(error)
                strafe = max(-MAX_STRAFE, min(MAX_STRAFE, error * KP_STRAFE))

                # X-Pattern Omni Kinematics [m1: FL, m2: BL, m3: FR, m4: BR]
                m1 = BASE_SPEED + strafe + turn
                m2 = BASE_SPEED - strafe + turn
                m3 = BASE_SPEED - strafe - turn
                m4 = BASE_SPEED + strafe - turn

                robot.set_motor(
                    int(np.clip(m1, -100, 100)),
                    int(np.clip(m2, -100, 100)),
                    int(np.clip(m3, -100, 100)),
                    int(np.clip(m4, -100, 100))
                )
            else:
                lost_count += 1
                if lost_count > 10:
                    robot.set_motor(0, 0, 0, 0)

            # Optional Debug Render
            if SHOW_DEBUG:
                if cx is not None:
                    cv2.circle(frame, (cx, roi_y0 + 15), 5, (0, 0, 255), -1)
                cv2.line(frame, (w // 2, 0), (w // 2, frame.shape[0]), (255, 0, 0), 1)
                cv2.imshow("Camera", frame)
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
