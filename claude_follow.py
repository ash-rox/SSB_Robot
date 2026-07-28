#!/usr/bin/env python3
# coding: utf-8
"""
SparkyBotMini Line Follower
Uses a front-mounted, downward-angled USB camera + PID control to keep
the robot centered on a line on the ground.

Wiring/assumption notes:
- Camera mounted front-center, ~45 deg down toward the ground.
- Motor mapping assumed: m1 = front-left, m2 = front-right,
  m3 = rear-left, m4 = rear-right (standard for this chassis).
  If the robot turns the WRONG way, swap the sign of `turn` below,
  or swap which motors get + vs - turn.
- Assumes a DARK line on a LIGHTER floor. If it's the reverse
  (light line, dark floor), set INVERT_THRESHOLD = False.

Place this script in the same folder as your sparkybotmini.py library
(rename the uploaded file back to sparkybotmini.py), then run:
    python3 line_follower.py
Press 'q' in the video window to quit (if SHOW_DEBUG is True), or
Ctrl+C from the terminal.
"""

import sys
import time

import cv2
import numpy as np

from sparkybotmini import SparkyBotMini

# ===================== CONFIG (tune these) =====================

SERIAL_PORT = "/dev/ttyUSB0"
CAMERA_INDEX = 0
FRAME_WIDTH = 320
FRAME_HEIGHT = 240

# Vision
ROI_TOP_FRAC = 0.55      # Only look at the bottom 45% of the frame (closest ground)
THRESH_VAL = 70          # Brightness cutoff for line vs floor (0-255); tune to your lighting
INVERT_THRESHOLD = True  # True = dark line on light floor. False = light line on dark floor.
MIN_LINE_AREA = 300      # Ignore blobs smaller than this (noise)
LOST_LINE_FRAMES = 10    # Stop motors after this many consecutive frames with no line found

# Motion
BASE_SPEED = 25          # Forward speed, -100 to 100 ("slowly" per your request)
MAX_TURN = 30            # Max steering correction applied on top of BASE_SPEED

# PID gains - start conservative, especially at low speed
KP = 0.35
KI = 0.02
KD = 0.15
INTEGRAL_LIMIT = 40

SHOW_DEBUG = True        # Set False if running headless (no monitor attached)

# =================================================================


class PID:
    """Simple PID controller with anti-windup and output clamping."""

    def __init__(self, kp, ki, kd, output_limit, integral_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.reset()

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def compute(self, error):
        now = time.time()
        dt = (now - self._prev_time) if self._prev_time is not None else 0.0
        self._prev_time = now

        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))

        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)
        return max(-self.output_limit, min(self.output_limit, output))


def find_line_center(frame, roi_top_frac, thresh_val, invert, min_area):
    """
    Locate the horizontal center (in full-frame pixel coords) of the
    largest line-like blob in the bottom portion of the frame.

    Returns (cx, roi_y0, mask) where cx is None if no line was found.
    """
    h, w = frame.shape[:2]
    roi_y0 = int(h * roi_top_frac)
    roi = frame[roi_y0:h, 0:w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(blurred, thresh_val, 255, thresh_type)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, roi_y0, mask

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None, roi_y0, mask

    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return None, roi_y0, mask

    cx = int(moments["m10"] / moments["m00"])
    return cx, roi_y0, mask


def main():
    robot = SparkyBotMini(port=SERIAL_PORT, debug=False)
    if not robot.connect():
        print("Failed to connect to robot. Check SERIAL_PORT.")
        return 1

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("Failed to open camera. Check CAMERA_INDEX.")
        robot.disconnect()
        return 1

    pid = PID(KP, KI, KD, output_limit=MAX_TURN, integral_limit=INTEGRAL_LIMIT)
    lost_count = 0

    print("Line follower running. Ctrl+C to stop.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed, stopping.")
                break

            h, w = frame.shape[:2]
            center_x = w // 2

            cx, roi_y0, mask = find_line_center(
                frame, ROI_TOP_FRAC, THRESH_VAL, INVERT_THRESHOLD, MIN_LINE_AREA
            )

            if cx is None:
                lost_count += 1
                if lost_count >= LOST_LINE_FRAMES:
                    robot.set_motor(0, 0, 0, 0)
                    pid.reset()
            else:
                lost_count = 0
                # Positive error = line is right of center -> robot should turn right
                error = cx - center_x
                turn = pid.compute(error)

                left_speed = max(-100, min(100, BASE_SPEED + turn))
                right_speed = max(-100, min(100, BASE_SPEED - turn))

                # m1/m3 = left side, m2/m4 = right side (see notes at top of file)
                robot.set_motor(int(left_speed), int(right_speed),
                                 int(left_speed), int(right_speed))

            if SHOW_DEBUG:
                debug_frame = frame.copy()
                cv2.line(debug_frame, (center_x, 0), (center_x, h), (255, 0, 0), 1)
                if cx is not None:
                    cv2.circle(debug_frame, (cx, roi_y0 + 10), 5, (0, 0, 255), -1)
                cv2.rectangle(debug_frame, (0, roi_y0), (w, h), (0, 255, 0), 1)
                cv2.imshow("camera", debug_frame)
                cv2.imshow("mask", mask)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        robot.set_motor(0, 0, 0, 0)
        robot.disconnect()
        cap.release()
        if SHOW_DEBUG:
            cv2.destroyAllWindows()
        print("Stopped cleanly.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
