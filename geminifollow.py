#!/usr/bin/env python3
# coding: utf-8
"""
SparkyBotMini High-FPS Line Follower (X-Pattern Omni Drive)
Features:
- Threaded camera stream to avoid I/O bottlenecks.
- ROI cropping prior to color conversion.
- Holonomic PID control (strafing + heading correction) for X-Omni chassis.
"""

import sys
import time
import threading
import cv2
import numpy as np

from sparkybotmini import SparkyBotMini

# ===================== CONFIGURATION =====================

SERIAL_PORT = "/dev/ttyUSB0"
CAMERA_INDEX = 0
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
TARGET_FPS = 60

# Vision
ROI_TOP_FRAC = 0.60      # Process bottom 40% of the frame
THRESH_VAL = 70          # Brightness cutoff for line vs floor
INVERT_THRESHOLD = True  # True = dark line on light floor, False = light line
MIN_LINE_AREA = 250      # Ignore small noise specs
LOST_LINE_FRAMES = 15    # Stop after consecutive lost frames

# Motion (X-Omni Drive)
BASE_SPEED_X = 20        # Forward speed (Vx)
MAX_TURN = 30            # Max rotational speed limit (omega)
MAX_STRAFE = 15          # Max lateral strafe limit (Vy)

# PID Gains for Turning (Heading Control)
KP_TURN = 0.22
KI_TURN = 0.001
KD_TURN = 0.006

# Proportional Gain for Lateral Strafing (Sideways Slide)
KP_STRAFE = 0.10

INTEGRAL_LIMIT = 300
SHOW_DEBUG = True        # Set False to run headless at maximum FPS

# =========================================================


class WebcamStream:
    """Threaded camera capture class to eliminate cap.read() latency."""

    def __init__(self, src=0, width=320, height=240, fps=60):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.stream.set(cv2.CAP_PROP_FPS, fps)
        
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            if not grabbed:
                self.stop()
                break
            self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.stream.release()


class PID:
    """PID Controller with dt timing and integral clamping."""

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
        self._prev_time = time.time()

    def compute(self, error):
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now

        if dt <= 0.0:
            return 0.0

        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))

        derivative = (error - self._prev_error) / dt
        self._prev_error = error

        output = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)
        return max(-self.output_limit, min(self.output_limit, output))


def find_line_center_fast(frame, roi_top_frac, thresh_val, invert, min_area):
    """Fast line centroid detection by cropping before color conversion."""
    h, w = frame.shape[:2]
    roi_y0 = int(h * roi_top_frac)
    
    # 1. Crop ROI FIRST to reduce pixels processed
    roi_color = frame[roi_y0:h, 0:w]

    # 2. Convert ROI to Grayscale
    gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)

    # 3. Fast Thresholding
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, mask = cv2.threshold(gray, thresh_val, 255, thresh_type)

    # 4. Find Contours
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
    # Initialize Robot Connection
    robot = SparkyBotMini(port=SERIAL_PORT, debug=False)
    if not robot.connect():
        print("Failed to connect to SparkyBotMini.")
        return 1

    # Start Threaded Camera Stream
    vs = WebcamStream(src=CAMERA_INDEX, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=TARGET_FPS).start()
    time.sleep(0.5)  # Warmup camera

    pid_turn = PID(KP_TURN, KI_TURN, KD_TURN, output_limit=MAX_TURN, integral_limit=INTEGRAL_LIMIT)
    lost_count = 0

    fps_counter = 0
    fps_start_time = time.time()

    print("X-Omni High-FPS Line Follower running. Press 'q' to stop.")

    try:
        while True:
            frame = vs.read()
            if frame is None:
                continue

            h, w = frame.shape[:2]
            center_x = w // 2

            # Image processing
            cx, roi_y0, mask = find_line_center_fast(
                frame, ROI_TOP_FRAC, THRESH_VAL, INVERT_THRESHOLD, MIN_LINE_AREA
            )

            if cx is None:
                lost_count += 1
                if lost_count >= LOST_LINE_FRAMES:
                    robot.set_motor(0, 0, 0, 0)
                    pid_turn.reset()
            else:
                lost_count = 0
                error = cx - center_x  # Offset distance

                # Calculate turning correction (yaw rotation)
                turn = pid_turn.compute(error)

                # Calculate proportional strafe correction (sideways movement)
                strafe = max(-MAX_STRAFE, min(MAX_STRAFE, error * KP_STRAFE))

                # X-PATTERN OMNI KINEMATICS
                # Vx = BASE_SPEED_X (Forward)
                # Vy = strafe       (Lateral)
                # w  = turn         (Rotation)
                
                m1_fl = BASE_SPEED_X + strafe + turn   # Front Left
                m2_bl = BASE_SPEED_X - strafe + turn   # Back Left
                m3_fr = BASE_SPEED_X - strafe - turn   # Front Right
                m4_br = BASE_SPEED_X + strafe - turn   # Back Right

                # Send commands to motors
                robot.set_motor(
                    int(np.clip(m1_fl, -100, 100)),
                    int(np.clip(m2_bl, -100, 100)),
                    int(np.clip(m3_fr, -100, 100)),
                    int(np.clip(m4_br, -100, 100))
                )

            # Measure & Display FPS
            fps_counter += 1
            if time.time() - fps_
