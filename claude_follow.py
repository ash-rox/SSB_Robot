#!/usr/bin/env python3
# coding: utf-8
"""
SparkyBotMini Line Follower
Uses an OpenCV video feed to find a line on the ground and a PID controller
to steer the robot so it stays centered on that line.

Camera assumption: mounted front-center, angled ~45 degrees down toward the
ground, so the line appears in the lower-middle portion of the frame.

Hardware assumption: 4-wheeled skid-steer drive, where:
    M1 = front-left,  M2 = front-right
    M3 = rear-left,   M4 = rear-right
If your wheels are wired differently, swap the assignments in
`drive(left_speed, right_speed)` below.
"""

import cv2
import time
import numpy as np

from sparkybotmini import SparkyBotMini


# ===================== PID Controller =====================

class PID:
    """Simple PID controller with output clamping and anti-windup."""

    def __init__(self, kp: float, ki: float, kd: float,
                 output_limit: float = 100.0, integral_limit: float = 50.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error: float) -> float:
        now = time.time()
        dt = (now - self._prev_time) if self._prev_time is not None else 0.0
        self._prev_time = now

        # Integral term (with clamping to avoid windup)
        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))

        # Derivative term (guard against divide-by-zero on first call)
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = (self.kp * error) + (self.ki * self._integral) + (self.kd * derivative)
        return max(-self.output_limit, min(self.output_limit, output))


# ===================== Line Detection =====================

class LineDetector:
    """
    Finds a dark line on a lighter floor within a horizontal band of the frame.
    Returns the horizontal offset of the line's centroid from the frame center,
    in pixels (positive = line is to the right of center).
    """

    def __init__(self, roi_top_ratio: float = 0.55, roi_bottom_ratio: float = 0.9,
                 min_contour_area: int = 500, invert: bool = False):
        """
        Args:
            roi_top_ratio / roi_bottom_ratio: vertical band of the frame to
                scan, as a fraction of frame height (0=top, 1=bottom). Given
                the 45-degree downward camera angle, the line will typically
                fall somewhere in the lower half of the frame.
            min_contour_area: ignore contours smaller than this (noise)
            invert: set True if your line is LIGHT on a DARK floor
        """
        self.roi_top_ratio = roi_top_ratio
        self.roi_bottom_ratio = roi_bottom_ratio
        self.min_contour_area = min_contour_area
        self.invert = invert

    def find_line_offset(self, frame):
        """
        Returns:
            (offset_px, debug_frame) where offset_px is None if no line found.
        """
        h, w = frame.shape[:2]
        y1 = int(h * self.roi_top_ratio)
        y2 = int(h * self.roi_bottom_ratio)
        roi = frame[y1:y2, 0:w]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh_type = cv2.THRESH_BINARY_INV if not self.invert else cv2.THRESH_BINARY
        _, thresh = cv2.threshold(blurred, 0, 255, thresh_type + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        offset_px = None
        debug_frame = frame.copy()
        cv2.rectangle(debug_frame, (0, y1), (w, y2), (255, 0, 0), 1)
        cv2.line(debug_frame, (w // 2, 0), (w // 2, h), (0, 255, 255), 1)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= self.min_contour_area:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    offset_px = cx - (w // 2)

                    # Draw debug info (offset back into full-frame coordinates)
                    cv2.drawContours(debug_frame, [largest], -1, (0, 255, 0), 2, offset=(0, y1))
                    cv2.circle(debug_frame, (cx, cy + y1), 5, (0, 0, 255), -1)

        return offset_px, debug_frame


# ===================== Line Follower App =====================

class LineFollower:
    def __init__(self, port: str = "/dev/ttyUSB0", camera_index: int = 0,
                 base_speed: int = 25, max_turn: int = 40,
                 show_video: bool = True, debug: bool = False):
        self.robot = SparkyBotMini(port=port, debug=debug)
        self.camera_index = camera_index
        self.base_speed = base_speed
        self.max_turn = max_turn
        self.show_video = show_video

        self.detector = LineDetector()
        # Start conservative; tune these for your floor/lighting/speed.
        self.pid = PID(kp=0.35, ki=0.0, kd=0.15, output_limit=max_turn)

        self.cap = None
        self._lost_line_frames = 0
        self._max_lost_frames = 15  # ~0.5s at 30fps before stopping

    def connect(self) -> bool:
        if not self.robot.connect():
            return False

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            print("? Could not open camera")
            self.robot.disconnect()
            return False

        return True

    def drive(self, left_speed: float, right_speed: float):
        """Map left/right speeds to the 4 motors (skid-steer)."""
        l = int(max(-100, min(100, left_speed)))
        r = int(max(-100, min(100, right_speed)))
        # M1=front-left, M2=front-right, M3=rear-left, M4=rear-right
        self.robot.set_motor(l, r, l, r)

    def stop(self):
        self.robot.set_motor(0, 0, 0, 0)

    def run(self):
        print("? Line following started. Press 'q' in the video window (or Ctrl+C) to stop.")
        self.pid.reset()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("? Camera read failed")
                    break

                offset_px, debug_frame = self.detector.find_line_offset(frame)

                if offset_px is not None:
                    self._lost_line_frames = 0

                    # Normalize offset to roughly [-1, 1] using frame half-width
                    half_width = frame.shape[1] / 2.0
                    normalized_error = offset_px / half_width

                    turn = self.pid.update(normalized_error)

                    left_speed = self.base_speed + turn
                    right_speed = self.base_speed - turn
                    self.drive(left_speed, right_speed)

                    cv2.putText(debug_frame, f"offset={offset_px}px turn={turn:.1f}",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    self._lost_line_frames += 1
                    if self._lost_line_frames >= self._max_lost_frames:
                        self.stop()
                        cv2.putText(debug_frame, "LINE LOST - stopped",
                                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    # else: keep coasting on last command briefly rather than
                    # jerking to a stop on a single dropped frame

                if self.show_video:
                    cv2.imshow("SparkyBot Line Follower", debug_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        except KeyboardInterrupt:
            print("\n??  Interrupted by user")

        finally:
            self.shutdown()

    def shutdown(self):
        self.stop()
        if self.cap:
            self.cap.release()
        if self.show_video:
            cv2.destroyAllWindows()
        self.robot.disconnect()
        print("? Stopped and disconnected")


# ===================== Entry Point =====================

if __name__ == "__main__":
    follower = LineFollower(
        port="/dev/ttyUSB0",
        camera_index=0,
        base_speed=25,     # forward speed while following (tune to your floor/robot)
        max_turn=40,       # max steering correction added/subtracted per side
        show_video=True,   # set False for headless operation (e.g. SSH without a display)
        debug=False,
    )

    if follower.connect():
        follower.run()
    else:
        print("? Failed to start line follower")
