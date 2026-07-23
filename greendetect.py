#!/usr/bin/env python3
"""
Green Object Detection - USB Camera
=====================================

Watches the USB camera feed and prints "green detected" whenever a
green-ish object appears in front of the robot, with a 0.5 second
cooldown between messages (so a continuously-visible green object
doesn't spam the console).

Runs indefinitely until you press the SPACEBAR (works over SSH,
headless - no mouse/window focus required).

Detection approach:
  1. Grab a frame from the camera.
  2. Convert it to HSV color space (much easier to isolate a color
     than in RGB, since hue is separated from brightness/lighting).
  3. Threshold the frame to a mask of "is this pixel green enough".
     A wide hue/saturation/value range is used on purpose to give
     some tolerance for imperfect, uneven, or dull greens rather than
     needing a pure, saturated green.
  4. Clean up the mask (remove small noise specks).
  5. If the green area is bigger than MIN_AREA, count it as a
     detection.

Install dependencies:
    pip install opencv-python numpy --break-system-packages

Run with:  python3 green_detector.py
Stop with: SPACEBAR (or Ctrl+C as a backup)
"""

import time
import sys
import termios
import tty
import select

try:
    import cv2
    import numpy as np
except ImportError:
    print("Missing dependency. Install with:")
    print("  pip install opencv-python numpy --break-system-packages")
    sys.exit(1)

# ------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------

CAMERA_INDEX = 0          # 0 is usually the first/only USB camera
FRAME_WIDTH = 320          # lower resolution = faster processing
FRAME_HEIGHT = 240

# HSV green range, widened to give "wiggle room" for imperfect,
# dull, shadowed, or slightly-off greens rather than needing a pure,
# fully-saturated green. OpenCV HSV ranges: H 0-179, S 0-255, V 0-255
#
#   - Hue range widened (35-85) to include yellow-green through teal
#   - Saturation floor lowered (40) to catch washed-out/dull greens
#   - Value floor lowered (40) to catch greens in dimmer lighting
#
# If you still get misses or false positives, tune these further -
# lower the floors more for extra tolerance, or raise them back up if
# it starts falsely triggering on background clutter.
GREEN_LOWER = np.array([35, 40, 40])
GREEN_UPPER = np.array([85, 255, 255])

MIN_AREA = 800             # minimum pixel area to count as "detected"
                           # (raise this if small green specs of noise
                           # falsely trigger detection; lower it to
                           # detect smaller/farther-away objects)

COOLDOWN_SECONDS = 0.5

SHOW_PREVIEW = False       # set True to display a debug window
                           # (only works if you have a display attached,
                           # not over a headless SSH session)


# ------------------------------------------------------------------
# NON-BLOCKING KEYBOARD CHECK (for spacebar-to-quit over SSH/headless)
# ------------------------------------------------------------------

def key_waiting():
    """Returns True if a key is waiting to be read from stdin."""
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    return bool(ready)


def read_key():
    """Reads a single waiting key (non-blocking; call key_waiting() first)."""
    return sys.stdin.read(1)


# ------------------------------------------------------------------
# DETECTION
# ------------------------------------------------------------------

def detect_green(frame):
    """Returns True if a green object above MIN_AREA is present."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)

    # Clean up noise: erode then dilate to remove small false positives
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, mask, None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area >= MIN_AREA:
        return True, mask, largest

    return False, mask, None


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"Could not open camera at index {CAMERA_INDEX}.")
        print("Try a different CAMERA_INDEX (1, 2, ...) or check 'ls /dev/video*'.")
        sys.exit(1)

    print("Green object detection running. Press SPACEBAR to stop (Ctrl+C also works).")

    last_detection_time = 0.0

    # Put the terminal into raw mode so we can catch a spacebar press
    # without the user needing to hit Enter afterward. Restored in
    # `finally` no matter how the loop exits.
    stdin_fd = sys.stdin.fileno()
    old_terminal_settings = termios.tcgetattr(stdin_fd)
    tty.setraw(stdin_fd)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                time.sleep(0.1)
                continue

            found, mask, contour = detect_green(frame)
            now = time.time()

            if found and (now - last_detection_time) >= COOLDOWN_SECONDS:
                print("green detected\r")
                last_detection_time = now

            if SHOW_PREVIEW:
                display = frame.copy()
                if contour is not None:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("Camera", display)
                cv2.imshow("Green Mask", mask)
                # When a preview window is open, it - not the terminal -
                # has keyboard focus, so also check for space there.
                if cv2.waitKey(1) & 0xFF == ord(' '):
                    break

            # Non-blocking check for a spacebar press in the terminal
            if key_waiting():
                key = read_key()
                if key == ' ':
                    print("\r\nSpacebar pressed - stopping.")
                    break

    except KeyboardInterrupt:
        print("\r\nStopping.")
    finally:
        termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_terminal_settings)
        cap.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
