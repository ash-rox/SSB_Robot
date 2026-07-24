#!/usr/bin/env python3

import cv2
import numpy as np
from sparkybotmini import SparkyBotMini

# --------------------
# Connect to robot
# --------------------
robot = SparkyBotMini("/dev/ttyUSB0")

if not robot.connect():
    print("Could not connect to robot.")
    exit()

camera = cv2.VideoCapture(0)

# Speed settings
FORWARD_SPEED = 30
TURN_SPEED = 20
STOP_SPEED = 0

# Minimum object size before stopping
TARGET_AREA = 30000

while True:

    ret, frame = camera.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 60, 60])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 500:

            x, y, w, h = cv2.boundingRect(largest)

            cx = x + w // 2
            cy = y + h // 2

            frame_center = frame.shape[1] // 2

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.circle(frame,(cx,cy),5,(0,0,255),-1)

            print("Green detected")

            # -------------------------
            # Steering
            # -------------------------

            error = cx - frame_center

            if area > TARGET_AREA:
                # Close enough
                robot.set_motor(0,0,0,0)

            elif error < -40:
                # Turn left
                robot.set_motor(-TURN_SPEED,
                                -TURN_SPEED,
                                 TURN_SPEED,
                                 TURN_SPEED)

            elif error > 40:
                # Turn right
                robot.set_motor(TURN_SPEED,
                                TURN_SPEED,
                               -TURN_SPEED,
                               -TURN_SPEED)

            else:
                # Drive straight
                robot.set_motor(FORWARD_SPEED,
                                FORWARD_SPEED,
                                FORWARD_SPEED,
                                FORWARD_SPEED)

    else:
        # Lost target
        robot.set_motor(0,0,0,0)

    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == ord('q'):
        break

robot.set_motor(0,0,0,0)
camera.release()
robot.disconnect()
cv2.destroyAllWindows()
