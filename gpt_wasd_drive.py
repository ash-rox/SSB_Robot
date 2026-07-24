#!/usr/bin/env python3

import keyboard
import time
from sparkybotmini import SparkyBotMini

# -------------------------
# Robot Setup
# -------------------------

robot = SparkyBotMini("/dev/ttyUSB0")

if not robot.connect():
    print("Failed to connect.")
    exit()

SPEED = 40

print("""
======== Robot Controls ========

W - Forward
S - Backward
A - Strafe Left
D - Strafe Right

Q - Rotate Left
E - Rotate Right

SPACE - Stop
ESC - Quit

===============================
""")

current = None

try:

    while True:

        # Quit
        if keyboard.is_pressed('esc'):
            break

        # Forward
        elif keyboard.is_pressed('w'):
            if current != "forward":
                robot.set_motor(SPEED, SPEED, SPEED, SPEED)
                current = "forward"

        # Backward
        elif keyboard.is_pressed('s'):
            if current != "backward":
                robot.set_motor(-SPEED, -SPEED, -SPEED, -SPEED)
                current = "backward"

        # Strafe Left
        elif keyboard.is_pressed('a'):
            if current != "left":
                robot.set_motor(-SPEED,
                                 SPEED,
                                 SPEED,
                                -SPEED)
                current = "left"

        # Strafe Right
        elif keyboard.is_pressed('d'):
            if current != "right":
                robot.set_motor(SPEED,
                               -SPEED,
                               -SPEED,
                                SPEED)
                current = "right"

        # Rotate Left
        elif keyboard.is_pressed('q'):
            if current != "rotate_left":
                robot.set_motor(-SPEED,
                                -SPEED,
                                 SPEED,
                                 SPEED)
                current = "rotate_left"

        # Rotate Right
        elif keyboard.is_pressed('e'):
            if current != "rotate_right":
                robot.set_motor(SPEED,
                                SPEED,
                               -SPEED,
                               -SPEED)
                current = "rotate_right"

        # Stop
        elif keyboard.is_pressed('space'):
            if current != "stop":
                robot.set_motor(0, 0, 0, 0)
                current = "stop"

        else:
            if current != "stop":
                robot.set_motor(0, 0, 0, 0)
                current = "stop"

        time.sleep(0.02)

except KeyboardInterrupt:
    pass

finally:
    robot.set_motor(0, 0, 0, 0)
    robot.disconnect()
