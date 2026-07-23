#!/usr/bin/env python3
import sys
import tty
import termios
from sparkybotmini import SparkyBotMini

# Define your desired driving speed magnitude (0 to 100)
SPEED = 60

def get_key():
    """Reads a single keypress character from the terminal without needing Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def move_robot(robot, vx: float, vy: float):
    """Calculates and sets the X-formation omni wheel motor speeds."""
    # Inverse kinematics formulas
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
    bot = SparkyBotMini(port="/dev/ttyUSB0", debug=False)
    if not bot.connect():
        print("Error: Could not connect to SparkyBotMini hardware.")
        sys.exit(1)

    print("\n=== SparkyBot Mini Keyboard Controller ===")
    print("  Q (Diag UL)   W (Forward)   E (Diag UR)")
    print("  A (Strafe L)  S (Backward)  D (Strafe R)")
    print("  Z (Diag DL)   [Space] Stop  C (Diag DR)")
    print("                Press 'X' to Quit")
    print("==========================================\n")

    try:
        while True:
            key = get_key().lower()

            if key == 'w':      # Forward
                print("Moving: Forward         ", end="\r")
                move_robot(bot, vx=0, vy=SPEED)
            elif key == 's':    # Backward
                print("Moving: Backward        ", end="\r")
                move_robot(bot, vx=0, vy=-SPEED)
            elif key == 'a':    # Strafe Left
                print("Moving: Strafe Left     ", end="\r")
                move_robot(bot, vx=-SPEED, vy=0)
            elif key == 'd':    # Strafe Right
                print("Moving: Strafe Right    ", end="\r")
                move_robot(bot, vx=SPEED, vy=0)
            elif key == 'q':    # Diagonal Up-Left
                print("Moving: Diagonal Up-L   ", end="\r")
                move_robot(bot, vx=-SPEED, vy=SPEED)
            elif key == 'e':    # Diagonal Up-Right
                print("Moving: Diagonal Up-R   ", end="\r")
                move_robot(bot, vx=SPEED, vy=SPEED)
            elif key == 'z':    # Diagonal Down-Left
                print("Moving: Diagonal Down-L ", end="\r")
                move_robot(bot, vx=-SPEED, vy=-SPEED)
            elif key == 'c':    # Diagonal Down-Right
                print("Moving: Diagonal Down-R ", end="\r")
                move_robot(bot, vx=SPEED, vy=-SPEED)
            elif key == ' ':    # Spacebar Stop
                print("Moving: STOPED          ", end="\r")
                move_robot(bot, vx=0, vy=0)
            elif key == 'x':    # Exit
                print("\nExiting controller.")
                break

    except KeyboardInterrupt:
        print("\nEmergency override triggered.")
    finally:
        bot.set_motor(0, 0, 0, 0)
        bot.disconnect()

if __name__ == "__main__":
    main()
