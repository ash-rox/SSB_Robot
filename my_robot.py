#!/usr/bin/env python3
import time
import sys
# Assuming SparkyBotMini library file is named sparky_bot.py in the same directory
from sparky_bot import SparkyBotMini

def move_robot(robot, vx: float, vy: float, vz: float = 0.0):
    """
    Calculates individual motor speeds for an X-formation omni wheel robot.
    
    Args:
        robot: The SparkyBotMini instance.
        vx: Side-to-side velocity (Strafe). Positive = Right, Negative = Left.
        vy: Forward/backward velocity. Positive = Forward, Negative = Backward.
        vz: Rotational velocity. Positive = Clockwise, Negative = Counter-Clockwise.
    """
    # X-formation inverse kinematics equations
    m1 = vy + vx + vz  # Front Left Motor
    m2 = vy - vx - vz  # Front Right Motor
    m3 = vy - vx + vz  # Rear Left Motor
    m4 = vy + vx - vz  # Rear Right Motor

    # Find the maximum computed magnitude to normalize values if they exceed max power limit
    max_val = max(abs(m1), abs(m2), abs(m3), abs(m4))
    
    # Scale velocities proportionally if any value goes past 100 (full speed)
    if max_val > 100.0:
        scale = 100.0 / max_val
        m1 *= scale
        m2 *= scale
        m3 *= scale
        m4 *= scale

    # Send values as integers to the underlying controller method
    robot.set_motor(int(m1), int(m2), int(m3), int(m4))

def main():
    # Initialize the controller mapped to the Pi 5's USB port interface
    bot = SparkyBotMini(port="/dev/ttyUSB0", debug=False)
    
    if not bot.connect():
        print("Error: Could not connect to SparkyBotMini hardware.")
        sys.exit(1)
        
    try:
        # 1. Forward Movement
        print("Moving Forward...")
        move_robot(bot, vx=0, vy=50) 
        time.sleep(1.5)
        
        # 2. Backward Movement
        print("Moving Backward...")
        move_robot(bot, vx=0, vy=-50) 
        time.sleep(1.5)
        
        # 3. Side-to-Side (Strafe Right)
        print("Strafing Right...")
        move_robot(bot, vx=60, vy=0) 
        time.sleep(1.5)
        
        # 4. Side-to-Side (Strafe Left)
        print("Strafing Left...")
        move_robot(bot, vx=-60, vy=0) 
        time.sleep(1.5)

        # 5. Diagonal Forward-Right
        print("Moving Diagonally Forward-Right...")
        move_robot(bot, vx=50, vy=50) 
        time.sleep(1.5)

        # 6. Diagonal Backward-Left
        print("Moving Diagonally Backward-Left...")
        move_robot(bot, vx=-50, vy=-50) 
        time.sleep(1.5)

        # Stop completely
        print("Stopping Robot.")
        move_robot(bot, vx=0, vy=0)

    except KeyboardInterrupt:
        print("\nHalting operations via emergency script override.")
    finally:
        # Ensure motors stop completely during disconnect procedures
        bot.set_motor(0, 0, 0, 0)
        bot.disconnect()

if __name__ == "__main__":
    main()
