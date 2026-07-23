#wait for robot to connect, then move forward for 5 seconds at 50% speed

#!/usr/bin/env python3
# coding: utf-8
"""
Simple Omniwheel Robot Forward Movement Example
Uses SparkyBotMini to drive the 4-wheel robot forward
"""

import time
import sys
from sparkybotmini import SparkyBotMini

def main():
    """Main entry point: connect, drive forward, stop, disconnect"""
    
    # ===== SETUP =====
    print("=" * 50)
    print("OMNIWHEEL ROBOT - FORWARD MOVEMENT TEST")
    print("=" * 50)
    
    # Create robot instance
    # Port: /dev/ttyUSB0 (SparkyBotMini serial connection)
    # Baudrate: 115200 (standard for SparkyBotMini)
    robot = SparkyBotMini(port="/dev/ttyUSB0", baudrate=115200, debug=False)
    
    try:
        # ===== CONNECT =====
        print("\n? Connecting to SparkyBotMini...")
        if not robot.connect():
            print("✗ Failed to connect to robot!")
            return False
        
        print("✓ Connected successfully!\n")
        
        # ===== ENABLE AUTO-REPORTING =====
        print("? Enabling sensor auto-reporting...")
        robot.set_auto_report(True)
        time.sleep(0.5)
        print("✓ Auto-reporting enabled\n")
 
        
        # Forward
        print("forward")
        robot.set_motor(m1=20, m2=20, m3=20, m4=20)
        time.sleep(5.0)
        robot.set_motor(m1=0, m2=0, m3=0, m4=0)
        time.sleep(0.5)

        #backward
        print("backward")
        robot.set_motor(m1=-20, m2=-20, m3=-20, m4=-20)
        time.sleep(5.0)
        robot.set_motor(m1=0, m2=0, m3=0, m4=0)
        time.sleep(0.5)
      
        #crableft
        print("crab left")
        robot.set_motor(m1=-20, m2=20, m3=20, m4=-20)
        time.sleep(5.0)
        robot.set_motor(m1=0, m2=0, m3=0, m4=0)
        time.sleep(0.5)

        #crabright
        print("crab right")
        robot.set_motor(m1=20, m2=-20, m3=-20, m4=20)
        time.sleep(5.0)
        robot.set_motor(m1=0, m2=0, m3=0, m4=0)
        time.sleep(0.5)

        #upleft
        print("up left")
        robot.set_motor(m1=0, m2=20, m3=20, m4=0)
        time.sleep(5.0)
        robot.set_motor(m1=0, m2=0, m3=0, m4=0)
        time.sleep(0.5)

        #downright
        print("down right")
        robot.set_motor(m1=0, m2=-20, m3=-20, m4=0)
        time.sleep(5.0)
        robot.set_motor(m1=0, m2=0, m3=0, m4=0)
        time.sleep(0.5)

        #downleft
        print("down left")
        robot.set_motor(m1=-20, m2=0, m3=0, m4=-20)
        time.sleep(5.0)
        robot.set_motor(m1=0, m2=0, m3=0, m4=0)
        time.sleep(0.5)

        #upright
        print("up right")
        robot.set_motor(m1=20, m2=0, m3=0, m4=20)
        time.sleep(5.0)
        robot.set_motor(m1=0, m2=0, m3=0, m4=0)
        time.sleep(0.5)
    
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user (Ctrl+C)")
        return False
    
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # ===== CLEANUP =====
        print("\n? Cleaning up...")
        robot.set_motor(0, 0, 0, 0)  # Ensure motors are stopped
        robot.disconnect()
        print("✓ Disconnected\n")


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
