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
        
        # ===== DRIVE FORWARD =====
        print("? Driving forward at 50% speed for 5 seconds...")
        print("  M1 (Left Front):  50")
        print("  M2 (Left Back):   50")
        print("  M3 (Right Front): 50")
        print("  M4 (Right Back):  50\n")
        
        # For omniwheel X-pattern: all motors same speed = forward
        robot.set_motor(m1=50, m2=50, m3=50, m4=50)
        time.sleep(5.0)
        
        # ===== STOP =====
        print("? Stopping motors...")
        robot.set_motor(m1=0, m2=0, m3=0, m4=0)
        time.sleep(0.5)
        print("✓ Robot stopped\n")
        
        # ===== READ FEEDBACK =====
        print("? Reading robot telemetry:")
        
        # Get velocity
        vx, vy, vz = robot.get_velocity()
        print(f"  Velocity: vx={vx:.2f} m/s, vy={vy:.2f} m/s, vz={vz:.2f} m/s")
        
        # Get battery voltage
        battery = robot.get_battery_voltage()
        print(f"  Battery: {battery:.1f}V")
        
        # Get encoders
        e1, e2, e3, e4 = robot.get_encoders()
        print(f"  Encoders: M1={e1}, M2={e2}, M3={e3}, M4={e4}")
        
        # Get attitude
        roll, pitch, yaw = robot.get_attitude(degrees=True)
        print(f"  Attitude: roll={roll:.1f}°, pitch={pitch:.1f}°, yaw={yaw:.1f}°\n")
        
        print("=" * 50)
        print("✓ TEST COMPLETE - All systems operational!")
        print("=" * 50)
        return True
        
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
