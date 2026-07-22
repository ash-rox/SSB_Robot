"""my_third_robot.py

Make the SparkyBotMini perform a crabwalk (lateral strafe) using omni wheels
arranged in an X pattern. Uses the set_motor(m1, m2, m3, m4) API from
sparkybotmini.SparkyBotMini.

Note: wheel sign/polarity depends on wiring. If the robot strafes the wrong
way, invert the `vy` sign or flip specific motor signs in the mapping below.
"""

import time
import sys
from sparkybotmini import SparkyBotMini


def _limit(v: int) -> int:
    """Clamp motor value to valid range (-100..100). 127 reserved for emergency stop."""
    if v == 127:
        return 127
    return max(-100, min(100, int(v)))


def crabwalk(robot: SparkyBotMini, speed: int = 40, duration: float = 2.0, direction: str = "right"):
    """
    Strafe (crabwalk) the robot left or right.

    Args:
        robot: connected SparkyBotMini instance
        speed: base speed percentage (0-100)
        duration: seconds to keep moving
        direction: 'right' or 'left'

    Uses a mecanum-style kinematic mapping for 4 omniwheels in an X pattern.
    Motor order for set_motor is assumed to be m1, m2, m3, m4 as in sparkybotmini.

    Wheel equations (vx forward, vy right, omega rotation):
        m1 =  vx - vy - omega
        m2 =  vx + vy + omega
        m3 =  vx + vy - omega
        m4 =  vx - vy + omega

    For pure lateral movement set vx=0 and omega=0.
    """
    if direction.lower() not in ("right", "left"):
        raise ValueError("direction must be 'right' or 'left'")

    s = max(0, min(100, int(speed)))
    # Positive vy -> strafe right in this mapping
    vy = s if direction.lower() == "right" else -s
    vx = 0
    omega = 0

    m1 = _limit(vx - vy - omega)
    m2 = _limit(vx + vy + omega)
    m3 = _limit(vx + vy - omega)
    m4 = _limit(vx - vy + omega)

    print(f"Crabwalk {direction}: motors = {m1}, {m2}, {m3}, {m4}")
    robot.set_motor(m1, m2, m3, m4)
    time.sleep(duration)
    robot.set_motor(0, 0, 0, 0)


if __name__ == "__main__":
    robot = SparkyBotMini(port="/dev/ttyUSB0")

    try:
        if not robot.connect():
            print("Failed to connect to robot")
            raise SystemExit(1)

        # Demo: crabwalk right then left
        print("Crabwalking right...")
        crabwalk(robot, speed=40, duration=2.0, direction="right")

        time.sleep(1.0)

        print("Crabwalking left...")
        crabwalk(robot, speed=40, duration=2.0, direction="left")

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        # Ensure motors stop and disconnect
        try:
            robot.set_motor(0, 0, 0, 0)
        except Exception:
            pass
        robot.disconnect()
        print("Stopped and disconnected")
