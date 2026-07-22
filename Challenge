#!/usr/bin/env python3
# Move straight 2s -> turn 90° -> move 2s -> turn 90° -> move 2s
# Uses SparkyBotMini.get_attitude() (yaw) to perform accurate 90° turns.

import time
from sparkybotmini import SparkyBotMini


def _normalize_deg(angle):
    """Normalize angle to (-180, 180]"""
    a = ((angle + 180) % 360) - 180
    return a


def _angle_diff(target, current):
    """Signed shortest difference (target - current) in degrees"""
    diff = (target - current + 180) % 360 - 180
    return diff


def move_forward(robot, duration=2.0, speed=40):
    print(f"> Moving forward for {duration}s at speed {speed}")
    robot.set_motor(speed, speed, speed, speed)
    time.sleep(duration)
    robot.set_motor(0, 0, 0, 0)
    time.sleep(0.05)


def rotate_exact_90(robot, direction="right", turn_speed=40, tol_deg=2.0, timeout=6.0):
    """
    Rotate approximately 90 degrees using yaw feedback.
    direction: "right" or "left"
    turn_speed: motor speed magnitude for turning
    tol_deg: stopping tolerance in degrees
    timeout: safety timeout in seconds
    """
    # Read initial yaw
    _, _, start_yaw = robot.get_attitude(degrees=True)
    if direction == "right":
        target = _normalize_deg(start_yaw + 90.0)
        # motor signs chosen to spin clockwise; if wrong, swap signs
        motors = (-turn_speed, turn_speed, -turn_speed, turn_speed)
    else:
        target = _normalize_deg(start_yaw - 90.0)
        motors = (turn_speed, -turn_speed, turn_speed, -turn_speed)

    print(f"> Rotating {direction} from {start_yaw:.1f}° to target {target:.1f}° (tol {tol_deg}°)")

    robot.set_motor(*motors)
    start_time = time.time()
    while True:
        _, _, yaw = robot.get_attitude(degrees=True)
        diff = _angle_diff(target, yaw)
        # diff is signed (target - current); close to 0 when reached
        # print debug optionally:
        # print(f"  yaw={yaw:.1f} target={target:.1f} diff={diff:.1f}")
        if abs(diff) <= tol_deg:
            print(f"  Reached target yaw {yaw:.1f}° (diff {diff:.1f}°)")
            break
        if time.time() - start_time > timeout:
            print("  Timeout while turning; stopping motors")
            break
        time.sleep(0.02)

    robot.set_motor(0, 0, 0, 0)
    time.sleep(0.05)


def main():
    robot = SparkyBotMini(port="/dev/ttyUSB0", debug=False)
    if not robot.connect():
        print("Failed to connect to robot")
        return

    try:
        robot.set_auto_report(True)
        time.sleep(0.3)  # allow sensors to start

        # Sequence: forward 2s -> turn 90 -> forward 2s -> turn 90 -> forward 2s
        move_forward(robot, duration=2.0, speed=40)
        rotate_exact_90(robot, direction="right", turn_speed=40)

        move_forward(robot, duration=2.0, speed=40)
        rotate_exact_90(robot, direction="right", turn_speed=40)

        move_forward(robot, duration=2.0, speed=40)

    finally:
        # Stop and cleanup
        robot.set_motor(0, 0, 0, 0)
        robot.set_auto_report(False)
        robot.disconnect()
        print("Done.")


if __name__ == "__main__":
    main()
