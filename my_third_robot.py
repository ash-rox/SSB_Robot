import time
from sparkybotmini import SparkyBotMini
robot = SparkyBotMini(port="/dev/ttyUSB0")
if robot.connect():
print("Moving forward!")
robot.set_motor(40, 40, 40, 40) # All wheels forward at 40% speed
time.sleep(2) # Keep moving for 2 seconds
print("Stopping...")
robot.set_motor(0, 0, 0, 0) # Stop all wheels
time.sleep(1)
7
robot.disconnect(
