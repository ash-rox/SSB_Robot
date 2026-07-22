import time
from sparkybotmini import SparkyBotMini
# 1. Create the robot object (Make sure your port matches your computer's setup)
# Linux usually look like "/dev/ttyUSB0"
robot = SparkyBotMini(port="/dev/ttyUSB0")
# 2. Connect to the robot
if robot.connect():
print("Successfully connected to SparkyBotMini!")
else:
print("Connection failed. Check your port and cable.")
exit()
# Always disconnect cleanly at the end!
robot.disconnect()
