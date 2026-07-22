import time
from sparkybotmini import SparkyBotMini
robot = SparkyBotMini(port="/dev/ttyUSB0")
if robot.connect():
  print("Successfully connected to SparkyBotMini!")
  print("Playing startup sound...")
  robot.beep(300) # Beep for 300 milliseconds
  time.sleep(1) # Wait for 1 second
robot.disconnect()
