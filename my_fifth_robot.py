import time
from sparkybotmini import SparkyBotMini

robot = SparkyBotMini(port="/dev/ttyUSB0")

if robot.connect():
  print("Turning on sensors...")
  robot.set_auto_report(True)
  time.sleep(0.5) # Give the sensors a moment to start up
  print("Reading battery and tilt for 5 seconds...")
  start_time = time.time()
  
  while time.time() - start_time < 5:
    # Check the battery
    voltage = robot.get_battery_voltage()
    # Check the tilt (Attitude)
    roll, pitch, yaw = robot.get_attitude(degrees=True)
    print(f"Battery: {voltage:.1f}V | Pitch (up/down tilt): {pitch:.1f} degrees")
    time.sleep(0.5) # Wait half a second before checking again
    
# Clean up and disconnect at the very end of your script
robot.set_auto_report(False)
robot.disconnect()
print("Program complete!")
