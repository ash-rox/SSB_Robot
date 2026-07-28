import cv2
import numpy as np
import time
from sparkybotmini import SparkyBotMini

# --- Dual-Axis PID Tuning Parameters ---
BASE_SPEED = 20       # Forward speed profile (Scale: -100 to 100)

# 1. STRAFE PID (Lateral sliding - handles side-to-side drift)
KP_STRAFE = 50.0      
KI_STRAFE = 0.10      
KD_STRAFE = 20.0      

# 2. YAW PID (Rotation - forces the nose to turn hard into loops)
KP_YAW = 75.0         
KI_YAW = 0.05         
KD_YAW = 25.0         

# --- Camera Optimization Parameters ---
CAMERA_WIDTH = 320    
CAMERA_HEIGHT = 240

# We now look at TWO horizontal rows to calculate the path's true slope/angle!
ROW_AHEAD = 30        # Looking far ahead (previewing incoming loops)
ROW_CLOSE = 100       # Looking close to the wheels

# Global hardware handle
robot = None

def control_omni_x_chassis(vx, vy, w):
    """
    Translates direction vectors into SparkyBot X-omni wheel kinematics.
    Combines Forward/Backward (vx), Lateral Strafe (vy), and Angular Rotation (w).
    """
    if robot is None:
        return

    # Full X-formation kinematics layout
    m1_speed = int((vx + vy + w) * BASE_SPEED)  # Front Left
    m2_speed = int((vx - vy - w) * BASE_SPEED)  # Front Right
    m3_speed = int((vx - vy + w) * BASE_SPEED)  # Rear Left
    m4_speed = int((vx + vy - w) * BASE_SPEED)  # Rear Right

    # Constrain to motor safe physical speed limits (-100 to 100)
    m1 = max(-100, min(100, m1_speed))
    m2 = max(-100, min(100, m2_speed))
    m3 = max(-100, min(100, m3_speed))
    m4 = max(-100, min(100, m4_speed))

    robot.set_motor(m1, m2, m3, m4)

def main():
    global robot
    
    # Establish hardware serial link
    robot = SparkyBotMini(port="/dev/ttyUSB0", debug=False)
    if not robot.connect():
        print("Hardware connection interface missing.")
        return

    # Initialize low-overhead camera stream
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Drop buffer queues to achieve zero latency

    print("Loop-Capable Dual-Axis PID Active. Press 'q' to stop.")
    
    # Initialize PID tracking memory registers
    last_err_strafe = 0.0
    integ_strafe = 0.0
    
    last_err_yaw = 0.0
    integ_yaw = 0.0
    
    last_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            dt = current_time - last_time
            if dt <= 0:
                dt = 0.001

            # --- High-Performance Downsampled Optimization Pipeline ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 85, 255, cv2.THRESH_BINARY_INV) # Black Tape = White Pixels

            # --- Dual-Row Slice Extraction ---
            slice_ahead = binary[ROW_AHEAD, :]
            slice_close = binary[ROW_CLOSE, :]
            
            pixels_ahead = np.where(slice_ahead == 255)[0]
            pixels_close = np.where(slice_close == 255)[0]

            cam_center_x = CAMERA_WIDTH // 2
            
            # Verify that both scan horizons can see the tape
            if len(pixels_ahead) > 0 and len(pixels_close) > 0:
                center_ahead = int(np.mean(pixels_ahead))
                center_close = int(np.mean(pixels_close))
                
                # 1. Strafe Error: Horizontal displacement relative to the closest row
                err_strafe = (center_close - cam_center_x) / cam_center_x
                
                # 2. Yaw Error: The delta/slope between the close row and ahead row.
                # If center_ahead is far away from center_close, the tape is curving heavily.
                err_yaw = (center_ahead - center_close) / cam_center_x
                
                # --- STRAFE PID MATH ENGINE ---
                integ_strafe += err_strafe * dt
                integ_strafe = max(-10.0, min(10.0, integ_strafe)) # Anti-windup
                deriv_strafe = (err_strafe - last_err_strafe) / dt
                vy_cmd = (err_strafe * KP_STRAFE) + (integ_strafe * KI_STRAFE) + (deriv_strafe * KD_STRAFE)
                
                # --- YAW PID MATH ENGINE ---
                integ_yaw += err_yaw * dt
                integ_yaw = max(-10.0, min(10.0, integ_yaw)) # Anti-windup
                deriv_yaw = (err_yaw - last_err_yaw) / dt
                w_cmd = (err_yaw * KP_YAW) + (integ_yaw * KI_YAW) + (deriv_yaw * KD_YAW)
                
                # Scale commands down to valid runtime floats (-1.0 to 1.0)
                vy_cmd = max(-1.0, min(1.0, vy_cmd / 100.0))
                w_cmd = max(-1.0, min(1.0, w_cmd / 100.0))
                
                # Execute full holistic path adjustment matrix
                control_omni_x_chassis(vx=1.0, vy=vy_cmd, w=w_cmd)
                
                # Save registers
                last_err_strafe = err_strafe
                last_err_yaw = err_yaw
                
                status_lbl = f"Loop Tracking | Yaw Err: {err_yaw:.2f}"
                robot.set_led(1, 0, 255, 0) # Green LED
                
                # For overlay graphics rendering
                target_draw_x1, target_draw_x2 = center_ahead, center_close
            else:
                # Emergency recovery backup plan if tape falls outside one of the row ranges
                integ_strafe, integ_yaw = 0.0, 0.0
                control_omni_x_chassis(vx=0.0, vy=0.0, w=0.0)
                status_lbl = "Loop Lost! Safety Halted."
                target_draw_x1, target_draw_x2 = cam_center_x, cam_center_x
                robot.set_led(1, 255, 0, 0) # Red LED

            # --- Diagnostic Visual Render Layer ---
            # Draw both sampling rows
            cv2.line(frame, (0, ROW_AHEAD), (CAMERA_WIDTH, ROW_AHEAD), (255, 0, 0), 1)
            cv2.line(frame, (0, ROW_CLOSE), (CAMERA_WIDTH, ROW_CLOSE), (255, 0, 0), 1)
            
            # Draw vector tracking line showing how the robot perceives the loop orientation
            cv2.line(frame, (target_draw_x2, ROW_CLOSE), (target_draw_x1, ROW_AHEAD), (0, 255, 255), 2)
            cv2.circle(frame, (target_draw_x1, ROW_AHEAD), 4, (0, 0, 255), -1)
            cv2.circle(frame, (target_draw_x2, ROW_CLOSE), 4, (0, 0, 255), -1)
            
            fps = 1.0 / dt
            last_time = current_time
            
            cv2.putText(frame, f"FPS: {int(fps)} | {status_lbl}", (10, 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.imshow("Omni Downward Video Feed", frame)
            cv2.imshow("Black and White Matrix Slice", binary)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        control_omni_x_chassis(0.0, 0.0, 0.0)
        robot.set_led(1, 0, 0, 0)
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("Robot safely halted.")

if __name__ == "__main__":
    main()
