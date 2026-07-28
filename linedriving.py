import cv2
import numpy as np
import time
from sparky_bot import SparkyBotMini

# --- Video Stream Optimization Parameters ---
CAMERA_WIDTH = 320    # Downscaled low resolution maximizes Raspberry Pi 5 framerate
CAMERA_HEIGHT = 240

# Horizontal scanning rows to visualize on screen (used to analyze pixel arrays later)
ROW_AHEAD = 45        
ROW_CLOSE = 110       
WINDOW_HALF_HEIGHT = 5

def main():
    # Initialize connection to SparkyBotMini hardware to keep it active
    robot = SparkyBotMini(port="/dev/ttyUSB0", debug=False)
    if not robot.connect():
        print("Hardware serial interface failed to connect. Check USB cable.")
        return

    # FORCE SAFETY: Ensure all robot motors are completely stopped and idle
    robot.set_motor(0, 0, 0, 0)
    robot.set_led(1, 0, 0, 0)  # Reset LED

    # Initialize low-overhead camera capture pipeline
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Flushes queues to get real-time frames with zero delay

    print("Stationary Image Processing Engine Active. Press 'q' in the window to quit.")
    
    last_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from camera.")
                break

            # Calculate live frame time delta for FPS tracking
            current_time = time.time()
            dt = current_time - last_time
            if dt <= 0:
                dt = 0.001
            fps = 1.0 / dt
            last_time = current_time

            # --- Optimization Pipeline ---
            # 1. Convert the standard color image into a 1-channel Grayscale image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 2. Apply a Gaussian Blur to smooth out lighting reflections, glare, or scuff marks
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # 3. Apply Adaptive Otsu's Thresholding with an INVERTED mask:
            # - Black tape will become PURE WHITE pixels (Value: 255)
            # - Light floor surface / anything else will become PURE BLACK pixels (Value: 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # --- Analysis Window Slicing (Verification) ---
            # Extract the horizontal slices where the robot will eventually evaluate line coordinates
            zone_ahead = binary[ROW_AHEAD - WINDOW_HALF_HEIGHT : ROW_AHEAD + WINDOW_HALF_HEIGHT, :]
            zone_close = binary[ROW_CLOSE - WINDOW_HALF_HEIGHT : ROW_CLOSE + WINDOW_HALF_HEIGHT, :]
            
            # Compress the 10-pixel deep boxes vertically into flat 1D horizontal vectors
            slice_ahead = np.mean(zone_ahead, axis=0)
            slice_close = np.mean(zone_close, axis=0)
            
            # Find the array indices where white pixels exist (value > 127)
            pixels_ahead = np.where(slice_ahead > 127)[0]
            pixels_close = np.where(slice_close > 127)[0]

            # Build diagnostic terminal status text based on what the pixels look like
            if pixels_ahead.size > 0 and pixels_close.size > 0:
                center_ahead = int(np.mean(pixels_ahead))
                center_close = int(np.mean(pixels_close))
                status_lbl = f"Tape Detected! Ahead Center: {center_ahead} | Close Center: {center_close}"
                print(f"STATUS: Line Visible -> Ahead: {center_ahead}, Close: {center_close}", flush=True)
            else:
                status_lbl = "No line in view or threshold mismatched."
                print("STATUS: Scanning... Line out of view.", flush=True)

            # --- Diagnostic Visual Overlay HUD ---
            # Draw blue horizontal bounding rectangles representing your tracking horizons
            cv2.rectangle(frame, (0, ROW_AHEAD - WINDOW_HALF_HEIGHT), (CAMERA_WIDTH, ROW_AHEAD + WINDOW_HALF_HEIGHT), (255, 0, 0), 1)
            cv2.rectangle(frame, (0, ROW_CLOSE - WINDOW_HALF_HEIGHT), (CAMERA_WIDTH, ROW_CLOSE + WINDOW_HALF_HEIGHT), (255, 0, 0), 1)
            
            # Render HUD text data onto the primary stream display window
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, status_lbl, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Display both windows side by side to verify pixel properties
            cv2.imshow("1. Real Camera Feed (HUD Overlay)", frame)
            cv2.imshow("2. Target Output (White = Tape, Black = Floor)", binary)

            # Check if user presses 'q' key to cleanly exit the execution loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Final safety cleanup routine: Keep motors completely off
        robot.set_motor(0, 0, 0, 0)
        cap.release()
        cv2.destroyAllWindows()
        robot.disconnect()
        print("\nImage processing session stopped. Motors remained idle.")

if __name__ == "__main__":
    main()
