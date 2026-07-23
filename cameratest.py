#!/usr/bin/env python3
# coding: utf-8
"""
SparkybotMini USB Camera Live Video Feed
Simple live video stream from USB camera connected to SparkybotMini robot
"""

import cv2
import time
from sparkybotmini import SparkyBotMini


# Define driving speed magnitude (0 to 100)
SPEED = 60


def move_robot(robot, vx: float, vy: float):
    """
    Calculates and sets the X-formation omni wheel motor speeds.
    
    Args:
        robot: SparkyBotMini robot instance
        vx: X velocity (strafe left/right)
        vy: Y velocity (forward/backward)
    """
    # Inverse kinematics formulas for omni wheels
    m1 = vy + vx  # Front Left Motor
    m2 = vy - vx  # Front Right Motor
    m3 = vy - vx  # Rear Left Motor
    m4 = vy + vx  # Rear Right Motor

    # Proportional scaling to ensure safety range limits inside [-100, 100]
    max_val = max(abs(m1), abs(m2), abs(m3), abs(m4))
    if max_val > 100.0:
        scale = 100.0 / max_val
        m1 *= scale
        m2 *= scale
        m3 *= scale
        m4 *= scale

    robot.set_motor(int(m1), int(m2), int(m3), int(m4))


class CameraViewer:
    """
    Display live video feed from USB camera on SparkybotMini.
    Provides simple controls and FPS display.
    """
    
    def __init__(self, camera_index=0, robot_port="/dev/ttyUSB0", robot_baudrate=115200, target_fps=60, display_scale=0.5):
        """
        Initialize camera viewer and robot connection.
        
        Args:
            camera_index: USB camera index (default: 0)
            robot_port: Serial port for SparkyBotMini (default: /dev/ttyUSB0)
            robot_baudrate: Baud rate for serial communication (default: 115200)
            target_fps: Target frames per second (default: 60)
            display_scale: Scale factor for display (default: 0.5 for 50% size)
        """
        self.camera_index = camera_index
        self.camera = None
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.display_scale = display_scale
        
        # Initialize robot
        self.robot = SparkyBotMini(port=robot_port, baudrate=robot_baudrate, debug=False)
        self.robot_connected = False
        
        # FPS tracking (optimized)
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        self.last_fps_update = time.time()
        self.fps_update_interval = 1.0  # Update FPS display every 1 second
        
        # Movement tracking
        self.last_movement = "STOPPED"
        
    def connect_robot(self):
        """
        Establish connection to SparkyBotMini robot.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        print("Connecting to SparkyBotMini robot...")
        if not self.robot.connect():
            print("✗ Failed to connect to robot!")
            print("  Continuing with camera feed only...")
            return False
        
        print("✓ Connected to robot successfully!")
        
        # Enable auto-reporting for sensor feedback
        self.robot.set_auto_report(True)
        time.sleep(0.5)
        print("✓ Auto-reporting enabled\n")
        
        self.robot_connected = True
        return True
    
    def initialize_camera(self):
        """
        Initialize USB camera with optimized settings.
        
        Returns:
            bool: True if camera initialized successfully, False otherwise
        """
        print(f"Initializing camera (index: {self.camera_index})...")
        self.camera = cv2.VideoCapture(self.camera_index)
        
        if not self.camera.isOpened():
            print("✗ Failed to open camera!")
            return False
        
        # Optimized camera properties for better performance - reduced resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.camera.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for low latency
        
        # Additional optimizations
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
        
        print("✓ Camera initialized successfully!")
        print(f"  Capture Resolution: {int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  Display Scale: {self.display_scale * 100:.0f}%")
        print(f"  Target FPS: {int(self.camera.get(cv2.CAP_PROP_FPS))}\n")
        
        return True
    
    def calculate_fps(self):
        """
        Calculate current FPS (optimized - only update periodically).
        
        Returns:
            float: Frames per second
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Only update FPS display every interval
        if current_time - self.last_fps_update >= self.fps_update_interval:
            elapsed = current_time - self.last_fps_update
            self.fps = self.frame_count / elapsed if elapsed > 0 else 0
            self.frame_count = 0
            self.last_fps_update = current_time
        
        return self.fps
    
    def display_info(self, frame):
        """
        Add information overlay to frame (FPS, instructions).
        Optimized to draw only necessary elements.
        
        Args:
            frame: Video frame to annotate
        """
        # Display FPS (only essential info on every frame)
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        # Display connection status
        if self.robot_connected:
            status_text = "Robot: ✓"
            status_color = (0, 255, 0)
        else:
            status_text = "Robot: ✗"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        
        # Display movement status
        movement_text = f"Movement: {self.last_movement}"
        cv2.putText(frame, movement_text, (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    def handle_movement_input(self, key):
        """
        Handle movement input from keyboard and update robot motors.
        
        Args:
            key: ASCII value of the key pressed
            
        Returns:
            bool: True if robot is moving, False if stopped
        """
        key_char = chr(key).lower()
        
        if key_char == 'w':  # Forward
            self.last_movement = "Forward"
            if self.robot_connected:
                move_robot(self.robot, vx=0, vy=SPEED)
            return True
        
        elif key_char == 's':  # Backward
            self.last_movement = "Backward"
            if self.robot_connected:
                move_robot(self.robot, vx=0, vy=-SPEED)
            return True
        
        elif key_char == 'a':  # Strafe Left
            self.last_movement = "Strafe Left"
            if self.robot_connected:
                move_robot(self.robot, vx=-SPEED, vy=0)
            return True
        
        elif key_char == 'd':  # Strafe Right
            self.last_movement = "Strafe Right"
            if self.robot_connected:
                move_robot(self.robot, vx=SPEED, vy=0)
            return True
        
        elif key_char == 'q':  # Diagonal Up-Left
            self.last_movement = "Diagonal UL"
            if self.robot_connected:
                move_robot(self.robot, vx=-SPEED, vy=SPEED)
            return True
        
        elif key_char == 'e':  # Diagonal Up-Right
            self.last_movement = "Diagonal UR"
            if self.robot_connected:
                move_robot(self.robot, vx=SPEED, vy=SPEED)
            return True
        
        elif key_char == 'z':  # Diagonal Down-Left
            self.last_movement = "Diagonal DL"
            if self.robot_connected:
                move_robot(self.robot, vx=-SPEED, vy=-SPEED)
            return True
        
        elif key_char == 'c':  # Diagonal Down-Right
            self.last_movement = "Diagonal DR"
            if self.robot_connected:
                move_robot(self.robot, vx=SPEED, vy=-SPEED)
            return True
        
        elif key_char == ' ':  # Spacebar Stop
            self.last_movement = "STOPPED"
            if self.robot_connected:
                move_robot(self.robot, vx=0, vy=0)
            return False
        
        return None
    
    def run(self):
        """
        Main loop: capture and display live video feed.
        Optimized for maximum frame rate.
        Controls:
            WASD: Move (W=forward, A=left, S=backward, D=right)
            Q/E/Z/C: Diagonal movement
            Space: Stop
            R: Reset FPS
            'x': Quit
            's' (in addition to movement): Screenshot (with Ctrl or shift modifier ideally)
        """
        print("Starting live video feed...")
        print("Controls:")
        print("  WASD: Move (W=forward, A=left, S=backward, D=right)")
        print("  Q/E/Z/C: Diagonal (Up-Left, Up-Right, Down-Left, Down-Right)")
        print("  Space: Stop")
        print("  R: Reset FPS")
        print("  X: Quit\n")
        
        screenshot_count = 0
        frame_timer = time.time()
        
        try:
            while True:
                ret, frame = self.camera.read()
                
                if not ret:
                    print("✗ Error reading frame from camera!")
                    break
                
                # Flip the frame vertically and horizontally to correct upside-down camera
                frame = cv2.flip(frame, -1)
                
                # Calculate FPS
                self.calculate_fps()
                
                # Add information overlay (minimal)
                self.display_info(frame)
                
                # Scale frame for display if needed
                if self.display_scale != 1.0:
                    display_width = int(frame.shape[1] * self.display_scale)
                    display_height = int(frame.shape[0] * self.display_scale)
                    frame = cv2.resize(frame, (display_width, display_height), interpolation=cv2.INTER_LINEAR)
                
                # Display frame
                cv2.imshow("SparkybotMini Camera Feed", frame)
                
                # Frame timing control
                elapsed = time.time() - frame_timer
                wait_time = max(1, int((self.frame_time - elapsed) * 1000))
                
                # Handle keyboard input (non-blocking)
                key = cv2.waitKey(wait_time) & 0xFF
                
                if key != 255:  # 255 means no key pressed
                    if key == ord('x'):
                        print("\n✓ Quit requested")
                        break
                    
                    elif key == ord('r'):
                        self.frame_count = 0
                        self.last_fps_update = time.time()
                        self.fps = 0
                        print("✓ FPS counter reset")
                    
                    else:
                        # Try to handle movement input (including 's' for backward movement)
                        result = self.handle_movement_input(key)
                
                frame_timer = time.time()
        
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user (Ctrl+C)")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up resources...")
        
        # Stop robot movement
        if self.robot_connected:
            move_robot(self.robot, vx=0, vy=0)
        
        # Release camera
        if self.camera:
            self.camera.release()
            print("✓ Camera released")
        
        # Disconnect robot
        if self.robot_connected:
            self.robot.disconnect()
            print("✓ Robot disconnected")
        
        # Close all OpenCV windows
        cv2.destroyAllWindows()
        print("✓ All windows closed")
        print("\nGoodbye!")


def main():
    """Main entry point."""
    print("=" * 60)
    print("SparkybotMini USB Camera Live Feed Viewer")
    print("=" * 60 + "\n")
    
    # Create viewer instance with optimized settings
    viewer = CameraViewer(
        camera_index=0,
        robot_port="/dev/ttyUSB0",
        robot_baudrate=115200,
        target_fps=60,
        display_scale=0.5  # Display at 50% size for smoother performance
    )
    
    # Connect to robot (optional - continues if fails)
    viewer.connect_robot()
    
    # Initialize camera (required)
    if not viewer.initialize_camera():
        print("✗ Cannot continue without camera!")
        return False
    
    # Run live feed
    viewer.run()
    
    return True


if __name__ == "__main__":
    main()
