#!/usr/bin/env python3
# coding: utf-8
"""
SparkybotMini USB Camera Live Video Feed
Simple live video stream from USB camera connected to SparkybotMini robot
"""

import cv2
import time
from sparkybotmini import SparkyBotMini


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
    
    def run(self):
        """
        Main loop: capture and display live video feed.
        Optimized for maximum frame rate.
        Press 'q' to quit, 's' to save a screenshot, 'r' to reset FPS.
        """
        print("Starting live video feed...")
        print("Press 'q' to quit, 's' to capture screenshot, 'r' to reset FPS\n")
        
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
                
                if key == ord('q'):
                    print("\n✓ Quit requested")
                    break
                
                elif key == ord('s'):
                    screenshot_count += 1
                    filename = f"camera_screenshot_{screenshot_count}.png"
                    cv2.imwrite(filename, frame)
                    print(f"✓ Screenshot saved: {filename}")
                
                elif key == ord('r'):
                    self.frame_count = 0
                    self.last_fps_update = time.time()
                    self.fps = 0
                    print("✓ FPS counter reset")
                
                frame_timer = time.time()
        
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user (Ctrl+C)")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up resources...")
        
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
