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
    
    def __init__(self, camera_index=0, robot_port="/dev/ttyUSB0", robot_baudrate=115200):
        """
        Initialize camera viewer and robot connection.
        
        Args:
            camera_index: USB camera index (default: 0)
            robot_port: Serial port for SparkyBotMini (default: /dev/ttyUSB0)
            robot_baudrate: Baud rate for serial communication (default: 115200)
        """
        self.camera_index = camera_index
        self.camera = None
        
        # Initialize robot
        self.robot = SparkyBotMini(port=robot_port, baudrate=robot_baudrate, debug=False)
        self.robot_connected = False
        
        # FPS tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
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
        Initialize USB camera.
        
        Returns:
            bool: True if camera initialized successfully, False otherwise
        """
        print(f"Initializing camera (index: {self.camera_index})...")
        self.camera = cv2.VideoCapture(self.camera_index)
        
        if not self.camera.isOpened():
            print("✗ Failed to open camera!")
            return False
        
        # Set camera properties for better performance
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        
        print("✓ Camera initialized successfully!")
        print(f"  Resolution: {int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  Target FPS: {int(self.camera.get(cv2.CAP_PROP_FPS))}\n")
        
        return True
    
    def calculate_fps(self):
        """
        Calculate current FPS.
        
        Returns:
            float: Frames per second
        """
        self.frame_count += 1
        
        if self.frame_count % 30 == 0:  # Update every 30 frames
            current_time = time.time()
            elapsed = current_time - self.start_time
            if elapsed > 0:
                self.fps = self.frame_count / elapsed
        
        return self.fps
    
    def display_info(self, frame):
        """
        Add information overlay to frame (FPS, instructions).
        
        Args:
            frame: Video frame to annotate
        """
        # Display FPS
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display connection status
        if self.robot_connected:
            status_text = "Robot: Connected"
            status_color = (0, 255, 0)
        else:
            status_text = "Robot: Not Connected"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Display instructions
        instructions = [
            "Controls:",
            "  's' - Capture screenshot",
            "  'r' - Reset FPS counter",
            "  'q' - Quit"
        ]
        
        y_offset = 120
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            y_offset += 25
    
    def run(self):
        """
        Main loop: capture and display live video feed.
        Press 'q' to quit, 's' to save a screenshot, 'r' to reset FPS.
        """
        print("Starting live video feed...")
        print("Press 'q' to quit, 's' to capture screenshot, 'r' to reset FPS\n")
        
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = self.camera.read()
                
                if not ret:
                    print("✗ Error reading frame from camera!")
                    break
                
                # Calculate FPS
                self.calculate_fps()
                
                # Add information overlay
                self.display_info(frame)
                
                # Display frame
                cv2.imshow("SparkybotMini Camera Feed", frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
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
                    self.start_time = time.time()
                    self.fps = 0
                    print("✓ FPS counter reset")
        
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
    
    # Create viewer instance
    viewer = CameraViewer(
        camera_index=0,
        robot_port="/dev/ttyUSB0",
        robot_baudrate=115200
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
