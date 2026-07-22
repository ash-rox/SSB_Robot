# Omnidirectional 4-Wheel Robot Control System Design
**SparkyBotMini-based X-Pattern Omniwheel Robot**

---

## 1. ROBOT CONFIGURATION

### Hardware Mapping
```
         FRONT
    M1 (LF) --- M3 (RF)
      \         /
       \       /
        \     /
         X
        /     \
       /       \
      /         \
    M2 (LB) --- M4 (RB)
         BACK

Legend:
- LF = Left Front (M1)
- LB = Left Back (M2)
- RF = Right Front (M3)
- RB = Right Back (M4)

Wheel Pattern: X-configuration (omniwheels)
Robot Center: Intersection of diagonals
Wheel Radius: 30mm (60mm wheel diameter)
Distance from center to wheel: √(L² + W²)/2
  where L = wheelbase, W = trackwidth
```

### Coordinate Frame
```
        Y (Left)
        ^
        |
        |
----+---+---+---- X (Forward)
    |   |   |
    |  [R]  |
    |   |   |
```

- **X-axis**: Forward/backward (positive = forward)
- **Y-axis**: Left/right (positive = left)
- **Z-axis**: Rotation (positive = counter-clockwise from above)
- **Origin**: Robot center (geometric centroid)

### Sensor Suite
| Sensor | Purpose | Connection |
|--------|---------|-----------|
| SparkyBotMini IMU | Orientation (yaw), acceleration | Serial (integrated) |
| Motor Encoders (M1-M4) | Wheel velocity feedback | GPIO/Serial |
| USB Camera | Obstacle detection, visual odometry | USB, left-facing |
| Raspberry Pi 5 | Main controller | - |

---

## 2. OMNIWHEEL KINEMATICS

### Forward Kinematics (Motor commands → Robot velocity)

Given motor speeds: **ω₁, ω₂, ω₃, ω₄** (rad/s or normalized -100 to +100)

For X-pattern omniwheels at 45° angles:

```
┌─────────────────────────────────────────┐
│ Velocity transformation matrix (4×3):   │
│                                         │
│ [ω₁]   1  -1   -1  [vx]               │
│ [ω₂] = 1  -1   +1  [vy]               │
│ [ω₃]   1  +1   +1  [ωz]               │
│ [ω₄]   1  +1   -1                     │
└─────────────────────────────────────────┘

Normalized (with wheel radius r, center-to-wheel d):
ω = (1/r) * M * v

Where:
- vx = forward velocity (m/s)
- vy = leftward velocity (m/s)
- ωz = rotation rate (rad/s)
- d = distance from center to wheel
```

### Inverse Kinematics (Robot velocity → Motor commands)

```
[vx]        [ω₁]
[vy] = M⁻¹ [ω₂]
[ωz]        [ω₃]
            [ω₄]

Explicit inverse (for X-pattern):
vx = (ω₁ + ω₂ + ω₃ + ω₄) / 4
vy = (-ω₁ - ω₂ + ω₃ + ω₄) / 4
ωz = (-ω₁ + ω₂ + ω₃ - ω₄) / (4*d)

Note: Motors are input as speeds (0-100 range)
Speed_to_angular = (speed / 100) * max_angular_velocity
```

### Example: Move Forward Only
```
Desired: vx=0.5 m/s, vy=0, ωz=0
Required motor speeds: ω₁=ω₂=ω₃=ω₄ = 50 (equal, forward)
```

### Example: Strafe Left Only
```
Desired: vx=0, vy=0.5 m/s, ωz=0
Required motor speeds: ω₁=ω₂=-50, ω₃=ω₄=+50 (diagonal opposite pairs)
```

### Example: Rotate CCW Only
```
Desired: vx=0, vy=0, ωz=1 rad/s
Required motor speeds: ω₁=ω₄=-50, ω₂=ω₃=+50 (alternating pattern)
```

---

## 3. SYSTEM ARCHITECTURE

### Module Hierarchy

```
┌──────────────────────────────────────────────────────────┐
│         MAIN NAVIGATION ORCHESTRATOR                      │
│    (omniwheel_navigator.py)                              │
└──────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │   ODOMETRY   │  │   PLANNER    │  │    MOTION    │
   │   MODULE     │  │   MODULE     │  │  CONTROLLER  │
   │ (pose from   │  │ (compute     │  │ (motor cmds) │
   │  encoders +  │  │  v, vy, ωz)  │  │              │
   │   IMU)       │  │              │  │              │
   └──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │  OMNIWHEEL   │  │   CAMERA     │  │  SAFETY      │
   │  KINEMATICS  │  │  PROCESSOR   │  │  WATCHDOG    │
   │  (4→3 or 3→4)│  │  (obstacle   │  │ (timeout,    │
   │              │  │   detection) │  │  bounds)     │
   └──────────────┘  └──────────────┘  └──────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
   │ SparkyBot    │  │ OpenCV       │  │ Error Log    │
   │ Library      │  │ / Vision API │  │              │
   │ (M1-M4 ctrl) │  │              │  │              │
   └──────────────┘  └──────────────┘  └──────────────┘
```

### Module Responsibilities

| Module | Responsibility | Input | Output |
|--------|---|---|---|
| **OmniwheelOdometry** | Fuse encoder + IMU → pose | Encoder ticks, IMU yaw | (x, y, θ) |
| **OmniwheelPlanner** | Compute desired (vx, vy, ωz) | Current pose, target pose | (vx, vy, ωz) |
| **OmniwheelMotionController** | Manage motor speeds | (vx, vy, ωz) | M1-M4 commands |
| **OmniwheelKinematics** | Convert 3-DOF ↔ 4-motor | (vx, vy, ωz) or [ω1-4] | [ω1-4] or (vx, vy, ωz) |
| **CameraProcessor** | Detect obstacles, markers | USB camera frames | Obstacles, lanes, markers |
| **SafetyWatchdog** | Timeout detection, bounds | State updates, timestamps | Stop cmd or alert |

---

## 4. INTERFACE SPECIFICATIONS

### 4.1 OmniwheelOdometry

```python
class OmniwheelOdometry:
    """
    6-DOF odometry for omniwheel robot
    Fuses wheel encoder data with IMU yaw
    """
    
    def __init__(self, wheel_radius: float, center_to_wheel: float,
                 encoder_ticks_per_rev: int, gyro_drift_threshold: float = 5.0):
        """
        Args:
            wheel_radius: 0.03 m (30mm)
            center_to_wheel: distance to each wheel from center (m)
            encoder_ticks_per_rev: counts per 360° (calibration)
            gyro_drift_threshold: max acceptable drift (°/s)
        """
        pass
    
    def update(self, encoder_ticks: Dict[str, int], imu_yaw: float,
               imu_yaw_rate: float, dt: float) -> 'Pose6DOF':
        """
        Update odometry with new sensor data
        
        Args:
            encoder_ticks: {'M1': count, 'M2': count, 'M3': count, 'M4': count}
            imu_yaw: heading in degrees or radians
            imu_yaw_rate: rotation rate (°/s or rad/s)
            dt: time step (seconds)
        
        Returns:
            Pose6DOF(x, y, vx, vy, theta, omega_z)
        """
        pass
    
    def reset(self, initial_pose: Optional['Pose6DOF'] = None):
        """Reset odometry to origin or given pose"""
        pass

@dataclass
class Pose6DOF:
    x: float           # meters
    y: float           # meters
    theta: float       # radians (yaw)
    vx: float          # m/s (forward)
    vy: float          # m/s (leftward)
    omega_z: float     # rad/s (rotation)
```

### 4.2 OmniwheelPlanner

```python
class OmniwheelPlanner:
    """
    Computes desired velocities for trajectory following
    Supports:
      - Point-to-point navigation
      - Velocity control
      - Heading correction
    """
    
    def __init__(self, kp_linear: float = 1.0, kp_angular: float = 2.0,
                 max_linear: float = 0.5, max_angular: float = 2.0,
                 distance_tol: float = 0.05, angle_tol: float = 0.1):
        """
        Args:
            kp_linear: Proportional gain for distance (m/s per meter error)
            kp_angular: Proportional gain for heading (rad/s per radian error)
            max_linear: Max forward speed (m/s)
            max_angular: Max rotation speed (rad/s)
            distance_tol: Distance threshold to consider target reached (m)
            angle_tol: Angle threshold (radians)
        """
        pass
    
    def compute_velocity(self, current: 'Pose6DOF', 
                        target: 'Pose6DOF') -> Tuple[float, float, float]:
        """
        Compute (vx, vy, omega_z) to reach target
        
        Returns:
            (vx, vy, omega_z) in m/s, m/s, rad/s
        """
        pass
    
    def is_target_reached(self, current: 'Pose6DOF', 
                         target: 'Pose6DOF') -> bool:
        """Check if current pose within tolerance of target"""
        pass
    
    def set_desired_velocity(self, vx: float, vy: float, omega_z: float):
        """Override trajectory planning; use direct velocity commands"""
        pass
```

### 4.3 OmniwheelMotionController

```python
class OmniwheelMotionController:
    """
    Translates desired (vx, vy, omega_z) into motor commands via SparkyBot
    """
    
    def __init__(self, sparkybot: SparkyBotMini,
                 wheel_radius: float = 0.03,
                 center_to_wheel: float = 0.15):
        """
        Args:
            sparkybot: Connected SparkyBotMini instance
            wheel_radius: 30mm for 60mm omniwheel
            center_to_wheel: distance from robot center to wheel
        """
        pass
    
    def command_velocity(self, vx: float, vy: float, 
                        omega_z: float) -> bool:
        """
        Command desired velocities; compute and send motor speeds
        
        Args:
            vx: forward velocity (m/s)
            vy: leftward velocity (m/s)
            omega_z: rotation rate (rad/s)
        
        Returns:
            True if motors updated, False on error
        """
        pass
    
    def command_motor_speeds(self, m1: int, m2: int, m3: int, 
                            m4: int) -> bool:
        """
        Direct motor control (bypasses kinematics)
        
        Args:
            m1-m4: Speed values -100 to +100
        
        Returns:
            True if successful
        """
        pass
    
    def stop(self):
        """Emergency stop: all motors to 0"""
        pass
```

### 4.4 OmniwheelKinematics

```python
class OmniwheelKinematics:
    """
    Pure kinematics: no state, just transformations
    """
    
    def __init__(self, wheel_radius: float, center_to_wheel: float):
        pass
    
    def velocity_to_motors(self, vx: float, vy: float, 
                          omega_z: float) -> Tuple[int, int, int, int]:
        """
        (vx, vy, omega_z) → [ω₁, ω₂, ω₃, ω₄]
        
        Returns:
            Motor speeds -100 to +100 (clamped)
        """
        pass
    
    def motors_to_velocity(self, m1: int, m2: int, m3: int, 
                          m4: int) -> Tuple[float, float, float]:
        """
        [ω₁, ω₂, ω₃, ω₄] → (vx, vy, omega_z)
        
        Used for odometry cross-check or diagnostics
        """
        pass
    
    def get_transformation_matrix(self) -> np.ndarray:
        """Return raw 4×3 or 3×4 matrix for advanced control"""
        pass
```

### 4.5 CameraProcessor

```python
class CameraProcessor:
    """
    Processes USB camera frames for perception
    """
    
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (640, 480)):
        """
        Args:
            camera_id: /dev/video<id> (0 for /dev/video0)
            resolution: (width, height)
        """
        pass
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Grab current frame from USB camera
        
        Returns:
            BGR image array or None on error
        """
        pass
    
    def detect_obstacles(self, frame: np.ndarray) -> List['Obstacle']:
        """
        Simple obstacle detection (e.g., contour-based)
        
        Returns:
            List of obstacles with (x, y, width, height, confidence)
        """
        pass
    
    def detect_lane_markers(self, frame: np.ndarray) -> Optional['Lane']:
        """Detect left/right boundary lines (if available)"""
        pass
    
    def detect_april_tags(self, frame: np.ndarray) -> List['AprilTag']:
        """Detect ArUco/AprilTag markers for absolute localization"""
        pass

@dataclass
class Obstacle:
    x: int          # pixel x in frame
    y: int          # pixel y in frame
    width: int      # bounding box width
    height: int     # bounding box height
    confidence: float  # 0.0-1.0

@dataclass
class AprilTag:
    tag_id: int
    corners: np.ndarray  # 4×2 pixel coordinates
    pose_estimate: Optional['Pose6DOF']  # if pose computed
```

### 4.6 SafetyWatchdog

```python
class SafetyWatchdog:
    """
    Monitor system health; detect stalls, timeouts, out-of-bounds
    """
    
    def __init__(self, timeout_sensor: float = 0.5,
                 timeout_motor: float = 1.0,
                 bounds: Optional[Tuple[float, float, float, float]] = None):
        """
        Args:
            timeout_sensor: Max time without encoder/IMU update (s)
            timeout_motor: Max time motor can be on without movement (s)
            bounds: (x_min, y_min, x_max, y_max) or None for unlimited
        """
        pass
    
    def tick(self, current_pose: 'Pose6DOF', 
            last_sensor_time: float) -> 'SafetyStatus':
        """
        Check health; return status
        
        Returns:
            SafetyStatus(healthy, warnings, errors)
        """
        pass
    
    def is_motor_stalled(self, current_velocity: float, 
                        desired_speed: int) -> bool:
        """Detect if motor commanded but not moving"""
        pass

@dataclass
class SafetyStatus:
    healthy: bool
    warnings: List[str]
    errors: List[str]
```

---

## 5. UNITS & SPECIFICATIONS

### Length & Distance
| Quantity | Unit | Typical Value |
|----------|------|---------------|
| Wheel radius | m | 0.030 (30mm) |
| Center-to-wheel | m | ~0.15 |
| Linear velocity | m/s | 0.0–0.5 |
| Distance tolerance | m | 0.05 |

### Angle & Rotation
| Quantity | Unit | Typical Value |
|----------|------|---------------|
| Heading (θ) | rad | -π to +π |
| Angular velocity | rad/s | 0.0–2.0 |
| Angle tolerance | rad | 0.1 (~5.7°) |

### Motor Commands
| Quantity | Unit | Range |
|----------|------|-------|
| Motor speed | - | -100 to +100 |
| Conversion factor | rad/s per speed unit | 0.01–0.05 * max_omega |

### Time
| Quantity | Unit | Typical |
|----------|------|---------|
| Control loop | Hz | 20–50 |
| Sensor update | ms | 10–50 |

---

## 6. FAILURE MODES & RECOVERY

### Encoder Failures

| Failure | Detection | Recovery |
|---------|-----------|----------|
| **Single wheel stalled** | One encoder static, others moving | Reduce all motor speeds; alert user |
| **Two opposite encoders fail** | Cannot distinguish rotation from strafing | Fall back to IMU yaw only |
| **All encoders fail** | Zero motion detected for >1s despite commands | **ERROR**: stop and report |

### IMU Failures

| Failure | Detection | Recovery |
|---------|-----------|----------|
| **Gyro drift** | Yaw rate unreasonable (>360°/s) | Ignore gyro; use encoders only |
| **IMU disconnected** | No update for 0.5s | Use encoder-derived yaw estimate |
| **Accel glitch** | Acceleration spikes unrealistic | Filter or ignore; log event |

### Motor Failures

| Failure | Detection | Recovery |
|---------|-----------|----------|
| **Motor stall** | Command sent, encoder frozen | Retry with higher PWM; escalate to ERROR |
| **Power loss to one motor** | Motor unresponsive to all commands | Disable that motor; use remaining 3 |
| **Short circuit** | Motor jerky or erratic | **STOP** immediately (safety) |

### Sensor Communication

| Failure | Detection | Recovery |
|---------|-----------|----------|
| **Serial timeout** | No response from SparkyBot for >100ms | Retry connection; escalate if persistent |
| **Camera disconnected** | OpenCV fails to open /dev/videoX | Disable vision; continue navigation |
| **USB camera frame drop** | Frame timestamp jumps >100ms | Skip frame; next capture continues |

### Navigation Failures

| Failure | Detection | Recovery |
|---------|-----------|----------|
| **Target unreachable** | No progress after timeout | Log failure; advance to next waypoint |
| **Out of bounds** | Pose exceeds limits | Reverse last command; alert user |
| **Oscillation** | Same position for >5 control loops | Reduce gains; pause and reset |

---

## 7. TEST PLAN

### Unit Tests (No Hardware)

#### 7.1 Kinematics Tests

```python
def test_forward_only():
    """
    Given: vx=0.5, vy=0, omega_z=0
    Expected: [ω₁, ω₂, ω₃, ω₄] = [50, 50, 50, 50]
    """
    assert kinematics.velocity_to_motors(0.5, 0, 0) == (50, 50, 50, 50)

def test_strafe_left_only():
    """
    Given: vx=0, vy=0.5, omega_z=0
    Expected: [ω₁, ω₂, ω₃, ω₄] = [-50, -50, 50, 50]
    """
    result = kinematics.velocity_to_motors(0, 0.5, 0)
    assert result == (-50, -50, 50, 50)

def test_rotate_ccw_only():
    """
    Given: vx=0, vy=0, omega_z=1
    Expected: [ω₁, ω₂, ω₃, ω₄] = [-50, 50, 50, -50]
    """
    result = kinematics.velocity_to_motors(0, 0, 1)
    assert result == (-50, 50, 50, -50)

def test_combined_motion():
    """
    Given: vx=0.3, vy=0.2, omega_z=0.5
    Expected: Motor speeds reasonable and within [-100, 100]
    """
    m1, m2, m3, m4 = kinematics.velocity_to_motors(0.3, 0.2, 0.5)
    assert all(-100 <= m <= 100 for m in [m1, m2, m3, m4])

def test_inverse_kinematics():
    """
    Given: Motor speeds from forward kinematics
    When: Apply inverse kinematics
    Then: Recover original (vx, vy, omega_z) ±tolerance
    """
    original = (0.3, 0.2, 0.5)
    motors = kinematics.velocity_to_motors(*original)
    recovered = kinematics.motors_to_velocity(*motors)
    assert all(abs(r - o) < 0.01 for r, o in zip(recovered, original))

def test_speed_saturation():
    """
    Given: Desired velocities exceed motor capabilities
    When: velocity_to_motors called
    Then: All motor speeds clamped to [-100, 100]
    """
    m1, m2, m3, m4 = kinematics.velocity_to_motors(5.0, 5.0, 10.0)
    assert all(-100 <= m <= 100 for m in [m1, m2, m3, m4])
```

#### 7.2 Odometry Tests

```python
def test_straight_line_odometry():
    """
    Given: All encoders incrementing equally, yaw constant
    When: Update odometry 100 times (simulated motion)
    Then: Final pose should have:
      - x proportional to total encoder count
      - y ≈ 0
      - theta ≈ initial_theta
    """
    pass

def test_pure_rotation():
    """
    Given: Encoders in opposite rotation pattern, gyro yaw changing
    When: Update odometry
    Then: Final x ≈ 0, y ≈ 0, theta = target angle
    """
    pass

def test_encoder_overflow():
    """
    Given: Encoder values wrap around (16-bit or 32-bit limit)
    When: Odometry processes wrap-around
    Then: Distance computed correctly (no negative jumps)
    """
    pass

def test_gyro_drift_detection():
    """
    Given: Gyro drift > threshold (e.g., 5°/s constant)
    When: Odometry checks encoder vs. gyro consistency
    Then: Triggers warning flag and uses encoders primarily
    """
    pass

def test_imu_yaw_fusion():
    """
    Given: Encoder yaw + IMU yaw (trust factor = 0.8)
    When: Odometry fused
    Then: Final yaw weighted by trust factor
    """
    pass
```

#### 7.3 Planner Tests

```python
def test_heading_to_waypoint():
    """
    Given: Current (0, 0, 0°), Target (1, 0, 0°)
    When: compute_velocity called
    Then: vx > 0, vy ≈ 0, omega_z ≈ 0
    """
    pass

def test_lateral_approach():
    """
    Given: Current (0, 0, 45°), Target (1, 0, 0°)
    When: compute_velocity called
    Then: omega_z < 0 (turn to correct heading first)
    """
    pass

def test_target_tolerance():
    """
    Given: Current (0.01, 0.01, 0.05 rad), Target (0, 0, 0)
      with tolerance = 0.05 m and 0.1 rad
    When: is_target_reached called
    Then: Returns True
    """
    pass

def test_gain_scaling():
    """
    Given: Distance error = 0.5 m, kp_linear = 1.0
    When: compute_velocity called
    Then: vx ≤ min(0.5 m/s, kp_linear * error) = 0.5 m/s
    """
    pass
```

#### 7.4 Vision Tests

```python
def test_obstacle_detection():
    """
    Given: Frame with solid colored rectangle (simulated obstacle)
    When: detect_obstacles called
    Then: Returns bounding box with reasonable confidence
    """
    pass

def test_camera_frame_grab():
    """
    Given: Camera available at /dev/video0
    When: capture_frame called
    Then: Returns frame array or None on error
    """
    pass

def test_april_tag_pose():
    """
    Given: Frame with known AprilTag at known location
    When: detect_april_tags called
    Then: Returns tag_id and approximate 3D pose
    """
    pass
```

### Integration Tests (Simulator or Bench)

```python
def test_full_trajectory():
    """
    Setup: Simulated robot, 3 waypoints
    Execute: OmniwheelNavigator.navigate(waypoints)
    Verify: Robot reaches each waypoint in sequence
    """
    pass

def test_obstacle_avoidance_with_vision():
    """
    Setup: Robot + camera, obstacle in front
    Execute: navigate_with_vision(target)
    Verify: Robot detects obstacle, deviates, continues
    """
    pass

def test_encoder_glitch_recovery():
    """
    Setup: Inject spike in M1 encoder (1000 ticks sudden)
    Execute: Navigation loop 10x
    Verify: Robot detects anomaly, filters it, continues
    """
    pass

def test_imu_compass_fusion():
    """
    Setup: Straight-line navigation, gyro with 2°/s bias
    Execute: 60 second run
    Verify: Final heading error < 5° (encoder + IMU fusion)
    """
    pass

def test_motor_loss_degradation():
    """
    Setup: 3 waypoints, disable M4 motor midway
    Execute: Navigation with 3 active motors
    Verify: Robot detects loss, adapts (or escalates to ERROR)
    """
    pass
```

### Acceptance Tests (Real Hardware)

```python
def test_real_straight_line():
    """
    Measure: Robot commanded to travel 1 m forward
    Verify: Actual distance traveled ∈ [0.95, 1.05] m
    """
    pass

def test_real_strafe():
    """
    Measure: Robot commanded to strafe 0.5 m left
    Verify: Lateral displacement ∈ [0.45, 0.55] m, no forward drift
    """
    pass

def test_real_rotation():
    """
    Measure: Robot rotates 360° with heading feedback
    Verify: Final heading within 10°, trajectory centered
    """
    pass

def test_real_camera_latency():
    """
    Measure: Time from frame capture to obstacle detection
    Verify: Latency < 100 ms (real-time capable)
    """
    pass

def test_battery_performance():
    """
    Run tests at 100%, 50%, 25% battery voltage
    Verify: Motion behavior consistent (with gain adjustments)
    """
    pass

def test_long_duration():
    """
    Execute 1-hour autonomous navigation loop
    Verify: No crashes, odometry drift acceptable
    """
    pass
```

---

## 8. CONFIGURATION FILE TEMPLATE

```yaml
# robot_config.yaml

hardware:
  wheel_radius_m: 0.030
  center_to_wheel_m: 0.15
  wheelbase_m: 0.3
  trackwidth_m: 0.3
  
encoders:
  ticks_per_revolution: 360
  m1: "LF"
  m2: "LB"
  m3: "RF"
  m4: "RB"

imu:
  gyro_scale: 0.0005          # rad/s per LSB
  accel_scale: 0.0001         # m/s² per LSB
  yaw_trust_factor: 0.8       # weight of IMU in fusion
  drift_threshold_dps: 5.0    # degrees/second

camera:
  device: "/dev/video0"
  resolution: [640, 480]
  fps: 30
  facing: "left"

planner:
  kp_linear: 1.0
  kp_angular: 2.0
  max_linear_m_per_s: 0.5
  max_angular_rad_per_s: 2.0
  distance_tolerance_m: 0.05
  angle_tolerance_rad: 0.1

control:
  loop_frequency_hz: 20
  sensor_timeout_s: 0.5
  motor_timeout_s: 1.0
  
safety:
  bounds_x: [-10.0, 10.0]
  bounds_y: [-10.0, 10.0]
  max_stall_time_s: 2.0
  emergency_stop_on_out_of_bounds: true
```

---

## 9. SUMMARY & NEXT STEPS

### Design Highlights ✅
- **X-pattern omniwheel kinematics** fully specified
- **4 independent modules** for easy testing & maintenance
- **Sensor fusion** (encoder + IMU) for robust odometry
- **Vision integration** ready for obstacle avoidance
- **Safety watchdog** for fault detection
- **Comprehensive test strategy** (unit → integration → acceptance)
- **Hardware-agnostic interfaces** (implement against SparkyBotMini)

### Ready for Implementation ✅
When you're ready, I can implement:

1. **`omniwheel_kinematics.py`** — Pure math, no hardware
2. **`omniwheel_odometry.py`** — Sensor fusion logic
3. **`omniwheel_planner.py`** — Control law (PID)
4. **`omniwheel_motion_controller.py`** — SparkyBot wrapper
5. **`camera_processor.py`** — OpenCV pipeline
6. **`safety_watchdog.py`** — Health monitoring
7. **`omniwheel_navigator.py`** — Main orchestrator
8. **Test suite** for all modules

### Questions for You

1. **Encoder details**: How many ticks per revolution? Are they connected to SparkyBot's `REPORT_ENCODER` or GPIO?
2. **Wheelbase/Trackwidth**: What's the actual distance from center to each wheel?
3. **Camera priority**: Is obstacle avoidance critical, or is it auxiliary?
4. **Waypoint format**: Will you provide absolute (x, y, θ) or relative (distance, angle) waypoints?
5. **Scaling preference**: Metric units (m/s, rad/s) or normalized motor scale (-100 to +100)?

---

**Ready to code? Let's build it modularly!** 🤖
