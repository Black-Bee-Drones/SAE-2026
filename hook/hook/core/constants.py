from ament_index_python.packages import get_package_share_directory
import os

# --- Altitude (meters) ---
INITIAL_TAKEOFF_ALTITUDE = 3.5
MAX_ASCEND_ALTITUDE = 6.8
WORK_ALTITUDE = 3.0
RELEASE_ALTITUDE = 2.0
RTL_ALTITUDE = 4.0

# --- Camera (Arducam 2MP IMX662, USB) ---
# Specs: 1920x1080, FOV 102(D) x 86(H) x 47(V), EFL 3.9mm, F1.0
IMAGE_SOURCE = "webcam"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGE_CENTER_X = IMAGE_WIDTH // 2
IMAGE_CENTER_Y = IMAGE_HEIGHT // 2
HORIZONTAL_FOV_DEG = 86.0
VERTICAL_FOV_DEG = 47.0

# --- Camera-to-hook offset in body frame (meters) ---
# Positive +x: hook forward of camera. Positive +y: hook left of camera.
# Hook is mounted near drone center; camera ~5 cm forward of hook.
CAMERA_TO_HOOK_BODY_X_M = -0.05
CAMERA_TO_HOOK_BODY_Y_M = 0.0

# --- Target heights in world ---
# Used by px_per_meter to remove the parallax error
SPHERE_HEIGHT_M = 1.7  # sphere mounted on hose at top of supports

# --- Segmentation model ---
SEG_MODEL_PATH = os.path.join(
    get_package_share_directory("hook"),
    "models",
    "sae-2026-hang-all-yolo26n-seg-v2-960.pt",
)
SEG_IMGSZ = 960
SEG_IOU = 0.6

SPHERE_CLASS = "sphere"
HOSE_CLASS = "rose"

SPHERE_CONF = 0.70
HOSE_CONF = 0.47
# Per-class filter refines further; predict cutoff is the min of all classes.
SEG_PREDICT_CONF = min(SPHERE_CONF, HOSE_CONF)

# --- Search and ascend (SEARCH_AND_ASCEND) ---
ASCEND_VELOCITY = 0.4  # m/s upward while searching
ASCENT_STOP_CONFIRMATIONS = 20
ASCENT_TIMEOUT = 25.0  # seconds

# --- Approach sphere (APPROACH_SPHERE) ---
APPROACH_TARGET_DISTANCE_M = (
    0.6  # real-world horizontal distance to sphere when parked (radial)
)
APPROACH_MIN_SAFE_DISTANCE_M = (
    0.50  # safety floor: never command motion that pushes closer
)
APPROACH_TOL_PX = 60  # error deadband: |err| < tol -> send 0 velocity
APPROACH_CONFIRMATIONS = 8  # consecutive frames inside tol to declare parked
APPROACH_KP = 0.0015  # px error -> m/s (same Kp on both axes)
APPROACH_MIN_OUTPUT_VELOCITY_XY = (
    0.04  # output deadband: |v| below this -> send 0 (no noise)
)
APPROACH_MAX_VELOCITY_XY = 0.45  # symmetric clamp
APPROACH_INIT_BEARING_FRAMES = 3  # median over first N detections to lock bearing
APPROACH_MIN_INIT_DIST_PX = (
    80  # if first detection is closer than this, skip bearing lock
)
APPROACH_DESCEND_VELOCITY = 0.2  # m/s descent during step 2
APPROACH_TIMEOUT = 90  # seconds
APPROACH_MAX_LOST_FRAMES = 50

# --- Hose side selection (SELECT_HOSE_SIDE) ---
SIDE_SAMPLE_FRAMES = 10
SIDE_LENGTH_RATIO = 1.4
HOSE_SHIFT_VELOCITY = 0.2
HOSE_SHIFT_DURATION = 1.5  # seconds
SIDE_REACQUIRE_FRAMES = 5
SIDE_TIMEOUT = 20.0

# --- Hose alignment (ALIGN_TO_HOSE) ---
HOSE_MIN_CONTOUR_AREA = 200  # px²
HOSE_ANGLE_TOLERANCE_DEG = 5.0
HOSE_ANGLE_KP = 0.01  # rad/s per degree
HOSE_ANGLE_MAX_VELOCITY = 0.4  # rad/s
HOSE_CENTER_TOLERANCE_PX = 40
HOSE_CENTER_KP = 0.001  # m/s per pixel
HOSE_CENTER_MAX_VELOCITY = 0.25  # m/s
HOSE_ALIGN_CONFIRMATIONS = 6
HOSE_ALIGN_TIMEOUT = 30  # seconds
HOSE_ALIGN_MAX_LOST_FRAMES = 60

# Sphere anchor (along-hose) used in ALIGN_TO_HOSE and DESCEND_AND_ALIGN
SPHERE_ANCHOR_DISTANCE_M = 0.6  # meters from sphere center along chosen hose direction
SPHERE_ANCHOR_TOLERANCE_PX = 35
SPHERE_ANCHOR_KP = 0.001  # m/s per pixel of sphere-y error -> vx

# --- Descent with alignment (DESCEND_AND_ALIGN) ---
DESCEND_VELOCITY = 0.1  # m/s downward
DESCEND_CENTER_KP = 0.0009
DESCEND_ANGLE_KP = 0.008
DESCEND_ANCHOR_KP = 0.0009
DESCEND_MAX_VELOCITY_XY = 0.2
DESCEND_MAX_YAW_VELOCITY = 0.3
DESCEND_CENTER_TOLERANCE_PX = 50
DESCEND_ANGLE_TOLERANCE_DEG = 8.0
DESCEND_ANCHOR_TOLERANCE_PX = 45
DESCEND_RELEASE_CONFIRMATIONS = 4
DESCEND_TIMEOUT = 45  # seconds
DESCEND_MAX_LOST_FRAMES = 60

# --- Servo (hook release) ---
SERVO_CHANNEL = 3
HOLD_PWM = 1000.0
RELEASE_PWM = 2000.0

# --- Saving detections ---
SAVE_DETECTIONS = True
DETECTION_SAVE_PATH = os.path.expanduser("~/sae2026")

# --- Simulation mode ---
SIM_MODE = os.environ.get("HOOK_SIM", "0") == "1"

# Subscribe to the republished compressed topic; the launch file
# (sae_hook.launch.py) runs an image_transport republish node that converts
# /down_camera (raw) -> /down_camera/compressed.
SIM_IMAGE_SOURCE = "/down_camera/compressed"
SIM_IMAGE_COMPRESSED = True

if SIM_MODE:
    IMAGE_SOURCE = SIM_IMAGE_SOURCE
