from ament_index_python.packages import get_package_share_directory
import os

# --- Altitude (meters) ---
SEARCH_ALTITUDE = 6.0
RELEASE_ALTITUDE = 2.0
RTL_ALTITUDE = 5.0

# --- Camera (Arducam 2MP IMX662, USB) ---
IMAGE_SOURCE = "webcam"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGE_CENTER_X = IMAGE_WIDTH // 2
IMAGE_CENTER_Y = IMAGE_HEIGHT // 2

# Approximate horizontal FOV in degrees (102° diagonal, 16:9)
HORIZONTAL_FOV_DEG = 87.0

# --- Model paths ---
SPHERE_MODEL_PATH = os.path.join(
    get_package_share_directory("hook"), "models", "sphere.pt"
)
ROPE_MODEL_PATH = os.path.join(
    get_package_share_directory("hook"), "models", "rope_seg.pt"
)

# --- Detection thresholds ---
SPHERE_CONF_THRESHOLD = 0.7
ROPE_CONF_THRESHOLD = 0.5

# --- Sphere detection ---
SPHERE_DETECTION_CONFIRMATIONS = 6
SPHERE_DETECT_TIMEOUT = 30  # seconds
YAW_SCAN_VELOCITY = 0.3  # rad/s during 360° scan

# --- Align and approach ---
YAW_ALIGN_TOLERANCE_PX = 80  # horizontal pixel tolerance for yaw alignment
YAW_ALIGN_KP = 0.002  # proportional gain for yaw correction (rad/s per pixel)
YAW_ALIGN_MAX_VELOCITY = 0.5  # rad/s
YAW_ALIGN_CONFIRMATIONS = 5

APPROACH_KP_Y = 0.0008  # lateral correction during approach
APPROACH_FORWARD_VELOCITY = 0.3  # m/s forward
APPROACH_MAX_LATERAL_VELOCITY = 0.3
APPROACH_SPHERE_AREA_THRESHOLD = 15000  # px², stop approaching when sphere bbox area exceeds this
APPROACH_TIMEOUT = 30  # seconds

# --- Rope centering (pre-descent) ---
CENTER_TOLERANCE_PX = 50
CENTERING_CONFIRMATIONS = 8
CENTER_KP_X = 0.001  # image Y error -> body X velocity
CENTER_KP_Y = 0.001  # image X error -> body Y velocity
CENTER_MAX_VELOCITY_XY = 0.3
CENTER_TIMEOUT = 40  # seconds
CENTER_MAX_LOST_FRAMES = 80

# --- Descent ---
DESCEND_VELOCITY = 0.15  # m/s downward
DESCEND_KP_X = 0.0009
DESCEND_KP_Y = 0.0009
DESCEND_MAX_VELOCITY_XY = 0.2
DESCEND_CENTER_TOLERANCE_PX = 60
DESCEND_TIMEOUT = 45  # seconds
DESCEND_MAX_LOST_FRAMES = 60

# --- Servo (hook release) ---
SERVO_CHANNEL = 3
HOLD_PWM = 1000.0
RELEASE_PWM = 2000.0

# --- Saving detections ---
SAVE_DETECTIONS = True
DETECTION_SAVE_PATH = os.path.expanduser("~/sae2026")
