from ament_index_python.packages import get_package_share_directory
import os

# --- Altitude (meters) ---
SEARCH_ALTITUDE = 6.0
WORK_ALTITUDE = 3.5
RELEASE_ALTITUDE = 2.0
RTL_ALTITUDE = 5.0

# --- Camera (Arducam 2MP IMX662, USB) ---
IMAGE_SOURCE = "webcam"
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
IMAGE_CENTER_X = IMAGE_WIDTH // 2
IMAGE_CENTER_Y = IMAGE_HEIGHT // 2

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

# --- Sphere detection (DETECT_SPHERE) ---
SPHERE_DETECTION_CONFIRMATIONS = 6
SPHERE_DETECT_TIMEOUT = 30  # seconds
YAW_SCAN_VELOCITY = 0.3  # rad/s during 360 scan

# --- Approach sphere (APPROACH_SPHERE) ---
APPROACH_KP_X = 0.001  # image Y error -> body X velocity
APPROACH_KP_Y = 0.001  # image X error -> body Y velocity
APPROACH_MAX_VELOCITY_XY = 0.3
APPROACH_DESCEND_VELOCITY = 0.2  # m/s descent while maintaining lateral tracking
APPROACH_CENTER_TOLERANCE_PX = 60  # px, close enough to sphere
SPHERE_OFFSET_PX = 120  # px, stop this far from image center to avoid lidar over sphere
APPROACH_CENTER_CONFIRMATIONS = 6
APPROACH_TIMEOUT = 40  # seconds
APPROACH_MAX_LOST_FRAMES = 50

# --- Hose alignment (ALIGN_TO_HOSE) ---
HOSE_MIN_CONTOUR_AREA = 200  # px², minimum mask area to trust
HOSE_ANGLE_TOLERANCE_DEG = 5.0  # degrees from perpendicular (angle -> 0)
HOSE_ANGLE_KP = 0.01  # rad/s per degree of angle error
HOSE_ANGLE_MAX_VELOCITY = 0.4  # rad/s
HOSE_CENTER_TOLERANCE_PX = 40  # perpendicular centering tolerance
HOSE_CENTER_KP = 0.001  # m/s per pixel of center_x error
HOSE_CENTER_MAX_VELOCITY = 0.25  # m/s
HOSE_ALIGN_CONFIRMATIONS = 6
HOSE_ALIGN_TIMEOUT = 30  # seconds
HOSE_ALIGN_MAX_LOST_FRAMES = 60
HOSE_OFFSET_DISTANCE = 0.5  # meters to shift along hose away from sphere

# --- Descent with alignment (DESCEND_AND_ALIGN) ---
DESCEND_VELOCITY = 0.1  # m/s downward
DESCEND_CENTER_KP = 0.0009  # m/s per pixel (center_x -> vy)
DESCEND_ANGLE_KP = 0.008  # rad/s per degree (angle -> vyaw)
DESCEND_MAX_VELOCITY_XY = 0.2
DESCEND_MAX_YAW_VELOCITY = 0.3
DESCEND_CENTER_TOLERANCE_PX = 50
DESCEND_ANGLE_TOLERANCE_DEG = 8.0
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
# Set SIM_MODE = True to use Gazebo SITL instead of real hardware.
# When True, overrides IMAGE_SOURCE and drone config.
SIM_MODE = os.environ.get("HOOK_SIM", "0") == "1"

SIM_IMAGE_SOURCE = "/down_camera"
SIM_IMAGE_WIDTH = 960
SIM_IMAGE_HEIGHT = 540

if SIM_MODE:
    IMAGE_SOURCE = SIM_IMAGE_SOURCE
    IMAGE_WIDTH = SIM_IMAGE_WIDTH
    IMAGE_HEIGHT = SIM_IMAGE_HEIGHT
    IMAGE_CENTER_X = IMAGE_WIDTH // 2
    IMAGE_CENTER_Y = IMAGE_HEIGHT // 2
