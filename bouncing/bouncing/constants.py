from ament_index_python.packages import get_package_share_directory
import os
from bouncing.setup import package_name

IS_INDOOR = True

# Camera parameters - IMX219 down-facing
IMAGE_SOURCE = "imx219"
IMX219_WIDTH = 1640
IMX219_HEIGHT = 1232

# Yolo settings
MODEL_PATH = os.path.join(
    get_package_share_directory(package_name), "models", "yolov11n.pt"
)
YOLO_CONFIDENCE_THRESHOLD = 0.8

# Detection saving paths
DETECTION_SAVE_PATH = os.path.expanduser(f"~/{package_name}")
SAVE_DETECTIONS = True  # Enable/disable saving detection images

# Takeoff parameters
TAKEOFF_ALTITUDE = 3.0
TAKEOFF_SLEEP = 5
TAKEOFF_TIMEOUT = 60
ALTITUDE_TOLERANCE = 0.1
