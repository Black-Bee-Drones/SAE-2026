from ament_index_python.packages import get_package_share_directory
import os
from bouncing.setup import package_name

IS_INDOOR = True

# Camera parameters - IMX219 down-facing
IMAGE_SOURCE = "imx219"
CAMERA_WIDTH = 1640
CAMERA_HEIGHT = 1232
FOV_H = 62.2
FOV_V = 48.8
PIXELS_PER_DEGREE_H = CAMERA_WIDTH  / FOV_H
PIXELS_PER_DEGREE_V = CAMERA_HEIGHT / FOV_V
PIXELS_PER_DEGREE = (PIXELS_PER_DEGREE_H + PIXELS_PER_DEGREE_V) / 2

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

# search_id
SEARCH_ALTITUDE = 5
SEARCH_POINTS_TIMEOUT = 30 # seconds
SEARCH_NUMBER_PHOTO = 1
SEARCH_POINTS = [
    {
        'x': 0,
        'y': 0,
    },
    {
        'x': 4.6,
        'y': 0,
    },
    {
        'x': 4.6,
        'y': 0,
    },
]
