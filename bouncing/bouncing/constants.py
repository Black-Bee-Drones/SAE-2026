from ament_index_python.packages import get_package_share_directory
from pathlib import Path

from nectar.vision import OpenCVConfig


CAMERA_IMAGE_SOURCE = 'webcam'
CAMERA_CONFIG = OpenCVConfig(
    width=1280,
    height=720,
    threaded=False,
    device_index=0,
)

MODEL_DETECTOR_SOURCE = str(
    Path(get_package_share_directory("bouncing")) / "models" / "best_detector_5_class.pt"
)
MODEL_DETECTOR_CONFIDENCE_THRESHOLD = 0.5
MODEL_CLASSIFIER_SOURCE = str(
    Path(get_package_share_directory("bouncing")) / "models" / "best_classifier.pt"
)
MODEL_CLASSIFIER_CONFIDENCE_THRESHOLD = 0.5

TAKEOFF_ALTITUDE = 1.0  # m

SEARCH_FIND_TOLERANCE = 4
SEARCH_LIMITE_ALTITUDE = 7.0  # m
SEARCH_TARGET_ALTITUDE = 6.0  # m
SEARCH_TIMEOUT = 3 * 60  # s
SEARCH_VERTICAL_SPEED = 0.5  # m/s
SEARCH_PHOTOS_PER_POINT = 20
SEARCH_POINTS = [
    {
        'x': 0.0,
        'y': 0.0,
    },
    {
        'x': 3.5,
        'y': 0.0,
    },
    {
        'x': 0.0,
        'y': 0.0,
    },
    {
        'x': -3.5,
        'y': 0.0,
    },
]

PRECISE_LIMITE_ALTITUDE = 7.0  # m
PRECISE_LIMITE_RECOVERY = 6.0  # m
PRECISE_HOVER_COUNT = 5  # number of consecutive detections to consider the landing successful
PRECISE_RESET_PID = 2  # number of detections lost to reset the PID controller
PRECISE_LOST_TOLERANCE = 5
PRECISE_TIMEOUT = 4 * 60  # s
PRECISE_VERTICAL_SPEED = 0.1  # m/s
PRECISE_DOWN_TOLERANCE_PX = 200  # px
PRECISE_ALING_TOLERANCE = 0.40  # m
PRECISE_LAND_ALTITUDE = 1.1  # m

CONTROLER_P_XY = 0.4
CONTROLER_I_XY = 0.0
CONTROLER_D_XY = 0.0
CONTROLER_OUTPUT_LIMITS_XY = (-0.22, 0.22)
CONTROLER_INTEGRAL_LIMITS_XY = (-0.01, 0.01)

CONTROLER_P_Z = 0.20
CONTROLER_I_Z = 0.0
CONTROLER_D_Z = 0.0
CONTROLER_OUTPUT_LIMITS_Z = (-0.8, 0.8)
CONTROLER_INTEGRAL_LIMITS_Z = (-0.1, 0.1)
