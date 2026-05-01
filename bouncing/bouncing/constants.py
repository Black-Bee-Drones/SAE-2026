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

MODEL_SOURCE = str(
    Path(get_package_share_directory("bouncing")) / "models" / "best.pt"
)
MODEL_CONFIDENCE_THRESHOLD = 0.5

TAKEOFF_ALTITUDE = 7.0  # m

SEARCH_NUMBER_DETECTIONS = 100
SEARCH_DETECTIONS_LOST_TOLERANCE = 10

RECOVERY_STEP = 0.5  # m
RECOVERY_LIMITE_ALTITUDE = 7.0  # m

PRECISE_LANDING_DETECTIONS_LOST_TOLERANCE = 15
PRECISE_LANDING_TIMEOUT = 3 * 60  # s
PRECISE_LANDING_VERTICAL_SPEED = -0.3  # m/s
PRECISE_LANDING_ALING_TOLERANCE = 50  # pixel
PRECISE_LANDING_LAND_ALTITUDE = 1.0  # m

CONTROLER_P_XY = 0.0013457
CONTROLER_I_XY = 0.0001
CONTROLER_D_XY = 0.0
