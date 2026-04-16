from ament_index_python.packages import get_package_share_directory
from pathlib import Path

from nectar.vision import OpenCVConfig


CAMERA_IMAGE_SOURCE = 'webcam'
CAMERA_CONFIG = OpenCVConfig(
    width=1280,
    height=720,
    # threaded=False,
    device_index=2,
)

MODEL_SOURCE = str(
    Path(get_package_share_directory("bouncing")) / "models" / "teste26n.pt"
)
MODEL_CONFIDENCE_THRESHOLD = 0.5

TAKEOFF_ALTITUDE = 3.0  # m
TAKEOFF_SLEEP = 0.1  # s

SEARCH_NUMBER_DETECTIONS = 100
SEARCH_DETECTIONS_LOST_TOLERANCE = 5

PRECISE_LANDING_DETECTIONS_LOST_TOLERANCE = 5
PRECISE_LANDING_TIMEOUT = 3 * 60  # s
PRECISE_LANDING_VERTICAL_SPEED = 0.3  # m/s
PRECISE_LANDING_ALING_TOLERANCE = 50  # pixel
PRECISE_LANDING_LAND_ALTITUDE = 1.0  # m

CONTROLER_P_XY = 0.001
CONTROLER_I_XY = 0.0
CONTROLER_D_XY = 0.0
