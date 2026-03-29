# Constants

from bouncing.setup import package_name

from ament_index_python.packages import get_package_share_directory
from pathlib import Path

from mirela_sdk.image_processing.camera import IMX219Config


IS_INDOOR = False

CAMERA_IMAGE_SOURCE = 'imx219'
CAMERA_CONFIG = IMX219Config(
    width=1640,
    height=1232,
    flip=2,
)

MODEL_SOURCE = str(
    Path(get_package_share_directory(package_name)) / "model" / "teste26n.pt"
)
MODEL_CONFIDENCE_THRESHOLD = 0.8

TAKEOFF_ALTITUDE = 5.0  # m
TAKEOFF_SLEEP = 5  # s

SEARCH_NUMBER_DETECTIONS = 10
SEARCH_DETECTIONS_LOST_TOLERANCE = 3

PRECISE_LANDING_ALTITUDE = TAKEOFF_ALTITUDE
PRECISE_LANDING_DETECTIONS_LOST_TOLERANCE = 3
PRECISE_LANDING_TIMEOUT = 60  # s
PRECISE_LANDING_VERTICAL_SPEED = 0.3  # m/s
PRECISE_LANDING_ALING_TOLERANCE = 50  # pixel
PRECISE_LANDING_LAND_ALTITUDE = 1.0  # m

CONTROLER_P_XY = 0.5
CONTROLER_I_XY = 0.0
CONTROLER_D_XY = 0.0
