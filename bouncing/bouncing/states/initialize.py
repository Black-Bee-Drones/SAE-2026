import yasmin
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone
from mirela_sdk.image_processing.camera import (
    ImageCalculus,
    ImageHandler,
    IMX219Config,
)
from mirela_sdk.ai import YOLODetector

from bouncing.constants import (
    IS_INDOOR,
    IMAGE_SOURCE,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    PIXELS_PER_DEGREE,
    MODEL_PATH,
    YOLO_CONFIDENCE_THRESHOLD,
)


class Initialize(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def execute(self, blackboard: Blackboard):
        try:
            yasmin.YASMIN_LOG_INFO("Initializing mission...")

            yasmin.YASMIN_LOG_INFO("Initialize Blackboard vars")
            blackboard['target_base'] = {} # {class, symbol}

            yasmin.YASMIN_LOG_INFO("Initializing MavDrone...")
            blackboard["mavdrone"] = MavDrone(
                node=YasminNode.get_instance(),
                mavros=False,
                indoor=IS_INDOOR,
            )
            mavdrone: MavDrone = blackboard["mavdrone"]

            yasmin.YASMIN_LOG_INFO("Initializing ImageCalculus...")
            image_calculus = ImageCalculus()
            image_calculus.update_camera_resolution(
                width = CAMERA_WIDTH,
                height = CAMERA_HEIGHT,
            )
            image_calculus.update_pixels_per_degree(PIXELS_PER_DEGREE)
            blackboard["image_calculus"] = image_calculus

            yasmin.YASMIN_LOG_INFO("Initializing ImageHandler...")
            blackboard["image_handler"] = ImageHandler(
                node=YasminNode.get_instance(),
                image_source=IMAGE_SOURCE,
                config=IMX219Config(
                    sensor_id=0, width=CAMERA_WIDTH, height=CAMERA_HEIGHT, flip=2
                ),
            )
            image_handler: ImageHandler = blackboard["image_handler"]
            mavdrone.delay(1)

            yasmin.YASMIN_LOG_INFO(f"Loading YOLO model from {MODEL_PATH}...")
            
            blackboard["yolo_detector"] = YOLODetector(
                model_source=MODEL_PATH,
                confidence_threshold=YOLO_CONFIDENCE_THRESHOLD,
                device="auto",
                auto_load=True,
            )
            yolo_detector: YOLODetector = blackboard["yolo_detector"]

            yasmin.YASMIN_LOG_INFO("Warming up YOLO model...")
            image_handler.open()
            frame = image_handler.take_photo()
            yolo_detector.detect(frame)
            yasmin.YASMIN_LOG_INFO("Yolo detector ready.")

            yasmin.YASMIN_LOG_INFO("Mission successfully initialized. Cameras ready.")
            return SUCCEED

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Failed to initialize: {e}")
            return ABORT
