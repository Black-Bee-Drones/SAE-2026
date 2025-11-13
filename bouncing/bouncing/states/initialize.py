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


class Initialize(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.IS_INDOOR = self.node.get_parameter_or('is_indoor', False)

        self.CAMERA_IMAGE_SOURCE = self.node.get_parameter_or('camera_image_source', 'imx219')
        self.CAMERA_WIDTH = self.node.get_parameter_or('camera_width', 1640)
        self.CAMERA_HEIGHT = self.node.get_parameter_or('camera_height', 1232)
        self.CAMERA_FLIP = self.node.get_parameter_or('camera_flip', 2)
        self.CAMERA_PIXELS_PER_DEGREE = self.node.get_parameter_or('camera_pixels_per_degree', 25.8)

        self.MODEL_PATH = self.node.get_parameter_or('model_path', 'models/yolov11n.pt')
        self.MODEL_CONFIDENCE_THRESHOLD = self.node.get_parameter_or('model_confidence_threshold', 0.8)

    @property
    def node(self):
        return YasminNode.get_instance()

    def execute(self, blackboard: Blackboard):
        try:
            yasmin.YASMIN_LOG_INFO("Initializing mission...")

            yasmin.YASMIN_LOG_INFO("Initializing parameter...")

            yasmin.YASMIN_LOG_INFO("Initialize Blackboard vars")
            blackboard['target_base'] = {} # {class, symbol}

            yasmin.YASMIN_LOG_INFO("Initializing MavDrone...")
            blackboard["mavdrone"] = MavDrone(
                node=self.node,
                mavros=False,
                indoor=self.IS_INDOOR,
            )
            mavdrone: MavDrone = blackboard["mavdrone"]

            yasmin.YASMIN_LOG_INFO("Initializing ImageCalculus...")
            image_calculus = ImageCalculus()
            image_calculus.update_camera_resolution(
                width = self.CAMERA_WIDTH,
                height = self.CAMERA_HEIGHT,
            )
            image_calculus.update_pixels_per_degree(self.CAMERA_PIXELS_PER_DEGREE)
            blackboard["image_calculus"] = image_calculus

            yasmin.YASMIN_LOG_INFO("Initializing ImageHandler...")
            if self.CAMERA_IMAGE_SOURCE == 'imx219':
                camera_config = IMX219Config(
                    width=self.CAMERA_WIDTH,
                    height=self.CAMERA_HEIGHT,
                    flip=self.CAMERA_FLIP,
                )

            blackboard["image_handler"] = ImageHandler(
                node=self.node,
                image_source=self.CAMERA_IMAGE_SOURCE,
                config=camera_config,
            )
            image_handler: ImageHandler = blackboard["image_handler"]
            mavdrone.delay(1)

            yasmin.YASMIN_LOG_INFO(f"Loading YOLO model from {self.MODEL_PATH}...")
            
            blackboard["yolo_detector"] = YOLODetector(
                model_source=self.MODEL_PATH,
                confidence_threshold=self.MODEL_CONFIDENCE_THRESHOLD,
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
