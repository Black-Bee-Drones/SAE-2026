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

from bouncing.utils import Detector


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
        self.MODEL_NUMBER_PATH = self.node.get_parameter_or('model_number_path', 'models/yolov11n.pt')
        self.MODEL_CONFIDENCE_THRESHOLD = self.node.get_parameter_or('model_confidence_threshold', 0.8)

    @property
    def node(self):
        return YasminNode.get_instance()

    @property
    def __state_name__(self):
        return f'{self.__class__.__name__}({', '.join([cls.__name__ for cls in self.__class__.__bases__])})'

    def execute(self, blackboard: Blackboard):
        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Start.')

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Initializing \"target_base\"...')
        blackboard['target_base'] = {} # {class, symbol}

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Initializing MavDrone...')
        try:
            blackboard['mavdrone'] = MavDrone(
                node=self.node,
                mavros=False,
                indoor=self.IS_INDOOR,
            )
        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f'{self.__state_name__}: Mavdrone failed: {e}.')
            return ABORT

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Initializing ImageCalculus...')
        try:
            image_calculus = ImageCalculus()
            image_calculus.update_camera_resolution(
                width = self.CAMERA_WIDTH,
                height = self.CAMERA_HEIGHT,
            )
            image_calculus.update_pixels_per_degree(self.CAMERA_PIXELS_PER_DEGREE)
            blackboard['image_calculus'] = image_calculus
        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f'{self.__state_name__}: ImageCalculus failed: {e}.')
            return ABORT

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Initializing ImageHandler...')
        try:
            if self.CAMERA_IMAGE_SOURCE == 'imx219':
                camera_config = IMX219Config(
                    width=self.CAMERA_WIDTH,
                    height=self.CAMERA_HEIGHT,
                    flip=self.CAMERA_FLIP,
                )
            blackboard['image_handler'] = ImageHandler(
                node=self.node,
                image_source=self.CAMERA_IMAGE_SOURCE,
                config=camera_config,
            )
        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f'{self.__state_name__}: ImageHandler failed: {e}.')
            return ABORT

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Initializing Detector...')
        try:
            blackboard['detector'] = Detector(
                model_path = self.MODEL_PATH,
                model_number_path = self.MODEL_NUMBER_PATH,
                model_confidence_threshold = self.MODEL_CONFIDENCE_THRESHOLD,
                image_calculus = image_calculus,
            )
        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f'{self.__state_name__}: Detector failed: {e}.')
            return ABORT

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Completed successfully.')
        return SUCCEED
