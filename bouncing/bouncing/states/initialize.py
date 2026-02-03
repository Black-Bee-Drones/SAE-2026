import yasmin
from yasmin import State, Blackboard
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone
from mirela_sdk.image_processing.camera import ImageHandler
from mirela_sdk.ai.detection import Detector

from bouncing.constants import (
    IS_INDOOR,
    CAMERA_IMAGE_SOURCE,
    CAMERA_CONFIG,
    MODEL_SOURCE,
    MODEL_CONFIDENCE_THRESHOLD,
)


class Initialize(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.node = YasminNode.get_instance()


    def execute(self, blackboard: Blackboard):
        yasmin.YASMIN_LOG_INFO('Initialize(State): Start.')


        yasmin.YASMIN_LOG_INFO('Initialize(State): Initializing \"target_base\"...')
        blackboard['target_base'] = {} # {class, symbol}


        yasmin.YASMIN_LOG_INFO('Initialize(State): Initializing MavDrone...')
        try:
            blackboard['mavdrone'] = MavDrone(
                node=self.node,
                mavros=False,
                indoor=IS_INDOOR,
            )
            yasmin.YASMIN_LOG_INFO('Initialize(State): Successfull start MavDrone...')
        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f'Initialize(State): Mavdrone failed: {e}.')
            return ABORT


        yasmin.YASMIN_LOG_INFO('Initialize(State): Initializing Detector...')
        try:
            detector = Detector(
                model_source = MODEL_SOURCE,
                framework = None,
                device = 'auto',
                confidence_threshold = MODEL_CONFIDENCE_THRESHOLD,
            )

            yasmin.YASMIN_LOG_INFO('Initialize(State): Loading Detector...')
            detector.load()

            blackboard['detector'] = detector
            yasmin.YASMIN_LOG_INFO('Initialize(State): Successfull start Detector...')

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f'Initialize(State): Detector failed: {e}.')
            return ABORT


        yasmin.YASMIN_LOG_INFO('Initialize(State): Initializing ImageHandler...')
        try:
            image_handler = ImageHandler(
                node=self.node,
                image_source=CAMERA_IMAGE_SOURCE,
                config=CAMERA_CONFIG,
                image_processing_callback=detector.detect,
            )

            yasmin.YASMIN_LOG_INFO('Initialize(State): Take testing photo...')
            image_handler.take_photo()

            blackboard['image_handler'] = image_handler
            yasmin.YASMIN_LOG_INFO('Initialize(State): Successfull start ImageHandler...')

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f'Initialize(State): ImageHandler failed: {e}.')
            return ABORT


        yasmin.YASMIN_LOG_INFO('Initialize(State): Completed successfully.')
        return SUCCEED
