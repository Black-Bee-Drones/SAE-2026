import yasmin
from yasmin import State, Blackboard
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.control import MavrosDrone, MavrosConfig
from nectar.vision import ImageHandler
from nectar.ai import Detector

from bouncing.constants import (
    CAMERA_IMAGE_SOURCE,
    CAMERA_CONFIG,
    MODEL_SOURCE,
    MODEL_CONFIDENCE_THRESHOLD,
)


class Initialize(State):
    def __init__(self, start_target_base={}, start_mavros=True, start_detector=True, start_image_handler=True):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.start_target_base = start_target_base

        self.start_mavros = start_mavros
        self.start_detector = start_detector
        self.start_image_handler = start_image_handler

        self.node = YasminNode.get_instance()


    def execute(self, blackboard: Blackboard):
        yasmin.YASMIN_LOG_INFO('Start.')


        yasmin.YASMIN_LOG_INFO('Initializing \"target_base\"...')
        blackboard['target_base'] = self.start_target_base  # {shape, number}


        if self.start_mavros:
            yasmin.YASMIN_LOG_INFO('Initializing MavrosDrone...')
            try:
                config = MavrosConfig()
                blackboard['drone'] = MavrosDrone(
                    config=config,
                    node=self.node,
                )
                yasmin.YASMIN_LOG_INFO('Successfull start MavrosDrone...')
            except Exception as e:
                yasmin.YASMIN_LOG_ERROR(f'Mavdrone failed: {e}.')
                return ABORT


        if self.start_detector:
            yasmin.YASMIN_LOG_INFO('Initializing Detector...')
            try:
                detector = Detector(
                    model_source = MODEL_SOURCE,
                    confidence_threshold = MODEL_CONFIDENCE_THRESHOLD,
                )

                yasmin.YASMIN_LOG_INFO('Loading Detector...')
                detector.load()

                blackboard['detector'] = detector
                yasmin.YASMIN_LOG_INFO('Successfull start Detector...')

            except Exception as e:
                yasmin.YASMIN_LOG_ERROR(f'Detector failed: {e}.')
                return ABORT


        if self.start_image_handler:
            yasmin.YASMIN_LOG_INFO('Initializing ImageHandler...')
            try:
                image_handler = ImageHandler(
                    node=self.node,
                    image_source=CAMERA_IMAGE_SOURCE,
                    config=CAMERA_CONFIG,
                    image_processing_callback=lambda img: (img, detector.detect(img)),
                )

                yasmin.YASMIN_LOG_INFO('Open camera...')
                image_handler.open()

                yasmin.YASMIN_LOG_INFO('Take testing photo...')
                _, result = image_handler.take_photo()
                yasmin.YASMIN_LOG_INFO(f'Result: {result}')

                blackboard['image_handler'] = image_handler
                yasmin.YASMIN_LOG_INFO('Successfull start ImageHandler...')

            except Exception as e:
                yasmin.YASMIN_LOG_ERROR(f'ImageHandler failed: {e}.')
                return ABORT


        yasmin.YASMIN_LOG_INFO('Completed successfully.')
        return SUCCEED
