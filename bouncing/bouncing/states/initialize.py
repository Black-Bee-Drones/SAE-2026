import yasmin
from yasmin import State, Blackboard
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.control import MavrosDrone, MavrosConfigpython
from nectar.vision import ImageHandler
from nectar.ai import Detector

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


    def log(self, msg, style='info'):
        class_name = f'{self.__class__.__name__}({', '.join([cls.__name__ for cls in self.__class__.__bases__])})'

        if style == 'info':
            yasmin.YASMIN_LOG_INFO(f'{class_name}: {msg}')
        elif style == 'error':
            yasmin.YASMIN_LOG_ERROR(f'{class_name}: {msg}')


    def execute(self, blackboard: Blackboard):
        self.log('Start.')


        self.log('Initializing \"target_base\"...')
        blackboard['target_base'] = {} # {shape, number}


        self.log('Initializing MavrosDrone...')
        try:
            config = MavrosConfigpython()
            blackboard['drone'] = MavrosDrone(
                config=config,
                node=self.node,
            )
            self.log('Successfull start MavrosDrone...')
        except Exception as e:
            self.log(
                f'Mavdrone failed: {e}.',
                style='error',
            )
            return ABORT


        self.log('Initializing Detector...')
        try:
            detector = Detector(
                model_source = MODEL_SOURCE,
                confidence_threshold = MODEL_CONFIDENCE_THRESHOLD,
            )

            self.log('Loading Detector...')
            detector.load()

            blackboard['detector'] = detector
            self.log('Successfull start Detector...')

        except Exception as e:
            self.log(
                f'Detector failed: {e}.',
                style='error',
            )
            return ABORT


        self.log('Initializing ImageHandler...')
        try:
            image_handler = ImageHandler(
                node=self.node,
                image_source=CAMERA_IMAGE_SOURCE,
                config=CAMERA_CONFIG,
                image_processing_callback=detector.detect,
            )

            self.log('Take testing photo...')
            image_handler.take_photo()

            blackboard['image_handler'] = image_handler
            self.log('Successfull start ImageHandler...')

        except Exception as e:
            self.log(
                f'ImageHandler failed: {e}.',
                style='error',
            )
            return ABORT


        self.log('Completed successfully.')
        return SUCCEED
