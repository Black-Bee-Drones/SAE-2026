import os
import cv2

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
    def __init__(self, start_target_base={}, start_mavros=True, start_camera=True):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.start_target_base = start_target_base

        self.start_mavros = start_mavros
        self.start_camera = start_camera

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


        if self.start_camera:
            yasmin.YASMIN_LOG_INFO('Initializing Detector...')
            try:
                self.detector = Detector(
                    model_source = MODEL_SOURCE,
                    confidence_threshold = MODEL_CONFIDENCE_THRESHOLD,
                )

                yasmin.YASMIN_LOG_INFO('Loading Detector...')
                self.detector.load()

                blackboard['detector'] = self.detector
                yasmin.YASMIN_LOG_INFO('Successfull start Detector...')

            except Exception as e:
                yasmin.YASMIN_LOG_ERROR(f'Detector failed: {e}.')
                return ABORT


            yasmin.YASMIN_LOG_INFO('Initializing ImageHandler...')
            try:
                self.image_handler = ImageHandler(
                    node=self.node,
                    image_source=CAMERA_IMAGE_SOURCE,
                    config=CAMERA_CONFIG,
                    image_processing_callback=self.callback_detector,
                )

                yasmin.YASMIN_LOG_INFO('Open camera...')
                self.image_handler.open()

                yasmin.YASMIN_LOG_INFO('Take testing photo...')
                result = self.image_handler.take_photo()

                blackboard['image_handler'] = self.image_handler
                yasmin.YASMIN_LOG_INFO('Successfull start ImageHandler...')

            except Exception as e:
                yasmin.YASMIN_LOG_ERROR(f'ImageHandler failed: {e}.')
                return ABORT


        yasmin.YASMIN_LOG_INFO('Completed successfully.')
        return SUCCEED


    def callback_detector(self, image):
        timestamp = self.node.get_clock().now().nanoseconds
        os.makedirs('photos', exist_ok=True)

        raw_path = os.path.join('photos', f'photos-{timestamp}.png')
        cv2.imwrite(raw_path, image)

        result = self.detector.detect(image)
        result.image = image

        annotated = self.detector.draw_detections(image, result)
        ann_path = os.path.join('photos', f'photos-{timestamp}-annotated.png')
        cv2.imwrite(ann_path, annotated)

        return result
