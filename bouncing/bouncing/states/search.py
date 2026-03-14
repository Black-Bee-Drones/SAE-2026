import cv2
import time

import yasmin
from yasmin import State, Blackboard
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, ABORT

from nectar.control import MavrosDrone, MoveReference
from nectar.vision import ImageHandler
from nectar.ai import Detector

from bouncing.constants import (
    SEARCH_ALTITUDE,
    SEARCH_POINTS,
)


class Search(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, FAIL, ABORT])


    def log(self, msg, style='info'):
        class_name = f'{self.__class__.__name__}({', '.join([cls.__name__ for cls in self.__class__.__bases__])})'

        if style == 'info':
            yasmin.YASMIN_LOG_INFO(f'{class_name}: {msg}')
        elif style == 'error':
            yasmin.YASMIN_LOG_ERROR(f'{class_name}: {msg}')


    def execute(self, blackboard: Blackboard):
        if ('drone' not in blackboard) or not blackboard['drone']:
            self.log(
                f'MavrosDrone not available.',
                style='error'
            )
            return ABORT
        drone: MavrosDrone = blackboard['drone']

        if ('image_handler' not in blackboard) or not blackboard['image_handler']:
            self.log(
                f'ImageHandler not available.',
                style='error'
            )
            return ABORT
        image_handler: ImageHandler = blackboard['image_handler']

        if ('detector' not in blackboard) or not blackboard['detector']:
            self.log(
                f'Detector not available.',
                style='error'
            )
            return ABORT
        detector: Detector = blackboard['detector']

        self.log('Start.')

        target_base = {
            'shape': None,
            'number': None,
            'x': None,
            'y': None,
        }
        results = []
        for i, point in enumerate(SEARCH_POINTS):
            self.log(f'Coordinate {i + 1}/{len(SEARCH_POINTS)}: {point}.')

            drone.move_to(
                x=point['x'],
                y=point['y'],
                z=SEARCH_ALTITUDE,
                yaw=0.0,
                reference = MoveReference.TAKEOFF,
            )

            self.log(f'Photo {i + 1}/{len(SEARCH_POINTS)}.')
            result = image_handler.take_photo()
            results.append(result)

            annotated = detector.draw_detections(result.image, result)
            cv2.imwrite(f'photo-{time.time_ns()}-{i}.png', annotated)

            # Search aruco
            if target_base['number'] is not None:
                aruco_detection = max(
                    result.filter_by_class(('aruco')),
                    key=lambda d: d.confidence,
                    default=None,
                )

                if aruco_detection:
                    self.log(f'Aruco detected.')

                    # number = get number of aruco
                    self.log(f'Aruco number: {number}.')

                    if number % 3 == 0:
                        multiple = 3
                    elif number % 4 == 0:
                        multiple = 4
                    else:
                        multiple = 5

                    self.log(f'Aruco multiplo: {multiple}.')
                    target_base['number'] = multiple

                    # Search aruco shape
                    for det in result:
                        if det.class_name != 'aruco' and \
                            (det.bbox[0] <= aruco_detection.center[0] <= det.bbox[2]) and \
                            (det.bbox[1] <= aruco_detection.center[1] <= det.bbox[3]):

                            self.log(f'Aruco shape: {det.class_name}.')
                            target_base['shape'] = det.class_name
                            break

            # Search correct land base
            if target_base['shape'] is not None:
                for j, result in enumerate(results):
                    same_shape = result.filter_by_class((target_base['shape']))
                    same_number = result.filter_by_class((target_base['number']))

                    for number in same_number:
                        for shape in same_shape:
                            if (shape.bbox[0] <= number.center[0] <= shape.bbox[2]) and \
                                (shape.bbox[1] <= number.center[1] <= shape.bbox[3]):

                                self.log('Find target base.')

                                target_base['x'] = SEARCH_POINTS[j]['x']
                                target_base['y'] = SEARCH_POINTS[j]['y']
                                blackboard['target_base'] = target_base

                                self.log('Completed successfully.')
                                return SUCCEED

        self.log(
            'Aruco/base not found.',
            style='error'
        )
        return FAIL
