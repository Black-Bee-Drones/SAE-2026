import cv2
import time

import yasmin
from yasmin import State, Blackboard
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, ABORT

from mirela_sdk.control.mavros import MavDrone
from mirela_sdk.image_processing.camera import ImageHandler
from mirela_sdk.ai.detection import Detector

from bouncing.constants import (
    SEARCH_ALTITUDE,
    SEARCH_POINTS_TIMEOUT,
    SEARCH_POINTS,
)

class Search(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, FAIL, ABORT])


    def execute(self, blackboard: Blackboard):
        if ('mavdrone' not in blackboard) or not blackboard['mavdrone']:
            yasmin.YASMIN_LOG_ERROR(f'Search(State): MavDrone not available.')
            return ABORT
        mavdrone: MavDrone = blackboard['mavdrone']

        if ('image_handler' not in blackboard) or not blackboard['image_handler']:
            yasmin.YASMIN_LOG_ERROR(f'Search(State): ImageHandler not available.')
            return ABORT
        image_handler: ImageHandler = blackboard['image_handler']

        if ('detector' not in blackboard) or not blackboard['detector']:
            yasmin.YASMIN_LOG_ERROR(f'Search(State): Detector not available.')
            return ABORT
        detector: Detector = blackboard['detector']

        yasmin.YASMIN_LOG_INFO('Search(State): Start.')

        target_base = {
            'shape': None,
            'number': None,
            'x': None,
            'y': None,
        }
        results = []
        for i, point in enumerate(SEARCH_POINTS):
            yasmin.YASMIN_LOG_INFO(f'Search(State): Coordinate {i + 1}/{len(SEARCH_POINTS)}: {point}.')

            mavdrone.offboard_position(
                x=point['x'],
                y=point['y'],
                z=SEARCH_ALTITUDE,
                yaw=0.0,
                ground_reference=True,
                timeout_sec=SEARCH_POINTS_TIMEOUT,
                precision_radius=0.1,
                strategy="PID",
            )

            yasmin.YASMIN_LOG_INFO(f'Search(State): Photo {i + 1}/{len(SEARCH_POINTS)}.')
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
                    yasmin.YASMIN_LOG_INFO(f'Search(State): Aruco detected.')

                    # number = get number of aruco
                    yasmin.YASMIN_LOG_INFO(f'Search(State): Aruco number: {number}.')

                    if number % 3 == 0:
                        multiple = 3
                    elif number % 4 == 0:
                        multiple = 4
                    else:
                        multiple = 5

                    yasmin.YASMIN_LOG_INFO(f'Search(State): Aruco multiplo: {multiple}.')
                    target_base['number'] = multiple

                    # Search aruco shape
                    for det in result:
                        if det.class_name != 'aruco' and \
                            (det.bbox[0] <= aruco_detection.center[0] <= det.bbox[2]) and \
                            (det.bbox[1] <= aruco_detection.center[1] <= det.bbox[3]):

                            yasmin.YASMIN_LOG_INFO(f'Search(State): Aruco shape: {det.class_name}.')
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

                                yasmin.YASMIN_LOG_INFO(f'Search(State): Find target base.')

                                target_base['x'] = SEARCH_POINTS[j]['x']
                                target_base['y'] = SEARCH_POINTS[j]['y']
                                blackboard['target_base'] = target_base

                                yasmin.YASMIN_LOG_INFO(f'Search(State): Completed successfully.')
                                return SUCCEED

        yasmin.YASMIN_LOG_ERROR(f'Search(State): Aruco/base not found.')
        return FAIL
