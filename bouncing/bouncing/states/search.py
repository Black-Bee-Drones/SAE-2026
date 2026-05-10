import cv2

import rclpy
from rclpy.duration import Duration

import yasmin
from yasmin_ros.yasmin_node import YasminNode
from yasmin import State, Blackboard
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, TIMEOUT, ABORT

from nectar.control import MavrosDrone
from nectar.vision import ImageHandler

from bouncing.constants import (
    SEARCH_FIND_TOLERANCE,
    SEARCH_LIMITE_ALTITUDE,
    SEARCH_TARGET_ALTITUDE,
    SEARCH_TIMEOUT,
    SEARCH_VERTICAL_SPEED,
)


class Search(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, FAIL, TIMEOUT, ABORT])
        self.node = YasminNode.get_instance()


    def execute(self, blackboard: Blackboard):
        if ('drone' not in blackboard) or not blackboard['drone']:
            yasmin.YASMIN_LOG_ERROR('MavrosDrone not available.')
            return ABORT
        drone: MavrosDrone = blackboard['drone']

        if ('image_handler' not in blackboard) or not blackboard['image_handler']:
            yasmin.YASMIN_LOG_ERROR(f'ImageHandler not available.')
            return ABORT
        image_handler: ImageHandler = blackboard['image_handler']

        yasmin.YASMIN_LOG_INFO('Start.')

        target_base = {}
        count = 0
        start = self.node.get_clock().now()
        duration = Duration(seconds=SEARCH_TIMEOUT)
        while self.node.get_clock().now() - start < duration:
            rclpy.spin_once(self.node, timeout_sec=0.1)

            if count >= SEARCH_FIND_TOLERANCE:
                yasmin.YASMIN_LOG_INFO('Completed successfully.')
                return SUCCEED

            if drone.get_altitude() >= SEARCH_LIMITE_ALTITUDE:
                yasmin.YASMIN_LOG_ERROR('Failed: limit altitude reached.')
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                drone.delay(1.0)
                return FAIL

            drone.move_velocity(
                vx = 0.0,
                vy = 0.0,
                vz = SEARCH_VERTICAL_SPEED if drone.get_altitude() < SEARCH_TARGET_ALTITUDE else 0.0,
                vyaw = 0.0,
            )

            result = image_handler.take_photo()

            aruco, number = self.get_target_number(result)
            if aruco is None or number is None:
                yasmin.YASMIN_LOG_ERROR('Target NOT found.')
                count = 0
                continue
            target_base['number'] = str(number)

            aruco_shape = self.get_aruco_shape(result, aruco)
            if not aruco_shape:
                yasmin.YASMIN_LOG_ERROR(f'Shape of aruco NOT found. Target number: {target_base["number"]}.')
                count = 0
                continue

            target_base['shape'] = aruco_shape.class_name
            if blackboard['target_base'] != target_base:
                yasmin.YASMIN_LOG_INFO(f'Target base: {target_base}.')
                blackboard['target_base'] = target_base
                count = 0

            landing_base_number = self.get_landing_base_number(target_base, result)
            if not landing_base_number:
                yasmin.YASMIN_LOG_ERROR('Landing base NOT found.')
                count = 0
                continue

            count += 1
            yasmin.YASMIN_LOG_INFO(f'Landing base found ({count}/{SEARCH_FIND_TOLERANCE}).')

        yasmin.YASMIN_LOG_ERROR('Timeout.')
        return TIMEOUT

    def get_aruco_shape(self, result, aruco):
        aruco_shapes = []
        for s in result.filter_by_class(['0', '1', '2']):  # all shapes
            if (abs(aruco.center[0] - s.center[0]) <= s.width / 2) and (abs(aruco.center[1] - s.center[1]) <= s.height / 2):
                aruco_shapes.append(s)

        if aruco_shapes:
            aruco_shape = max(
                aruco_shapes,
                key=lambda shape: (shape.center[0] - aruco.center[0]) ** 2 + (shape.center[1] - aruco.center[1]) ** 2
            )
            return aruco_shape
        return None


    def get_target_number(self, result):
        for d in result.filter_by_class(['6']):
            x1, y1, x2, y2 = d.bbox

            h, w = result.image.shape[:2]

            x1 = min(w, max(0, int(x1 - w / 2)))
            y1 = min(h, max(0, int(y1 - h / 2)))
            x2 = min(w, max(0, int(x2 + w / 2)))
            y2 = min(h, max(0, int(y2 + h / 2)))

            crop = result.image[y1:y2, x1:x2]

            n = self.get_number_of_aruco(crop)

            if not n:
                return None, None

            if n % 3 == 0:
                return d, 3
            elif n % 4 == 0:
                return d, 4
            elif n % 5 == 0:
                return d, 5
        return None, None


    def get_number_of_aruco(self, img):
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dict_options = [
            cv2.aruco.DICT_5X5_50,
            cv2.aruco.DICT_5X5_100,
            cv2.aruco.DICT_5X5_250,
            cv2.aruco.DICT_5X5_1000,
        ]

        parameters = cv2.aruco.DetectorParameters()

        for dict_id in dict_options:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                return ids.flatten()

        return None
    
    def get_landing_base_number(self, target_base, result):
        area_img = result.image.shape[0] * result.image.shape[1]
        landing_bases = []
        for n in result.filter_by_class([target_base['number']]):
            for s in result.filter_by_class(['0', '1', '2']):
                if ((s.area / area_img) >= 0.6):
                    continue

                if (s.class_name == target_base['shape']):
                    yasmin.YASMIN_LOG_INFO(f'mesmo shape')
                    if (abs(n.center[0] - s.center[0]) <= s.width / 2) and (abs(n.center[1] - s.center[1]) <= s.height / 2):
                        landing_bases.append(n)

        if landing_bases:
            landing_base_number = max(
                landing_bases,
                key=lambda l: l.confidence
            )
            return landing_base_number

        return None
