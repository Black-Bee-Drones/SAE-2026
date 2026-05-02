import cv2

import yasmin
from yasmin_ros.yasmin_node import YasminNode
from yasmin import State, Blackboard
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, ABORT

from nectar.vision import ImageHandler

from bouncing.constants import (
    SEARCH_NUMBER_DETECTIONS,
    SEARCH_DETECTIONS_LOST_TOLERANCE,
)


class Search(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, FAIL, ABORT])
        self.node = YasminNode.get_instance()


    def execute(self, blackboard: Blackboard):
        if ('image_handler' not in blackboard) or not blackboard['image_handler']:
            yasmin.YASMIN_LOG_ERROR(f'ImageHandler not available.')
            return ABORT
        image_handler: ImageHandler = blackboard['image_handler']

        yasmin.YASMIN_LOG_INFO('Start.')

        target_base = {} # {shape, number}

        count = 0
        for i in range(SEARCH_NUMBER_DETECTIONS):
            yasmin.YASMIN_LOG_INFO(f'Take and process photo {i+1}/{SEARCH_NUMBER_DETECTIONS}.')
            result = image_handler.take_photo()

            # Search aruco number
            find_arucos = []
            for d in result.filter_by_class(['6']): # aruco detections
                x1, y1, x2, y2 = d.bbox

                h, w = result.image.shape[:2]

                x1 = min(w, max(0, int(x1 - w / 2)))
                y1 = min(h, max(0, int(y1 - h / 2)))
                x2 = min(w, max(0, int(x2 + w / 2)))
                y2 = min(h, max(0, int(y2 + h / 2)))

                crop = result.image[y1:y2, x1:x2]

                n = self.get_number_of_aruco(crop)

                if not n:
                    continue

                if n % 3 == 0:
                    find_arucos.append((d, 3))
                elif n % 4 == 0:
                    find_arucos.append((d, 4))
                elif n % 5 == 0:
                    find_arucos.append((d, 5))

            if not find_arucos:
                yasmin.YASMIN_LOG_ERROR('Aruco not found.')
                count = 0
                continue

            aruco, number = max(find_arucos, key=lambda a: a[0].confidence)
            target_base['number'] = str(number)
            yasmin.YASMIN_LOG_INFO(f'Aruco found. Target number: {target_base["number"]}.')

            # Search aruco shape
            aruco_shapes = []
            for s in result.filter_by_class(['0', '1', '2']):  # all shapes
                if (abs(aruco.center[0] - s.center[0]) <= s.width / 2) and (abs(aruco.center[1] - s.center[1]) <= s.height / 2):
                    aruco_shapes.append(s)

            if not aruco_shapes:
                yasmin.YASMIN_LOG_ERROR('Shape of aruco not found.')
                count = 0
                continue

            aruco_shape = max(
                aruco_shapes,
                key=lambda shape: (shape.center[0] - aruco.center[0]) ** 2 + (shape.center[1] - aruco.center[1]) ** 2
            )

            target_base['shape'] = aruco_shape.class_name
            yasmin.YASMIN_LOG_INFO(f'Shape of aruco found. Target shape: {target_base["shape"]}.')

            # Save target base on blackboard
            blackboard['target_base'] = target_base

            # Search Landing base
            landing_bases = []
            for s in result.filter_by_class([target_base['shape']]):  # all target shapes
                for n in result.filter_by_class([target_base['number']]):  # all target numbers
                    if (abs(n.center[0] - s.center[0]) <= s.width / 2) and (abs(n.center[1] - s.center[1]) <= s.height / 2):
                        landing_bases.append((s, n))

            if not landing_bases:
                yasmin.YASMIN_LOG_ERROR('Target base found, but landing base not found.')
                count = 0
                continue

            landing_bases_shape, landing_base_number = max(
                landing_bases,
                key=lambda l: l[0].confidence * l[1].confidence
            )

            yasmin.YASMIN_LOG_INFO(f'Landing base found.')
            count += 1

            if count >= SEARCH_DETECTIONS_LOST_TOLERANCE:
                yasmin.YASMIN_LOG_INFO('Completed successfully.')
                return SUCCEED

        yasmin.YASMIN_LOG_ERROR('Target and/or landing base not found.')
        return FAIL


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
