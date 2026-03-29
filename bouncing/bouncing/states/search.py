import cv2

import yasmin
from yasmin_ros.yasmin_node import YasminNode
from yasmin import State, Blackboard
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, ABORT

from nectar.vision import ImageHandler, Detector

from bouncing.constants import (
    SEARCH_NUMBER_DETECTIONS,
    SEARCH_DETECTIONS_LOST_TOLERANCE,
)


class Search(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, FAIL, ABORT])
        self.node = YasminNode.get_instance()


    def log(self, msg, style='info'):
        class_name = f'{self.__class__.__name__}({', '.join([cls.__name__ for cls in self.__class__.__bases__])})'

        if style == 'info':
            yasmin.YASMIN_LOG_INFO(f'{class_name}: {msg}')
        elif style == 'error':
            yasmin.YASMIN_LOG_ERROR(f'{class_name}: {msg}')


    def execute(self, blackboard: Blackboard):
        if ('image_handler' not in blackboard) or not blackboard['image_handler']:
            self.log(
                f'ImageHandler not available.',
                style='error'
            )
            return ABORT
        image_handler: ImageHandler = blackboard['image_handler']

        self.log('Start.')

        target_base = {} # {shape, number}

        count = 0
        for i in range(SEARCH_NUMBER_DETECTIONS):
            self.log(f'Take and process photo {i+1}/{SEARCH_NUMBER_DETECTIONS}.')
            result = image_handler.take_photo()

            now = self.node.get_clock().now().nanoseconds
            name = f'photo-{now}.png'
            cv2.imwrite(name, result.image)
            self.log(f'Save raw photo: {name}.')

            annotated = Detector.draw_detections(result.image, result)
            annotated_name = f'photo-{now}-annotated.png'
            cv2.imwrite(annotated_name, annotated)
            self.log(f'Save annotated photo: {annotated_name}.')

            # Search aruco number
            find_arucos = []
            for a in list(d for d in result if d.class_name == '6'): # aruco detections
                x1, y1, x2, y2 = a.bbox

                h, w = result.image.shape[:2]

                x1 = min(w, max(0, int(x1 - a.width / 2)))
                y1 = min(h, max(0, int(y1 - a.height / 2)))
                x2 = min(w, max(0, int(x2 + a.width / 2)))
                y2 = min(h, max(0, int(y2 + a.height / 2)))

                crop = a.image[y1:y2, x1:x2]

                n = self.get_number_of_aruco(crop)

                if not n:
                    continue

                if n % 3 == 0:
                    find_arucos.append((a, 3))
                elif n % 4 == 0:
                    find_arucos.append((a, 4))
                elif n % 5 == 0:
                    find_arucos.append((a, 5))

            if not find_arucos:
                self.log(
                    'Aruco not found.',
                    style='error'
                )
                count = 0
                continue

            aruco, target_base['number'] = max(find_arucos, key=lambda a: a[0].confidence)
            self.log(f'Aruco found. Target number: {target_base["number"]}.')

            # Search aruco shape
            aruco_shapes = []
            for s in list(d for d in result if d.class_name == ('0', '1', '2')):  # all shapes
                if (abs(aruco.center[0] - s.center[0]) <= s.width / 2) and (abs(aruco.center[1] - s.center[1]) <= s.height / 2):
                    aruco_shapes.append((s, n))

            if not aruco_shapes:
                self.log(
                    'Shape of aruco not found.',
                    style='error'
                )
                count = 0
                continue

            aruco_shape = max(
                aruco_shapes,
                key=lambda shape: (shape.center[0] - aruco.center[0]) ** 2 + (shape.center[1] - aruco.center[1]) ** 2
            )

            target_base['shape'] = aruco_shape.class_name
            self.log(f'Shape of aruco found. Target shape: {target_base["shape"]}.')

            # Save target base on blackboard
            blackboard['target_base'] = target_base

            # Search Landing base
            landing_bases = []
            for s in list(d for d in result if d.class_name == target_base['shape']):  # all target shapes
                for n in list(d for d in result if d.class_name == target_base['number']):  # all target numbers
                    if (abs(n.center[0] - s.center[0]) <= s.width / 2) and (abs(n.center[1] - s.center[1]) <= s.height / 2):
                        landing_bases.append((s, n))

            if not landing_bases:
                self.log(
                    'Target base found, but landing base not found.',
                    style='error'
                )
                count = 0
                continue

            landing_bases_shape, landing_base_number = max(
                landing_bases,
                key=lambda l: l[0].confidence * l[1].confidence
            )

            count += 1

            if count >= SEARCH_DETECTIONS_LOST_TOLERANCE:
                self.log('Completed successfully.')
                return SUCCEED

        self.log(
            'Target and/or landing base not found.',
            style='error'
        )
        return FAIL


    def get_number_of_aruco(img):
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
                return {
                    "dict": dict_id,
                    "ids": ids.flatten()
                }

        return None