import cv2
from typing import Iterable

from mirela_sdk.ai import YOLODetector


class Detector:
    def __init__(self, model_path, model_number_path, model_confidence_threshold, image_calculus):
        self.yolo_detector = YOLODetector(
            model_source=model_path,
            confidence_threshold=model_confidence_threshold,
            device='auto',
            auto_load=True,
        )

        self.number_detector = YOLODetector(
            model_source=model_number_path,
            confidence_threshold=model_confidence_threshold,
            device='auto',
            auto_load=True,
        )

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()

        self.image_calculus = image_calculus

    def detect_figure(self, frame):
        result = self.yolo_detector.detect(frame)
        detections = []

        for i, (bbox, cls_id) in enumerate(zip(result.boxes_xyxy, result.class_ids)):
            class_name = self.yolo_detector.class_names.get(cls_id, f'class_{cls_id}')

            x1, y1, x2, y2 = map(int, bbox)

            target_pixel = (
                (x2 - x1) / 2,
                (y2 - y1) / 2,
            )

            crop = frame[y1:y2, x1:x2].copy()

            detection = {
                'class': class_name,
                'target_pixel': target_pixel,
                'crop': crop,
            }

            print(f'Detecção {i}: {detection}')

            detections.append(detection)

        return detections

    def detect_symbol(self, frame):
        aruco_id = self.detect_aruco_id(frame)
        if aruco_id:
            return {
                'number': aruco_id,
                'is_gabarito': True,
            }
        return {
            'number': self.detect_number(frame),
            'is_gabarito': True,
        }

    def detect_aruco_id(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            return int(ids[0][0])
        return None

    def detect_number(self, frame):
        result = self.number_detector.detect(frame)
        class_name = self.yolo_detector.class_names.get(result.class_ids[0], f'class_{result.class_ids[0]}')
        return class_name

    def detect(self, frame, position=None) -> Iterable[dict]:
        for figure_detection in self.detect_figure(frame):
            if position:
                vector = self.image_calculus.calculate_vector_from_drone_to_ground(
                    altura=position[2],
                    target_pixel=figure_detection['target_pixel'],
                )
                figure_detection['x'] = position[0] + vector[0]
                figure_detection['y'] = position[1] + vector[1]
            symbol_detection = self.detect_symbol(figure_detection.pop('crop'))
            figure_detection.update(symbol_detection)
            yield figure_detection
