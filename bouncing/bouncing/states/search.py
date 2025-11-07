import yasmin
from yasmin import State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone
from mirela_sdk.image_processing.camera import ImageHandler, ImageCalculus
from mirela_sdk.ai import YOLODetector

from bouncing.constants import (
    SEARCH_ALTITUDE,
    SEARCH_POINTS_TIMEOUT,
    SEARCH_POINTS,
)


class Search(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def detect_aruco(self, frame):
        return
    def detect_number(self, frame):
        return

    def check_succeed(blackboard):
        target_base = blackboard['target_base']

        if not target_base:
            return False

        detections = blackboard['detections']
        class_name = target_base.get('class')
        symbol = target_base.get('symbol')

        for detection in detections:
            if class_name == detection['class'] and symbol == detection['symbol']:
                return True
        return False


    def execute(self, blackboard):
        if ("mavdrone" not in blackboard) or not blackboard["mavdrone"]:
            yasmin.YASMIN_LOG_ERROR("MavDrone not available in SearchID state.")
            return ABORT
        mavdrone: MavDrone = blackboard["mavdrone"]

        if ("image_calculus" not in blackboard) or not blackboard["image_calculus"]:
            yasmin.YASMIN_LOG_ERROR("ImageCalculus not available in SearchID state.")
            return ABORT
        image_calculus: ImageCalculus = blackboard["image_calculus"]

        if ("image_handler" not in blackboard) or not blackboard["image_handler"]:
            yasmin.YASMIN_LOG_ERROR("ImageHandler not available in SearchID state.")
            return ABORT
        image_handler: ImageHandler = blackboard["image_handler"]

        if ("yolo_detector" not in blackboard) or not blackboard["yolo_detector"]:
            yasmin.YASMIN_LOG_ERROR("YOLODetector not available in SearchID state.")
            return ABORT
        yolo_detector: YOLODetector = blackboard["yolo_detector"]

        yasmin.YASMIN_LOG_INFO("Start SEARCH.")
        for point in SEARCH_POINTS:
            mavdrone.offboard_position(
                x=point['x'],
                y=point['y'],
                z=SEARCH_ALTITUDE,
                yaw = 0.0,
                ground_reference=True,
                timeout_sec=SEARCH_POINTS_TIMEOUT,
                precision_radius=0.1,
                strategy="PID",
            )

            frame = image_handler.take_photo()
            result = yolo_detector.detect(frame)

            for i, (bbox, cls_id) in enumerate(zip(result.boxes_xyxy, result.class_ids)):
                class_name = yolo_detector.class_names.get(cls_id, f"class_{cls_id}")

                x1, y1, x2, y2 = map(int, bbox)

                center_x = (x2 - x1) / 2
                center_y = (y2 - y1) / 2

                vector = image_calculus.calculate_vector_from_drone_to_ground(
                    altura = SEARCH_ALTITUDE,
                    target_pixel = (center_x, center_y),
                )

                crop = frame[y1:y2, x1:x2].copy()
                symbol = self.detect_symbol(crop)

                aruco = self.detect_aruco(crop)
                if aruco:
                    target_base = {
                        'class': class_name,
                        'symbol': aruco,
                    }
                    blackboard['target_base'] = target_base
                    yasmin.YASMIN_LOG_INFO(f'Detecção {i}: TARGET_BASE = {target_base}')

                else:
                    symbol = self.detect_number(crop)

                    x1, y1, x2, y2 = map(int, bbox)
                    center_x = (x2 - x1) / 2
                    center_y = (y2 - y1) / 2

                    detection = {
                        'class'  : class_name,
                        'symbol' : symbol,
                        'x'      : point['x'] + vector[0],
                        'y'      : point['y'] + vector[1],
                    }

                    blackboard['detections'].append(detection)
                    yasmin.YASMIN_LOG_INFO(f"Detecção {i}: {detection}")

                if self.check_succeed(blackboard):
                    return SUCCEED

        yasmin.YASMIN_LOG_WARN('ARUCO E/OU BASE NÃO ENCONTRADA')
        return ABORT
