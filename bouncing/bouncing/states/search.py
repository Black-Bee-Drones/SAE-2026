import yasmin
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone
from mirela_sdk.image_processing.camera import ImageHandler, ImageCalculus
from mirela_sdk.ai import YOLODetector


class Search(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.SEARCH_ALTITUDE = self.node.get_parameter_or('search_altitude', 5)
        self.SEARCH_POINTS_TIMEOUT = self.node.get_parameter_or('search_points_timeout', 30)
        self.SEARCH_POINTS = self.node.get_parameter_or('search.points', [
            {'x': 0.0, 'y': 0.0},
            {'x': 4.6, 'y': 0.0},
            {'x': 4.6, 'y': 0.0},
        ])

    @property
    def node(self):
        return YasminNode.get_instance()

    def detect_aruco(self, frame):
        return
    def detect_number(self, frame):
        return

    def check_succeed(self, blackboard):
        if not self.target_base:
            return False

        class_name = self.target_base.get('class')
        symbol = self.target_base.get('symbol')

        for detection in self.detections:
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
        self.target_base = {}
        self.detections = []
        for point in self.SEARCH_POINTS:
            mavdrone.offboard_position(
                x=point['x'],
                y=point['y'],
                z=self.SEARCH_ALTITUDE,
                yaw = 0.0,
                ground_reference=True,
                timeout_sec=self.SEARCH_POINTS_TIMEOUT,
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
                    altura = self.SEARCH_ALTITUDE,
                    target_pixel = (center_x, center_y),
                )

                crop = frame[y1:y2, x1:x2].copy()
                symbol = self.detect_symbol(crop)

                aruco = self.detect_aruco(crop)
                if aruco:
                    self.target_base = {
                        'class': class_name,
                        'symbol': aruco,
                    }
                    yasmin.YASMIN_LOG_INFO(f'Detecção {i}: TARGET_BASE = {self.target_base}')

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

                    self.detections.append(detection)
                    yasmin.YASMIN_LOG_INFO(f"Detecção {i}: {detection}")

                if self.check_succeed(blackboard):
                    blackboard['target_base'] = self.target_base
                    return SUCCEED

        yasmin.YASMIN_LOG_WARN('ARUCO E/OU BASE NÃO ENCONTRADA')
        return ABORT
