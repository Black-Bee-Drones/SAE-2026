import yasmin
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, ABORT

from mirela_sdk.control.mavros import MavDrone
from mirela_sdk.image_processing.camera import ImageHandler

from bouncing.utils import Detector


class Search(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, FAIL, ABORT])
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
    
    @property
    def __state_name__(self):
        return f'{self.__class__.__name__}({', '.join([cls.__name__ for cls in self.__class__.__bases__])})'

    def execute(self, blackboard):
        if ('mavdrone' not in blackboard) or not blackboard['mavdrone']:
            yasmin.YASMIN_LOG_ERROR(f'{self.__state_name__}: MavDrone not available.')
            return ABORT
        mavdrone: MavDrone = blackboard['mavdrone']

        if ('image_handler' not in blackboard) or not blackboard['image_handler']:
            yasmin.YASMIN_LOG_ERROR(f'{self.__state_name__}: ImageHandler not available.')
            return ABORT
        image_handler: ImageHandler = blackboard['image_handler']

        if ('detector' not in blackboard) or not blackboard['detector']:
            yasmin.YASMIN_LOG_ERROR(f'{self.__state_name__}: Detector not available.')
            return ABORT
        detector: Detector = blackboard['detector']

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Start.')

        target_base = None
        detections = []
        for i, point in enumerate(self.SEARCH_POINTS):
            yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Coordinate {i}/{len(self.SEARCH_POINTS)}: {point}.')

            mavdrone.offboard_position(
                x=point['x'],
                y=point['y'],
                z=self.SEARCH_ALTITUDE,
                yaw=0.0,
                ground_reference=True,
                timeout_sec=self.SEARCH_POINTS_TIMEOUT,
                precision_radius=0.1,
                strategy="PID",
            )
            position = (point['x'], point['y'], self.SEARCH_ALTITUDE)

            yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Photo {i}/{len(self.SEARCH_POINTS)}.')
            frame = image_handler.take_photo()
            for detection in detector.detect(frame, position):
                if detection.get('is_gabarito'):
                    yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Target base: {detection}.')
                    target_base = detection
                else:
                    yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Detection: {detection}.')
                    detections.append(detection)

                if self.get_target_base(target_base, detections):
                    blackboard['target_base'] = self.get_target_base(target_base, detections)

                    yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Completed successfully.')
                    return SUCCEED

        yasmin.YASMIN_LOG_ERROR(f'{self.__state_name__}: Aruco/base not found.')
        return FAIL

    def get_target_base(self, target_base, detections):
        if target_base is None:
            return False

        class_name = self.target_base.get('class')
        number = self.target_base.get('number')

        for detection in detections:
            if detection.get('class') == class_name and detection.get('number') == number:
                return detection
        return None
