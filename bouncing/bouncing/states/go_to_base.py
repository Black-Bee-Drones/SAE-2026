import yasmin
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, ABORT

from mirela_sdk.control.mavros import MavDrone
from mirela_sdk.image_processing.camera import ImageHandler

from bouncing.utils import Detector


class GoToBase(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, FAIL, ABORT])

        self.GO_TO_BASE_ALTITUDE = self.node.get_parameter_or('go_to_base_altitude', 3)
        self.GO_TO_BASE_NUMBER_PHOTO = self.node.get_parameter_or('go_to_base_number_photo', 3)
        self.GO_TO_BASE_TIMEOUT = self.node.get_parameter_or('go_to_base_timeout', 30)

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

        if ('target_base' not in blackboard) or not blackboard['target_base']:
            yasmin.YASMIN_LOG_ERROR(f'{self.__state_name__}: \"target_base\" not available.')
            return ABORT
        target_base: dict = blackboard['target_base']

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Start.')

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Go to target base: {target_base}')
        mavdrone.offboard_position(
            x=target_base['x'],
            y=target_base['y'],
            z=self.GO_TO_BASE_ALTITUDE,
            yaw=0.0,
            ground_reference=True,
            timeout_sec=self.GO_TO_BASE_TIMEOUT,
            precision_radius=0.1,
            strategy='PID',
        )

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Check target base ({target_base}).')
        for i in range(self.GO_TO_BASE_NUMBER_PHOTO):
            yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Take photo {i+1}/{len(self.GO_TO_BASE_NUMBER_PHOTO)}.')
            frame = image_handler.take_photo()
            for detection in detector.detect(frame):
                yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Detection: {detection}.')
                if detection.get('class') == target_base.get('class') \
                    and detection.get('number') == target_base.get('number') \
                    and not detection.get('is_gabarito'):
                    yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Completed successfully.')
                    return SUCCEED

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Not found target base ({target_base}).')
        return FAIL
