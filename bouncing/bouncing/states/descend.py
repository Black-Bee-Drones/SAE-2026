import yasmin
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone
from mirela_sdk.image_processing.camera import ImageHandler

from bouncing.utils import Detector


class Descend(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.DESCEND_STAP_SIZE = self.node.get_parameter_or('descend_stap_size', 0.5)
        self.DESCEND_TIMEOUT = self.node.get_parameter_or('descend_timeout', 10)

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

        while True:
            yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Check target base.')
            frame = image_handler.take_photo()
            found = False
            for d in detector.detect(frame):
                if d.get('class') == target_base.get('class') \
                    and d.get('number') == target_base.get('number') \
                    and not d.get('is_gabarito'):
                    target_base = d
                    found = True
                    break

            if not found:
                yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Completed successfully.')
                return SUCCEED

            yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Go to target base: {target_base}')
            mavdrone.offboard_position(
                x=target_base['x'],
                y=target_base['y'],
                z=-self.DESCEND_STAP_SIZE,
                yaw=0.0,
                ground_reference=False,
                timeout_sec=self.DESCEND_TIMEOUT,
                precision_radius=0.1,
                strategy='PID',
            )
