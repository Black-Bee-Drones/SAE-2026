import yasmin
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, ABORT

from mirela_sdk.control.mavros import MavDrone
from mirela_sdk.image_processing.camera import ImageHandler
from mirela_sdk.control.pid import PIDController

from bouncing.utils import Detector


class Descend(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, FAIL, ABORT])

        self.DESCEND_VERTICAL_SPEED = self.node.get_parameter_or('descend_vertical_speed', 0.3)
        self.DESCEND_TIMEOUT = self.node.get_parameter_or('descend_timeout', 10)

        self.pid_x = PIDController(
            kp=self.node.get_parameter_or('controler_p_xy', 0.5),
            ki=self.node.get_parameter_or('controler_i_xy', 0),
            kd=self.node.get_parameter_or('controler_d_xy', 0),
            output_limits=0,
            integral_limits=0,
        )
        self.pid_y = PIDController(
            kp=self.node.get_parameter_or('controler_p_xy', 0.5),
            ki=self.node.get_parameter_or('controler_i_xy', 0),
            kd=self.node.get_parameter_or('controler_d_xy', 0),
            output_limits=0,
            integral_limits=0,
        )

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

        start = self.node.get_clock().now()
        while (self.node.get_clock().now() - start).nanoseconds / 1e9 < self.DESCEND_TIMEOUT:
            mavdrone.delay(0.1)
            yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Check target base.')
            frame = image_handler.take_photo()

            alt = mavdrone.get_heading.data
            for d in detector.detect(frame, position=(0,0, alt)):
                if d.get('class') == target_base.get('class') \
                    and d.get('number') == target_base.get('number') \
                    and not d.get('is_gabarito'):
                    error_x = d.get('x')
                    error_y = d.get('y')
                    break
            else:
                mavdrone.offboard_velocity()
                yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Completed successfully.')
                return SUCCEED

            output_x = self.pid_x.update(error_x)
            output_y = self.pid_y.update(error_y)

            yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Target base: {target_base}')
            yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Detection: error_x={error_x}, error_y={error_y}, output_x={output_x}, output_y={output_y}')
            mavdrone.offboard_velocity(
                self,
                linear_x = output_x,
                linear_y = output_y,
                linear_z = -self.DESCEND_VERTICAL_SPEED,
                angular_z = 0.0,
                ground_reference = False,
            )

        yasmin.YASMIN_LOG_INFO(f'{self.__state_name__}: Timeout.')
        return FAIL