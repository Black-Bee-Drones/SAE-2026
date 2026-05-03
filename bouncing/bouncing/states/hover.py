from rclpy.duration import Duration

import yasmin
from yasmin import State, Blackboard
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, TIMEOUT, ABORT

from nectar.control import MavrosDrone, PIDController
from nectar.vision import ImageHandler

from bouncing.constants import (
    HOVER_DETECTIONS_LOST_TOLERANCE,
    HOVER_TIMEOUT,
    HOVER_ALING_TOLERANCE,
    CONTROLER_P_XY,
    CONTROLER_I_XY,
    CONTROLER_D_XY,
)


class Hover(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, FAIL, TIMEOUT, ABORT])

        self.node = YasminNode.get_instance()

        self.pid_x = PIDController(
            kp=CONTROLER_P_XY,
            ki=CONTROLER_I_XY,
            kd=CONTROLER_D_XY,
            output_limits=(-0.3, 0.3),
        )

        self.pid_y = PIDController(
            kp=CONTROLER_P_XY,
            ki=CONTROLER_I_XY,
            kd=CONTROLER_D_XY,
            output_limits=(-0.3, 0.3),
        )


    def execute(self, blackboard: Blackboard):
        if ('drone' not in blackboard) or not blackboard['drone']:
            yasmin.YASMIN_LOG_ERROR('MavrosDrone not available.')
            return ABORT
        drone: MavrosDrone = blackboard['drone']

        if ('image_handler' not in blackboard) or not blackboard['image_handler']:
            yasmin.YASMIN_LOG_ERROR('ImageHandler not available.')
            return ABORT
        image_handler: ImageHandler = blackboard['image_handler']

        if ('target_base' not in blackboard) or not blackboard['target_base']:
            yasmin.YASMIN_LOG_ERROR('\"target_base\" not available.')
            return ABORT
        target_base: dict = blackboard['target_base']

        yasmin.YASMIN_LOG_INFO('Start.')

        yasmin.YASMIN_LOG_INFO(f'Start PID in landing base: {target_base}.')
        lost_detection_count = 0
        start = self.node.get_clock().now()
        duration = Duration(seconds=HOVER_TIMEOUT)
        while self.node.get_clock().now() - start < duration:

            yasmin.YASMIN_LOG_INFO(f'Take and process photo.')
            result = image_handler.take_photo()

            # Search number
            h, w = result.image.shape[:2]
            numbers = max(
                result.filter_by_class([target_base['number']]),
                key=lambda l: -((l.center[1] - (h / 2))**2 + (l.center[0] - (w / 2))**2)
            )

            if not numbers:
                lost_detection_count += 1
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                yasmin.YASMIN_LOG_ERROR(f'Lost detection ({lost_detection_count}/{HOVER_DETECTIONS_LOST_TOLERANCE}).')

                if lost_detection_count >= HOVER_DETECTIONS_LOST_TOLERANCE:
                    yasmin.YASMIN_LOG_ERROR('It lost detection many times.')
                    return FAIL
                continue

            else:
                landing_base_number = max(
                    numbers,
                    key=lambda n: n.confidence * n.area
                )

            lost_detection_count = 0

            h, w = result.image.shape[:2]
            center = landing_base_number.center

            # normalized error
            error_x = (center[1] - (h / 2)) / drone.get_altitude()
            error_y = (center[0] - (w / 2)) / drone.get_altitude()

            output_x = self.pid_x.update(error_x)
            output_y = self.pid_y.update(error_y)

            yasmin.YASMIN_LOG_INFO(f'Detection: error_x={error_x:.2f}, error_y={error_y:.2f}, output_x={output_x:.2f}, output_y={output_y:.2f}')

            if (error_x ** 2 + error_y ** 2 <= HOVER_ALING_TOLERANCE ** 2):
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                drone.delay(0.5)
                return SUCCEED
            else:
                drone.move_velocity(
                    vx = output_x,
                    vy = output_y,
                    vz = 0.0,
                    vyaw = 0.0,
                )

        yasmin.YASMIN_LOG_INFO('Timeout.')
        return TIMEOUT
