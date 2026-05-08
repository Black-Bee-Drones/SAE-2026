from rclpy.duration import Duration

import yasmin
from yasmin import State, Blackboard
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, TIMEOUT, ABORT

from nectar.control import MavrosDrone, PIDController
from nectar.vision import ImageHandler

from bouncing.constants import (
    PRECISE_LIMITE_ALTITUDE,
    PRECISE_HOVER_COUNT,
    PRECISE_RESET_PID,
    PRECISE_LOST_TOLERANCE,
    PRECISE_TIMEOUT,
    PRECISE_VERTICAL_SPEED,
    PRECISE_ALING_TOLERANCE,
    PRECISE_LAND_ALTITUDE,
    CONTROLER_P_XY,
    CONTROLER_I_XY,
    CONTROLER_D_XY,
    CONTROLER_OUTPUT_LIMITS_XY,
    CONTROLER_INTEGRAL_LIMITS_XY,
)


class PreciseLanding(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, FAIL, TIMEOUT, ABORT])

        self.node = YasminNode.get_instance()

        self.pid_x = PIDController(
            kp=CONTROLER_P_XY,
            ki=CONTROLER_I_XY,
            kd=CONTROLER_D_XY,
            output_limits=CONTROLER_OUTPUT_LIMITS_XY,
            integral_limits=CONTROLER_INTEGRAL_LIMITS_XY,
        )

        self.pid_y = PIDController(
            kp=CONTROLER_P_XY,
            ki=CONTROLER_I_XY,
            kd=CONTROLER_D_XY,
            output_limits=CONTROLER_OUTPUT_LIMITS_XY,
            integral_limits=CONTROLER_INTEGRAL_LIMITS_XY,
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
        hover_count = 0
        start = self.node.get_clock().now()
        duration = Duration(seconds=PRECISE_TIMEOUT)
        while self.node.get_clock().now() - start < duration:

            if hover_count >= PRECISE_HOVER_COUNT:
                yasmin.YASMIN_LOG_INFO(f'Completed successfully.')
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                drone.delay(1.0)
                return SUCCEED

            if drone.get_altitude() >= PRECISE_LIMITE_ALTITUDE:
                yasmin.YASMIN_LOG_ERROR('Failed: limit altitude reached.')
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                drone.delay(1.0)
                return FAIL

            result = image_handler.take_photo()
            landing_base_number = self.get_landing_base(target_base, result)

            if landing_base_number is None:
                lost_detection_count += 1

                if lost_detection_count <= PRECISE_LOST_TOLERANCE:
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    yasmin.YASMIN_LOG_ERROR(f'Lost detection ({lost_detection_count}/{PRECISE_LOST_TOLERANCE}).')

                else:
                    yasmin.YASMIN_LOG_ERROR('Recovery: It lost detection many times.')
                    drone.move_velocity(vz=PRECISE_VERTICAL_SPEED)

                continue

            if lost_detection_count >= PRECISE_RESET_PID:
                self.pid_x.reset()
                self.pid_y.reset()
            lost_detection_count = 0

            h, w = result.image.shape[:2]
            center = landing_base_number.center

            error_x = (center[1] - (h / 2)) / drone.get_altitude()
            error_y = (center[0] - (w / 2)) / drone.get_altitude()

            centralized = error_x ** 2 + error_y ** 2 <= PRECISE_ALING_TOLERANCE ** 2
            height_is_low = drone.get_altitude() <= PRECISE_LAND_ALTITUDE

            output_x = self.pid_x.update(error_x)
            output_y = self.pid_y.update(error_y)
            output_z = -PRECISE_VERTICAL_SPEED if (centralized and not height_is_low) else 0.0

            yasmin.YASMIN_LOG_INFO(f'Detection: error_x={error_x:.2f}, error_y={error_y:.2f}, output_x={output_x:.2f}, output_y={output_y:.2f}')

            if centralized and height_is_low:
                hover_count += 1
                yasmin.YASMIN_LOG_INFO(f'Hovering ({hover_count}/{PRECISE_HOVER_COUNT}).')
            else:
                hover_count = 0

            drone.move_velocity(
                vx = output_x,
                vy = output_y,
                vz = output_z,
                vyaw = 0.0,
            )

        yasmin.YASMIN_LOG_INFO('Timeout.')
        return TIMEOUT


    def get_landing_base(self, target_base: dict, result):
        area_img = result.image.shape[0] * result.image.shape[1]
        landing_bases = []
        numbers = []
        for n in result.filter_by_class([target_base['number']]):
            valid_number = True
            for s in result.filter_by_class(['0', '1', '2']):
                if ((s.area / area_img) <= 0.6):
                    continue

                if (s.class_name == target_base['shape']):
                    if (abs(n.center[0] - s.center[0]) <= s.width / 2) and (abs(n.center[1] - s.center[1]) <= s.height / 2):
                        landing_bases.append(n)

                else:
                    if (abs(n.center[0] - s.center[0]) <= s.width / 2) and (abs(n.center[1] - s.center[1]) <= s.height / 2):
                        valid_number = False

            if valid_number:
                numbers.append(n)

        if landing_bases:
            landing_base_number = max(
                landing_bases,
                key=lambda l: l.confidence
            )
            return landing_base_number

        elif numbers:
            landing_base_number = max(
                numbers,
                key=lambda n: n.confidence
            )
            return landing_base_number
        return None
