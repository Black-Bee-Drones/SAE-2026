import math

import rclpy
from rclpy.duration import Duration

import yasmin
from yasmin import State, Blackboard
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, TIMEOUT, ABORT

from nectar.control import MavrosDrone, PIDController
from nectar.vision import ImageHandler

from bouncing.constants import (
    PRECISE_LIMITE_ALTITUDE,
    PRECISE_LIMITE_RECOVERY,
    PRECISE_HOVER_COUNT,
    PRECISE_RESET_PID,
    PRECISE_LOST_TOLERANCE,
    PRECISE_TIMEOUT,
    PRECISE_VERTICAL_SPEED,
    PRECISE_ALING_TOLERANCE,
    PRECISE_LAND_ALTITUDE,
    PRECISE_DOWN_TOLERANCE_PX,
    CONTROLER_P_XY,
    CONTROLER_I_XY,
    CONTROLER_D_XY,
    CONTROLER_OUTPUT_LIMITS_XY,
    CONTROLER_INTEGRAL_LIMITS_XY,
    CONTROLER_P_Z,
    CONTROLER_I_Z,
    CONTROLER_D_Z,
    CONTROLER_OUTPUT_LIMITS_Z,
    CONTROLER_INTEGRAL_LIMITS_Z,
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

        self.pid_z = PIDController(
            kp=CONTROLER_P_Z,
            ki=CONTROLER_I_Z,
            kd=CONTROLER_D_Z,
            output_limits=CONTROLER_OUTPUT_LIMITS_Z,
            integral_limits=CONTROLER_INTEGRAL_LIMITS_Z,
        )
    
    def ppm(self, altitude_m: float, fov_deg: float, width: float):
        half_fov_rad = math.radians(fov_deg/2.0)
        return width / (2.0 * altitude_m * math.tan(half_fov_rad))


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
                    if drone.get_altitude() >= PRECISE_LIMITE_RECOVERY:
                        yasmin.YASMIN_LOG_ERROR('Recovery: Limit.')
                        drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                        continue

                    yasmin.YASMIN_LOG_ERROR('Recovery: It lost detection many times.')
                    drone.move_velocity(vz=PRECISE_VERTICAL_SPEED)
                continue

            if lost_detection_count >= PRECISE_RESET_PID:
                self.pid_x.reset()
                self.pid_y.reset()
            lost_detection_count = 0

            h, w = result.image.shape[:2]
            center = landing_base_number.center

            error_x_px = (center[1] - (h / 2))
            error_y_px = (center[0] - (w / 2))

            alt = drone.get_altitude()
            error_x = error_x_px / self.ppm(alt, 86, w)
            error_y = error_y_px / self.ppm(alt, 47, h)
            error_z = PRECISE_LAND_ALTITUDE - drone.get_altitude()

            ert_dig_px = math.hypot(error_x_px, error_y_px)
            ert_dig = math.hypot(error_x, error_y)

            output_x = self.pid_x.update(error_x)
            output_y = self.pid_y.update(error_y)
            output_z = self.pid_z.update(error_z)

            yasmin.YASMIN_LOG_INFO(f'Detection: alt={alt:.1f}, ert_dig={ert_dig:.2f}, error_x={error_x:.2f}, error_y={error_y:.2f}, output_x={output_x:.2f}, output_y={output_y:.2f}')

            if (ert_dig <= PRECISE_ALING_TOLERANCE) and (alt <= PRECISE_LAND_ALTITUDE):
                hover_count += 1
                yasmin.YASMIN_LOG_INFO(f'Hovering ({hover_count}/{PRECISE_HOVER_COUNT}).')
            else:
                hover_count = 0

            drone.move_velocity(
                vx = output_x,
                vy = output_y,
                vz = output_z if (ert_dig_px <= PRECISE_DOWN_TOLERANCE_PX) else 0.0,
                vyaw = 0.0,
            )

        yasmin.YASMIN_LOG_ERROR('Timeout.')
        return TIMEOUT


    def get_landing_base(self, target_base: dict, result):
        area_img = result.image.shape[0] * result.image.shape[1]
        landing_bases = []
        numbers = []
        for n in result.filter_by_class([target_base['number']]):
            valid_number = True
            for s in result.filter_by_class(['0', '1', '2']):
                if ((s.area / area_img) >= 0.6):
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
