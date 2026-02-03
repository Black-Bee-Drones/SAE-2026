import time
import cv2

import yasmin
from yasmin import State, Blackboard
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, ABORT

from mirela_sdk.control.mavros import MavDrone
from mirela_sdk.control.pid import PIDController
from mirela_sdk.image_processing.camera import ImageHandler
from mirela_sdk.ai.detection import Detector


from bouncing.constants import (
    PRECISE_LANDING_ALTITUDE,
    PRECISE_LANDING_GO_TO_POINT_TIMEOUT,
    PRECISE_LANDING_DETECTIONS_LOST_TOLERANCE,
    PRECISE_LANDING_TIMEOUT,
    PRECISE_LANDING_VERTICAL_SPEED,
    PRECISE_LANDING_ALING_TOLERANCE,
    PRECISE_LANDING_LAND_ALTITUDE,
    CONTROLER_P_XY,
    CONTROLER_I_XY,
    CONTROLER_D_XY,
)


class PreciseLanding(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, FAIL, ABORT])

        self.pid_x = PIDController(
            kp=CONTROLER_P_XY,
            ki=CONTROLER_I_XY,
            kd=CONTROLER_D_XY,
            output_limits=0,
            integral_limits=0,
        )

        self.pid_y = PIDController(
            kp=CONTROLER_P_XY,
            ki=CONTROLER_I_XY,
            kd=CONTROLER_D_XY,
            output_limits=0,
            integral_limits=0,
        )


    def execute(self, blackboard: Blackboard):
        if ('mavdrone' not in blackboard) or not blackboard['mavdrone']:
            yasmin.YASMIN_LOG_ERROR('PreciseLanding(State): MavDrone not available.')
            return ABORT
        mavdrone: MavDrone = blackboard['mavdrone']

        if ('image_handler' not in blackboard) or not blackboard['image_handler']:
            yasmin.YASMIN_LOG_ERROR('PreciseLanding(State): ImageHandler not available.')
            return ABORT
        image_handler: ImageHandler = blackboard['image_handler']

        if ('detector' not in blackboard) or not blackboard['detector']:
            yasmin.YASMIN_LOG_ERROR('PreciseLanding(State): Detector not available.')
            return ABORT
        detector: Detector = blackboard['detector']

        if ('target_base' not in blackboard) or not blackboard['target_base']:
            yasmin.YASMIN_LOG_ERROR('PreciseLanding(State): \"target_base\" not available.')
            return ABORT
        target_base: dict = blackboard['target_base']


        yasmin.YASMIN_LOG_INFO('PreciseLanding(State): Start.')


        yasmin.YASMIN_LOG_INFO(f'PreciseLanding(State): Go to target base: {target_base}')
        mavdrone.offboard_position(
            x=target_base['x'],
            y=target_base['y'],
            z=PRECISE_LANDING_ALTITUDE,
            yaw=0.0,
            ground_reference=True,
            timeout_sec=PRECISE_LANDING_GO_TO_POINT_TIMEOUT,
            precision_radius=0.1,
            strategy='PID',
        )


        yasmin.YASMIN_LOG_INFO(f'PreciseLanding(State): Start PID in target base: {target_base}.')
        lost_detection_count = 0
        start = self.node.get_clock().now()
        while (self.node.get_clock().now() - start).nanoseconds / 1e9 < PRECISE_LANDING_TIMEOUT:

            if mavdrone.rel_alt <= PRECISE_LANDING_LAND_ALTITUDE:
                yasmin.YASMIN_LOG_INFO(f'PreciseLanding(State): Completed successfully.')
                mavdrone.offboard_velocity(
                    self,
                    linear_x = 0.0,
                    linear_y = 0.0,
                    linear_z = 0.0,
                    angular_z = 0.0,
                    ground_reference = False,
                )
                mavdrone.delay(1.0)
                return SUCCEED

            yasmin.YASMIN_LOG_INFO('PreciseLanding(State): Take photo.')
            result = image_handler.take_photo()

            annotated = detector.draw_detections(result.image, result)
            name = f'photo-{time.time_ns()}.png'
            cv2.imwrite(name, annotated)
            yasmin.YASMIN_LOG_INFO(f'PreciseLanding(State): Save annotated photo: {name}.')

            error_x, error_y = None, None

            same_number = result.filter_by_class_id((target_base['number']))

            if len(number) == 1:
                # Aling number
                error_x, error_y = same_number[0].center
            else:
                same_shape = result.filter_by_class_id((target_base['shape']))

                if len(number) == 1:
                    # Aling shape
                    error_x, error_y = same_shape[0].center

                else:
                    for number in same_number:
                        for shape in same_shape:
                            if (shape.bbox[0] <= number.center[0] <= shape.bbox[2]) and \
                                (shape.bbox[1] <= number.center[1] <= shape.bbox[3]):

                                # Aling number inside the shape
                                error_x, error_y = same_number[0].center
                                break

            if (error_x is not None) and (error_y is not None):
                lost_detection_count = 0

                output_x = self.pid_x.update(error_x)
                output_y = self.pid_y.update(error_y)

                yasmin.YASMIN_LOG_INFO(f'Descend(State): Detection: error_x={error_x}, error_y={error_y}, output_x={output_x}, output_y={output_y}')
                mavdrone.offboard_velocity(
                    self,
                    linear_x = output_x,
                    linear_y = output_y,
                    linear_z = -PRECISE_LANDING_VERTICAL_SPEED if (error_x ** 2 + error_y ** 2 <= PRECISE_LANDING_ALING_TOLERANCE ** 2) else 0.0,
                    angular_z = 0.0,
                    ground_reference = False,
                )

            else:
                lost_detection_count += 1
                yasmin.YASMIN_LOG_INFO(f'PreciseLanding(State): Lost detection ({lost_detection_count}/{PRECISE_LANDING_DETECTIONS_LOST_TOLERANCE}).')

                if lost_detection_count >= PRECISE_LANDING_DETECTIONS_LOST_TOLERANCE:
                    yasmin.YASMIN_LOG_INFO(f'PreciseLanding(State): FAIL, lost detection.')
                    return FAIL

        yasmin.YASMIN_LOG_INFO(f'PreciseLanding(State): FAIL, timeout.')
        return FAIL
