import time
import cv2

import yasmin
from yasmin import State, Blackboard
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, ABORT

from nectar.control import MavrosDrone, MoveReference, PIDController
from nectar.vision import ImageHandler
from nectar.ai import Detector

from bouncing.constants import (
    PRECISE_LANDING_ALTITUDE,
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


    def log(self, msg, style='info'):
        class_name = f'{self.__class__.__name__}({', '.join([cls.__name__ for cls in self.__class__.__bases__])})'

        if style == 'info':
            yasmin.YASMIN_LOG_INFO(f'{class_name}: {msg}')
        elif style == 'error':
            yasmin.YASMIN_LOG_ERROR(f'{class_name}: {msg}')


    def execute(self, blackboard: Blackboard):
        if ('drone' not in blackboard) or not blackboard['drone']:
            self.log(
                'MavrosDrone not available.',
                style='error'
            )
            return ABORT
        drone: MavrosDrone = blackboard['drone']

        if ('image_handler' not in blackboard) or not blackboard['image_handler']:
            self.log(
                'ImageHandler not available.',
                style='error'
            )
            return ABORT
        image_handler: ImageHandler = blackboard['image_handler']

        if ('detector' not in blackboard) or not blackboard['detector']:
            self.log(
                'Detector not available.',
                style='error'
            )
            return ABORT
        detector: Detector = blackboard['detector']

        if ('target_base' not in blackboard) or not blackboard['target_base']:
            self.log(
                '\"target_base\" not available.',
                style='error'
            )
            return ABORT
        target_base: dict = blackboard['target_base']

        self.log('Start.')

        self.log(f'Go to target base: {target_base}')
        drone.offboard_position(
            x=target_base['x'],
            y=target_base['y'],
            z=PRECISE_LANDING_ALTITUDE,
            yaw=0.0,
            reference = MoveReference.TAKEOFF,
        )

        self.log(f'Start PID in target base: {target_base}.')
        lost_detection_count = 0
        start = self.node.get_clock().now()
        while (self.node.get_clock().now() - start).nanoseconds / 1e9 < PRECISE_LANDING_TIMEOUT:

            if drone.rel_alt <= PRECISE_LANDING_LAND_ALTITUDE:
                self.log(f'Completed successfully.')
                drone.move_velocity()
                drone.delay(1.0)
                return SUCCEED

            self.log('Take photo.')
            result = image_handler.take_photo()

            annotated = detector.draw_detections(result.image, result)
            name = f'photo-{time.time_ns()}.png'
            cv2.imwrite(name, annotated)
            self.log(f'Save annotated photo: {name}.')

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

                self.log(f'Detection: error_x={error_x}, error_y={error_y}, output_x={output_x}, output_y={output_y}')
                drone.move_velocity(
                    vx = output_x,
                    vy = output_y,
                    vz = -PRECISE_LANDING_VERTICAL_SPEED if (error_x ** 2 + error_y ** 2 <= PRECISE_LANDING_ALING_TOLERANCE ** 2) else 0.0,
                    vyaw = 0.0,
                )

            else:
                lost_detection_count += 1
                self.log(f'Lost detection ({lost_detection_count}/{PRECISE_LANDING_DETECTIONS_LOST_TOLERANCE}).')

                if lost_detection_count >= PRECISE_LANDING_DETECTIONS_LOST_TOLERANCE:
                    self.log(f'FAIL, lost detection.')
                    return FAIL

        self.log(f'FAIL, timeout.')
        return FAIL
