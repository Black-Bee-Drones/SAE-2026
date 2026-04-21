import cv2

from rclpy.duration import Duration

import yasmin
from yasmin import State, Blackboard
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, TIMEOUT, ABORT

from nectar.control import MavrosDrone, PIDController
from nectar.vision import ImageHandler
from nectar.ai import Detector

from bouncing.constants import (
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
        super().__init__(outcomes=[SUCCEED, FAIL, TIMEOUT, ABORT])

        self.node = YasminNode.get_instance()

        self.pid_x = PIDController(
            kp=CONTROLER_P_XY,
            ki=CONTROLER_I_XY,
            kd=CONTROLER_D_XY,
        )

        self.pid_y = PIDController(
            kp=CONTROLER_P_XY,
            ki=CONTROLER_I_XY,
            kd=CONTROLER_D_XY,
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

        if ('detector' not in blackboard) or not blackboard['detector']:
            yasmin.YASMIN_LOG_ERROR(f'Detector not available.')
            return ABORT
        detector: Detector = blackboard['detector']

        yasmin.YASMIN_LOG_INFO('Start.')

        yasmin.YASMIN_LOG_INFO(f'Start PID in landing base: {target_base}.')
        lost_detection_count = 0
        start = self.node.get_clock().now()
        duration = Duration(seconds=PRECISE_LANDING_TIMEOUT)
        while self.node.get_clock().now() - start < duration:

            if drone.rel_alt <= PRECISE_LANDING_LAND_ALTITUDE:
                yasmin.YASMIN_LOG_INFO(f'Completed successfully.')
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                drone.delay(1.0)
                return SUCCEED

            yasmin.YASMIN_LOG_INFO(f'Take and process photo.')
            img, result = image_handler.take_photo()

            now = self.node.get_clock().now().nanoseconds
            name = f'photo-{now}.png'
            cv2.imwrite(name, img)
            yasmin.YASMIN_LOG_INFO(f'Save raw photo: {name}.')

            annotated = detector.draw_detections(img, result)
            annotated_name = f'photo-{now}-annotated.png'
            cv2.imwrite(annotated_name, annotated)
            yasmin.YASMIN_LOG_INFO(f'Save annotated photo: {annotated_name}.')

            # Search number inside shape
            landing_bases = []
            for s in list(d for d in result if d.class_name == target_base['shape']):  # all target shapes
                for n in list(d for d in result if d.class_name == str(target_base['number'])):  # all target numbers
                    if (abs(n.center[0] - s.center[0]) <= s.width / 2) and (abs(n.center[1] - s.center[1]) <= s.height / 2):
                        landing_bases.append((s, n))

            if landing_bases:
                landing_bases_shape, landing_base_number = max(
                    landing_bases,
                    key=lambda l: l[0].confidence * l[1].confidence
                )

            # Search number
            else:
                yasmin.YASMIN_LOG_INFO('Shape not found. Try get number.')
                numbers = list(d for d in result if d.class_name == str(target_base['number']))

                if not numbers:
                    lost_detection_count += 1
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    yasmin.YASMIN_LOG_ERROR(f'Lost detection ({lost_detection_count}/{PRECISE_LANDING_DETECTIONS_LOST_TOLERANCE}).')

                    if lost_detection_count >= PRECISE_LANDING_DETECTIONS_LOST_TOLERANCE:
                        yasmin.YASMIN_LOG_ERROR('It lost detection many times.')
                        return FAIL
                    continue

                elif len(numbers) == 1:
                    landing_base_number = numbers.pop()

                else:
                    landing_base_number = max(
                        numbers,
                        key=lambda n: n.confidence * n.area
                    )

            lost_detection_count = 0

            h, w = img.shape[:2]
            center = landing_base_number.center

            # normalized error
            error_x = (center[0] - (w / 2)) / drone.rel_alt
            error_y = (center[1] - (h / 2)) / drone.rel_alt

            output_x = self.pid_x.update(error_x)
            output_y = self.pid_y.update(error_y)

            yasmin.YASMIN_LOG_INFO(f'Detection: \n\terror_x={error_x}, \n\terror_y={error_y}, \n\toutput_x={output_x}, \n\toutput_y={output_y}')
            drone.move_velocity(
                vx = output_x,
                vy = output_y,
                vz = -PRECISE_LANDING_VERTICAL_SPEED if (error_x ** 2 + error_y ** 2 <= PRECISE_LANDING_ALING_TOLERANCE ** 2) else 0.0,
                vyaw = 0.0,
            )

        yasmin.YASMIN_LOG_INFO('Timeout.')
        return TIMEOUT
