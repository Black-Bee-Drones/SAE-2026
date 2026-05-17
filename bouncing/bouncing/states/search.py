import math
import cv2

import rclpy
from rclpy.duration import Duration

import yasmin
from yasmin_ros.yasmin_node import YasminNode
from yasmin import State, Blackboard
from yasmin_ros.basic_outcomes import SUCCEED, FAIL, TIMEOUT, ABORT

from nectar.control import MavrosDrone, PIDController
from nectar.vision import ImageHandler

from bouncing.constants import (
    SEARCH_FIND_TOLERANCE,
    SEARCH_LIMITE_ALTITUDE,
    SEARCH_TARGET_ALTITUDE,
    SEARCH_TIMEOUT,
    SEARCH_VERTICAL_SPEED,
    SEARCH_POINTS,
    SEARCH_PHOTOS_PER_POINT,
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


class Search(State):
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
            yasmin.YASMIN_LOG_ERROR(f'ImageHandler not available.')
            return ABORT
        image_handler: ImageHandler = blackboard['image_handler']

        yasmin.YASMIN_LOG_INFO('Start.')
        has_rotated = False
        try:
            count_to_next_point = 0
            point_index = 0
            start = self.node.get_clock().now()
            count_gabarito = 0
            duration = Duration(seconds=SEARCH_TIMEOUT)
            while self.node.get_clock().now() - start < duration:

                if drone.get_altitude() >= SEARCH_LIMITE_ALTITUDE:
                    yasmin.YASMIN_LOG_ERROR('Failed: limit altitude reached.')
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    drone.delay(1.0)
                    return FAIL

                result = image_handler.take_photo()

                target_base = self.get_target_base(result)

                if not target_base:
                    if not blackboard['target_base']:
                        yasmin.YASMIN_LOG_ERROR('Target NOT found.')

                elif (blackboard['target_base'] != target_base and count_gabarito < 3):
                    yasmin.YASMIN_LOG_INFO(f'Target base: {target_base}.')
                    blackboard['target_base'] = target_base
                    count_gabarito = 0

                else:
                    count_gabarito += 1

                if blackboard['target_base']:
                    landing_base_number = self.get_landing_base_number(blackboard['target_base'], result)

                    if not landing_base_number:
                        yasmin.YASMIN_LOG_ERROR('Landing base NOT found.')
                    else:
                        count_landind_base += 1
                        yasmin.YASMIN_LOG_INFO(f'Landing base found ({count_landind_base}/{SEARCH_FIND_TOLERANCE}).')
                        if count_landind_base >= SEARCH_FIND_TOLERANCE:
                            yasmin.YASMIN_LOG_INFO('Completed successfully.')
                            return SUCCEED

                if count_to_next_point >= SEARCH_PHOTOS_PER_POINT:
                    count_to_next_point = 0
                    point_index += 1
                    if point_index >= len(SEARCH_POINTS):
                        point_index = 0

                    yasmin.YASMIN_LOG_INFO(f'Next point reached. x={SEARCH_POINTS[point_index]["x"]}, y={SEARCH_POINTS[point_index]["y"]}')
                    drone.move_to(
                        SEARCH_POINTS[point_index]['x'] - SEARCH_POINTS[point_index-1]['x'],
                        SEARCH_POINTS[point_index]['y'] - SEARCH_POINTS[point_index-1]['y'],
                        0.0,
                        0.0,
                    )

                if (drone.get_altitude() >= SEARCH_TARGET_ALTITUDE):
                    count_to_next_point += 1
                
                if (drone.get_altitude() < SEARCH_TARGET_ALTITUDE):
                    output_z = SEARCH_VERTICAL_SPEED
                    yasmin.YASMIN_LOG_INFO(f'UP: vz={output_z:.2f}')
                else:
                    output_z = 0.0

                drone.move_velocity(
                    vx = 0.0,
                    vy = 0.0,
                    vz = output_z,
                    vyaw = 0.0,
                )

            yasmin.YASMIN_LOG_ERROR('Timeout.')
            return TIMEOUT

        except:
            yasmin.YASMIN_LOG_ERROR('Error: ABORT.')
            return ABORT

    def get_target_base(self, result):
        target_base = {}
        aruco, number = self.get_target_number(result)
        if aruco is None or number is None:
            return None
        target_base['number'] = str(number)

        aruco_shape = self.get_aruco_shape(result, aruco)
        if not aruco_shape:
            return None
        target_base['shape'] = aruco_shape.class_name
        return target_base

    def get_takeoff_base_error(self, result, alt):
        d = result.filter_by_class(['7'])
        if d:
            h, w = result.image.shape[:2]

            center = d[0].center

            error_x = (center[1] - (h / 2))
            error_y = (center[0] - (w / 2))

            error_x = error_x / self.ppm(alt, 86, w)
            error_y = error_y / self.ppm(alt, 47, h)

            return error_x, error_y
        return 0.0, 0.0

    def get_aruco_shape(self, result, aruco):
        aruco_shapes = []
        for s in result.filter_by_class(['0', '1', '2']):  # all shapes
            if (abs(aruco.center[0] - s.center[0]) <= s.width / 2) and (abs(aruco.center[1] - s.center[1]) <= s.height / 2):
                aruco_shapes.append(s)

        if aruco_shapes:
            aruco_shape = max(
                aruco_shapes,
                key=lambda shape: (shape.center[0] - aruco.center[0]) ** 2 + (shape.center[1] - aruco.center[1]) ** 2
            )
            return aruco_shape
        return None


    def get_target_number(self, result):
        for d in result.filter_by_class(['6']):
            x1, y1, x2, y2 = d.bbox

            h, w = result.image.shape[:2]

            x1 = min(w, max(0, int(x1 - w / 2)))
            y1 = min(h, max(0, int(y1 - h / 2)))
            x2 = min(w, max(0, int(x2 + w / 2)))
            y2 = min(h, max(0, int(y2 + h / 2)))

            crop = result.image[y1:y2, x1:x2]

            n = self.get_number_of_aruco(crop)

            if not n:
                return None, None

            if n % 3 == 0:
                return d, 3
            elif n % 4 == 0:
                return d, 4
            elif n % 5 == 0:
                return d, 5
        return None, None


    def get_number_of_aruco(self, img):
        if img is None:
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        dict_options = [
            cv2.aruco.DICT_5X5_1000,
            cv2.aruco.DICT_5X5_250,
            cv2.aruco.DICT_5X5_100,
            cv2.aruco.DICT_5X5_50,
        ]

        parameters = cv2.aruco.DetectorParameters()

        for dict_id in dict_options:
            aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None:
                try:
                    yasmin.YASMIN_LOG_INFO(f'Aruco detected: ids={ids.flatten()[0]}')
                    return ids.flatten()[0]
                except:
                    return None

        return None
    
    def get_landing_base_number(self, target_base, result):
        area_img = result.image.shape[0] * result.image.shape[1]
        landing_bases = []
        for n in result.filter_by_class([target_base['number']]):
            for s in result.filter_by_class(['0', '1', '2']):
                if ((s.area / area_img) >= 0.6):
                    continue

                if (s.class_name == target_base['shape']):
                    if (abs(n.center[0] - s.center[0]) <= s.width / 2) and (abs(n.center[1] - s.center[1]) <= s.height / 2):
                        landing_bases.append(n)

        if landing_bases:
            landing_base_number = max(
                landing_bases,
                key=lambda l: l.confidence
            )
            return landing_base_number

        return None
