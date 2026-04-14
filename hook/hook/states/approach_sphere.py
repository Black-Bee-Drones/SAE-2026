import time

import cv2
import rclpy
from datetime import datetime
from pathlib import Path

import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
from yasmin_ros.yasmin_node import YasminNode

from nectar.control import MavrosDrone, MoveReference, PIDController, AltitudeSource
from nectar.vision import ImageHandler
from nectar.ai.detection import Detector

from hook.core.constants import (
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    WORK_ALTITUDE,
    SPHERE_CONF_THRESHOLD,
    APPROACH_KP_X,
    APPROACH_KP_Y,
    APPROACH_MAX_VELOCITY_XY,
    APPROACH_DESCEND_VELOCITY,
    APPROACH_CENTER_TOLERANCE_PX,
    SPHERE_OFFSET_PX,
    APPROACH_CENTER_CONFIRMATIONS,
    APPROACH_TIMEOUT,
    APPROACH_MAX_LOST_FRAMES,
    SAVE_DETECTIONS,
    DETECTION_SAVE_PATH,
)


def _sphere_close_enough(cx, cy, center_x, center_y, tolerance, offset):
    """Check if the sphere is close enough to image center without being directly centered.

    The drone should get near the sphere but not fly directly over it.
    Returns True when the sphere is within (tolerance + offset) pixels of
    image center and at least tolerance pixels close on one axis.
    """
    dx = abs(cx - center_x)
    dy = abs(cy - center_y)
    max_dist = tolerance + offset
    return dx < max_dist and dy < max_dist and (dx < tolerance or dy < tolerance)


class ApproachSphere(State):
    """Approach the sphere at search altitude, then descend to WORK_ALTITUDE.

    Step 1: Center XY near the sphere at current altitude (with offset to
    avoid flying directly above it and triggering lidar terrain-follow).
    Step 2: Descend to WORK_ALTITUDE while maintaining lateral tracking.
    """

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.pid_x = None
        self.pid_y = None
        self.save_dir = None
        self.frame_count = 0

    def _detect_sphere(self, camera, detector):
        frame = camera.take_photo()
        if frame is None:
            return None, None, None
        result = detector.detect(frame, conf=SPHERE_CONF_THRESHOLD)
        if len(result) == 0:
            return frame, None, result
        best = max(result.detections, key=lambda d: d.confidence)
        return frame, best, result

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]
        camera: ImageHandler = blackboard["camera"]
        detector: Detector = blackboard["sphere_detector"]

        if self.pid_x is None:
            self.pid_x = PIDController(
                kp=APPROACH_KP_X,
                setpoint=IMAGE_CENTER_Y,
                output_limits=(-APPROACH_MAX_VELOCITY_XY, APPROACH_MAX_VELOCITY_XY),
            )
        if self.pid_y is None:
            self.pid_y = PIDController(
                kp=APPROACH_KP_Y,
                setpoint=IMAGE_CENTER_X,
                output_limits=(-APPROACH_MAX_VELOCITY_XY, APPROACH_MAX_VELOCITY_XY),
            )

        if SAVE_DETECTIONS:
            ts = blackboard.get("mission_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "approach_sphere"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # --- Step 1: Center XY near sphere at search altitude ---
        yasmin.YASMIN_LOG_INFO("Step 1: Moving near sphere at search altitude...")

        centered_count = 0
        lost_count = 0
        start_time = time.time()

        while time.time() - start_time < APPROACH_TIMEOUT:
            frame, det, result = self._detect_sphere(camera, detector)

            if det is None:
                lost_count += 1
                if lost_count > APPROACH_MAX_LOST_FRAMES:
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    yasmin.YASMIN_LOG_ERROR("Lost sphere during XY approach.")
                    return ABORT
                time.sleep(0.05)
                continue

            lost_count = 0
            cx, cy = det.center

            if _sphere_close_enough(
                cx, cy, IMAGE_CENTER_X, IMAGE_CENTER_Y,
                APPROACH_CENTER_TOLERANCE_PX, SPHERE_OFFSET_PX,
            ):
                centered_count += 1
                if centered_count >= APPROACH_CENTER_CONFIRMATIONS:
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    yasmin.YASMIN_LOG_INFO(
                        f"Near sphere (offset). Sphere at ({cx:.0f}, {cy:.0f})."
                    )
                    break
            else:
                centered_count = 0

            vel_x = self.pid_x.update(cy)
            vel_y = self.pid_y.update(cx)

            drone.move_velocity(
                vx=vel_x, vy=vel_y, vz=0.0, vyaw=0.0,
                reference=MoveReference.BODY,
            )

            if SAVE_DETECTIONS and self.save_dir and self.frame_count % 5 == 0:
                self.frame_count += 1
                annotated = detector.draw_detections(frame, result)
                cv2.imwrite(
                    str(self.save_dir / f"center_{self.frame_count:04d}.jpg"),
                    annotated,
                )

            if int(time.time()) % 2 == 0:
                yasmin.YASMIN_LOG_INFO(
                    f"Centering: sphere=({cx:.0f}, {cy:.0f}), "
                    f"centered={centered_count}/{APPROACH_CENTER_CONFIRMATIONS}"
                )

            time.sleep(0.03)
        else:
            drone.move_velocity(0.0, 0.0, 0.0, 0.0)
            yasmin.YASMIN_LOG_ERROR("XY centering timed out.")
            return ABORT

        # --- Step 2: Descend to WORK_ALTITUDE while tracking sphere ---
        yasmin.YASMIN_LOG_INFO(
            f"Step 2: Descending to {WORK_ALTITUDE}m..."
        )

        self.pid_x.reset()
        self.pid_y.reset()
        lost_count = 0
        start_time = time.time()

        while time.time() - start_time < APPROACH_TIMEOUT:
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)

            current_alt = drone.get_altitude(AltitudeSource.AUTO)
            if current_alt is None:
                time.sleep(0.05)
                continue

            if current_alt <= WORK_ALTITUDE:
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                yasmin.YASMIN_LOG_INFO(
                    f"Reached work altitude ({current_alt:.2f}m)."
                )
                return SUCCEED

            frame, det, result = self._detect_sphere(camera, detector)

            if SAVE_DETECTIONS and self.save_dir and frame is not None and result and len(result) > 0:
                self.frame_count += 1
                if self.frame_count % 5 == 0:
                    annotated = detector.draw_detections(frame, result)
                    cv2.putText(
                        annotated, f"Alt: {current_alt:.2f}m",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    )
                    cv2.imwrite(
                        str(self.save_dir / f"descend_{self.frame_count:04d}.jpg"),
                        annotated,
                    )

            if det is None:
                lost_count += 1
                if lost_count > APPROACH_MAX_LOST_FRAMES:
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    yasmin.YASMIN_LOG_ERROR("Lost sphere during descent.")
                    return ABORT
                drone.move_velocity(
                    vx=0.0, vy=0.0, vz=-APPROACH_DESCEND_VELOCITY, vyaw=0.0,
                    reference=MoveReference.BODY,
                )
                time.sleep(0.05)
                continue

            lost_count = 0
            cx, cy = det.center

            vel_x = self.pid_x.update(cy)
            vel_y = self.pid_y.update(cx)

            drone.move_velocity(
                vx=vel_x,
                vy=vel_y,
                vz=-APPROACH_DESCEND_VELOCITY,
                vyaw=0.0,
                reference=MoveReference.BODY,
            )

            if int(time.time()) % 2 == 0:
                yasmin.YASMIN_LOG_INFO(
                    f"Descending: alt={current_alt:.2f}m, "
                    f"sphere=({cx:.0f}, {cy:.0f})"
                )

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0)
        yasmin.YASMIN_LOG_ERROR("Descent to work altitude timed out.")
        return ABORT
