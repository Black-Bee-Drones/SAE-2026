import time

import cv2
from datetime import datetime
from pathlib import Path

import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.control import MavrosDrone, MoveReference
from nectar.vision import ImageHandler
from nectar.ai.detection import Detector

from hook.core.constants import (
    IMAGE_CENTER_X,
    SPHERE_CONF_THRESHOLD,
    YAW_ALIGN_TOLERANCE_PX,
    YAW_ALIGN_KP,
    YAW_ALIGN_MAX_VELOCITY,
    YAW_ALIGN_CONFIRMATIONS,
    APPROACH_KP_Y,
    APPROACH_FORWARD_VELOCITY,
    APPROACH_MAX_LATERAL_VELOCITY,
    APPROACH_SPHERE_AREA_THRESHOLD,
    APPROACH_TIMEOUT,
    SAVE_DETECTIONS,
    DETECTION_SAVE_PATH,
)


class AlignAndApproach(State):
    """Align yaw toward the sphere, then approach the rope while keeping sphere centered."""

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
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

        if SAVE_DETECTIONS:
            ts = blackboard.get("mission_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "align_approach"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # --- Phase 1: Yaw alignment ---
        yasmin.YASMIN_LOG_INFO("Phase 1: Aligning yaw toward sphere...")
        aligned_count = 0
        start_time = time.time()

        while time.time() - start_time < APPROACH_TIMEOUT:
            frame, det, result = self._detect_sphere(camera, detector)
            if det is None:
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                time.sleep(0.05)
                continue

            cx, _ = det.center
            error_x = cx - IMAGE_CENTER_X

            if abs(error_x) < YAW_ALIGN_TOLERANCE_PX:
                aligned_count += 1
                yasmin.YASMIN_LOG_INFO(
                    f"Yaw aligned ({aligned_count}/{YAW_ALIGN_CONFIRMATIONS}), "
                    f"error={error_x:.0f}px"
                )
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                if aligned_count >= YAW_ALIGN_CONFIRMATIONS:
                    break
            else:
                aligned_count = 0
                yaw_vel = -YAW_ALIGN_KP * error_x
                yaw_vel = max(-YAW_ALIGN_MAX_VELOCITY, min(YAW_ALIGN_MAX_VELOCITY, yaw_vel))
                drone.move_velocity(vyaw=yaw_vel, reference=MoveReference.BODY)

            time.sleep(0.05)
        else:
            drone.move_velocity(0.0, 0.0, 0.0, 0.0)
            yasmin.YASMIN_LOG_ERROR("Yaw alignment timed out.")
            return ABORT

        # --- Phase 2: Approach ---
        yasmin.YASMIN_LOG_INFO("Phase 2: Approaching rope...")
        start_time = time.time()
        lost_count = 0

        while time.time() - start_time < APPROACH_TIMEOUT:
            frame, det, result = self._detect_sphere(camera, detector)

            if SAVE_DETECTIONS and self.save_dir and frame is not None and result and len(result) > 0:
                self.frame_count += 1
                if self.frame_count % 5 == 0:
                    annotated = detector.draw_detections(frame, result)
                    cv2.imwrite(
                        str(self.save_dir / f"approach_{self.frame_count:04d}.jpg"),
                        annotated,
                    )

            if det is None:
                lost_count += 1
                if lost_count > 40:
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    yasmin.YASMIN_LOG_ERROR("Lost sphere during approach.")
                    return ABORT
                time.sleep(0.05)
                continue

            lost_count = 0
            cx, _ = det.center

            if det.area >= APPROACH_SPHERE_AREA_THRESHOLD:
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                yasmin.YASMIN_LOG_INFO(
                    f"Close to rope (sphere area={det.area}px²). Approach complete."
                )
                return SUCCEED

            lateral_error = cx - IMAGE_CENTER_X
            vel_y = -APPROACH_KP_Y * lateral_error
            vel_y = max(-APPROACH_MAX_LATERAL_VELOCITY, min(APPROACH_MAX_LATERAL_VELOCITY, vel_y))

            drone.move_velocity(
                vx=APPROACH_FORWARD_VELOCITY,
                vy=vel_y,
                vz=0.0,
                vyaw=0.0,
                reference=MoveReference.BODY,
            )

            if int(time.time()) % 2 == 0:
                yasmin.YASMIN_LOG_INFO(
                    f"Approaching: area={det.area}, lateral_err={lateral_error:.0f}px, "
                    f"vel=({APPROACH_FORWARD_VELOCITY:.2f}, {vel_y:.3f})"
                )

            time.sleep(0.05)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0)
        yasmin.YASMIN_LOG_ERROR("Approach timed out.")
        return ABORT
