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
    YAW_ALIGN_TOLERANCE_PX,
    YAW_ALIGN_KP,
    YAW_ALIGN_MAX_VELOCITY,
    YAW_ALIGN_CONFIRMATIONS,
    APPROACH_KP_X,
    APPROACH_KP_Y,
    APPROACH_MAX_VELOCITY_XY,
    APPROACH_DESCEND_VELOCITY,
    APPROACH_CENTER_TOLERANCE_PX,
    APPROACH_TIMEOUT,
    APPROACH_MAX_LOST_FRAMES,
    SAVE_DETECTIONS,
    DETECTION_SAVE_PATH,
)


class ApproachSphere(State):
    """Yaw toward sphere, then center XY and descend to WORK_ALTITUDE using sphere detection."""

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

        if SAVE_DETECTIONS:
            ts = blackboard.get("mission_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "approach_sphere"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # --- Phase 1: Yaw toward sphere ---
        yasmin.YASMIN_LOG_INFO("Phase 1: Aligning yaw toward sphere...")
        aligned_count = 0
        start_time = time.time()

        while time.time() - start_time < APPROACH_TIMEOUT:
            _, det, _ = self._detect_sphere(camera, detector)
            if det is None:
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                time.sleep(0.05)
                continue

            error_x = det.center[0] - IMAGE_CENTER_X

            if abs(error_x) < YAW_ALIGN_TOLERANCE_PX:
                aligned_count += 1
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                if aligned_count >= YAW_ALIGN_CONFIRMATIONS:
                    yasmin.YASMIN_LOG_INFO("Yaw aligned toward sphere.")
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

        # --- Phase 2: Center XY on sphere + descend to WORK_ALTITUDE ---
        yasmin.YASMIN_LOG_INFO(
            f"Phase 2: Centering on sphere and descending to {WORK_ALTITUDE}m..."
        )

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

        lost_count = 0
        start_time = time.time()

        while time.time() - start_time < APPROACH_TIMEOUT:
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)

            current_alt = drone.get_altitude(AltitudeSource.REL_ALT)
            if current_alt is None:
                time.sleep(0.05)
                continue

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
                        str(self.save_dir / f"approach_{self.frame_count:04d}.jpg"),
                        annotated,
                    )

            if det is None:
                lost_count += 1
                if lost_count > APPROACH_MAX_LOST_FRAMES:
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    yasmin.YASMIN_LOG_ERROR("Lost sphere during approach.")
                    return ABORT
                time.sleep(0.05)
                continue

            lost_count = 0
            cx, cy = det.center
            error_x = cx - IMAGE_CENTER_X
            error_y = cy - IMAGE_CENTER_Y

            centered = (
                abs(error_x) < APPROACH_CENTER_TOLERANCE_PX
                and abs(error_y) < APPROACH_CENTER_TOLERANCE_PX
            )

            if centered and current_alt <= WORK_ALTITUDE:
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                yasmin.YASMIN_LOG_INFO(
                    f"Above sphere at {current_alt:.2f}m. Transitioning to hose alignment."
                )
                return SUCCEED

            vel_x = self.pid_x.update(cy)
            vel_y = self.pid_y.update(cx)
            vel_z = -APPROACH_DESCEND_VELOCITY if current_alt > WORK_ALTITUDE else 0.0

            drone.move_velocity(
                vx=vel_x, vy=vel_y, vz=vel_z, vyaw=0.0,
                reference=MoveReference.BODY,
            )

            if int(time.time()) % 2 == 0:
                yasmin.YASMIN_LOG_INFO(
                    f"Approach: alt={current_alt:.2f}m, "
                    f"error=({error_x:.0f}, {error_y:.0f})px, "
                    f"vel=({vel_x:.3f}, {vel_y:.3f}, {vel_z:.3f})"
                )

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0)
        yasmin.YASMIN_LOG_ERROR("Approach timed out.")
        return ABORT
