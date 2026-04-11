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
from nectar.ai.segmentation import Segmentor

from hook.core.constants import (
    IMAGE_CENTER_X,
    ROPE_CONF_THRESHOLD,
    RELEASE_ALTITUDE,
    HOSE_MIN_CONTOUR_AREA,
    DESCEND_VELOCITY,
    DESCEND_CENTER_KP,
    DESCEND_ANGLE_KP,
    DESCEND_MAX_VELOCITY_XY,
    DESCEND_MAX_YAW_VELOCITY,
    DESCEND_CENTER_TOLERANCE_PX,
    DESCEND_ANGLE_TOLERANCE_DEG,
    DESCEND_TIMEOUT,
    DESCEND_MAX_LOST_FRAMES,
    SAVE_DETECTIONS,
    DETECTION_SAVE_PATH,
)
from hook.states.align_to_hose import estimate_hose_pose


class DescendAndAlign(State):
    """Descend while maintaining perpendicular alignment and centering on hose."""

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.pid_center = None
        self.pid_angle = None
        self.save_dir = None
        self.frame_count = 0

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]
        camera: ImageHandler = blackboard["camera"]
        segmentor: Segmentor = blackboard["rope_segmentor"]

        if self.pid_center is None:
            self.pid_center = PIDController(
                kp=DESCEND_CENTER_KP,
                setpoint=IMAGE_CENTER_X,
                output_limits=(-DESCEND_MAX_VELOCITY_XY, DESCEND_MAX_VELOCITY_XY),
            )
        if self.pid_angle is None:
            self.pid_angle = PIDController(
                kp=DESCEND_ANGLE_KP,
                setpoint=0.0,
                output_limits=(-DESCEND_MAX_YAW_VELOCITY, DESCEND_MAX_YAW_VELOCITY),
            )

        if SAVE_DETECTIONS:
            ts = blackboard.get("mission_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "descend"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        yasmin.YASMIN_LOG_INFO(
            f"Descending to {RELEASE_ALTITUDE}m with hose alignment..."
        )

        lost_count = 0
        start_time = time.time()

        while time.time() - start_time < DESCEND_TIMEOUT:
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)

            current_alt = drone.get_altitude(AltitudeSource.REL_ALT)
            if current_alt is None:
                time.sleep(0.05)
                continue

            frame = camera.take_photo()
            if frame is None:
                time.sleep(0.05)
                continue

            result = segmentor.segment(frame, conf=ROPE_CONF_THRESHOLD)
            self.frame_count += 1

            seg_mask = None
            if len(result) > 0 and result[0].mask is not None:
                seg_mask = result[0].mask

            if seg_mask is None:
                lost_count += 1
                if lost_count > DESCEND_MAX_LOST_FRAMES:
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    yasmin.YASMIN_LOG_WARN("Lost hose during descent.")
                    return ABORT
                drone.move_velocity(
                    vx=0.0, vy=0.0, vz=-DESCEND_VELOCITY, vyaw=0.0,
                    reference=MoveReference.BODY,
                )
                time.sleep(0.05)
                continue

            pose = estimate_hose_pose(seg_mask, HOSE_MIN_CONTOUR_AREA)
            if pose is None:
                lost_count += 1
                time.sleep(0.05)
                continue

            lost_count = 0
            cx, cy, angle = pose

            error_center = cx - IMAGE_CENTER_X
            error_angle = angle

            if SAVE_DETECTIONS and self.save_dir and self.frame_count % 10 == 0:
                annotated = segmentor.draw_segmentations(frame, result)
                cv2.drawMarker(
                    annotated, (IMAGE_CENTER_X, int(cy)),
                    (0, 255, 0), cv2.MARKER_CROSS, 30, 2,
                )
                cv2.putText(
                    annotated,
                    f"Alt: {current_alt:.2f}m  Angle: {angle:.1f}deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.imwrite(
                    str(self.save_dir / f"descend_{self.frame_count:04d}.jpg"),
                    annotated,
                )

            centered = abs(error_center) < DESCEND_CENTER_TOLERANCE_PX
            angle_ok = abs(error_angle) < DESCEND_ANGLE_TOLERANCE_DEG

            if centered and angle_ok and current_alt <= RELEASE_ALTITUDE:
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                yasmin.YASMIN_LOG_INFO(
                    f"Release altitude reached ({current_alt:.2f}m), "
                    f"centered and aligned."
                )
                return SUCCEED

            vy = self.pid_center.update(cx)
            vyaw = -self.pid_angle.update(angle)

            drone.move_velocity(
                vx=0.0,
                vy=vy,
                vz=-DESCEND_VELOCITY,
                vyaw=vyaw,
                reference=MoveReference.BODY,
            )

            if int(time.time() * 2) % 2 == 0:
                yasmin.YASMIN_LOG_INFO(
                    f"Descending: alt={current_alt:.2f}m, "
                    f"cx_err={error_center:.0f}px, angle={angle:.1f}deg, "
                    f"vy={vy:.3f}, vyaw={vyaw:.3f}"
                )

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0)
        yasmin.YASMIN_LOG_ERROR("Descent timed out.")
        return ABORT
