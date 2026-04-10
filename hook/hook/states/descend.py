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
    IMAGE_CENTER_Y,
    ROPE_CONF_THRESHOLD,
    RELEASE_ALTITUDE,
    DESCEND_VELOCITY,
    DESCEND_KP_X,
    DESCEND_KP_Y,
    DESCEND_MAX_VELOCITY_XY,
    DESCEND_CENTER_TOLERANCE_PX,
    DESCEND_TIMEOUT,
    DESCEND_MAX_LOST_FRAMES,
    SAVE_DETECTIONS,
    DETECTION_SAVE_PATH,
)
from hook.states.center_on_rope import compute_mask_centroid


class DescendAndCenter(State):
    """Descend while maintaining centering on rope via segmentation + PID."""

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.pid_x = None
        self.pid_y = None
        self.save_dir = None
        self.frame_count = 0

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]
        camera: ImageHandler = blackboard["camera"]
        segmentor: Segmentor = blackboard["rope_segmentor"]

        if self.pid_x is None:
            self.pid_x = PIDController(
                kp=DESCEND_KP_X,
                setpoint=IMAGE_CENTER_Y,
                output_limits=(-DESCEND_MAX_VELOCITY_XY, DESCEND_MAX_VELOCITY_XY),
            )
        if self.pid_y is None:
            self.pid_y = PIDController(
                kp=DESCEND_KP_Y,
                setpoint=IMAGE_CENTER_X,
                output_limits=(-DESCEND_MAX_VELOCITY_XY, DESCEND_MAX_VELOCITY_XY),
            )

        if SAVE_DETECTIONS:
            ts = blackboard.get("mission_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "descend"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        yasmin.YASMIN_LOG_INFO(
            f"Descending to {RELEASE_ALTITUDE}m while centering on rope..."
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

            if len(result) == 0:
                lost_count += 1
                if lost_count > DESCEND_MAX_LOST_FRAMES:
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    yasmin.YASMIN_LOG_WARN(
                        "Lost rope during descent. Holding position."
                    )
                    return ABORT
                drone.move_velocity(
                    vx=0.0, vy=0.0, vz=-DESCEND_VELOCITY, vyaw=0.0,
                    reference=MoveReference.BODY,
                )
                time.sleep(0.05)
                continue

            lost_count = 0
            seg = result[0]

            if seg.mask is not None:
                centroid = compute_mask_centroid(seg.mask)
                if centroid is None:
                    cx, cy = seg.center
                else:
                    cx, cy = centroid
            else:
                cx, cy = seg.center

            error_x = cx - IMAGE_CENTER_X
            error_y = cy - IMAGE_CENTER_Y

            if SAVE_DETECTIONS and self.save_dir and self.frame_count % 10 == 0:
                annotated = segmentor.draw_segmentations(frame, result)
                cv2.drawMarker(
                    annotated,
                    (IMAGE_CENTER_X, IMAGE_CENTER_Y),
                    (0, 255, 0),
                    cv2.MARKER_CROSS, 30, 2,
                )
                cv2.putText(
                    annotated,
                    f"Alt: {current_alt:.2f}m",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                )
                cv2.imwrite(
                    str(self.save_dir / f"descend_{self.frame_count:04d}.jpg"),
                    annotated,
                )

            centered = (
                abs(error_x) < DESCEND_CENTER_TOLERANCE_PX
                and abs(error_y) < DESCEND_CENTER_TOLERANCE_PX
            )

            if centered and current_alt <= RELEASE_ALTITUDE:
                drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                yasmin.YASMIN_LOG_INFO(
                    f"Release altitude reached ({current_alt:.2f}m) and centered."
                )
                return SUCCEED

            vel_x = self.pid_x.update(cy)
            vel_y = self.pid_y.update(cx)

            drone.move_velocity(
                vx=vel_x,
                vy=vel_y,
                vz=-DESCEND_VELOCITY,
                vyaw=0.0,
                reference=MoveReference.BODY,
            )

            if int(time.time() * 2) % 2 == 0:
                yasmin.YASMIN_LOG_INFO(
                    f"Descending: alt={current_alt:.2f}m, "
                    f"error=({error_x:.0f}, {error_y:.0f})px, "
                    f"vel=({vel_x:.3f}, {vel_y:.3f}, {-DESCEND_VELOCITY:.3f})"
                )

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0)
        yasmin.YASMIN_LOG_ERROR("Descent timed out.")
        return ABORT
