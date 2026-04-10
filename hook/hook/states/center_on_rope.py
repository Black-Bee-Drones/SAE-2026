import time

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.control import MavrosDrone, MoveReference, PIDController
from nectar.vision import ImageHandler
from nectar.ai.segmentation import Segmentor

from hook.core.constants import (
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    ROPE_CONF_THRESHOLD,
    CENTER_TOLERANCE_PX,
    CENTERING_CONFIRMATIONS,
    CENTER_KP_X,
    CENTER_KP_Y,
    CENTER_MAX_VELOCITY_XY,
    CENTER_TIMEOUT,
    CENTER_MAX_LOST_FRAMES,
    SAVE_DETECTIONS,
    DETECTION_SAVE_PATH,
)


def compute_mask_centroid(mask: np.ndarray):
    """Compute centroid from a binary mask using image moments."""
    m = cv2.moments(mask.astype(np.uint8))
    if m["m00"] == 0:
        return None
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return cx, cy


class CenterOnRope(State):
    """Fine centering on the rope using segmentation + PID before descent."""

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
                kp=CENTER_KP_X,
                setpoint=IMAGE_CENTER_Y,
                output_limits=(-CENTER_MAX_VELOCITY_XY, CENTER_MAX_VELOCITY_XY),
            )
        if self.pid_y is None:
            self.pid_y = PIDController(
                kp=CENTER_KP_Y,
                setpoint=IMAGE_CENTER_X,
                output_limits=(-CENTER_MAX_VELOCITY_XY, CENTER_MAX_VELOCITY_XY),
            )

        if SAVE_DETECTIONS:
            ts = blackboard.get("mission_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "center_rope"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        yasmin.YASMIN_LOG_INFO("Centering on rope with segmentation...")

        centered_count = 0
        lost_count = 0
        start_time = time.time()

        while time.time() - start_time < CENTER_TIMEOUT:
            frame = camera.take_photo()
            if frame is None:
                time.sleep(0.05)
                continue

            result = segmentor.segment(frame, conf=ROPE_CONF_THRESHOLD)
            self.frame_count += 1

            if len(result) == 0:
                lost_count += 1
                if lost_count > CENTER_MAX_LOST_FRAMES:
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    yasmin.YASMIN_LOG_ERROR("Lost rope during centering.")
                    return ABORT
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

            if SAVE_DETECTIONS and self.save_dir and self.frame_count % 5 == 0:
                annotated = segmentor.draw_segmentations(frame, result)
                cv2.drawMarker(
                    annotated,
                    (IMAGE_CENTER_X, IMAGE_CENTER_Y),
                    (0, 255, 0),
                    cv2.MARKER_CROSS, 30, 2,
                )
                cv2.imwrite(
                    str(self.save_dir / f"center_{self.frame_count:04d}.jpg"),
                    annotated,
                )

            if abs(error_x) < CENTER_TOLERANCE_PX and abs(error_y) < CENTER_TOLERANCE_PX:
                centered_count += 1
                yasmin.YASMIN_LOG_INFO(
                    f"Centered ({centered_count}/{CENTERING_CONFIRMATIONS}), "
                    f"error=({error_x:.0f}, {error_y:.0f})px"
                )
                if centered_count >= CENTERING_CONFIRMATIONS:
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    yasmin.YASMIN_LOG_INFO("Rope centering complete.")
                    return SUCCEED
            else:
                centered_count = 0

            vel_x = self.pid_x.update(cy)
            vel_y = self.pid_y.update(cx)

            drone.move_velocity(
                vx=vel_x,
                vy=vel_y,
                vz=0.0,
                vyaw=0.0,
                reference=MoveReference.BODY,
            )

            if int(time.time()) % 2 == 0:
                yasmin.YASMIN_LOG_INFO(
                    f"Centering: error=({error_x:.0f}, {error_y:.0f})px, "
                    f"vel=({vel_x:.3f}, {vel_y:.3f})"
                )

            time.sleep(0.02)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0)
        yasmin.YASMIN_LOG_ERROR("Rope centering timed out.")
        return ABORT
