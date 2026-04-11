import time
from typing import Optional, Tuple

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
    ROPE_CONF_THRESHOLD,
    HOSE_MIN_CONTOUR_AREA,
    HOSE_ANGLE_TOLERANCE_DEG,
    HOSE_ANGLE_KP,
    HOSE_ANGLE_MAX_VELOCITY,
    HOSE_CENTER_TOLERANCE_PX,
    HOSE_CENTER_KP,
    HOSE_CENTER_MAX_VELOCITY,
    HOSE_ALIGN_CONFIRMATIONS,
    HOSE_ALIGN_TIMEOUT,
    HOSE_ALIGN_MAX_LOST_FRAMES,
    HOSE_OFFSET_DISTANCE,
    SAVE_DETECTIONS,
    DETECTION_SAVE_PATH,
)


def estimate_hose_pose(
    mask: np.ndarray, min_area: int = 200
) -> Optional[Tuple[float, float, float]]:
    """Extract hose center and angle from a segmentation mask via minAreaRect.

    Returns (center_x, center_y, angle) where angle is the deviation from
    vertical in degrees (0 = hose is vertical in image = drone perpendicular).
    Returns None if the mask has no valid contour.
    """
    mask_u8 = mask.astype(np.uint8)
    if mask_u8.max() == 1:
        mask_u8 = mask_u8 * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return None

    rect = cv2.minAreaRect(largest)
    (cx, cy), (w, h), raw_angle = rect

    # Normalize angle so 0 = hose vertical in image.
    # minAreaRect returns angle in [-90, 0) with width as the first side.
    # We want: if the long axis is vertical, angle = 0.
    if w < h:
        angle = raw_angle + 90
    else:
        angle = raw_angle
    if angle > 90:
        angle -= 180
    if angle < -90:
        angle += 180

    return cx, cy, angle


class AlignToHose(State):
    """Align yaw perpendicular to hose and center above it using segmentation."""

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.pid_yaw = None
        self.pid_center = None
        self.save_dir = None
        self.frame_count = 0

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]
        camera: ImageHandler = blackboard["camera"]
        segmentor: Segmentor = blackboard["rope_segmentor"]

        if self.pid_yaw is None:
            self.pid_yaw = PIDController(
                kp=HOSE_ANGLE_KP,
                setpoint=0.0,
                output_limits=(-HOSE_ANGLE_MAX_VELOCITY, HOSE_ANGLE_MAX_VELOCITY),
            )
        if self.pid_center is None:
            self.pid_center = PIDController(
                kp=HOSE_CENTER_KP,
                setpoint=IMAGE_CENTER_X,
                output_limits=(-HOSE_CENTER_MAX_VELOCITY, HOSE_CENTER_MAX_VELOCITY),
            )

        if SAVE_DETECTIONS:
            ts = blackboard.get("mission_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "align_hose"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        yasmin.YASMIN_LOG_INFO("Aligning yaw perpendicular to hose...")

        aligned_count = 0
        lost_count = 0
        start_time = time.time()

        while time.time() - start_time < HOSE_ALIGN_TIMEOUT:
            frame = camera.take_photo()
            if frame is None:
                time.sleep(0.05)
                continue

            result = segmentor.segment(frame, conf=ROPE_CONF_THRESHOLD)
            self.frame_count += 1

            # Get the hose mask
            seg_mask = None
            if len(result) > 0:
                seg = result[0]
                seg_mask = seg.mask

            if seg_mask is None:
                lost_count += 1
                if lost_count > HOSE_ALIGN_MAX_LOST_FRAMES:
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    yasmin.YASMIN_LOG_ERROR("Lost hose during alignment.")
                    return ABORT
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

            if SAVE_DETECTIONS and self.save_dir and self.frame_count % 5 == 0:
                annotated = segmentor.draw_segmentations(frame, result)
                cv2.drawMarker(
                    annotated, (IMAGE_CENTER_X, int(cy)),
                    (0, 255, 0), cv2.MARKER_CROSS, 30, 2,
                )
                cv2.putText(
                    annotated, f"Angle: {angle:.1f} deg",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                )
                cv2.imwrite(
                    str(self.save_dir / f"align_{self.frame_count:04d}.jpg"),
                    annotated,
                )

            angle_ok = abs(error_angle) < HOSE_ANGLE_TOLERANCE_DEG
            center_ok = abs(error_center) < HOSE_CENTER_TOLERANCE_PX

            if angle_ok and center_ok:
                aligned_count += 1
                yasmin.YASMIN_LOG_INFO(
                    f"Aligned ({aligned_count}/{HOSE_ALIGN_CONFIRMATIONS}): "
                    f"angle={angle:.1f}deg, cx_err={error_center:.0f}px"
                )
                if aligned_count >= HOSE_ALIGN_CONFIRMATIONS:
                    drone.move_velocity(0.0, 0.0, 0.0, 0.0)
                    yasmin.YASMIN_LOG_INFO("Hose alignment complete.")

                    if HOSE_OFFSET_DISTANCE > 0:
                        yasmin.YASMIN_LOG_INFO(
                            f"Shifting {HOSE_OFFSET_DISTANCE}m along hose..."
                        )
                        drone.move_to(
                            x=HOSE_OFFSET_DISTANCE,
                            reference=MoveReference.BODY,
                            timeout=10.0,
                            precision=0.2,
                        )

                    return SUCCEED
            else:
                aligned_count = 0

            vyaw = -self.pid_yaw.update(angle)
            vy = self.pid_center.update(cx)

            drone.move_velocity(
                vx=0.0, vy=vy, vz=0.0, vyaw=vyaw,
                reference=MoveReference.BODY,
            )

            if int(time.time()) % 2 == 0:
                yasmin.YASMIN_LOG_INFO(
                    f"Aligning: angle={angle:.1f}deg, cx_err={error_center:.0f}px, "
                    f"vyaw={vyaw:.3f}, vy={vy:.3f}"
                )

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0)
        yasmin.YASMIN_LOG_ERROR("Hose alignment timed out.")
        return ABORT
