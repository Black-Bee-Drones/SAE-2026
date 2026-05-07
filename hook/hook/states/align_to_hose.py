"""Perpendicular alignment over the chosen hose with the sphere as a side anchor.

Convention: angle = 0 means hose runs horizontal in image, drone heading is
perpendicular to the rope. Image axes mapped to body via FLU down-cam:
  image -y -> body +x (forward)
  image +x -> body -y (right)

Control wiring:
  vx   <- pid_center.update(hose_cy)           # forward/back to stay over hose
  vy   <- pid_anchor.update(sphere_cx)         # lateral to anchor sphere offset
  vyaw <- pid_yaw.update(angle_from_horizontal)
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.ai.detection import PerClassConfidenceFilter
from nectar.ai.segmentation import Segmentor
from nectar.control import AltitudeSource, MavrosDrone, MoveReference, PIDController
from nectar.vision import ImageHandler

from hook.core.constants import (
    DETECTION_SAVE_PATH,
    HOSE_ALIGN_CONFIRMATIONS,
    HOSE_ALIGN_MAX_LOST_FRAMES,
    HOSE_ALIGN_TIMEOUT,
    HOSE_ANGLE_KP,
    HOSE_ANGLE_MAX_VELOCITY,
    HOSE_ANGLE_TOLERANCE_DEG,
    HOSE_CENTER_KP,
    HOSE_CENTER_MAX_VELOCITY,
    HOSE_CENTER_TOLERANCE_PX,
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    SAVE_DETECTIONS,
    SPHERE_ANCHOR_DISTANCE_M,
    SPHERE_ANCHOR_KP,
    SPHERE_ANCHOR_TOLERANCE_PX,
)
from hook.core.perception import (
    best_sphere,
    hook_image_offset,
    hose_pose,
    hose_segments,
    pick_hose_by_dir,
    px_per_meter,
    run_seg,
)


def _anchor_sign(blackboard: Blackboard, sphere_cx: float, hook_dx: float) -> int:
    """Sign of the target sphere x-offset from the hook-corrected image center.

    Snapshotted on first AlignToHose tick (or reused from blackboard) so it
    stays stable through the state's yaw convergence.
    """
    if "anchor_sign" in blackboard:
        return blackboard["anchor_sign"]
    sign = 1 if sphere_cx >= IMAGE_CENTER_X + hook_dx else -1
    blackboard["anchor_sign"] = sign
    return sign


class AlignToHose(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.pid_yaw = None
        self.pid_center = None
        self.pid_anchor = None
        self.save_dir = None
        self.frame_count = 0

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]
        camera: ImageHandler = blackboard["camera"]
        segmentor: Segmentor = blackboard["segmentor"]
        class_filter: PerClassConfidenceFilter = blackboard["class_filter"]
        side_unit: Tuple[float, float] = blackboard["hose_side_image_unit"]

        if self.pid_yaw is None:
            self.pid_yaw = PIDController(
                kp=HOSE_ANGLE_KP,
                setpoint=0.0,
                output_limits=(-HOSE_ANGLE_MAX_VELOCITY, HOSE_ANGLE_MAX_VELOCITY),
            )
        if self.pid_center is None:
            self.pid_center = PIDController(
                kp=HOSE_CENTER_KP,
                setpoint=IMAGE_CENTER_Y,
                output_limits=(-HOSE_CENTER_MAX_VELOCITY, HOSE_CENTER_MAX_VELOCITY),
            )
        if self.pid_anchor is None:
            self.pid_anchor = PIDController(
                kp=SPHERE_ANCHOR_KP,
                setpoint=0.0,
                output_limits=(-HOSE_CENTER_MAX_VELOCITY, HOSE_CENTER_MAX_VELOCITY),
            )

        if SAVE_DETECTIONS:
            ts = (
                blackboard["mission_timestamp"]
                if "mission_timestamp" in blackboard
                else datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "align_hose"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        yasmin.YASMIN_LOG_INFO("Aligning perpendicular to hose with sphere anchor...")

        aligned = 0
        lost = 0
        start = time.time()

        while time.time() - start < HOSE_ALIGN_TIMEOUT:
            altitude = drone.get_altitude(AltitudeSource.LIDAR)
            if altitude is None:
                altitude = drone.get_altitude(AltitudeSource.AUTO)

            frame, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)
            chosen = pick_hose_by_dir(sphere, hose_segments(result), side_unit)
            pose = hose_pose(chosen) if chosen is not None else None

            if sphere is None or pose is None:
                lost += 1
                if lost > HOSE_ALIGN_MAX_LOST_FRAMES:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY
                    )
                    yasmin.YASMIN_LOG_ERROR(
                        "Lost sphere or chosen hose during alignment."
                    )
                    return ABORT
                time.sleep(0.05)
                continue

            lost = 0
            hose_cx, hose_cy, angle, _, _ = pose
            sphere_cx, _ = sphere.center

            ppm = px_per_meter(altitude) if altitude else 0.0
            hook_dx, hook_dy = hook_image_offset(altitude)
            anchor_px = SPHERE_ANCHOR_DISTANCE_M * ppm
            sign = _anchor_sign(blackboard, sphere_cx, hook_dx)
            target_hose_cy = IMAGE_CENTER_Y + hook_dy
            target_sphere_cx = IMAGE_CENTER_X + hook_dx + sign * anchor_px

            self.pid_center.setpoint = target_hose_cy
            err_center = hose_cy - target_hose_cy
            err_anchor = sphere_cx - target_sphere_cx
            err_angle = angle

            angle_ok = abs(err_angle) < HOSE_ANGLE_TOLERANCE_DEG
            center_ok = abs(err_center) < HOSE_CENTER_TOLERANCE_PX
            anchor_ok = abs(err_anchor) < SPHERE_ANCHOR_TOLERANCE_PX

            self._save_frame(
                frame,
                result,
                segmentor,
                hose_cx,
                target_sphere_cx,
                angle,
                altitude,
            )

            if angle_ok and center_ok and anchor_ok:
                aligned += 1
                if aligned >= HOSE_ALIGN_CONFIRMATIONS:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY
                    )
                    yasmin.YASMIN_LOG_INFO("Hose alignment complete.")
                    return SUCCEED
            else:
                aligned = 0

            vx = self.pid_center.update(hose_cy)
            vy = self.pid_anchor.update(err_anchor)
            vyaw = self.pid_yaw.update(angle)

            drone.move_velocity(
                vx=vx,
                vy=vy,
                vz=0.0,
                vyaw=vyaw,
                reference=MoveReference.BODY,
            )

            if int(time.time()) % 2 == 0:
                yasmin.YASMIN_LOG_INFO(
                    f"Align: angle={angle:+.1f} hose_cy_err={err_center:+.0f} "
                    f"anchor_err={err_anchor:+.0f} vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}"
                )

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
        yasmin.YASMIN_LOG_ERROR("Hose alignment timed out.")
        return ABORT

    def _save_frame(
        self, frame, result, segmentor, hose_cx, target_cx, angle, altitude
    ):
        if not (SAVE_DETECTIONS and self.save_dir and frame is not None and result):
            return
        self.frame_count += 1
        if self.frame_count % 5 != 0:
            return
        annotated = segmentor.draw_segmentations(frame, result)
        cv2.drawMarker(
            annotated,
            (int(hose_cx), IMAGE_CENTER_Y),
            (0, 255, 0),
            cv2.MARKER_CROSS,
            30,
            2,
        )
        cv2.drawMarker(
            annotated,
            (int(target_cx), IMAGE_CENTER_Y),
            (0, 0, 255),
            cv2.MARKER_TRIANGLE_UP,
            30,
            2,
        )
        alt_txt = f"{altitude:.2f}m" if altitude is not None else "n/a"
        cv2.putText(
            annotated,
            f"alt={alt_txt} angle={angle:+.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imwrite(str(self.save_dir / f"align_{self.frame_count:04d}.jpg"), annotated)
