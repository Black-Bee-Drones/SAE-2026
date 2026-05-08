"""Sphere-anchored stand-off alignment over the chosen hose.

Converges to a stand-off pose: rope held ``ALIGN_STANDOFF_M`` in front of
the hook (image upper half), sphere anchored at ``SPHERE_ANCHOR_DISTANCE_M``
along the chosen hose direction (image x offset = ``anchor_sign * D``),
drone yaw perpendicular to the rope (rope horizontal in image).

PIDs feed on metric errors so behavior is invariant to altitude.

Wiring (image-to-body: ``image -y -> body +x``, ``image +x -> body -y``):
    vx   <- pid_center.update((hose_cy - target_hose_cy) / ppm)
    vy   <- pid_anchor.update((sphere_cx - target_sphere_cx) / ppm)
    vyaw <- pid_yaw.update(angle_deg)
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import rclpy
import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
from yasmin_ros.yasmin_node import YasminNode

from nectar.ai.detection import PerClassConfidenceFilter
from nectar.ai.segmentation import Segmentor
from nectar.control import AltitudeSource, MavrosDrone, MoveReference, PIDController
from nectar.vision import ImageHandler

from hook.core import overlay
from hook.core.constants import (
    ALIGN_STANDOFF_M,
    ALIGN_YAW_FIRST_TOLERANCE_DEG,
    DETECTION_SAVE_PATH,
    HOSE_ALIGN_CONFIRMATIONS,
    HOSE_ALIGN_MAX_LOST_FRAMES,
    HOSE_ALIGN_TIMEOUT,
    HOSE_ANGLE_KP,
    HOSE_ANGLE_MAX_VELOCITY,
    HOSE_ANGLE_TOLERANCE_DEG,
    HOSE_CENTER_KP,
    HOSE_CENTER_MAX_VELOCITY,
    HOSE_CENTER_TOLERANCE_M,
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    PID_MIN_OUTPUT_VELOCITY_XY,
    PID_MIN_OUTPUT_VYAW,
    SAVE_DETECTIONS,
    SPHERE_ANCHOR_KP,
    SPHERE_ANCHOR_TOLERANCE_M,
    SPHERE_HEIGHT_M,
)
from hook.core.perception import (
    alignment_targets,
    best_sphere,
    hook_image_offset,
    hose_pose,
    hose_segments,
    pick_hose_by_dir,
    px_per_meter,
    run_seg,
)


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
        anchor_sign: int = blackboard["anchor_sign"]

        if self.pid_yaw is None:
            self.pid_yaw = PIDController(
                kp=HOSE_ANGLE_KP,
                setpoint=0.0,
                output_limits=(-HOSE_ANGLE_MAX_VELOCITY, HOSE_ANGLE_MAX_VELOCITY),
                output_deadband=PID_MIN_OUTPUT_VYAW,
            )
        if self.pid_center is None:
            self.pid_center = PIDController(
                kp=HOSE_CENTER_KP,
                setpoint=0.0,
                output_limits=(-HOSE_CENTER_MAX_VELOCITY, HOSE_CENTER_MAX_VELOCITY),
                output_deadband=PID_MIN_OUTPUT_VELOCITY_XY,
            )
        if self.pid_anchor is None:
            self.pid_anchor = PIDController(
                kp=SPHERE_ANCHOR_KP,
                setpoint=0.0,
                output_limits=(-HOSE_CENTER_MAX_VELOCITY, HOSE_CENTER_MAX_VELOCITY),
                output_deadband=PID_MIN_OUTPUT_VELOCITY_XY,
            )

        if SAVE_DETECTIONS:
            ts = (
                blackboard["mission_timestamp"]
                if "mission_timestamp" in blackboard
                else datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "align_hose"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        yasmin.YASMIN_LOG_INFO(
            f"Aligning to stand-off pose (standoff={ALIGN_STANDOFF_M:.2f}m, "
            f"anchor_sign={anchor_sign:+d})..."
        )

        aligned = 0
        lost = 0
        start = time.time()

        while time.time() - start < HOSE_ALIGN_TIMEOUT:
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)
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
            sphere_cx, sphere_cy = sphere.center

            ppm = px_per_meter(altitude, SPHERE_HEIGHT_M)
            hook_dx, hook_dy = hook_image_offset(altitude, SPHERE_HEIGHT_M)
            target_sphere_cx, target_hose_cy, _ = alignment_targets(
                side_unit, altitude, standoff_m=ALIGN_STANDOFF_M
            )

            err_center_px = hose_cy - target_hose_cy
            err_anchor_px = sphere_cx - target_sphere_cx
            err_center_m = err_center_px / ppm if ppm > 0 else 0.0
            err_anchor_m = err_anchor_px / ppm if ppm > 0 else 0.0
            err_angle = angle

            angle_ok = abs(err_angle) < HOSE_ANGLE_TOLERANCE_DEG
            center_ok = abs(err_center_m) < HOSE_CENTER_TOLERANCE_M
            anchor_ok = abs(err_anchor_m) < SPHERE_ANCHOR_TOLERANCE_M

            vyaw = self.pid_yaw.update(err_angle)
            if abs(err_angle) > ALIGN_YAW_FIRST_TOLERANCE_DEG:
                vx, vy = 0.0, 0.0
                self.pid_center.reset()
                self.pid_anchor.reset()
                phase = "yaw"
            else:
                vx = self.pid_center.update(err_center_m)
                vy = self.pid_anchor.update(err_anchor_m)
                phase = "full"

            drone.move_velocity(
                vx=vx, vy=vy, vz=0.0, vyaw=vyaw, reference=MoveReference.BODY
            )

            self._save_frame(
                frame=frame,
                result=result,
                hook_xy=(int(IMAGE_CENTER_X + hook_dx), int(IMAGE_CENTER_Y + hook_dy)),
                hose_pose=pose,
                sphere_center=(sphere_cx, sphere_cy),
                target_sphere_xy=(target_sphere_cx, target_hose_cy),
                target_hose_cy=target_hose_cy,
                ppm=ppm,
                anchor_sign=anchor_sign,
                phase=phase,
                altitude=altitude,
                err_center_m=err_center_m,
                err_anchor_m=err_anchor_m,
                err_angle_deg=err_angle,
                err_center_px=err_center_px,
                err_anchor_px=err_anchor_px,
                vx=vx,
                vy=vy,
                vyaw=vyaw,
            )

            if angle_ok and center_ok and anchor_ok:
                aligned += 1
                if aligned >= HOSE_ALIGN_CONFIRMATIONS:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY
                    )
                    yasmin.YASMIN_LOG_INFO("Stand-off pose reached.")
                    return SUCCEED
            else:
                aligned = 0

            if int(time.time()) % 2 == 0:
                yasmin.YASMIN_LOG_INFO(
                    f"Align[{phase:>4s}] | "
                    f"err: hose_cy={err_center_m:+.3f}m ({err_center_px:+5.0f}px) "
                    f"angle={err_angle:+5.1f}deg "
                    f"anchor={err_anchor_m:+.3f}m ({err_anchor_px:+5.0f}px) | "
                    f"cmd: vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}"
                )

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
        yasmin.YASMIN_LOG_ERROR("Hose alignment timed out.")
        return ABORT

    def _save_frame(self, *, frame, result, **kwargs) -> None:
        if not (SAVE_DETECTIONS and self.save_dir and frame is not None):
            return
        annotated = overlay.annotate_seg(frame, result)
        overlay.draw_align(
            annotated,
            image_center=(IMAGE_CENTER_X, IMAGE_CENTER_Y),
            tol_center_m=HOSE_CENTER_TOLERANCE_M,
            tol_anchor_m=SPHERE_ANCHOR_TOLERANCE_M,
            standoff_m=ALIGN_STANDOFF_M,
            **kwargs,
        )
        self.frame_count += 1
        cv2.imwrite(str(self.save_dir / f"align_{self.frame_count:04d}.jpg"), annotated)
