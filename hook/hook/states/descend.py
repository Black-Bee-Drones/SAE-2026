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
    DESCEND_ANCHOR_KP,
    DESCEND_ANCHOR_TOLERANCE_M,
    DESCEND_ANGLE_KP,
    DESCEND_ANGLE_TOLERANCE_DEG,
    DESCEND_CENTER_KP,
    DESCEND_CENTER_TOLERANCE_M,
    DESCEND_MAX_LOST_FRAMES,
    DESCEND_MAX_VELOCITY_XY,
    DESCEND_MAX_YAW_VELOCITY,
    DESCEND_RELEASE_CONFIRMATIONS,
    DESCEND_SPHERE_TARGET_MARGIN_PX,
    DESCEND_STANDOFF_RAMP_TICKS,
    DESCEND_TIMEOUT,
    DESCEND_VZ_KP,
    DESCEND_VZ_MAX,
    DESCEND_VZ_MIN,
    DETECTION_SAVE_PATH,
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    IMAGE_WIDTH,
    PID_MIN_OUTPUT_VELOCITY_XY,
    PID_MIN_OUTPUT_VYAW,
    RELEASE_ALTITUDE,
    SAVE_DETECTIONS,
    SPHERE_HEIGHT_M,
    WORK_ALTITUDE,
)


def _descend_vz(altitude: float) -> float:
    """Proportional vertical command. Returns the body-frame vz to send.

    Above RELEASE_ALTITUDE: vz = -clip(KP * (alt - floor), VZ_MIN, VZ_MAX).
    At and below the floor: vz = 0 (hard altitude floor; the drone never
    descends past RELEASE_ALTITUDE regardless of lateral convergence).
    """
    above_floor = altitude - RELEASE_ALTITUDE
    if above_floor <= 0.0:
        return 0.0
    speed = DESCEND_VZ_KP * above_floor
    speed = max(DESCEND_VZ_MIN, min(DESCEND_VZ_MAX, speed))
    return -speed
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


class DescendAndAlign(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.pid_center = None
        self.pid_angle = None
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

        if self.pid_center is None:
            self.pid_center = PIDController(
                kp=DESCEND_CENTER_KP,
                setpoint=0.0,
                output_limits=(-DESCEND_MAX_VELOCITY_XY, DESCEND_MAX_VELOCITY_XY),
                output_deadband=PID_MIN_OUTPUT_VELOCITY_XY,
            )
        if self.pid_angle is None:
            self.pid_angle = PIDController(
                kp=DESCEND_ANGLE_KP,
                setpoint=0.0,
                output_limits=(-DESCEND_MAX_YAW_VELOCITY, DESCEND_MAX_YAW_VELOCITY),
                output_deadband=PID_MIN_OUTPUT_VYAW,
            )
        if self.pid_anchor is None:
            self.pid_anchor = PIDController(
                kp=DESCEND_ANCHOR_KP,
                setpoint=0.0,
                output_limits=(-DESCEND_MAX_VELOCITY_XY, DESCEND_MAX_VELOCITY_XY),
                output_deadband=PID_MIN_OUTPUT_VELOCITY_XY,
            )

        if SAVE_DETECTIONS:
            ts = (
                blackboard["mission_timestamp"]
                if "mission_timestamp" in blackboard
                else datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "descend"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        yasmin.YASMIN_LOG_INFO(
            f"Descending to {RELEASE_ALTITUDE}m on LIDAR (anchor_sign={anchor_sign:+d})..."
        )

        confirmed = 0
        hose_lost = 0
        ramp_tick = 0
        was_anchored = True
        start = time.time()

        while time.time() - start < DESCEND_TIMEOUT:
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)

            altitude = drone.get_altitude(AltitudeSource.LIDAR)
            if altitude is None:
                altitude = drone.get_altitude(AltitudeSource.AUTO)
            if altitude is None:
                time.sleep(0.05)
                continue

            frame, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)
            hoses = hose_segments(result)
            chosen = pick_hose_by_dir(sphere, hoses, side_unit)
            if chosen is None and hoses:
                # pick_hose_by_dir needs the sphere as origin. At low altitude
                # the sphere leaves the FOV before the hose does; the other
                # rope is far away in world and outside the frustum, so the
                # most-confident remaining detection is the chosen rope.
                chosen = hoses[0]
            pose = hose_pose(chosen) if chosen is not None else None

            if pose is None:
                hose_lost += 1
                if hose_lost > DESCEND_MAX_LOST_FRAMES:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=1.0
                    )
                    yasmin.YASMIN_LOG_WARN(
                        f"No hose detection for {hose_lost} frames; aborting."
                    )
                    return ABORT
                drone.move_velocity(
                    vx=0.0, vy=0.0, vz=_descend_vz(altitude), vyaw=0.0,
                    reference=MoveReference.BODY,
                )
                drone.delay(0.05)
                continue

            hose_lost = 0
            hose_cx, hose_cy, angle, _, _ = pose

            ramp_tick = min(ramp_tick + 1, DESCEND_STANDOFF_RAMP_TICKS)
            ramp = ramp_tick / DESCEND_STANDOFF_RAMP_TICKS
            standoff = ALIGN_STANDOFF_M * (1.0 - ramp)
            ppm = px_per_meter(altitude, SPHERE_HEIGHT_M)
            hook_dx, hook_dy = hook_image_offset(altitude, SPHERE_HEIGHT_M)
            target_sphere_cx, target_hose_cy, _ = alignment_targets(
                side_unit, altitude, standoff_m=standoff
            )

            target_in_frame = (
                DESCEND_SPHERE_TARGET_MARGIN_PX
                <= target_sphere_cx
                <= IMAGE_WIDTH - DESCEND_SPHERE_TARGET_MARGIN_PX
            )
            sphere_can_anchor = sphere is not None and target_in_frame

            err_center_px = hose_cy - target_hose_cy
            err_center_m = err_center_px / ppm if ppm > 0 else 0.0
            err_angle = angle
            sphere_xy = sphere.center if sphere is not None else None
            if sphere_can_anchor:
                err_anchor_px = sphere_xy[0] - target_sphere_cx
                err_anchor_m = err_anchor_px / ppm if ppm > 0 else 0.0
                anchor_ok = abs(err_anchor_m) < DESCEND_ANCHOR_TOLERANCE_M
            else:
                err_anchor_px = 0.0
                err_anchor_m = 0.0
                anchor_ok = True

            centered = abs(err_center_m) < DESCEND_CENTER_TOLERANCE_M
            angle_ok = abs(err_angle) < DESCEND_ANGLE_TOLERANCE_DEG

            if was_anchored and not sphere_can_anchor:
                yasmin.YASMIN_LOG_INFO(
                    f"Sphere anchor released at alt={altitude:.2f}m "
                    f"(target_cx={target_sphere_cx:+.0f} outside FOV margin); "
                    f"finishing descent on hose-only references."
                )
                was_anchored = False
            elif not was_anchored and sphere_can_anchor:
                yasmin.YASMIN_LOG_INFO(
                    f"Sphere anchor re-acquired at alt={altitude:.2f}m."
                )
                was_anchored = True

            if altitude <= RELEASE_ALTITUDE and centered and angle_ok and anchor_ok:
                confirmed += 1
                if confirmed >= DESCEND_RELEASE_CONFIRMATIONS:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=2.5
                    )
                    yasmin.YASMIN_LOG_INFO(
                        f"Release pose reached: alt={altitude:.2f}m "
                        f"(anchored={sphere_can_anchor})"
                    )
                    return SUCCEED
            else:
                confirmed = 0

            vyaw = self.pid_angle.update(err_angle)
            if abs(err_angle) > ALIGN_YAW_FIRST_TOLERANCE_DEG:
                vx, vy = 0.0, 0.0
                self.pid_center.reset()
                self.pid_anchor.reset()
            else:
                vx = self.pid_center.update(err_center_m)
                if sphere_can_anchor:
                    vy = self.pid_anchor.update(err_anchor_m)
                else:
                    vy = 0.0
                    self.pid_anchor.reset()

            vz = _descend_vz(altitude)
            drone.move_velocity(
                vx=vx, vy=vy, vz=vz, vyaw=vyaw, reference=MoveReference.BODY
            )

            self._save_frame(
                frame=frame,
                result=result,
                hook_xy=(int(IMAGE_CENTER_X + hook_dx), int(IMAGE_CENTER_Y + hook_dy)),
                hose_pose=pose,
                sphere_center=sphere_xy,
                target_sphere_xy=(target_sphere_cx, target_hose_cy),
                target_hose_cy=target_hose_cy,
                ppm=ppm,
                anchor_sign=anchor_sign,
                sphere_can_anchor=sphere_can_anchor,
                altitude=altitude,
                err_center_m=err_center_m,
                err_anchor_m=err_anchor_m,
                err_angle_deg=err_angle,
                err_center_px=err_center_px,
                err_anchor_px=err_anchor_px,
                vx=vx,
                vy=vy,
                vz=vz,
                vyaw=vyaw,
            )

            if int(time.time() * 2) % 2 == 0:
                anchor_str = (
                    f"anchor={err_anchor_m:+.3f}m ({err_anchor_px:+5.0f}px)"
                    if sphere_can_anchor
                    else "anchor=hose-only"
                )
                yasmin.YASMIN_LOG_INFO(
                    f"Descend alt={altitude:.2f}m standoff={standoff:.2f}m | "
                    f"err: hose_cy={err_center_m:+.3f}m ({err_center_px:+5.0f}px) "
                    f"angle={err_angle:+5.1f}deg "
                    f"{anchor_str} | "
                    f"cmd: vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f} vyaw={vyaw:+.2f}"
                )

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=2.0)
        yasmin.YASMIN_LOG_ERROR("Descent timed out.")
        return ABORT

    def _save_frame(self, *, frame, result, **kwargs) -> None:
        if not (SAVE_DETECTIONS and self.save_dir and frame is not None):
            return
        annotated = overlay.annotate_seg(frame, result)
        overlay.draw_descend(
            annotated,
            image_center=(IMAGE_CENTER_X, IMAGE_CENTER_Y),
            tol_center_m=DESCEND_CENTER_TOLERANCE_M,
            tol_anchor_m=DESCEND_ANCHOR_TOLERANCE_M,
            alt_min=RELEASE_ALTITUDE - 0.5,
            alt_max=WORK_ALTITUDE + 0.5,
            release_alt=RELEASE_ALTITUDE,
            **kwargs,
        )
        self.frame_count += 1
        cv2.imwrite(str(self.save_dir / f"descend_{self.frame_count:04d}.jpg"), annotated)
