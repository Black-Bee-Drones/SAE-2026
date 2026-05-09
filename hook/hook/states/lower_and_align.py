"""Sphere-anchored stand-off alignment + LIDAR descent in one state.

Merges what used to be ``ALIGN_TO_HOSE`` and ``DESCEND_AND_ALIGN``. Same
three image-frame PIDs throughout, three internal phases:

1. ``yaw``     — only ``vyaw`` runs while ``|rope_angle| > ALIGN_YAW_FIRST_TOLERANCE_DEG``.
                 Position PIDs are reset; ``hose_cy`` and ``sphere_cx``
                 are not geometrically meaningful until the rope is close
                 to horizontal in image.
2. ``align``   — full ``(vx, vy, vyaw)`` PIDs with ``standoff = ALIGN_STANDOFF_M``,
                 ``vz = 0``. Exits to ``descend`` when angle/center/anchor
                 errors all stay within tolerance for
                 ``HOSE_ALIGN_CONFIRMATIONS`` consecutive ticks.
3. ``descend`` — same controllers; ``standoff`` linearly ramps
                 ``ALIGN_STANDOFF_M -> 0`` over ``DESCEND_STANDOFF_RAMP_TICKS``;
                 ``vz = -clip(DESCEND_VZ_KP·(alt - RELEASE_ALTITUDE),
                              VZ_MIN, VZ_MAX)`` (hard zero at/below floor).
                 SUCCEED when ``alt <= RELEASE_ALTITUDE`` AND angle/center
                 within tolerance for ``DESCEND_RELEASE_CONFIRMATIONS`` ticks.

Sphere-loss fallback (uniform across all phases): if ``best_sphere`` is
``None`` OR ``target_sphere_cx`` falls outside the FOV margin, the
sphere-anchor PID is disabled, ``vy = 0``, ``anchor_ok`` is forced True,
and the chosen rope falls back to the most-confident ``rose`` instance
(at low altitude the wrong rope is well outside the FOV). Sphere loss
does NOT abort. Only chosen-rope loss for ``LOWER_MAX_LOST_FRAMES``
consecutive frames aborts.

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
    DESCEND_RELEASE_CONFIRMATIONS,
    DESCEND_SPHERE_TARGET_MARGIN_PX,
    DESCEND_STANDOFF_RAMP_TICKS,
    DESCEND_TIMEOUT,
    DESCEND_VZ_KP,
    DESCEND_VZ_MAX,
    DESCEND_VZ_MIN,
    DETECTION_SAVE_PATH,
    HOSE_ALIGN_CONFIRMATIONS,
    HOSE_ALIGN_TIMEOUT,
    HOSE_ANGLE_KP,
    HOSE_ANGLE_MAX_VELOCITY,
    HOSE_ANGLE_TOLERANCE_DEG,
    HOSE_CENTER_KP,
    HOSE_CENTER_MAX_VELOCITY,
    HOSE_CENTER_TOLERANCE_M,
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    IMAGE_WIDTH,
    LOWER_MAX_LOST_FRAMES,
    PID_MIN_OUTPUT_VELOCITY_XY,
    PID_MIN_OUTPUT_VYAW,
    RELEASE_ALTITUDE,
    SAVE_DETECTIONS,
    SPHERE_ANCHOR_KP,
    SPHERE_ANCHOR_TOLERANCE_M,
    SPHERE_HEIGHT_M,
    WORK_ALTITUDE,
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


def _descend_vz(altitude: float) -> float:
    """Proportional vertical command above the floor; hard zero at/below."""
    above_floor = altitude - RELEASE_ALTITUDE
    if above_floor <= 0.0:
        return 0.0
    speed = DESCEND_VZ_KP * above_floor
    speed = max(DESCEND_VZ_MIN, min(DESCEND_VZ_MAX, speed))
    return -speed


class LowerAndAlign(State):
    """Three-phase hose alignment + LIDAR descent, single PID stack."""

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

        self._build_pids()

        if SAVE_DETECTIONS:
            ts = (
                blackboard["mission_timestamp"]
                if "mission_timestamp" in blackboard
                else datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "lower_and_align"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        yasmin.YASMIN_LOG_INFO(
            f"LowerAndAlign: standoff={ALIGN_STANDOFF_M:.2f}m anchor_sign={anchor_sign:+d} "
            f"release={RELEASE_ALTITUDE:.2f}m"
        )

        phase = "yaw"
        align_confirmed = 0
        descend_confirmed = 0
        hose_lost = 0
        ramp_tick = 0
        was_anchored = True
        align_start = time.time()
        descend_start = None

        while True:
            now = time.time()
            if phase in ("yaw", "align") and now - align_start > HOSE_ALIGN_TIMEOUT:
                drone.move_velocity(
                    0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=2.0
                )
                yasmin.YASMIN_LOG_ERROR("LowerAndAlign: align phase timed out.")
                return ABORT
            if (
                phase == "descend"
                and descend_start is not None
                and now - descend_start > DESCEND_TIMEOUT
            ):
                drone.move_velocity(
                    0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=2.0
                )
                yasmin.YASMIN_LOG_ERROR("LowerAndAlign: descend phase timed out.")
                return ABORT

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
                # Sphere-loss fallback: most-confident remaining rope.
                # At low altitude the wrong rope is outside the frustum
                # so the surviving detection is the chosen one.
                chosen = hoses[0]
            pose = hose_pose(chosen) if chosen is not None else None

            if pose is None:
                hose_lost += 1
                if hose_lost > LOWER_MAX_LOST_FRAMES:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=1.0
                    )
                    yasmin.YASMIN_LOG_ERROR(
                        f"LowerAndAlign: chosen hose lost for {hose_lost} frames."
                    )
                    return ABORT
                vz = _descend_vz(altitude) if phase == "descend" else 0.0
                drone.move_velocity(
                    vx=0.0, vy=0.0, vz=vz, vyaw=0.0,
                    reference=MoveReference.BODY,
                )
                time.sleep(0.05)
                continue

            hose_lost = 0
            _, hose_cy, angle, _, _ = pose

            ppm = px_per_meter(altitude, SPHERE_HEIGHT_M)
            hook_dx, hook_dy = hook_image_offset(altitude, SPHERE_HEIGHT_M)

            if phase == "descend":
                ramp_tick = min(ramp_tick + 1, DESCEND_STANDOFF_RAMP_TICKS)
                ramp = ramp_tick / DESCEND_STANDOFF_RAMP_TICKS
                standoff = ALIGN_STANDOFF_M * (1.0 - ramp)
            else:
                standoff = ALIGN_STANDOFF_M

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
                anchor_ok = abs(err_anchor_m) < SPHERE_ANCHOR_TOLERANCE_M
            else:
                err_anchor_px = 0.0
                err_anchor_m = 0.0
                anchor_ok = True

            centered = abs(err_center_m) < HOSE_CENTER_TOLERANCE_M
            angle_ok = abs(err_angle) < HOSE_ANGLE_TOLERANCE_DEG

            if was_anchored and not sphere_can_anchor:
                yasmin.YASMIN_LOG_INFO(
                    f"LowerAndAlign[{phase}]: sphere anchor released "
                    f"(alt={altitude:.2f}m, target_cx={target_sphere_cx:+.0f} "
                    f"{'outside FOV' if not target_in_frame else 'sphere lost'}); "
                    f"holding vy=0 on hose-only references."
                )
                was_anchored = False
            elif not was_anchored and sphere_can_anchor:
                yasmin.YASMIN_LOG_INFO(
                    f"LowerAndAlign[{phase}]: sphere anchor re-acquired "
                    f"at alt={altitude:.2f}m."
                )
                was_anchored = True

            vyaw = self.pid_yaw.update(err_angle)
            if abs(err_angle) > ALIGN_YAW_FIRST_TOLERANCE_DEG:
                vx, vy = 0.0, 0.0
                self.pid_center.reset()
                self.pid_anchor.reset()
                phase_label = "yaw"
            else:
                vx = self.pid_center.update(err_center_m)
                if sphere_can_anchor:
                    vy = self.pid_anchor.update(err_anchor_m)
                else:
                    vy = 0.0
                    self.pid_anchor.reset()
                if phase == "yaw":
                    phase = "align"
                phase_label = phase

            vz = _descend_vz(altitude) if phase == "descend" else 0.0

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
                phase=phase_label,
                sphere_can_anchor=sphere_can_anchor,
                altitude=altitude,
                standoff_m=standoff,
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

            if phase == "align":
                if angle_ok and centered and anchor_ok:
                    align_confirmed += 1
                    if align_confirmed >= HOSE_ALIGN_CONFIRMATIONS:
                        yasmin.YASMIN_LOG_INFO(
                            "LowerAndAlign: stand-off pose reached; entering descend."
                        )
                        phase = "descend"
                        descend_start = time.time()
                        ramp_tick = 0
                        align_confirmed = 0
                else:
                    align_confirmed = 0
            elif phase == "descend":
                if altitude <= RELEASE_ALTITUDE and centered and angle_ok and anchor_ok:
                    descend_confirmed += 1
                    if descend_confirmed >= DESCEND_RELEASE_CONFIRMATIONS:
                        drone.move_velocity(
                            0.0, 0.0, 0.0, 0.0,
                            reference=MoveReference.BODY, duration=2.5,
                        )
                        yasmin.YASMIN_LOG_INFO(
                            f"LowerAndAlign: release pose reached at "
                            f"alt={altitude:.2f}m (anchored={sphere_can_anchor})."
                        )
                        return SUCCEED
                else:
                    descend_confirmed = 0

            if int(time.time() * 2) % 2 == 0:
                anchor_str = (
                    f"anchor={err_anchor_m:+.3f}m ({err_anchor_px:+5.0f}px)"
                    if sphere_can_anchor
                    else "anchor=hose-only"
                )
                vz_str = f" vz={vz:+.2f}" if phase == "descend" else ""
                yasmin.YASMIN_LOG_INFO(
                    f"LowerAndAlign[{phase_label:>7s}] alt={altitude:.2f}m "
                    f"standoff={standoff:.2f}m | "
                    f"err: hose_cy={err_center_m:+.3f}m ({err_center_px:+5.0f}px) "
                    f"angle={err_angle:+5.1f}deg "
                    f"{anchor_str} | "
                    f"cmd: vx={vx:+.2f} vy={vy:+.2f}{vz_str} vyaw={vyaw:+.2f}"
                )

            time.sleep(0.03)

    def _build_pids(self) -> None:
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

    def _save_frame(self, *, frame, result, **kwargs) -> None:
        if not (SAVE_DETECTIONS and self.save_dir and frame is not None):
            return
        annotated = overlay.annotate_seg(frame, result)
        overlay.draw_lower_and_align(
            annotated,
            image_center=(IMAGE_CENTER_X, IMAGE_CENTER_Y),
            tol_center_m=HOSE_CENTER_TOLERANCE_M,
            tol_anchor_m=SPHERE_ANCHOR_TOLERANCE_M,
            alt_min=RELEASE_ALTITUDE - 0.5,
            alt_max=WORK_ALTITUDE + 0.5,
            release_alt=RELEASE_ALTITUDE,
            **kwargs,
        )
        self.frame_count += 1
        cv2.imwrite(
            str(self.save_dir / f"lower_{self.frame_count:04d}.jpg"), annotated
        )
