"""Pre-rotate the drone so the chosen rope ends up perpendicular to body
+x AND in front of the drone (image upper half).

Closed-loop body yaw, vision-only. After SELECT_SIDE we know the chosen
rope's direction in image (``hose_side_image_unit``). The ``LOWER_AND_ALIGN``
controllers downstream are 180°-symmetric: rope-in-front and rope-behind
are both fixed points of the rope-angle PID, and the position PID then
drags the drone backward across the rope when we land in the wrong half.

This state breaks that symmetry by computing the closest yaw rotation
``Δ_target`` such that, after applying it, the rope is horizontal in image
AND the sphere ends up in the upper half (rope in front). Then it spins
to that target using the sphere's image-frame polar angle as the only
yaw-invariant feedback, and updates ``hose_side_image_unit`` /
``anchor_sign`` to reflect the rotation so downstream states see the new
body frame transparently.

See :func:`hook.core.perception.predict_orient_yaw` for the math.

Image-to-body convention (down camera, FLU body):
    image -y -> body +x (forward)
    image +x -> body -y (right)
"""

import math
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import rclpy
import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
from yasmin_ros.yasmin_node import YasminNode

from nectar.ai.detection import PerClassConfidenceFilter
from nectar.ai.segmentation import Segmentor
from nectar.control import MavrosDrone, MoveReference, PIDController
from nectar.vision import ImageHandler

from hook.core import overlay
from hook.core.constants import (
    DETECTION_SAVE_PATH,
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    ORIENT_ANGLE_TOLERANCE_RAD,
    ORIENT_CONFIRMATIONS,
    ORIENT_MAX_LOST_FRAMES,
    ORIENT_MAX_YAW_VELOCITY,
    ORIENT_SAMPLE_FRAMES,
    ORIENT_SKIP_THRESHOLD_RAD,
    ORIENT_TIMEOUT,
    ORIENT_YAW_KP,
    PID_MIN_OUTPUT_VYAW,
    SAVE_DETECTIONS,
)
from hook.core.perception import (
    anchor_sign_for_side,
    best_sphere,
    hose_pose,
    hose_segments,
    pick_hose_by_dir,
    predict_orient_yaw,
    run_seg,
)


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi] using atan2 for numerical robustness."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _circular_mean(angles: List[float]) -> float:
    """Vector mean over (cos, sin) of the input angles. Robust to wrap."""
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c)


def _rotate_vec(
    v: Tuple[float, float], delta: float
) -> Tuple[float, float]:
    """Rotate a 2D image-frame vector by ``delta`` (atan2 sense, image
    y-down). Same convention as :func:`predict_orient_yaw`.
    """
    cosd, sind = math.cos(delta), math.sin(delta)
    vx, vy = v
    return (cosd * vx - sind * vy, sind * vx + cosd * vy)


class OrientToHook(State):
    """Predictive yaw to put the chosen rope perpendicular AND in front.

    Sample phase: collect ``ORIENT_SAMPLE_FRAMES`` good frames (sphere
    AND chosen rope visible). Per-frame ``Δ_target = predict_orient_yaw``
    using the rope's current image direction (axis_unit aligned with the
    blackboard's ``hose_side_image_unit`` for sign consistency) and the
    sphere image position. Vector-mean across frames -> ``Δ_total``.

    If ``|Δ_total| < ORIENT_SKIP_THRESHOLD_RAD``: SUCCEED, leave
    ``hose_side_image_unit`` / ``anchor_sign`` untouched.

    Otherwise spin: closed loop on sphere image-frame polar angle around
    image center, target = ``theta_initial + Δ_total``. PID setpoint=0,
    feedback = ``wrap_pi(theta_current - theta_target)``. Stops when
    the wrapped error stays under ``ORIENT_ANGLE_TOLERANCE_RAD`` for
    ``ORIENT_CONFIRMATIONS`` consecutive frames.

    Commit: rotate ``hose_side_image_unit`` by the OBSERVED yaw delta
    (``theta_final - theta_initial``, the actual sphere-polar shift) and
    derive the new ``anchor_sign``. Robust to PID convergence tolerance.
    """

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.pid_yaw = None
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
                kp=ORIENT_YAW_KP,
                setpoint=0.0,
                output_limits=(-ORIENT_MAX_YAW_VELOCITY, ORIENT_MAX_YAW_VELOCITY),
                output_deadband=PID_MIN_OUTPUT_VYAW,
            )

        if SAVE_DETECTIONS:
            ts = (
                blackboard["mission_timestamp"]
                if "mission_timestamp" in blackboard
                else datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "orient_to_hook"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        sample = self._sample_decision(camera, segmentor, class_filter, side_unit)
        if sample is None:
            yasmin.YASMIN_LOG_ERROR(
                "OrientToHook: could not gather any usable sphere+rope sample."
            )
            return ABORT

        delta_target, last_sphere_xy = sample

        if abs(delta_target) < ORIENT_SKIP_THRESHOLD_RAD:
            yasmin.YASMIN_LOG_INFO(
                f"OrientToHook: rope already perpendicular and in front "
                f"(|Δ|={math.degrees(abs(delta_target)):.1f}deg < "
                f"{math.degrees(ORIENT_SKIP_THRESHOLD_RAD):.1f}deg); "
                f"skipping pre-rotation."
            )
            return SUCCEED

        sx, sy = last_sphere_xy
        theta_initial = math.atan2(sy - IMAGE_CENTER_Y, sx - IMAGE_CENTER_X)
        theta_target = _wrap_pi(theta_initial + delta_target)
        yasmin.YASMIN_LOG_INFO(
            f"OrientToHook: spinning Δ={math.degrees(delta_target):+6.1f}deg "
            f"(theta_0={math.degrees(theta_initial):+6.1f}deg -> "
            f"theta_target={math.degrees(theta_target):+6.1f}deg)."
        )

        result = self._spin_to_target(
            drone, camera, segmentor, class_filter, theta_target, side_unit
        )
        if result is None:
            return ABORT
        theta_final = result

        delta_actual = _wrap_pi(theta_final - theta_initial)
        new_side_unit = _rotate_vec(side_unit, delta_actual)
        new_anchor_sign = anchor_sign_for_side(new_side_unit)
        blackboard["hose_side_image_unit"] = new_side_unit
        blackboard["anchor_sign"] = new_anchor_sign

        drone.move_velocity(
            0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=1.0
        )
        yasmin.YASMIN_LOG_INFO(
            f"OrientToHook: spin complete. "
            f"Δ_actual={math.degrees(delta_actual):+6.1f}deg, "
            f"side_unit=({side_unit[0]:+.2f},{side_unit[1]:+.2f}) -> "
            f"({new_side_unit[0]:+.2f},{new_side_unit[1]:+.2f}), "
            f"anchor_sign={anchor_sign:+d}->{new_anchor_sign:+d}."
        )
        return SUCCEED

    def _sample_decision(
        self,
        camera,
        segmentor,
        class_filter,
        side_unit,
    ) -> Optional[Tuple[float, Tuple[float, float]]]:
        """Collect ``ORIENT_SAMPLE_FRAMES`` good frames and return
        ``(delta_target, last_sphere_xy)`` or ``None`` on timeout.

        Per frame: pick the chosen rope (``pick_hose_by_dir``), use its
        ``axis_unit`` aligned with ``side_unit`` (sign-consistent), and
        compute the per-frame ``Δ`` from sphere position + rope axis.
        Vector-mean across frames yields a wrap-safe ``Δ_total``.
        """
        deltas: List[float] = []
        last_sphere_xy: Optional[Tuple[float, float]] = None
        deadline = time.time() + 5.0

        while len(deltas) < ORIENT_SAMPLE_FRAMES and time.time() < deadline:
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)

            frame, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)
            chosen = pick_hose_by_dir(sphere, hose_segments(result), side_unit)
            pose = hose_pose(chosen) if chosen is not None else None

            if sphere is None or pose is None:
                self._save_overlay(
                    frame, result,
                    phase="sample",
                    sphere_xy=sphere.center if sphere is not None else None,
                    pose=pose,
                    delta_target=None,
                    theta_current=None,
                    theta_target=None,
                    err_rad=None,
                    vyaw=0.0,
                    sample_idx=len(deltas),
                )
                time.sleep(0.05)
                continue

            sx, sy = sphere.center
            _, _, _, _, axis_unit = pose
            # Align axis_unit sign with side_unit (axis from minAreaRect has
            # arbitrary sign; use the side direction as the sign reference).
            if axis_unit[0] * side_unit[0] + axis_unit[1] * side_unit[1] < 0:
                axis_unit = (-axis_unit[0], -axis_unit[1])

            delta = predict_orient_yaw(axis_unit, (sx, sy))
            deltas.append(delta)
            last_sphere_xy = (sx, sy)

            self._save_overlay(
                frame, result,
                phase="sample",
                sphere_xy=last_sphere_xy,
                pose=pose,
                delta_target=delta,
                theta_current=None,
                theta_target=None,
                err_rad=None,
                vyaw=0.0,
                sample_idx=len(deltas),
            )

            time.sleep(0.03)

        if not deltas or last_sphere_xy is None:
            return None

        delta_target = _circular_mean(deltas)
        return delta_target, last_sphere_xy

    def _spin_to_target(
        self,
        drone,
        camera,
        segmentor,
        class_filter,
        theta_target: float,
        side_unit: Tuple[float, float],
    ) -> Optional[float]:
        """Closed-loop spin. Returns the final ``theta_current`` on success,
        or ``None`` on timeout / sphere-loss abort. Holds last vyaw on
        transient sphere loss.
        """
        confirmed = 0
        lost = 0
        last_vyaw = 0.0
        last_log = 0.0
        last_theta = 0.0
        start = time.time()

        while time.time() - start < ORIENT_TIMEOUT:
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)

            frame, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)

            if sphere is None:
                lost += 1
                if lost > ORIENT_MAX_LOST_FRAMES:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY
                    )
                    yasmin.YASMIN_LOG_ERROR(
                        f"OrientToHook: lost sphere for {lost} frames during spin."
                    )
                    return None
                drone.move_velocity(
                    vx=0.0, vy=0.0, vz=0.0, vyaw=last_vyaw,
                    reference=MoveReference.BODY,
                )
                self._save_overlay(
                    frame, result,
                    phase="spin",
                    sphere_xy=None,
                    pose=None,
                    delta_target=None,
                    theta_current=None,
                    theta_target=theta_target,
                    err_rad=None,
                    vyaw=last_vyaw,
                    sample_idx=None,
                )
                time.sleep(0.03)
                continue

            lost = 0
            sx, sy = sphere.center
            theta_current = math.atan2(sy - IMAGE_CENTER_Y, sx - IMAGE_CENTER_X)
            err_rad = _wrap_pi(theta_current - theta_target)
            vyaw = self.pid_yaw.update(err_rad)
            last_vyaw = vyaw
            last_theta = theta_current

            drone.move_velocity(
                vx=0.0, vy=0.0, vz=0.0, vyaw=vyaw, reference=MoveReference.BODY
            )

            chosen = pick_hose_by_dir(sphere, hose_segments(result), side_unit)
            pose = hose_pose(chosen) if chosen is not None else None
            self._save_overlay(
                frame, result,
                phase="spin",
                sphere_xy=(sx, sy),
                pose=pose,
                delta_target=None,
                theta_current=theta_current,
                theta_target=theta_target,
                err_rad=err_rad,
                vyaw=vyaw,
                sample_idx=None,
            )

            now = time.time()
            if now - last_log > 0.3:
                yasmin.YASMIN_LOG_INFO(
                    f"OrientToHook[spin] | "
                    f"theta={math.degrees(theta_current):+6.1f}° "
                    f"target={math.degrees(theta_target):+6.1f}° "
                    f"err={math.degrees(err_rad):+6.1f}° | "
                    f"vyaw={vyaw:+.2f}"
                )
                last_log = now

            if abs(err_rad) < ORIENT_ANGLE_TOLERANCE_RAD:
                confirmed += 1
                if confirmed >= ORIENT_CONFIRMATIONS:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY
                    )
                    return theta_current
            else:
                confirmed = 0

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
        yasmin.YASMIN_LOG_ERROR(
            f"OrientToHook: spin timed out (last theta="
            f"{math.degrees(last_theta):+6.1f}°, target="
            f"{math.degrees(theta_target):+6.1f}°)."
        )
        return None

    def _save_overlay(
        self,
        frame,
        result,
        *,
        phase: str,
        sphere_xy: Optional[Tuple[float, float]],
        pose: Optional[Tuple[float, float, float, float, Tuple[float, float]]],
        delta_target: Optional[float],
        theta_current: Optional[float],
        theta_target: Optional[float],
        err_rad: Optional[float],
        vyaw: float,
        sample_idx: Optional[int],
    ) -> None:
        if not (SAVE_DETECTIONS and self.save_dir and frame is not None):
            return
        annotated = overlay.annotate_seg(frame, result)
        overlay.draw_orient(
            annotated,
            image_center=(IMAGE_CENTER_X, IMAGE_CENTER_Y),
            phase=phase,
            sphere_xy=sphere_xy,
            hose_pose=pose,
            delta_target=delta_target,
            theta_current=theta_current,
            theta_target=theta_target,
            err_rad=err_rad,
            tol_rad=ORIENT_ANGLE_TOLERANCE_RAD,
            vyaw=vyaw,
            sample_idx=sample_idx,
            sample_total=ORIENT_SAMPLE_FRAMES,
        )
        self.frame_count += 1
        cv2.imwrite(
            str(self.save_dir / f"orient_{self.frame_count:04d}.jpg"), annotated
        )
