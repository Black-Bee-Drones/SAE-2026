"""Pre-rotate the drone so the chosen rope ends up perpendicular to body
+x AND in front of the drone (image upper half).

Closed-loop body yaw, vision-only, anisotropic-correct. After SELECT_SIDE
we know the chosen rope's direction in image (``hose_side_image_unit``).
The ``LOWER_AND_ALIGN`` controllers downstream are 180°-symmetric:
rope-in-front and rope-behind are both fixed points of the rope-angle PID,
and the position PID then drags the drone backward across the rope when
we land in the wrong half. This state breaks that symmetry by computing
the closest body-yaw rotation ``Δ_target`` (using both ppm_x and ppm_y so
the math is right under the wide-angle lens's anisotropic projection)
that makes the rope perpendicular to body +x AND keeps the sphere in
front. It then spins the drone to that target.

Spin feedback is the **body-frame** sphere polar angle (``atan2(body_y,
body_x)`` derived from the sphere's image position by anisotropic
deprojection). When the body yaws CCW by Δ, that angle decreases by
exactly Δ — independent of ``ppm_y/ppm_x``. The image-frame polar angle
does **not** have that property, so the loop transient is now linear in
body yaw instead of being warped by the FOV asymmetry.

On commit, ``hose_side_image_unit`` is updated by applying the
anisotropic image-frame transformation that corresponds to the observed
body-yaw delta, so ``LOWER_AND_ALIGN`` sees the post-spin frame
transparently.

Image-to-body convention (down camera, FLU body):
    image -y -> body +x (forward)
    image +x -> body -y (right)
"""

import math
import time
from typing import List, Optional, Tuple

import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.ai.detection import PerClassConfidenceFilter
from nectar.ai.segmentation import Segmentor
from nectar.control import AltitudeSource, MavrosDrone, MoveReference, PIDController
from nectar.vision import ImageHandler

from hook.core import overlay
from hook.core.constants import (
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
    SPHERE_HEIGHT_M,
)
from hook.core.frame_sink import FrameSink, build_state_sink
from hook.core.perception import (
    anchor_sign_for_side,
    best_sphere,
    hose_pose,
    hose_segments,
    pick_hose_by_dir,
    predict_orient_yaw,
    px_per_meter_x,
    px_per_meter_y,
    rotate_image_vector_under_body_yaw,
    run_seg,
    sphere_body_angle,
)


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi] using atan2 for numerical robustness."""
    return math.atan2(math.sin(angle), math.cos(angle))


def _circular_mean(angles: List[float]) -> float:
    """Vector mean over (cos, sin) of the input angles. Robust to wrap."""
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c)


class OrientToHook(State):
    """Predictive yaw to put the chosen rope perpendicular AND in front.

    Sample phase: collect ``ORIENT_SAMPLE_FRAMES`` good frames (sphere
    AND chosen rope visible). Per-frame ``Δ = predict_orient_yaw(side_unit,
    sphere_xy, ppm_x, ppm_y)`` with the rope's current image direction
    (sign-aligned to the blackboard's ``hose_side_image_unit``) and the
    sphere image position. Vector-mean across frames -> ``Δ_target``.

    If ``|Δ_target| < ORIENT_SKIP_THRESHOLD_RAD``: SUCCEED, leave
    ``hose_side_image_unit`` and ``anchor_sign`` untouched.

    Otherwise spin: closed loop on the **body-frame** sphere polar
    angle. The cumulative body-yaw delta since the spin started equals
    ``wrap_pi(phi_initial - phi_current)`` (pure rotation of the body
    frame). PID drives this cumulative delta to ``Δ_target``. Stops
    when the wrapped error stays under ``ORIENT_ANGLE_TOLERANCE_RAD``
    for ``ORIENT_CONFIRMATIONS`` consecutive frames.

    Commit: rotate ``hose_side_image_unit`` using the OBSERVED body-yaw
    delta (anisotropic image transformation), refresh ``anchor_sign``.
    """

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.pid_yaw = None
        self.sink: FrameSink = None

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

        self.sink = build_state_sink(blackboard, "orient_to_hook")

        sample = self._sample_decision(
            drone, camera, segmentor, class_filter, side_unit
        )
        if sample is None:
            yasmin.YASMIN_LOG_ERROR(
                "OrientToHook: could not gather any usable sphere+rope sample."
            )
            return ABORT

        delta_target, last_sphere_xy, last_ppm = sample
        ppm_x_last, ppm_y_last = last_ppm

        if abs(delta_target) < ORIENT_SKIP_THRESHOLD_RAD:
            yasmin.YASMIN_LOG_INFO(
                f"OrientToHook: rope already perpendicular and in front "
                f"(|Δ|={math.degrees(abs(delta_target)):.1f}deg < "
                f"{math.degrees(ORIENT_SKIP_THRESHOLD_RAD):.1f}deg); "
                f"skipping pre-rotation."
            )
            return SUCCEED

        phi_initial = sphere_body_angle(last_sphere_xy, ppm_x_last, ppm_y_last)
        yasmin.YASMIN_LOG_INFO(
            f"OrientToHook: spinning Δ={math.degrees(delta_target):+6.1f}deg "
            f"(body-frame; sphere body angle phi_0={math.degrees(phi_initial):+6.1f}deg)."
        )

        delta_actual = self._spin_to_target(
            drone, camera, segmentor, class_filter,
            delta_target, phi_initial, side_unit,
        )
        if delta_actual is None:
            return ABORT

        # Commit: anisotropic image transformation of side_unit under the
        # observed body yaw, then refresh anchor_sign.
        new_side_unit = rotate_image_vector_under_body_yaw(
            side_unit, delta_actual, ppm_x_last, ppm_y_last,
        )
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
        drone,
        camera,
        segmentor,
        class_filter,
        side_unit,
    ) -> Optional[Tuple[float, Tuple[float, float], Tuple[float, float]]]:
        """Collect ``ORIENT_SAMPLE_FRAMES`` good frames. Returns
        ``(delta_target, last_sphere_xy, (ppm_x, ppm_y))`` or ``None`` on
        timeout. Per frame Δ uses anisotropic ppm at sample-time altitude.
        """
        deltas: List[float] = []
        last_sphere_xy: Optional[Tuple[float, float]] = None
        last_ppm: Optional[Tuple[float, float]] = None
        deadline = time.time() + 5.0

        while len(deltas) < ORIENT_SAMPLE_FRAMES and time.time() < deadline:
            drone.delay(0.05)

            altitude = drone.get_altitude(AltitudeSource.LIDAR)
            if altitude is None:
                altitude = drone.get_altitude(AltitudeSource.AUTO)

            frame, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)
            chosen = pick_hose_by_dir(sphere, hose_segments(result), side_unit)
            pose = hose_pose(chosen) if chosen is not None else None

            ppm_x = px_per_meter_x(altitude, SPHERE_HEIGHT_M)
            ppm_y = px_per_meter_y(altitude, SPHERE_HEIGHT_M)

            if sphere is None or pose is None or ppm_x <= 0.0 or ppm_y <= 0.0:
                self._save_overlay(
                    frame, result,
                    phase="sample",
                    sphere_xy=sphere.center if sphere is not None else None,
                    pose=pose,
                    delta_target=None,
                    err_rad=None,
                    vyaw=0.0,
                    sample_idx=len(deltas),
                )
                continue

            sx, sy = sphere.center
            _, _, _, _, axis_unit = pose
            # Align axis_unit sign with side_unit (axis from minAreaRect has
            # arbitrary sign; use the side direction as the sign reference).
            if axis_unit[0] * side_unit[0] + axis_unit[1] * side_unit[1] < 0:
                axis_unit = (-axis_unit[0], -axis_unit[1])

            delta = predict_orient_yaw(axis_unit, (sx, sy), ppm_x, ppm_y)
            deltas.append(delta)
            last_sphere_xy = (sx, sy)
            last_ppm = (ppm_x, ppm_y)

            self._save_overlay(
                frame, result,
                phase="sample",
                sphere_xy=last_sphere_xy,
                pose=pose,
                delta_target=delta,
                err_rad=None,
                vyaw=0.0,
                sample_idx=len(deltas),
            )

        if not deltas or last_sphere_xy is None or last_ppm is None:
            return None

        return _circular_mean(deltas), last_sphere_xy, last_ppm

    def _spin_to_target(
        self,
        drone,
        camera,
        segmentor,
        class_filter,
        delta_target: float,
        phi_initial: float,
        side_unit: Tuple[float, float],
    ) -> Optional[float]:
        """Closed-loop spin with body-frame feedback.

        At each tick: compute the sphere's body-frame polar angle from
        its image position; the cumulative body-yaw since the spin
        started is ``wrap_pi(phi_initial - phi_current_body)``. Drive
        that to ``delta_target``. Returns the observed ``delta_actual``
        on success, ``None`` on timeout / sphere-loss abort.
        """
        confirmed = 0
        lost = 0
        last_vyaw = 0.0
        last_log = 0.0
        delta_observed = 0.0
        start = time.time()

        while time.time() - start < ORIENT_TIMEOUT:
            drone.delay(0.05)
            altitude = drone.get_altitude(AltitudeSource.LIDAR)
            if altitude is None:
                altitude = drone.get_altitude(AltitudeSource.AUTO)

            frame, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)
            ppm_x = px_per_meter_x(altitude, SPHERE_HEIGHT_M)
            ppm_y = px_per_meter_y(altitude, SPHERE_HEIGHT_M)

            if sphere is None or ppm_x <= 0.0 or ppm_y <= 0.0:
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
                    delta_target=delta_target,
                    err_rad=None,
                    vyaw=last_vyaw,
                    sample_idx=None,
                )
                continue

            lost = 0
            phi_current = sphere_body_angle(sphere.center, ppm_x, ppm_y)
            delta_observed = _wrap_pi(phi_initial - phi_current)
            err_rad = _wrap_pi(delta_target - delta_observed)
            # PID setpoint=0, feedback=err_rad -> output ≈ -kp·err. We want
            # err > 0 (target > observed) to produce vyaw > 0 (CCW), so we
            # feed `-err_rad` to keep the kp positive convention sane.
            vyaw = self.pid_yaw.update(-err_rad)
            last_vyaw = vyaw

            drone.move_velocity(
                vx=0.0, vy=0.0, vz=0.0, vyaw=vyaw, reference=MoveReference.BODY
            )

            chosen = pick_hose_by_dir(sphere, hose_segments(result), side_unit)
            pose = hose_pose(chosen) if chosen is not None else None
            self._save_overlay(
                frame, result,
                phase="spin",
                sphere_xy=sphere.center,
                pose=pose,
                delta_target=delta_target,
                err_rad=err_rad,
                vyaw=vyaw,
                sample_idx=None,
            )

            now = time.time()
            if now - last_log > 0.3:
                yasmin.YASMIN_LOG_INFO(
                    f"OrientToHook[spin] | "
                    f"Δ_obs={math.degrees(delta_observed):+6.1f}° "
                    f"target={math.degrees(delta_target):+6.1f}° "
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
                    return delta_observed
            else:
                confirmed = 0

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
        yasmin.YASMIN_LOG_ERROR(
            f"OrientToHook: spin timed out (Δ_obs="
            f"{math.degrees(delta_observed):+6.1f}°, target="
            f"{math.degrees(delta_target):+6.1f}°)."
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
        err_rad: Optional[float],
        vyaw: float,
        sample_idx: Optional[int],
    ) -> None:
        if frame is None:
            return
        annotated = overlay.annotate_seg(frame, result)
        overlay.draw_orient(
            annotated,
            image_center=(IMAGE_CENTER_X, IMAGE_CENTER_Y),
            phase=phase,
            sphere_xy=sphere_xy,
            hose_pose=pose,
            delta_target=delta_target,
            err_rad=err_rad,
            tol_rad=ORIENT_ANGLE_TOLERANCE_RAD,
            vyaw=vyaw,
            sample_idx=sample_idx,
            sample_total=ORIENT_SAMPLE_FRAMES,
        )
        self.sink.emit(annotated)
