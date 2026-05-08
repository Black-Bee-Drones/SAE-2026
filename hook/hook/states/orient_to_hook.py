"""Pre-rotate the drone so the chosen rope ends up in front of the hook.

ALIGN's three PIDs (rope-angle / hose-row / sphere-anchor) are 180° symmetric
about the chosen rope axis: rope-in-front and rope-behind are both fixed
points. When SELECT_SIDE leaves the drone yawed such that the chosen rope
is behind in body frame (rope mask centroid below image center), ALIGN's
center PID would otherwise back the drone over the rope to drag the rope
to the standoff target above center.

This state breaks the symmetry by spinning the drone 180° using the sphere's
image-frame polar angle around image center as the closed-loop reference.
The sphere is uniquely identified (highest-confidence sphere instance), so
the spin is robust to rope mis-identification mid-rotation. After the spin,
``hose_side_image_unit`` and ``anchor_sign`` on the blackboard are negated
so ALIGN/DESCEND see the new body frame transparently.

No-op when the chosen rope is already in front of the drone (median body-x
over ``ORIENT_BEHIND_SAMPLE_FRAMES`` frames is above
``-ORIENT_BEHIND_THRESHOLD_M``).

Image-to-body convention (down camera, FLU body):
    image -y -> body +x (forward)
    image +x -> body -y (right)
"""

import math
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

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
    DETECTION_SAVE_PATH,
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    ORIENT_ANGLE_TOLERANCE_RAD,
    ORIENT_BEHIND_SAMPLE_FRAMES,
    ORIENT_BEHIND_THRESHOLD_M,
    ORIENT_CONFIRMATIONS,
    ORIENT_MAX_LOST_FRAMES,
    ORIENT_MAX_YAW_VELOCITY,
    ORIENT_TIMEOUT,
    ORIENT_YAW_KP,
    PID_MIN_OUTPUT_VYAW,
    SAVE_DETECTIONS,
    SPHERE_HEIGHT_M,
)
from hook.core.perception import (
    best_sphere,
    hose_pose,
    hose_segments,
    pick_hose_by_dir,
    px_per_meter,
    run_seg,
)


def _wrap_pi(angle: float) -> float:
    """Wrap an angle to (-pi, pi] using atan2 for numerical robustness."""
    return math.atan2(math.sin(angle), math.cos(angle))


class OrientToHook(State):
    """Pre-rotate so the chosen rope is in front of the drone.

    Sample-and-decide phase: median chosen-rope body-x over N frames; trigger
    the 180° flip iff the rope is behind by more than
    ``ORIENT_BEHIND_THRESHOLD_M``. Otherwise SUCCEED immediately.

    Spin phase: closed loop on sphere image-frame polar angle around image
    center, target = current_angle + pi (mirror image position). Stops when
    the wrapped error stays under ``ORIENT_ANGLE_TOLERANCE_RAD`` for
    ``ORIENT_CONFIRMATIONS`` consecutive frames.

    Commit: negate ``hose_side_image_unit`` and ``anchor_sign`` on the
    blackboard so ALIGN's ``pick_hose_by_dir`` correctly identifies the
    original physical rope after the body 180° yaw.
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

        sample = self._sample_decision(drone, camera, segmentor, class_filter, side_unit)
        if sample is None:
            yasmin.YASMIN_LOG_ERROR(
                "OrientToHook: could not gather any usable sphere+rope sample."
            )
            return ABORT

        rope_body_x_med, sphere_xy = sample
        if rope_body_x_med >= -ORIENT_BEHIND_THRESHOLD_M:
            yasmin.YASMIN_LOG_INFO(
                f"OrientToHook: rope already in front "
                f"(body_x={rope_body_x_med:+.3f}m >= "
                f"-{ORIENT_BEHIND_THRESHOLD_M:.2f}m); skipping pre-rotation."
            )
            return SUCCEED

        sx, sy = sphere_xy
        theta_0 = math.atan2(sy - IMAGE_CENTER_Y, sx - IMAGE_CENTER_X)
        # Epsilon-bias the target so the wrapped error at entry is +(pi-eps),
        # giving a deterministic CCW spin direction (positive vyaw in FLU).
        # Without the bias, +pi and -pi alias and the initial sign is undefined.
        theta_target = _wrap_pi(theta_0 + math.pi - 1e-3)
        yasmin.YASMIN_LOG_INFO(
            f"OrientToHook: rope behind (body_x={rope_body_x_med:+.3f}m). "
            f"Spinning 180°: theta_0={math.degrees(theta_0):+6.1f}° -> "
            f"theta_target={math.degrees(theta_target):+6.1f}°."
        )

        if not self._spin_to_target(
            drone, camera, segmentor, class_filter, theta_target, side_unit
        ):
            return ABORT

        new_side_unit = (-side_unit[0], -side_unit[1])
        new_anchor_sign = -anchor_sign
        blackboard["hose_side_image_unit"] = new_side_unit
        blackboard["anchor_sign"] = new_anchor_sign

        drone.move_velocity(
            0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=1.0
        )
        yasmin.YASMIN_LOG_INFO(
            f"OrientToHook: spin complete. "
            f"side_unit={side_unit}->{new_side_unit}, "
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
    ) -> Optional[Tuple[float, Tuple[float, float]]]:
        """Median rope body-x and the most recent sphere image position over
        ``ORIENT_BEHIND_SAMPLE_FRAMES`` frames. Returns None if no valid
        sphere+chosen-rope frame arrives within the timeout."""
        body_xs = []
        last_sphere_xy: Optional[Tuple[float, float]] = None
        deadline = time.time() + 5.0

        while (
            len(body_xs) < ORIENT_BEHIND_SAMPLE_FRAMES
            and time.time() < deadline
        ):
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)
            altitude = drone.get_altitude(AltitudeSource.LIDAR)
            if altitude is None:
                altitude = drone.get_altitude(AltitudeSource.AUTO)

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
                    rope_body_x=None,
                    theta_current=None,
                    theta_target=None,
                    err_rad=None,
                    vyaw=0.0,
                    sample_idx=len(body_xs),
                )
                time.sleep(0.05)
                continue

            ppm = px_per_meter(altitude, SPHERE_HEIGHT_M) if altitude else 0.0
            if ppm <= 0.0:
                time.sleep(0.05)
                continue

            _, hose_cy, _, _, _ = pose
            rope_body_x = -(hose_cy - IMAGE_CENTER_Y) / ppm
            body_xs.append(rope_body_x)
            last_sphere_xy = sphere.center

            self._save_overlay(
                frame, result,
                phase="sample",
                sphere_xy=last_sphere_xy,
                pose=pose,
                rope_body_x=rope_body_x,
                theta_current=None,
                theta_target=None,
                err_rad=None,
                vyaw=0.0,
                sample_idx=len(body_xs),
            )

            time.sleep(0.03)

        if not body_xs or last_sphere_xy is None:
            return None

        body_xs.sort()
        median = body_xs[len(body_xs) // 2]
        return median, last_sphere_xy

    def _spin_to_target(
        self,
        drone,
        camera,
        segmentor,
        class_filter,
        theta_target: float,
        side_unit: Tuple[float, float],
    ) -> bool:
        """Closed-loop spin. Holds last vyaw on transient sphere loss; aborts
        after ``ORIENT_MAX_LOST_FRAMES`` consecutive misses or
        ``ORIENT_TIMEOUT`` seconds."""
        confirmed = 0
        lost = 0
        last_vyaw = 0.0
        last_log = 0.0
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
                    return False
                drone.move_velocity(
                    vx=0.0, vy=0.0, vz=0.0, vyaw=last_vyaw,
                    reference=MoveReference.BODY,
                )
                self._save_overlay(
                    frame, result,
                    phase="spin",
                    sphere_xy=None,
                    pose=None,
                    rope_body_x=None,
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
                rope_body_x=None,
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
                    return True
            else:
                confirmed = 0

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
        yasmin.YASMIN_LOG_ERROR("OrientToHook: spin timed out before reaching target.")
        return False

    def _save_overlay(
        self,
        frame,
        result,
        *,
        phase: str,
        sphere_xy: Optional[Tuple[float, float]],
        pose: Optional[Tuple[float, float, float, float, Tuple[float, float]]],
        rope_body_x: Optional[float],
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
            rope_body_x=rope_body_x,
            behind_threshold_m=ORIENT_BEHIND_THRESHOLD_M,
            theta_current=theta_current,
            theta_target=theta_target,
            err_rad=err_rad,
            tol_rad=ORIENT_ANGLE_TOLERANCE_RAD,
            vyaw=vyaw,
            sample_idx=sample_idx,
            sample_total=ORIENT_BEHIND_SAMPLE_FRAMES,
        )
        self.frame_count += 1
        cv2.imwrite(
            str(self.save_dir / f"orient_{self.frame_count:04d}.jpg"), annotated
        )
