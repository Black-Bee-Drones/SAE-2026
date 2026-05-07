"""Lateral approach to the sphere along a frozen bearing, then descend.

Setpoint is a single image-space point on the line from the camera nadir
toward the sphere captured at the start of approach. The drone always
parks on that line, APPROACH_TARGET_DISTANCE_M short of the sphere
(measured at the hook). The bearing is the only frozen quantity; the
sphere position is re-detected every tick.

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
from nectar.control import AltitudeSource, MavrosDrone, MoveReference
from nectar.vision import ImageHandler

from hook.core.constants import (
    APPROACH_CONFIRMATIONS,
    APPROACH_DESCEND_VELOCITY,
    APPROACH_INIT_BEARING_FRAMES,
    APPROACH_KP,
    APPROACH_MAX_LOST_FRAMES,
    APPROACH_MAX_VELOCITY_XY,
    APPROACH_MIN_INIT_DIST_PX,
    APPROACH_MIN_OUTPUT_VELOCITY_XY,
    APPROACH_MIN_SAFE_DISTANCE_M,
    APPROACH_TARGET_DISTANCE_M,
    APPROACH_TIMEOUT,
    APPROACH_TOL_PX,
    DETECTION_SAVE_PATH,
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    SAVE_DETECTIONS,
    SPHERE_HEIGHT_M,
    WORK_ALTITUDE,
)
from hook.core.perception import (
    approach_setpoint,
    best_sphere,
    px_per_meter,
    run_seg,
)


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


class ApproachSphere(State):
    """Move horizontally onto the line drone-base -> sphere, stopping
    APPROACH_TARGET_DISTANCE_M short, then descend to WORK_ALTITUDE on the
    same line."""

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.save_dir = None
        self.frame_count = 0

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]
        camera: ImageHandler = blackboard["camera"]
        segmentor: Segmentor = blackboard["segmentor"]
        class_filter: PerClassConfidenceFilter = blackboard["class_filter"]

        if SAVE_DETECTIONS:
            ts = (
                blackboard["mission_timestamp"]
                if "mission_timestamp" in blackboard
                else datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "approach_sphere"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        bearing = self._capture_initial_bearing(camera, segmentor, class_filter)
        if bearing is None:
            yasmin.YASMIN_LOG_ERROR("Could not capture initial bearing.")
            return ABORT
        blackboard["approach_bearing_unit"] = bearing
        yasmin.YASMIN_LOG_INFO(
            f"Bearing locked: ux={bearing[0]:+.3f} uy={bearing[1]:+.3f}"
        )

        if not self._lateral_approach(
            drone, camera, segmentor, class_filter, blackboard, bearing
        ):
            return ABORT

        return self._descend_with_offset(
            drone, camera, segmentor, class_filter, blackboard, bearing
        )

    # ------------------------------------------------------------------
    # Bearing capture
    # ------------------------------------------------------------------

    def _capture_initial_bearing(
        self, camera, segmentor, class_filter
    ) -> Optional[Tuple[float, float]]:
        """Median bearing over the first APPROACH_INIT_BEARING_FRAMES valid
        detections. If the very first detection is closer than
        APPROACH_MIN_INIT_DIST_PX, returns (0,0) -> drone is already over
        the sphere; descent proceeds straight down. Returns None only if
        no detection arrives within the timeout."""
        samples = []
        deadline = time.time() + 5.0
        while len(samples) < APPROACH_INIT_BEARING_FRAMES and time.time() < deadline:
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)
            _, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)
            if sphere is None:
                time.sleep(0.05)
                continue
            cx, cy = sphere.center
            dx = cx - IMAGE_CENTER_X
            dy = cy - IMAGE_CENTER_Y
            if not samples and math.hypot(dx, dy) < APPROACH_MIN_INIT_DIST_PX:
                yasmin.YASMIN_LOG_INFO(
                    "Sphere already centered; skipping bearing lock."
                )
                return (0.0, 0.0)
            samples.append((dx, dy))

        if not samples:
            return None

        samples.sort(key=lambda s: s[0])
        dx_med = samples[len(samples) // 2][0]
        samples.sort(key=lambda s: s[1])
        dy_med = samples[len(samples) // 2][1]
        norm = math.hypot(dx_med, dy_med)
        if norm < 1e-3:
            return (0.0, 0.0)
        return (dx_med / norm, dy_med / norm)

    # ------------------------------------------------------------------
    # Velocity step
    # ------------------------------------------------------------------

    def _step_velocity(self, ex: float, ey: float) -> Tuple[float, float]:
        """Pure proportional control with two deadbands.

        - Error deadband: if the image error is inside APPROACH_TOL_PX,
          send (0, 0) so perception noise is not turned into commands.
        - Output deadband: if the proportional output is below
          APPROACH_MIN_OUTPUT_VELOCITY_XY, also send (0, 0). Avoids
          shipping sub-noise commands to MAVROS.
        """
        err = math.hypot(ex, ey)
        if err < APPROACH_TOL_PX:
            return 0.0, 0.0
        vx = -APPROACH_KP * ey
        vy = -APPROACH_KP * ex
        if math.hypot(vx, vy) < APPROACH_MIN_OUTPUT_VELOCITY_XY:
            return 0.0, 0.0
        return _clamp(vx, APPROACH_MAX_VELOCITY_XY), _clamp(vy, APPROACH_MAX_VELOCITY_XY)

    def _enforce_safety(
        self,
        vx: float,
        vy: float,
        sphere_image: Tuple[float, float],
        ppm: float,
    ) -> Tuple[float, float]:
        """Strip any velocity component that pushes the drone closer than
        APPROACH_MIN_SAFE_DISTANCE_M to the live sphere centroid."""
        if ppm <= 0.0:
            return vx, vy
        sx, sy = sphere_image
        dx_img = sx - IMAGE_CENTER_X
        dy_img = sy - IMAGE_CENTER_Y
        dist_px = math.hypot(dx_img, dy_img)
        if dist_px < 1e-3:
            return 0.0, 0.0
        if dist_px / ppm >= APPROACH_MIN_SAFE_DISTANCE_M:
            return vx, vy
        # Body unit vector pointing toward the sphere from the camera nadir
        # (image -y -> body +x, image +x -> body -y).
        bsx = -dy_img / dist_px
        bsy = -dx_img / dist_px
        v_toward = vx * bsx + vy * bsy
        if v_toward > 0.0:
            vx -= v_toward * bsx
            vy -= v_toward * bsy
        return vx, vy

    # ------------------------------------------------------------------
    # Step 1: lateral approach
    # ------------------------------------------------------------------

    def _lateral_approach(
        self, drone, camera, segmentor, class_filter, blackboard, bearing
    ) -> bool:
        yasmin.YASMIN_LOG_INFO("Step 1: lateral approach to setpoint...")
        confirmed = 0
        lost = 0
        start = time.time()
        last_log = 0.0

        while time.time() - start < APPROACH_TIMEOUT:
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)
            altitude = drone.get_altitude(AltitudeSource.LIDAR)
            if altitude is None:
                altitude = drone.get_altitude(AltitudeSource.AUTO)

            frame, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)

            if sphere is None:
                lost += 1
                if lost > APPROACH_MAX_LOST_FRAMES:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY
                    )
                    yasmin.YASMIN_LOG_ERROR("Lost sphere during lateral approach.")
                    return False
                time.sleep(0.05)
                continue

            lost = 0
            cx, cy = sphere.center
            ppm = px_per_meter(altitude, SPHERE_HEIGHT_M) if altitude else 0.0
            target_px = APPROACH_TARGET_DISTANCE_M * ppm if ppm > 0 else 0.0
            target_x, target_y = approach_setpoint(
                bearing, target_px, altitude, SPHERE_HEIGHT_M
            )

            ex = cx - target_x
            ey = cy - target_y
            err = math.hypot(ex, ey)

            if err < APPROACH_TOL_PX:
                confirmed += 1
                if confirmed >= APPROACH_CONFIRMATIONS:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=1.0
                    )
                    blackboard["approach_image_offset"] = (
                        cx - IMAGE_CENTER_X,
                        cy - IMAGE_CENTER_Y,
                    )
                    yasmin.YASMIN_LOG_INFO(
                        f"Approach reached: err={err:.0f}px target_px={target_px:.0f} "
                        f"sphere=({cx:.0f},{cy:.0f}) setpoint=({target_x:.0f},{target_y:.0f})"
                    )
                    return True
            else:
                confirmed = 0

            vx, vy = self._step_velocity(ex, ey)
            vx, vy = self._enforce_safety(vx, vy, (cx, cy), ppm)
            drone.move_velocity(
                vx=vx, vy=vy, vz=0.0, vyaw=0.0, reference=MoveReference.BODY,
            )

            self._save_overlay(
                frame, result, segmentor, "approach",
                bearing, (cx, cy), (target_x, target_y), (ex, ey), target_px, altitude,
            )

            now = time.time()
            if now - last_log > 0.5:
                yasmin.YASMIN_LOG_INFO(
                    f"Approach: err={err:.0f}px target_px={target_px:.0f} "
                    f"ex={ex:+.0f} ey={ey:+.0f} vx={vx:+.2f} vy={vy:+.2f}"
                )
                last_log = now

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
        yasmin.YASMIN_LOG_ERROR("Lateral approach timed out.")
        return False

    # ------------------------------------------------------------------
    # Step 2: descend keeping the same line
    # ------------------------------------------------------------------

    def _descend_with_offset(
        self, drone, camera, segmentor, class_filter, blackboard, bearing
    ):
        yasmin.YASMIN_LOG_INFO(f"Step 2: descending to {WORK_ALTITUDE}m...")
        lost = 0
        last_log = 0.0
        start = time.time()

        while time.time() - start < APPROACH_TIMEOUT:
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)

            altitude = drone.get_altitude(AltitudeSource.LIDAR)
            if altitude is None:
                altitude = drone.get_altitude(AltitudeSource.AUTO)
            if altitude is None:
                time.sleep(0.05)
                continue

            if altitude <= WORK_ALTITUDE:
                drone.move_velocity(
                    0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=2.0
                )
                yasmin.YASMIN_LOG_INFO(f"Reached work altitude ({altitude:.2f}m).")
                return SUCCEED

            frame, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)
            ppm = px_per_meter(altitude, SPHERE_HEIGHT_M)
            target_px = APPROACH_TARGET_DISTANCE_M * ppm
            target_x, target_y = approach_setpoint(
                bearing, target_px, altitude, SPHERE_HEIGHT_M
            )

            if sphere is None:
                lost += 1
                if lost > APPROACH_MAX_LOST_FRAMES:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY
                    )
                    yasmin.YASMIN_LOG_ERROR("Lost sphere during descent.")
                    return ABORT
                drone.move_velocity(
                    vx=0.0, vy=0.0, vz=-APPROACH_DESCEND_VELOCITY, vyaw=0.0,
                    reference=MoveReference.BODY,
                )
                time.sleep(0.05)
                continue

            lost = 0
            cx, cy = sphere.center
            blackboard["approach_image_offset"] = (
                cx - IMAGE_CENTER_X,
                cy - IMAGE_CENTER_Y,
            )
            ex = cx - target_x
            ey = cy - target_y
            vx, vy = self._step_velocity(ex, ey)
            vx, vy = self._enforce_safety(vx, vy, (cx, cy), ppm)

            drone.move_velocity(
                vx=vx, vy=vy, vz=-APPROACH_DESCEND_VELOCITY, vyaw=0.0,
                reference=MoveReference.BODY,
            )

            self._save_overlay(
                frame, result, segmentor, "descend",
                bearing, (cx, cy), (target_x, target_y), (ex, ey), target_px, altitude,
            )

            now = time.time()
            if now - last_log > 0.5:
                err = math.hypot(ex, ey)
                yasmin.YASMIN_LOG_INFO(
                    f"Descending: alt={altitude:.2f}m err={err:.0f}px "
                    f"target_px={target_px:.0f}"
                )
                last_log = now

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
        yasmin.YASMIN_LOG_ERROR("Descent to work altitude timed out.")
        return ABORT

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _save_overlay(
        self, frame, result, segmentor, phase,
        bearing, sphere_center, target, err, target_px, altitude,
    ):
        if not (SAVE_DETECTIONS and self.save_dir and frame is not None and result):
            return
        self.frame_count += 1
        annotated = segmentor.draw_segmentations(frame, result)
        _draw_approach_overlay(
            annotated, bearing, sphere_center, target, err, target_px, altitude,
        )
        cv2.imwrite(
            str(self.save_dir / f"{phase}_{self.frame_count:04d}.jpg"),
            annotated,
        )


# ----------------------------------------------------------------------
# Overlay helper
# ----------------------------------------------------------------------

_CYAN = (255, 255, 0)
_MAGENTA = (255, 0, 255)
_YELLOW = (0, 255, 255)
_RED = (0, 0, 255)
_GREEN = (0, 255, 0)


def _draw_approach_overlay(
    img,
    bearing: Tuple[float, float],
    sphere_center: Tuple[float, float],
    target: Tuple[float, float],
    err: Tuple[float, float],
    target_px: float,
    altitude: Optional[float],
):
    """Radial setpoint visualization (drone-centric).

    Drone target = where the cyan + should sit when parked. With the radial
    scheme, drone_target lies on the line from current camera nadir along
    the captured bearing direction, at the position that puts the sphere
    onto the bearing-aligned setpoint. Equivalently: drone_target =
    image_center + err. As the drone moves and the live sphere detection
    updates, drone_target moves; the controller stops when cyan + overlaps
    the yellow circle.
    """
    cx_img, cy_img = IMAGE_CENTER_X, IMAGE_CENTER_Y
    sx, sy = int(sphere_center[0]), int(sphere_center[1])
    ex, ey = err
    dt_x = cx_img + int(ex)
    dt_y = cy_img + int(ey)
    ux, uy = bearing

    ppm = target_px / APPROACH_TARGET_DISTANCE_M if APPROACH_TARGET_DISTANCE_M > 0 else 0.0
    to_m = (lambda p: p / ppm) if ppm > 1e-3 else (lambda p: 0.0)

    cv2.drawMarker(img, (cx_img, cy_img), _CYAN, cv2.MARKER_CROSS, 30, 2)
    cv2.circle(img, (sx, sy), 8, _RED, -1)

    cv2.line(img, (cx_img, cy_img), (sx, sy), _MAGENTA, 2)
    sphere_dist_px = math.hypot(sx - cx_img, sy - cy_img)
    cv2.putText(
        img, f"sphere={to_m(sphere_dist_px):.2f}m ({sphere_dist_px:.0f}px)",
        (int((cx_img + sx) / 2) + 10, int((cy_img + sy) / 2) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _MAGENTA, 2,
    )

    if abs(ux) > 1e-6 or abs(uy) > 1e-6:
        scale = max(IMAGE_WIDTH, IMAGE_HEIGHT)
        x1 = int(cx_img - ux * scale)
        y1 = int(cy_img - uy * scale)
        x2 = int(cx_img + ux * scale)
        y2 = int(cy_img + uy * scale)
        _draw_dashed_line(img, (x1, y1), (x2, y2), _YELLOW, 1, dash=14, gap=10)

    diag_px = math.hypot(ex, ey)
    cv2.arrowedLine(img, (cx_img, cy_img), (dt_x, dt_y), _GREEN, 2, tipLength=0.15)
    cv2.putText(
        img, f"err={diag_px:.0f}px ({to_m(diag_px):.2f}m)",
        (int((cx_img + dt_x) / 2) + 10, int((cy_img + dt_y) / 2) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREEN, 2,
    )

    cv2.circle(img, (dt_x, dt_y), int(max(APPROACH_TOL_PX, 8)), _YELLOW, 2)
    cv2.drawMarker(img, (dt_x, dt_y), _YELLOW, cv2.MARKER_TILTED_CROSS, 18, 2)

    alt_txt = f"{altitude:.2f}m" if altitude is not None else "n/a"
    text_lines = [
        f"alt={alt_txt}  D={APPROACH_TARGET_DISTANCE_M:.2f}m  target_px={target_px:.0f}  ppm={ppm:.0f}",
        f"err={diag_px:.0f}px ({to_m(diag_px):.2f}m)  err_xy=({ex:+.0f},{ey:+.0f})",
        f"bearing=({ux:+.2f},{uy:+.2f})  drone_target=({dt_x},{dt_y})  sphere=({sx},{sy})",
    ]
    for i, line in enumerate(text_lines):
        cv2.putText(
            img, line, (10, 30 + 28 * i),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, _GREEN, 2,
        )


def _draw_dashed_line(img, p1, p2, color, thickness, dash=15, gap=10):
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1.0:
        return
    dx = (x2 - x1) / length
    dy = (y2 - y1) / length
    step = dash + gap
    n = int(length // step) + 1
    for i in range(n):
        s = i * step
        e = min(s + dash, length)
        a = (int(x1 + dx * s), int(y1 + dy * s))
        b = (int(x1 + dx * e), int(y1 + dy * e))
        cv2.line(img, a, b, color, thickness)
