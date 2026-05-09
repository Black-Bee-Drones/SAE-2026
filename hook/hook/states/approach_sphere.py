"""Lateral approach to the sphere along a frozen body-frame bearing,
then descend.

Setpoint is a single image-space point on the line from the camera nadir
toward the sphere captured at the start of approach. The drone always
parks on that line, ``APPROACH_TARGET_DISTANCE_M`` short of the sphere
(measured at the hook). The bearing is captured ONCE in **body frame**
so the per-tick image-target reconstruction is anisotropic-correct (the
camera's HFOV and VFOV imply different ppm in image-x vs image-y).

PIDs feed on metric body-frame errors (``ex_m`` along body-y from
image-x via ``ppm_x``; ``ey_m`` along body-x from image-y via ``ppm_y``)
so behavior is invariant to altitude across the descent from
``ascent_alt`` down to ``WORK_ALTITUDE``.

Image-to-body convention (down camera, FLU body):
  image -y -> body +x (forward)
  image +x -> body -y (right)
"""

import math
import time
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
    APPROACH_CONFIRMATIONS,
    APPROACH_DESCEND_VELOCITY,
    APPROACH_INIT_BEARING_FRAMES,
    APPROACH_KP_M,
    APPROACH_MAX_LOST_FRAMES,
    APPROACH_MAX_VELOCITY_XY,
    APPROACH_MIN_INIT_DIST_M,
    APPROACH_MIN_SAFE_DISTANCE_M,
    APPROACH_TARGET_DISTANCE_M,
    APPROACH_TIMEOUT,
    APPROACH_TOL_M,
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    PID_MIN_OUTPUT_VELOCITY_XY,
    SPHERE_HEIGHT_M,
    WORK_ALTITUDE,
)
from hook.core.frame_sink import FrameSink, build_state_sink
from hook.core.perception import (
    approach_setpoint,
    best_sphere,
    image_bearing_to_body_unit,
    image_offset_to_body,
    px_per_meter_x,
    px_per_meter_y,
    run_seg,
)


class ApproachSphere(State):
    """Move horizontally onto the line drone-base -> sphere, stopping
    APPROACH_TARGET_DISTANCE_M short, then descend to WORK_ALTITUDE on the
    same line."""

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.pid_x = None
        self.pid_y = None
        self.sink: FrameSink = None

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]
        camera: ImageHandler = blackboard["camera"]
        segmentor: Segmentor = blackboard["segmentor"]
        class_filter: PerClassConfidenceFilter = blackboard["class_filter"]

        if self.pid_x is None:
            self.pid_x = PIDController(
                kp=APPROACH_KP_M,
                setpoint=0.0,
                output_limits=(-APPROACH_MAX_VELOCITY_XY, APPROACH_MAX_VELOCITY_XY),
                output_deadband=PID_MIN_OUTPUT_VELOCITY_XY,
            )
        if self.pid_y is None:
            self.pid_y = PIDController(
                kp=APPROACH_KP_M,
                setpoint=0.0,
                output_limits=(-APPROACH_MAX_VELOCITY_XY, APPROACH_MAX_VELOCITY_XY),
                output_deadband=PID_MIN_OUTPUT_VELOCITY_XY,
            )

        self.sink = build_state_sink(blackboard, "approach_sphere")

        body_bearing = self._capture_initial_bearing(
            drone, camera, segmentor, class_filter
        )
        if body_bearing is None:
            yasmin.YASMIN_LOG_ERROR("Could not capture initial bearing.")
            return ABORT
        blackboard["approach_bearing_unit"] = body_bearing
        yasmin.YASMIN_LOG_INFO(
            f"Bearing locked (body frame): "
            f"bx={body_bearing[0]:+.3f} by={body_bearing[1]:+.3f}"
        )

        if not self._lateral_approach(
            drone, camera, segmentor, class_filter, blackboard, body_bearing
        ):
            return ABORT

        return self._descend_with_offset(
            drone, camera, segmentor, class_filter, blackboard, body_bearing
        )

    def _capture_initial_bearing(
        self, drone, camera, segmentor, class_filter
    ) -> Optional[Tuple[float, float]]:
        """Median image-frame bearing over the first
        ``APPROACH_INIT_BEARING_FRAMES`` valid detections, deprojected to
        a body-frame unit vector and frozen.

        Returns ``(0.0, 0.0)`` if the very first detection is already
        closer than ``APPROACH_MIN_INIT_DIST_M`` (true 2-D body distance,
        anisotropic-deprojected) — drone is over the sphere, descent
        goes straight down. Returns ``None`` only if no detection
        arrives within the timeout.
        """
        samples = []
        deadline = time.time() + 5.0
        while len(samples) < APPROACH_INIT_BEARING_FRAMES and time.time() < deadline:
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)
            altitude = drone.get_altitude(AltitudeSource.LIDAR)
            if altitude is None:
                altitude = drone.get_altitude(AltitudeSource.AUTO)
            _, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)
            if sphere is None:
                time.sleep(0.05)
                continue
            cx, cy = sphere.center
            dx = cx - IMAGE_CENTER_X
            dy = cy - IMAGE_CENTER_Y
            bx, by = image_offset_to_body((dx, dy), altitude, SPHERE_HEIGHT_M)
            dist_m = math.hypot(bx, by)
            if not samples and dist_m < APPROACH_MIN_INIT_DIST_M:
                yasmin.YASMIN_LOG_INFO(
                    f"Sphere already over drone (dist={dist_m:.2f}m); "
                    f"skipping bearing lock."
                )
                return (0.0, 0.0)
            samples.append((dx, dy, altitude))

        if not samples:
            return None

        samples.sort(key=lambda s: s[0])
        dx_med = samples[len(samples) // 2][0]
        samples.sort(key=lambda s: s[1])
        dy_med = samples[len(samples) // 2][1]
        alt_med = samples[len(samples) // 2][2]
        norm_image = math.hypot(dx_med, dy_med)
        if norm_image < 1e-3:
            return (0.0, 0.0)
        image_bearing = (dx_med / norm_image, dy_med / norm_image)
        return image_bearing_to_body_unit(image_bearing, alt_med, SPHERE_HEIGHT_M)

    def _enforce_safety(
        self,
        vx: float,
        vy: float,
        sphere_image: Tuple[float, float],
        altitude: Optional[float],
    ) -> Tuple[float, float]:
        """Strip any velocity component that would push the drone closer
        than ``APPROACH_MIN_SAFE_DISTANCE_M`` to the live sphere centroid,
        using true 2-D body-frame distance (anisotropic-deprojected from
        the image position).
        """
        sx, sy = sphere_image
        dx_img = sx - IMAGE_CENTER_X
        dy_img = sy - IMAGE_CENTER_Y
        bx, by = image_offset_to_body((dx_img, dy_img), altitude, SPHERE_HEIGHT_M)
        dist_m = math.hypot(bx, by)
        if dist_m < 1e-3:
            return 0.0, 0.0
        if dist_m >= APPROACH_MIN_SAFE_DISTANCE_M:
            return vx, vy
        # Body-frame unit toward the sphere; v_toward is the projection
        # of the body-frame velocity onto that unit.
        bsx = bx / dist_m
        bsy = by / dist_m
        v_toward = vx * bsx + vy * bsy
        if v_toward > 0.0:
            vx -= v_toward * bsx
            vy -= v_toward * bsy
        return vx, vy

    def _step_velocity(
        self, ex_m: float, ey_m: float
    ) -> Tuple[float, float]:
        """Two metric PIDs feeding the body-frame velocities.
        ``ex_m`` is body-y meters (image-x via ppm_x), ``ey_m`` is
        body-x meters (image-y via ppm_y). The image-to-body mapping
        is ``image -y -> body +x``, ``image +x -> body -y``, so vx is
        driven by ``ey_m`` (body-x error) and vy by ``ex_m`` (body-y
        error).
        """
        err_m = math.hypot(ex_m, ey_m)
        if err_m < APPROACH_TOL_M:
            self.pid_x.reset()
            self.pid_y.reset()
            return 0.0, 0.0
        vx = self.pid_x.update(ey_m)
        vy = self.pid_y.update(ex_m)
        return vx, vy

    def _lateral_approach(
        self, drone, camera, segmentor, class_filter, blackboard, body_bearing
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
            ppm_x = px_per_meter_x(altitude, SPHERE_HEIGHT_M)
            ppm_y = px_per_meter_y(altitude, SPHERE_HEIGHT_M)
            target_x, target_y = approach_setpoint(
                body_bearing, APPROACH_TARGET_DISTANCE_M, altitude, SPHERE_HEIGHT_M
            )

            ex_px = cx - target_x
            ey_px = cy - target_y
            ex_m = ex_px / ppm_x if ppm_x > 0 else 0.0
            ey_m = ey_px / ppm_y if ppm_y > 0 else 0.0
            err_m = math.hypot(ex_m, ey_m)

            if err_m < APPROACH_TOL_M:
                confirmed += 1
                if confirmed >= APPROACH_CONFIRMATIONS:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=2.5
                    )
                    blackboard["approach_image_offset"] = (
                        cx - IMAGE_CENTER_X,
                        cy - IMAGE_CENTER_Y,
                    )
                    yasmin.YASMIN_LOG_INFO(
                        f"Approach reached: err={err_m:.3f}m "
                        f"sphere=({cx:.0f},{cy:.0f}) setpoint=({target_x:.0f},{target_y:.0f})"
                    )
                    return True
            else:
                confirmed = 0

            vx, vy = self._step_velocity(ex_m, ey_m)
            vx, vy = self._enforce_safety(vx, vy, (cx, cy), altitude)
            drone.move_velocity(
                vx=vx, vy=vy, vz=0.0, vyaw=0.0, reference=MoveReference.BODY,
            )

            self._save_overlay(
                frame, result,
                body_bearing, (cx, cy), (target_x, target_y), (ex_px, ey_px),
                ppm_x, ppm_y, altitude, vx, vy, 0.0,
            )

            now = time.time()
            if now - last_log > 0.5:
                yasmin.YASMIN_LOG_INFO(
                    f"Approach alt={altitude:.2f}m | "
                    f"err={err_m:+.3f}m "
                    f"ex={ex_m:+.3f}m ey={ey_m:+.3f}m | "
                    f"cmd: vx={vx:+.2f} vy={vy:+.2f}"
                )
                last_log = now

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
        yasmin.YASMIN_LOG_ERROR("Lateral approach timed out.")
        return False

    def _descend_with_offset(
        self, drone, camera, segmentor, class_filter, blackboard, body_bearing
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
            ppm_x = px_per_meter_x(altitude, SPHERE_HEIGHT_M)
            ppm_y = px_per_meter_y(altitude, SPHERE_HEIGHT_M)
            target_x, target_y = approach_setpoint(
                body_bearing, APPROACH_TARGET_DISTANCE_M, altitude, SPHERE_HEIGHT_M
            )

            if sphere is None:
                lost += 1
                if lost > APPROACH_MAX_LOST_FRAMES:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY
                    )
                    yasmin.YASMIN_LOG_ERROR("Lost sphere during descent.")
                    return ABORT
                vz_blind = -APPROACH_DESCEND_VELOCITY if altitude > WORK_ALTITUDE else 0.0
                drone.move_velocity(
                    vx=0.0, vy=0.0, vz=vz_blind, vyaw=0.0,
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
            ex_px = cx - target_x
            ey_px = cy - target_y
            ex_m = ex_px / ppm_x if ppm_x > 0 else 0.0
            ey_m = ey_px / ppm_y if ppm_y > 0 else 0.0
            vx, vy = self._step_velocity(ex_m, ey_m)
            vx, vy = self._enforce_safety(vx, vy, (cx, cy), altitude)

            drone.move_velocity(
                vx=vx, vy=vy, vz=-APPROACH_DESCEND_VELOCITY, vyaw=0.0,
                reference=MoveReference.BODY,
            )

            self._save_overlay(
                frame, result,
                body_bearing, (cx, cy), (target_x, target_y), (ex_px, ey_px),
                ppm_x, ppm_y, altitude, vx, vy, -APPROACH_DESCEND_VELOCITY,
            )

            now = time.time()
            if now - last_log > 0.5:
                err_m = math.hypot(ex_m, ey_m)
                yasmin.YASMIN_LOG_INFO(
                    f"Descending alt={altitude:.2f}m | "
                    f"err={err_m:+.3f}m | "
                    f"cmd: vx={vx:+.2f} vy={vy:+.2f} vz={-APPROACH_DESCEND_VELOCITY:+.2f}"
                )
                last_log = now

            time.sleep(0.03)

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
        yasmin.YASMIN_LOG_ERROR("Descent to work altitude timed out.")
        return ABORT

    def _save_overlay(
        self, frame, result,
        body_bearing, sphere_center, target, err_px,
        ppm_x, ppm_y, altitude, vx, vy, vz,
    ):
        if frame is None or not result:
            return
        annotated = overlay.annotate_seg(frame, result)
        _draw_approach_overlay(
            annotated, body_bearing, sphere_center, target, err_px,
            ppm_x, ppm_y, altitude, vx, vy, vz,
        )
        self.sink.emit(annotated)


_CYAN = (255, 255, 0)
_MAGENTA = (255, 0, 255)
_YELLOW = (0, 255, 255)
_RED = (0, 0, 255)
_GREEN = (0, 255, 0)


def _draw_approach_overlay(
    img,
    body_bearing: Tuple[float, float],
    sphere_center: Tuple[float, float],
    target: Tuple[float, float],
    err_px: Tuple[float, float],
    ppm_x: float,
    ppm_y: float,
    altitude: Optional[float],
    vx: float,
    vy: float,
    vz: float,
):
    """Radial setpoint visualization (drone-centric).

    drone_target = where the cyan + should sit when parked. With the
    radial scheme this is image_center + err_px; the controller stops
    when cyan + overlaps the yellow circle.

    Anisotropic px↔m: sphere distance and the err arrow are in real
    body-frame meters (image-x via ``ppm_x``, image-y via ``ppm_y``).
    """
    cx_img, cy_img = IMAGE_CENTER_X, IMAGE_CENTER_Y
    sx, sy = int(sphere_center[0]), int(sphere_center[1])
    ex, ey = err_px
    dt_x = cx_img + int(ex)
    dt_y = cy_img + int(ey)
    bx, by = body_bearing

    sphere_dx = sx - cx_img
    sphere_dy = sy - cy_img
    sphere_bx = -sphere_dy / ppm_y if ppm_y > 0 else 0.0
    sphere_by = -sphere_dx / ppm_x if ppm_x > 0 else 0.0
    sphere_dist_m = math.hypot(sphere_bx, sphere_by)
    err_bx = -ey / ppm_y if ppm_y > 0 else 0.0
    err_by = -ex / ppm_x if ppm_x > 0 else 0.0
    err_dist_m = math.hypot(err_bx, err_by)
    # Tolerance ring radius: render in pixels at the average ppm so the
    # circle is visually clean (the tolerance itself is a true 2-D body
    # distance; we render an isotropic stand-in).
    ppm_avg = 0.5 * (ppm_x + ppm_y) if (ppm_x > 0 and ppm_y > 0) else 0.0
    tol_px = max(int(APPROACH_TOL_M * ppm_avg), 8) if ppm_avg > 0 else 8

    cv2.drawMarker(img, (cx_img, cy_img), _CYAN, cv2.MARKER_CROSS, 30, 2)
    cv2.circle(img, (sx, sy), 8, _RED, -1)

    cv2.line(img, (cx_img, cy_img), (sx, sy), _MAGENTA, 2)
    sphere_dist_px = math.hypot(sx - cx_img, sy - cy_img)
    cv2.putText(
        img, f"sphere={sphere_dist_m:.2f}m ({sphere_dist_px:.0f}px)",
        (int((cx_img + sx) / 2) + 10, int((cy_img + sy) / 2) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _MAGENTA, 2,
    )

    # Body bearing rendered in image: project the body unit through the
    # anisotropic pinhole so the ray actually points where the sphere
    # was at sample time (body +y → image -x · ppm_x, body +x → image
    # -y · ppm_y).
    if abs(bx) > 1e-6 or abs(by) > 1e-6:
        bearing_image_x = -by * ppm_x
        bearing_image_y = -bx * ppm_y
        norm = math.hypot(bearing_image_x, bearing_image_y)
        if norm > 1e-3:
            bearing_image_x /= norm
            bearing_image_y /= norm
            scale = max(IMAGE_WIDTH, IMAGE_HEIGHT)
            x1 = int(cx_img - bearing_image_x * scale)
            y1 = int(cy_img - bearing_image_y * scale)
            x2 = int(cx_img + bearing_image_x * scale)
            y2 = int(cy_img + bearing_image_y * scale)
            _draw_dashed_line(img, (x1, y1), (x2, y2), _YELLOW, 1, dash=14, gap=10)

    diag_px = math.hypot(ex, ey)
    cv2.arrowedLine(img, (cx_img, cy_img), (dt_x, dt_y), _GREEN, 2, tipLength=0.15)
    cv2.putText(
        img, f"err={err_dist_m:.3f}m ({diag_px:.0f}px)",
        (int((cx_img + dt_x) / 2) + 10, int((cy_img + dt_y) / 2) - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, _GREEN, 2,
    )

    cv2.circle(img, (dt_x, dt_y), tol_px, _YELLOW, 2)
    cv2.drawMarker(img, (dt_x, dt_y), _YELLOW, cv2.MARKER_TILTED_CROSS, 18, 2)

    alt_txt = f"{altitude:.2f}m" if altitude is not None else "n/a"
    text_lines = [
        f"alt={alt_txt}  D={APPROACH_TARGET_DISTANCE_M:.2f}m  "
        f"ppm_x={ppm_x:.0f}  ppm_y={ppm_y:.0f}",
        f"err={err_dist_m:+.3f}m  err_body=({err_bx:+.3f}m,{err_by:+.3f}m)",
        f"body_bearing=({bx:+.2f},{by:+.2f})  drone_target=({dt_x},{dt_y})  sphere=({sx},{sy})",
        f"cmd: vx={vx:+.2f}  vy={vy:+.2f}  vz={vz:+.2f}",
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
