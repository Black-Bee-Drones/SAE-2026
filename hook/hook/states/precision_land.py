"""Visual precision landing on the blue base.

Three phases:

- ``rtl``: ``move_to(0, 0, PRECISION_LAND_RTL_ALTITUDE)`` against the
  TAKEOFF reference via PID_EKF. GPS / EKF drift moves the drone close to
  the takeoff point but not exactly onto it; the next phase corrects the
  remainder visually.
- ``align``: hover at RTL altitude, drive the base centroid onto the
  image center with two metric P-controllers (image-x via ``ppm_x``,
  image-y via ``ppm_y``). Exits when ``|err| < PRECISION_LAND_TOL_M``
  holds for ``PRECISION_LAND_INIT_CONFIRMATIONS`` frames.
- ``descend``: proportional ``vz = -clip(KP·(alt - target), 0, VZ_MAX)``
  toward ``PRECISION_LAND_TARGET_ALTITUDE`` while the xy P-controllers
  keep the base centered. SUCCEED when
  ``alt ≤ target + ALT_TOLERANCE`` AND
  ``|err| < PRECISION_LAND_TOL_M`` for ``LAND_CONFIRMATIONS`` frames;
  the subsequent ``LAND`` state issues the actual touchdown.

Image-to-body convention (down camera, FLU body):
  image -y -> body +x (forward)
  image +x -> body -y (right)
"""

import gc
import math
import threading
import time
from typing import Optional, Tuple

import numpy as np
import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.ai.detection import Detection, Detector
from nectar.control import (
    AltitudeSource,
    MavrosDrone,
    MoveReference,
    NavigationMethod,
    PIDController,
)
from nectar.vision import ImageHandler

from hook.core import overlay
from hook.core.constants import (
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    PID_MIN_OUTPUT_VELOCITY_XY,
    PRECISION_LAND_ALT_TOLERANCE_M,
    PRECISION_LAND_BASE_HEIGHT_M,
    PRECISION_LAND_CLASS_ID,
    PRECISION_LAND_CONF,
    PRECISION_LAND_IMGSZ,
    PRECISION_LAND_INIT_CONFIRMATIONS,
    PRECISION_LAND_IOU,
    PRECISION_LAND_KP,
    PRECISION_LAND_LAND_CONFIRMATIONS,
    PRECISION_LAND_MAX_LOST_FRAMES,
    PRECISION_LAND_MAX_VELOCITY_XY,
    PRECISION_LAND_MODEL_PATH,
    PRECISION_LAND_RTL_ALTITUDE,
    PRECISION_LAND_TARGET_ALTITUDE,
    PRECISION_LAND_TIMEOUT,
    PRECISION_LAND_TOL_M,
    PRECISION_LAND_VZ_KP,
    PRECISION_LAND_VZ_MAX,
)
from hook.core.frame_sink import FrameSink, build_state_sink
from hook.core.perception import px_per_meter_x, px_per_meter_y


def _best_base(result) -> Optional[Detection]:
    """Highest-confidence class-`PRECISION_LAND_CLASS_ID` detection, or None."""
    if result is None or not result.detections:
        return None
    candidates = [d for d in result.detections if d.class_id == PRECISION_LAND_CLASS_ID]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.confidence)


class PrecisionLand(State):
    """RTL to (0, 0, RTL_ALT), align on the blue base, descend to TARGET_ALT.

    SUCCEED hands off to the regular ``LAND`` state for the actual
    touchdown; ABORT also routes to ``LAND`` so the drone always lands.
    """

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.pid_x: Optional[PIDController] = None
        self.pid_y: Optional[PIDController] = None
        self.detector: Optional[Detector] = None
        self.sink: FrameSink = None
        self._load_done: Optional[threading.Event] = None
        self._load_error: Optional[BaseException] = None

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]
        camera: ImageHandler = blackboard["camera"]

        self.sink = build_state_sink(blackboard, "precision_land")

        try:
            # Load the detector in parallel with the RTL move so the heavy
            # YOLO startup is hidden under the drone's travel time. The
            # worker also frees the segmentor first to keep GPU memory low.
            self._start_detector_load(blackboard)

            if not self._rtl_to_origin(drone):
                return ABORT
            if not self._wait_detector():
                return ABORT
            if not self._align_initial(drone, camera):
                return ABORT
            return self._descend_and_succeed(drone, camera)
        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Precision land failed: {e}")
            return ABORT

    def _rtl_to_origin(self, drone: MavrosDrone) -> bool:
        yasmin.YASMIN_LOG_INFO(
            f"RTL to takeoff (0, 0, {PRECISION_LAND_RTL_ALTITUDE:.2f}m)..."
        )
        drone.move_to(
            x=0.0,
            y=0.0,
            z=PRECISION_LAND_RTL_ALTITUDE,
            reference=MoveReference.TAKEOFF,
            method=NavigationMethod.PID_EKF,
            precision=0.25,
            timeout=60.0,
        )
        drone.delay(1.0)
        return True

    def _start_detector_load(self, blackboard: Blackboard) -> None:
        """Spawn a daemon thread that frees the segmentor and loads the
        detector. Re-entry safe: a no-op if the load is already running or
        the detector is already loaded.
        """
        if self.detector is not None or self._load_done is not None:
            return
        self._load_done = threading.Event()
        self._load_error = None
        yasmin.YASMIN_LOG_INFO(
            f"Preloading blue-base detector in background: "
            f"{PRECISION_LAND_MODEL_PATH}"
        )
        threading.Thread(
            target=self._detector_load_worker,
            args=(blackboard,),
            daemon=True,
            name="precision-land-detector-loader",
        ).start()

    def _detector_load_worker(self, blackboard: Blackboard) -> None:
        try:
            self._free_segmentor(blackboard)
            t0 = time.perf_counter()
            detector = Detector(
                PRECISION_LAND_MODEL_PATH,
                confidence_threshold=PRECISION_LAND_CONF,
            )
            detector.load()
            detector.detect(
                np.zeros(
                    (PRECISION_LAND_IMGSZ, PRECISION_LAND_IMGSZ, 3),
                    dtype=np.uint8,
                ),
                conf=PRECISION_LAND_CONF,
                iou=PRECISION_LAND_IOU,
            )
            self.detector = detector
            yasmin.YASMIN_LOG_INFO(
                f"Detector ready after {time.perf_counter() - t0:.2f}s "
                f"(background)."
            )
        except BaseException as e:
            self._load_error = e
        finally:
            self._load_done.set()

    @staticmethod
    def _free_segmentor(blackboard: Blackboard) -> None:
        """Drop the segmentor reference and reclaim its GPU memory.

        Cheap insurance for tight GPU budgets (e.g., Jetson Orin Nano):
        after RELEASE the segmentation model is unused for the rest of
        the mission, so freeing it before loading the detector keeps
        peak GPU memory at one model.
        """
        if blackboard.contains("segmentor"):
            try:
                blackboard.remove("segmentor")
            except Exception:
                blackboard["segmentor"] = None
        if blackboard.contains("class_filter"):
            try:
                blackboard.remove("class_filter")
            except Exception:
                blackboard["class_filter"] = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _wait_detector(self) -> bool:
        if self.detector is not None:
            return True
        if self._load_done is None:
            return False
        self._load_done.wait()
        if self._load_error is not None:
            yasmin.YASMIN_LOG_ERROR(
                f"Detector load failed: {self._load_error}"
            )
            return False
        return self.detector is not None and self.detector.is_loaded

    def _ensure_pids(self) -> None:
        if self.pid_x is None:
            self.pid_x = PIDController(
                kp=PRECISION_LAND_KP,
                setpoint=0.0,
                output_limits=(
                    -PRECISION_LAND_MAX_VELOCITY_XY,
                    PRECISION_LAND_MAX_VELOCITY_XY,
                ),
                output_deadband=PID_MIN_OUTPUT_VELOCITY_XY,
            )
        if self.pid_y is None:
            self.pid_y = PIDController(
                kp=PRECISION_LAND_KP,
                setpoint=0.0,
                output_limits=(
                    -PRECISION_LAND_MAX_VELOCITY_XY,
                    PRECISION_LAND_MAX_VELOCITY_XY,
                ),
                output_deadband=PID_MIN_OUTPUT_VELOCITY_XY,
            )

    def _step_velocity(self, ex_m: float, ey_m: float) -> Tuple[float, float]:
        """Two metric P-controllers feeding body-frame velocities.

        ``ex_m`` is body-y meters (image-x via ``ppm_x``), ``ey_m`` is
        body-x meters (image-y via ``ppm_y``). With ``image -y -> body +x``
        and ``image +x -> body -y``, ``vx`` is driven by ``ey_m`` and
        ``vy`` by ``ex_m``.
        """
        err_m = math.hypot(ex_m, ey_m)
        if err_m < PRECISION_LAND_TOL_M:
            self.pid_x.reset()
            self.pid_y.reset()
            return 0.0, 0.0
        vx = self.pid_x.update(ey_m)
        vy = self.pid_y.update(ex_m)
        return vx, vy

    def _read(self, drone, camera):
        altitude = drone.get_altitude(AltitudeSource.LIDAR)
        if altitude is None:
            altitude = drone.get_altitude(AltitudeSource.AUTO)
        frame = camera.take_photo()
        if frame is None:
            return None, None, altitude
        result = self.detector.detect(
            frame, conf=PRECISION_LAND_CONF, iou=PRECISION_LAND_IOU
        )
        return frame, result, altitude

    def _errors(
        self, base: Detection, altitude: Optional[float]
    ) -> Tuple[float, float, float, float]:
        bx, by = base.center
        ex_px = bx - IMAGE_CENTER_X
        ey_px = by - IMAGE_CENTER_Y
        ppm_x = px_per_meter_x(altitude, PRECISION_LAND_BASE_HEIGHT_M)
        ppm_y = px_per_meter_y(altitude, PRECISION_LAND_BASE_HEIGHT_M)
        ex_m = ex_px / ppm_x if ppm_x > 0 else 0.0
        ey_m = ey_px / ppm_y if ppm_y > 0 else 0.0
        return ex_m, ey_m, ex_px, ey_px

    def _align_initial(self, drone: MavrosDrone, camera: ImageHandler) -> bool:
        self._ensure_pids()
        yasmin.YASMIN_LOG_INFO("Aligning on blue base at RTL altitude...")
        confirmed = 0
        lost = 0
        start = time.time()
        last_log = 0.0

        while time.time() - start < PRECISION_LAND_TIMEOUT:
            drone.delay(0.05)
            frame, result, altitude = self._read(drone, camera)
            if frame is None:
                continue

            base = _best_base(result)
            if base is None:
                lost += 1
                if lost > PRECISION_LAND_MAX_LOST_FRAMES:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY
                    )
                    yasmin.YASMIN_LOG_ERROR("Base lost during initial align.")
                    return False
                drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
                self._save_frame(
                    frame,
                    base=None,
                    altitude=altitude,
                    phase="align",
                    confirmations=confirmed,
                    target_confirmations=PRECISION_LAND_INIT_CONFIRMATIONS,
                    ex_m=0.0,
                    ey_m=0.0,
                    vx=0.0,
                    vy=0.0,
                    vz=0.0,
                )
                continue

            lost = 0
            ex_m, ey_m, _, _ = self._errors(base, altitude)
            err_m = math.hypot(ex_m, ey_m)

            if err_m < PRECISION_LAND_TOL_M:
                confirmed += 1
                drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
                self._save_frame(
                    frame,
                    base=base,
                    altitude=altitude,
                    phase="align",
                    confirmations=confirmed,
                    target_confirmations=PRECISION_LAND_INIT_CONFIRMATIONS,
                    ex_m=ex_m,
                    ey_m=ey_m,
                    vx=0.0,
                    vy=0.0,
                    vz=0.0,
                )
                if confirmed >= PRECISION_LAND_INIT_CONFIRMATIONS:
                    yasmin.YASMIN_LOG_INFO(
                        f"Align reached: err={err_m:.3f}m "
                        f"base=({base.center[0]:.0f},{base.center[1]:.0f})"
                    )
                    return True
                continue

            confirmed = 0
            vx, vy = self._step_velocity(ex_m, ey_m)
            drone.move_velocity(
                vx=vx,
                vy=vy,
                vz=0.0,
                vyaw=0.0,
                reference=MoveReference.BODY,
            )
            self._save_frame(
                frame,
                base=base,
                altitude=altitude,
                phase="align",
                confirmations=confirmed,
                target_confirmations=PRECISION_LAND_INIT_CONFIRMATIONS,
                ex_m=ex_m,
                ey_m=ey_m,
                vx=vx,
                vy=vy,
                vz=0.0,
            )

            now = time.time()
            if now - last_log > 0.5:
                yasmin.YASMIN_LOG_INFO(
                    f"Align alt={altitude:.2f}m | "
                    f"err={err_m:.3f}m ex={ex_m:+.3f}m ey={ey_m:+.3f}m | "
                    f"cmd: vx={vx:+.2f} vy={vy:+.2f}"
                )
                last_log = now

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
        yasmin.YASMIN_LOG_ERROR("Initial align timed out.")
        return False

    def _descend_and_succeed(self, drone: MavrosDrone, camera: ImageHandler) -> str:
        yasmin.YASMIN_LOG_INFO(
            f"Descending to {PRECISION_LAND_TARGET_ALTITUDE:.2f}m..."
        )
        confirmed = 0
        lost = 0
        start = time.time()
        last_log = 0.0

        while time.time() - start < PRECISION_LAND_TIMEOUT:
            drone.delay(0.05)
            frame, result, altitude = self._read(drone, camera)
            if frame is None or altitude is None:
                continue

            alt_err = altitude - PRECISION_LAND_TARGET_ALTITUDE
            vz = -max(0.0, min(PRECISION_LAND_VZ_KP * alt_err, PRECISION_LAND_VZ_MAX))

            base = _best_base(result)
            if base is None:
                lost += 1
                if lost > PRECISION_LAND_MAX_LOST_FRAMES:
                    drone.move_velocity(
                        0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY
                    )
                    yasmin.YASMIN_LOG_ERROR("Base lost during descent.")
                    return ABORT
                drone.move_velocity(
                    vx=0.0,
                    vy=0.0,
                    vz=vz,
                    vyaw=0.0,
                    reference=MoveReference.BODY,
                )
                self._save_frame(
                    frame,
                    base=None,
                    altitude=altitude,
                    phase="descend",
                    confirmations=confirmed,
                    target_confirmations=PRECISION_LAND_LAND_CONFIRMATIONS,
                    ex_m=0.0,
                    ey_m=0.0,
                    vx=0.0,
                    vy=0.0,
                    vz=vz,
                )
                continue

            lost = 0
            ex_m, ey_m, _, _ = self._errors(base, altitude)
            err_m = math.hypot(ex_m, ey_m)
            vx, vy = self._step_velocity(ex_m, ey_m)

            in_band = (
                altitude
                <= PRECISION_LAND_TARGET_ALTITUDE + PRECISION_LAND_ALT_TOLERANCE_M
            )
            in_tol = err_m < PRECISION_LAND_TOL_M

            if in_band and in_tol:
                confirmed += 1
                drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
                self._save_frame(
                    frame,
                    base=base,
                    altitude=altitude,
                    phase="descend",
                    confirmations=confirmed,
                    target_confirmations=PRECISION_LAND_LAND_CONFIRMATIONS,
                    ex_m=ex_m,
                    ey_m=ey_m,
                    vx=0.0,
                    vy=0.0,
                    vz=0.0,
                )
                if confirmed >= PRECISION_LAND_LAND_CONFIRMATIONS:
                    yasmin.YASMIN_LOG_INFO(
                        f"Precision land reached: alt={altitude:.2f}m "
                        f"err={err_m:.3f}m"
                    )
                    return SUCCEED
                continue

            confirmed = 0
            drone.move_velocity(
                vx=vx,
                vy=vy,
                vz=vz,
                vyaw=0.0,
                reference=MoveReference.BODY,
            )
            self._save_frame(
                frame,
                base=base,
                altitude=altitude,
                phase="descend",
                confirmations=confirmed,
                target_confirmations=PRECISION_LAND_LAND_CONFIRMATIONS,
                ex_m=ex_m,
                ey_m=ey_m,
                vx=vx,
                vy=vy,
                vz=vz,
            )

            now = time.time()
            if now - last_log > 0.5:
                yasmin.YASMIN_LOG_INFO(
                    f"Descend alt={altitude:.2f}m | "
                    f"err={err_m:.3f}m | "
                    f"cmd: vx={vx:+.2f} vy={vy:+.2f} vz={vz:+.2f}"
                )
                last_log = now

        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
        yasmin.YASMIN_LOG_ERROR("Descend-and-land timed out.")
        return ABORT

    def _save_frame(
        self,
        frame,
        *,
        base: Optional[Detection],
        altitude: Optional[float],
        phase: str,
        confirmations: int,
        target_confirmations: int,
        ex_m: float,
        ey_m: float,
        vx: float,
        vy: float,
        vz: float,
    ) -> None:
        if frame is None:
            return
        annotated = frame.copy()
        overlay.draw_precision_land(
            annotated,
            image_center=(IMAGE_CENTER_X, IMAGE_CENTER_Y),
            base_center=base.center if base is not None else None,
            base_bbox=base.bbox if base is not None else None,
            altitude=altitude,
            target_altitude=PRECISION_LAND_TARGET_ALTITUDE,
            err_x_m=ex_m,
            err_y_m=ey_m,
            err_total_m=math.hypot(ex_m, ey_m),
            tol_m=PRECISION_LAND_TOL_M,
            phase=phase,
            confirmations=confirmations,
            target_confirmations=target_confirmations,
            vx=vx,
            vy=vy,
            vz=vz,
        )
        self.sink.emit(annotated)
