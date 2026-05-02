import cv2
import time
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.ai.detection.models.ultralytics import UltralyticsModel

from .navigation import Navigation

from ..parameters import SIMULATION

from zaxis.drone import Drone

from sensor_msgs.msg import CompressedImage

import threading

from zaxis.runtime import CommandHandle

PIXELS_PER_METER = 1.0 / 0.0010210406  # k = 0.0010210406 m/px

# ---------------------------------------------------------------------------
# PID controller (single axis, stateless between calls via returned state)
# ---------------------------------------------------------------------------

class PIDController:
    """Minimal single-axis PID."""

    def __init__(self, kp: float, ki: float, kd: float, output_limit: float = 1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def update(self, error: float) -> float:
        now = time.time()
        dt = (now - self._prev_time) if self._prev_time is not None else 0.0
        self._prev_time = now

        self._integral += error * dt
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, output))


# ---------------------------------------------------------------------------
# GaugeReading state
# ---------------------------------------------------------------------------

class GaugeReading(State):
    """
    YASMIN state that:
      1. Searches for a manometer by climbing in steps if nothing is detected.
         Climbing is relative to the current position (non-accumulative).
      2. Once found (coarse model), runs a PID centering loop.
      3. Descends to the *trusted zone* altitude.
      4. Reads the gauge class with the main classification model.

    Parameters
    ----------
    model_path : str
        Path to the **classification** model (detects pressure classes).
    coarse_model_path : str
        Path to the **coarse detection** model (detects the manometer without
        classifying it — used only for centering).
    confidence_threshold : float
        Minimum confidence for the classification model.
    coarse_confidence_threshold : float
        Minimum confidence for the coarse centering model.
    altitude_limit_m : float
        Maximum altitude the drone is allowed to climb to during the search
        phase.  Expressed as a positive number; converted to negative NED Z
        internally (e.g. 5.0 → z = -5.0 m NED).  The drone will stop
        climbing and wait if it reaches this ceiling.
    climb_step_m : float
        How many metres to climb each time no detection is found (relative,
        non-accumulative).
    climb_timeout_s : float
        Seconds to wait without detection before triggering another climb step.
    trusted_zone_altitude_m : float
        Absolute altitude (NED Z, so *negative* means higher) to descend to
        before running the classification model.  Pass as a *positive* value —
        the sign inversion is handled internally.
    pid_kp, pid_ki, pid_kd : float
        PID gains for the centering loop (applied identically to both axes).
    pid_output_limit_m : float
        Maximum single-step correction in metres for the PID.
    centering_tolerance_norm : float
        Centering is considered done when the normalised error on *both* axes
        is below this value.  Expressed as a fraction of the frame dimension,
        so 0.05 means "within 5 % of the frame width/height from centre."
        Using normalised units keeps the threshold altitude-independent.
    cam : optional
        Camera object.  Falls back to ``cv2.VideoCapture(0)`` if None.
    """

    def __init__(
        self,
        model_path: str,
        coarse_model_path: str,
        confidence_threshold: float = 0.85,
        coarse_confidence_threshold: float = 0.60,
        climb_step_m: float = 0.5,
        climb_timeout_s: float = 5.0,
        altitude_limit_m: float = 5.0,
        trusted_zone_altitude_m: float = 1.5,
        pid_kp: float = 0.4,
        pid_ki: float = 0.0,
        pid_kd: float = 0.05,
        pid_output_limit_m: float = 0.5,
        centering_tolerance_norm: float = 0.05,
        cam=None,
    ):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.node = YasminNode.get_instance()
        self.confidence_threshold = confidence_threshold
        self.coarse_confidence_threshold = coarse_confidence_threshold
        self.climb_step_m = climb_step_m
        self.climb_timeout_s = climb_timeout_s
        self.altitude_limit_m = altitude_limit_m
        self._climbing = False
        # Store as positive; sign inversion happens when commanding the drone.
        self.trusted_zone_altitude_m = abs(trusted_zone_altitude_m)
        self.centering_tolerance_norm = centering_tolerance_norm
        self.cam = cam
        self._stop_event = threading.Event()

        # PID controllers (one per axis)
        self._pid_fwd = PIDController(pid_kp, pid_ki, pid_kd, pid_output_limit_m)
        self._pid_right = PIDController(pid_kp, pid_ki, pid_kd, pid_output_limit_m)

        # --- Classification model (main) ---
        try:
            self.detector = UltralyticsModel(model_name=model_path)
            self.detector.load_model()
        except Exception as e:
            self.node.get_logger().error(f"Error loading classification model: {e}")
            self.detector = None

        # --- Coarse detection model (centering only) ---
        try:
            self.coarse_detector = UltralyticsModel(model_name=coarse_model_path)
            self.coarse_detector.load_model()
        except Exception as e:
            self.node.get_logger().error(f"Error loading coarse detection model: {e}")
            self.coarse_detector = None

        self.drone: Drone = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_frame(self):
        """Read one frame from the camera.  Returns (ok, frame)."""
        if SIMULATION:
            frame = self.cam.frame
            return frame is not None, frame
        return self.cam.read()

    def get_square_crops(self, frame, overlap=0.3):
        """Sliding-window square crops covering the full FOV."""
        h, w = frame.shape[:2]
        size = min(h, w)
        step = int(size * (1 - overlap))
        crops = []

        if w >= h:
            x = 0
            while x + size <= w:
                crops.append((frame[0:size, x : x + size], x, 0))
                x += step
            if not crops or crops[-1][1] + size < w:
                crops.append((frame[0:size, w - size : w], w - size, 0))
        else:
            y = 0
            while y + size <= h:
                crops.append((frame[y : y + size, 0:size], 0, y))
                y += step
            if not crops or crops[-1][2] + size < h:
                crops.append((frame[h - size : h, 0:size], 0, h - size))

        return crops

    def _best_detection_in_frame(self, frame, detector, conf_threshold):
        """
        Run *detector* over all sliding-window crops of *frame* and return
        (best_detection, x_off, y_off, annotated_crop) or (None, 0, 0, None).
        """
        best_detection = None
        best_confidence = 0.0
        best_x_off = 0
        best_y_off = 0
        best_annotated = None

        for crop, x_off, y_off in self.get_square_crops(frame):
            result = detector.detect(crop, conf=conf_threshold)
            detections = result if isinstance(result, list) else result.detections

            for det in detections:
                if det.confidence > best_confidence:
                    best_detection = det
                    best_confidence = det.confidence
                    best_x_off = x_off
                    best_y_off = y_off
                    best_annotated = detector.draw_detections(crop, result)

        return best_detection, best_x_off, best_y_off, best_annotated

    def _climb_relative(self, delta_m: float):
        """
        Climb *delta_m* metres relative to the current position (non-accumulative).
        Blocks until the move completes or the stop event is set.

        NED convention: Z is positive *downward*, so higher altitude = more
        negative Z.  ``altitude_limit_m`` is stored as a positive number and
        converted to ``-altitude_limit_m`` here as the ceiling.
        """
        self._climbing = True
        try:
            current_z = self.drone.local_position.z
            altitude_ceiling = -abs(self.altitude_limit_m)

            if current_z <= altitude_ceiling:
                self.node.get_logger().warn(
                    f"[GaugeReading] Altitude limit reached "
                    f"(z={current_z:.2f} m, ceiling={altitude_ceiling:.2f} m NED). "
                    "Stopping climb."
                )
                return

            target_z = max(current_z - abs(delta_m), altitude_ceiling)

            self.node.get_logger().info(
                f"[GaugeReading] Climbing {delta_m:.2f} m "
                f"(z: {current_z:.2f} → {target_z:.2f}, ceiling={altitude_ceiling:.2f})"
            )

            task = self.drone.goto_local(
                x=self.drone.local_position.x,
                y=self.drone.local_position.y,
                z=target_z,
            )
            if task is None:
                return

            while not task.done():
                if self._stop_event.is_set():
                    task.stop()
                    return
                time.sleep(0.05)
        finally:
            self._climbing = False

    def _pid_center_and_descend(self) -> bool:
        """
        Combined PID centering + descent loop using the *coarse* model.

        Error is expressed in **normalised image coordinates** [-0.5, +0.5],
        computed as (bbox_centre - frame_centre) / frame_dimension.  This makes
        the loop scale-invariant: the PID gains stay valid regardless of
        altitude, resolution, or optics, because the error is always relative
        to the frame size rather than an absolute pixel count that would change
        with the drone's height.

        The PID gains (pid_kp / ki / kd) are therefore tuned in units of
        metres per unit of normalised error — i.e. kp=1.0 means "move 1 m
        when the target is at the frame edge".

        **Descent behaviour**: on every iteration where the drone is already
        within ``centering_tolerance_norm`` on both axes, the target Z is
        stepped down by ``descent_step_m`` toward ``trusted_zone_altitude_m``.
        Lateral correction and descent are issued as a single ``goto_local``
        command so the drone moves diagonally rather than stop-go.  If the
        drone drifts outside tolerance the descent step is skipped that
        iteration and the PID corrects position first.

        Returns True once at the trusted zone altitude, False if the object
        is lost for too long or the loop times out.
        """
        self._pid_fwd.reset()
        self._pid_right.reset()

        # trusted_zone_altitude_m is stored as a positive value (e.g. 1.5 m).
        # In NED, higher altitude = more negative Z, so the trusted zone sits
        # at z = -1.5 m.  The drone may be above OR below that when this loop
        # starts, so we compute the step direction dynamically each iteration
        # rather than assuming we always descend.
        trusted_z = -abs(self.trusted_zone_altitude_m)
        ALTITUDE_STEP_M = 0.1   # metres to move per iteration when centred
        ALTITUDE_TOLERANCE_M = 0.05  # close enough to trusted zone
        LOST_TIMEOUT_S = 4.0
        MAX_CENTERING_S = 60.0
        start = time.time()
        lost_since = None

        self.node.get_logger().info(
            f"[GaugeReading] Starting PID centering + altitude correction "
            f"(trusted zone z={trusted_z:.2f} m NED)."
        )

        while time.time() - start < MAX_CENTERING_S:
            if self._stop_event.is_set():
                return False

            ok, frame = self._read_frame()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            detection, x_off, y_off, _ = self._best_detection_in_frame(
                frame, self.coarse_detector, self.coarse_confidence_threshold
            )

            if detection is None:
                if lost_since is None:
                    lost_since = time.time()
                elif time.time() - lost_since > LOST_TIMEOUT_S:
                    self.node.get_logger().warn(
                        "[GaugeReading] Object lost during centering — aborting PID."
                    )
                    return False
                self.node.get_logger().info(
                    "[GaugeReading] Coarse model: no detection.",
                    throttle_duration_sec=1.0,
                )
                time.sleep(0.05)
                continue

            lost_since = None  # object found again
            self.node.get_logger().info(
                f"[GaugeReading] Coarse model: detection conf={detection.confidence:.3f}."
            )

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = detection.xyxy
            bbox_cx = (x1 + x2) / 2.0 + x_off
            bbox_cy = (y1 + y2) / 2.0 + y_off

            # Normalised error: 0.0 = centred, ±0.5 = at the frame edge.
            # Dividing by the frame dimension makes this independent of
            # resolution and altitude.
            norm_dx = (bbox_cx - w / 2.0) / w   # + → target is right of centre
            norm_dy = (bbox_cy - h / 2.0) / h   # + → target is below centre

            is_centred = (
                abs(norm_dx) <= self.centering_tolerance_norm
                and abs(norm_dy) <= self.centering_tolerance_norm
            )

            current_z = self.drone.local_position.z

            at_trusted_zone = abs(current_z - trusted_z) <= ALTITUDE_TOLERANCE_M

            # When centred, step toward trusted_z regardless of whether that
            # means ascending (current_z < trusted_z, e.g. -5.0 → -1.5) or
            # descending (current_z > trusted_z).  The step direction is
            # derived from the sign of the remaining error each iteration.
            if is_centred and not at_trusted_zone:
                z_error = trusted_z - current_z          # + = need to ascend in NED
                step = min(ALTITUDE_STEP_M, abs(z_error)) * (1 if z_error > 0 else -1)
                next_z = current_z + step
                direction = "ascending" if step > 0 else "descending"
                self.node.get_logger().info(
                    f"[GaugeReading] Centred — {direction} toward trusted zone "
                    f"(z: {current_z:.2f} → {next_z:.2f} m NED, target={trusted_z:.2f})."
                )
            else:
                next_z = current_z  # hold altitude while correcting position

            # Done: centred AND within tolerance of trusted zone.
            if is_centred and at_trusted_zone:
                self.node.get_logger().info(
                    f"[GaugeReading] Reached trusted zone while centred "
                    f"(z={current_z:.2f} m NED) — done."
                )
                return True

            # PID outputs in metres.
            # norm_dy is negated because image Y increases downward while
            # the drone's forward axis (FRD X) increases forward.
            fwd_cmd   = self._pid_fwd.update(-norm_dy)   # image Y inverted → FRD X
            right_cmd = self._pid_right.update(norm_dx)

            # Zero out lateral command when centred to avoid jitter.
            if is_centred:
                fwd_cmd   = 0.0
                right_cmd = 0.0

            self.node.get_logger().info(
                f"[GaugeReading] PID: norm_dx={norm_dx:.4f} right={right_cmd:.3f}m  "
                f"norm_dy={norm_dy:.4f} fwd={fwd_cmd:.3f}m  "
                f"z={current_z:.2f}→{next_z:.2f}m  centred={is_centred}"
            )

            frame_location = self.drone.capture_origin()
            task = self.drone.goto_local(
                x=fwd_cmd,
                y=right_cmd,
                z=next_z,
                origin=frame_location,
            )
            if task is not None:
                move_deadline = time.time() + 2.0  # never block longer than 2 s
                while not task.done():
                    if self._stop_event.is_set() or time.time() > move_deadline:
                        task.stop()
                        if time.time() > move_deadline:
                            self.node.get_logger().warn(
                                "[GaugeReading] goto_local timed out — continuing PID."
                            )
                        break
                    time.sleep(0.05)

        self.node.get_logger().warn("[GaugeReading] PID centering + descent timed out.")
        return False

    def _publish_frame(self, blackboard: Blackboard, frame):
        """Compress *frame* and publish it on the debug topic."""
        _, buffer = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
        )
        msg = CompressedImage()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = buffer.tobytes()
        blackboard["inference_image_publisher"].publish(msg)

    # ------------------------------------------------------------------
    # Main execute
    # ------------------------------------------------------------------

    def execute(self, blackboard: Blackboard):
        # ----------------------------------------------------------------
        # Setup
        # ----------------------------------------------------------------
        if self.detector is None or self.coarse_detector is None:
            self.node.get_logger().error("One or more models failed to load.")
            return ABORT

        if "inference_image_publisher" not in blackboard:
            blackboard["inference_image_publisher"] = self.node.create_publisher(
                CompressedImage, "/gauge/compressed", 10
            )

        if "drone" in blackboard:
            self.drone = blackboard["drone"]

        self._stop_event.clear()

        if self.cam is None:
            self.cam = cv2.VideoCapture(0)

        goto_handler: CommandHandle = blackboard["goto_handler"]

        # ----------------------------------------------------------------
        # Phase 1 — Search + classify (with continuous relative climbing)
        # ----------------------------------------------------------------
        max_duration = 100.0
        consecutive_limit = 5

        start_time = time.time()
        time_last_det = None
        consecutive_count = 0
        last_class_id = None
        best_detection_overall = None
        best_confidence_overall = 0.0
        best_inference_image_overall = None
        last_inference_image = None
        frame_count = 0

        # Climbing state — tracks when we last climbed so we do NOT climb
        # again until *climb_timeout_s* has elapsed without a detection.
        last_climb_time: float = None  # None = haven't climbed yet this phase

        # Track when the goto_handler finished so we can enforce the
        # post-navigation settling delay before starting to climb.
        GOTO_SETTLE_S = 5.0
        goto_finished_at: float = None  # None = handler not yet done

        try:
            while (time.time() - start_time) < max_duration:
                # ----------------------------------------------------------
                # Stop condition: detection found and confirmed
                # ----------------------------------------------------------
                if (
                    best_detection_overall is not None
                    and time_last_det is not None
                    and time.time() - time_last_det > 10.0
                ):
                    self._stop_event.set()
                    break

                # Track when the navigation handler transitions to done so
                # we can measure the settling delay from that moment.
                if not goto_handler.done():
                    start_time = time.time()
                    goto_finished_at = None  # still moving — reset the clock
                elif goto_finished_at is None:
                    goto_finished_at = time.time()  # just became done — latch

                # ----------------------------------------------------------
                # Climbing logic (non-accumulative, continuous)
                # ----------------------------------------------------------
                # We climb if ALL of these are true:
                #   - goto_handler is done AND has been done for >= GOTO_SETTLE_S
                #   - No detection has ever been seen OR detection was lost
                #   - The previous climb has fully finished (_climbing=False)
                #   - Enough time has passed since the last climb started
                #   - The altitude ceiling has not been reached
                handler_settled = (
                    goto_finished_at is not None
                    and (time.time() - goto_finished_at) >= GOTO_SETTLE_S
                )
                no_recent_detection = (
                    time_last_det is None
                    or (time.time() - time_last_det) > self.climb_timeout_s
                )
                enough_time_since_last_climb = (
                    last_climb_time is None
                    or (time.time() - last_climb_time) > self.climb_timeout_s
                )
                below_ceiling = (
                    self.drone.local_position.z > -abs(self.altitude_limit_m)
                )

                if (
                    handler_settled
                    and no_recent_detection
                    and enough_time_since_last_climb
                    and not self._climbing
                    and below_ceiling
                ):
                    last_climb_time = time.time()
                    climb_thread = threading.Thread(
                        target=self._climb_relative,
                        args=(self.climb_step_m,),
                        daemon=True,
                    )
                    climb_thread.start()
                elif not handler_settled and goto_finished_at is not None:
                    remaining = GOTO_SETTLE_S - (time.time() - goto_finished_at)
                    self.node.get_logger().info(
                        f"[GaugeReading] Waiting {remaining:.1f}s after navigation "
                        "before climbing.",
                        throttle_duration_sec=2.0,
                    )
                elif not below_ceiling:
                    self.node.get_logger().warn(
                        "[GaugeReading] Altitude ceiling reached — holding position.",
                        throttle_duration_sec=5.0,
                    )

                # ----------------------------------------------------------
                # Frame capture
                # ----------------------------------------------------------
                ok, frame = self._read_frame()
                frame_location = self.drone.capture_origin()

                if not ok or frame is None:
                    self.node.get_logger().warn(
                        "No frame captured from camera, retrying...",
                        throttle_duration_sec=2.5,
                    )
                    continue

                frame_count += 1

                # ----------------------------------------------------------
                # Detection (classification model)
                # ----------------------------------------------------------
                best_detection, bx_off, by_off, annotated = (
                    self._best_detection_in_frame(
                        frame, self.detector, self.confidence_threshold
                    )
                )

                if best_detection is not None:
                    last_inference_image = annotated
                    time_last_det = time.time()

                # Publish debug image
                self._publish_frame(
                    blackboard,
                    last_inference_image if best_detection is not None else frame,
                )

                num_det = 1 if best_detection is not None else 0
                self.node.get_logger().info(
                    f"Frame {frame_count}: {num_det} detection(s)"
                )

                if best_detection is not None:
                    class_id = best_detection.class_id
                    conf = best_detection.confidence

                    if class_id == last_class_id:
                        consecutive_count += 1
                    else:
                        consecutive_count = 1
                        last_class_id = class_id

                    if conf > best_confidence_overall:
                        best_detection_overall = best_detection
                        best_confidence_overall = conf
                        best_inference_image_overall = last_inference_image

                    self.node.get_logger().info(
                        f"Frame {frame_count}: Class {class_id}, "
                        f"Conf {conf:.3f}, "
                        f"Consecutive {consecutive_count}/{consecutive_limit}"
                    )

                    if consecutive_count >= consecutive_limit:
                        self.node.get_logger().info(
                            f"[GaugeReading] {consecutive_limit} consecutive detections "
                            f"of class {class_id} confirmed — switching to PID centering."
                        )
                        self._stop_event.set()  # halt any ongoing climb
                        break
                else:
                    # Reset streak on any missed frame so a glitch can't
                    # accumulate toward the consecutive threshold.
                    consecutive_count = 0

        finally:
            self._stop_event.set()

        # ----------------------------------------------------------------
        # Phase 2 — PID centering + descent (coarse model)
        # ----------------------------------------------------------------
        # Only enter if we actually found something.
        if best_detection_overall is not None or last_class_id is not None:
            self._stop_event.clear()
            reached_trusted_zone = self._pid_center_and_descend()

            if reached_trusted_zone:
                self.node.get_logger().info(
                    "[GaugeReading] At trusted zone — running final classification."
                )
                # Single clean classification pass at the trusted altitude.
                for _ in range(consecutive_limit):
                    ok, frame = self._read_frame()
                    if not ok or frame is None:
                        continue

                    det, _, _, ann = self._best_detection_in_frame(
                        frame, self.detector, self.confidence_threshold
                    )
                    if det is not None:
                        last_class_id = det.class_id
                        best_detection_overall = det
                        best_inference_image_overall = ann
                        last_inference_image = ann
                        self._publish_frame(blackboard, ann)
            else:
                self.node.get_logger().warn(
                    "[GaugeReading] PID centering + descent failed — "
                    "using best detection so far."
                )

        # ----------------------------------------------------------------
        # Finalise blackboard
        # ----------------------------------------------------------------
        try:
            self.cam.close()
        except Exception:
            pass

        if last_class_id is not None:
            blackboard["gauge_reading"] = last_class_id
            blackboard["inference_image_cv"] = last_inference_image
            return SUCCEED

        if best_detection_overall is not None:
            blackboard["gauge_reading"] = best_detection_overall.class_id
            blackboard["inference_image_cv"] = best_inference_image_overall
            return SUCCEED

        self.node.get_logger().warn(
            "Timeout reached without valid detections. Continuing mission..."
        )
        blackboard["gauge_reading"] = -1
        blackboard["inference_image_cv"] = None
        return SUCCEED