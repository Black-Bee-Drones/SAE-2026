from operator import index
import cv2
import time
from yasmin import Blackboard, State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.ai.detection.models.ultralytics import UltralyticsModel
from zaxis.drone import Drone
from sensor_msgs.msg import CompressedImage

import os

MANOMETER_HEIGHT_M = 1.7
APPROACH_ALTITUDE_M = MANOMETER_HEIGHT_M + 1.0

PID_KP_BASE = 0.5
PID_KP_BBOX_SCALE = 0.1
PID_KP_MIN = 0.4
PID_KP_MAX = 1.2
PID_MAX_VEL = 0.2
PID_KP_Z = 0.15

CENTERED_THRESHOLD_BASE = 0.05
CENTERED_THRESHOLD_MAX = 0.08
CENTERED_THRESHOLD_MIN = 0.25

CONSECUTIVE_LIMIT = 3
MAX_DURATION_S = 15.0

class GaugeReading(State):

    def __init__(
        self,
        model_path: str,
        coarse_model_path: str,
        confidence_threshold: float = 0.4,
        coarse_confidence_threshold: float = 0.4,
    ):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.node = YasminNode.get_instance()
        self.confidence_threshold = confidence_threshold
        self.coarse_confidence_threshold = coarse_confidence_threshold

        self.detector = self._load_model(model_path, "classification")
        self.coarse_detector = self._load_model(coarse_model_path, "coarse")

        self.drone: Drone = None

    def _load_model(self, path: str, label: str):
        try:
            m = UltralyticsModel(model_name=path)
            m.load_model()
            return m
        except Exception as e:
            self.node.get_logger().error(f"Error loading {label} model: {e}")
            return None

    def _read_frame(self, blackboard: Blackboard):
        if blackboard["simulation"]:
            frame = self.cam.frame
            return frame is not None, frame
        return self.cam.read()

    def _publish_frame(self, blackboard: Blackboard, frame):
        save_dir = os.path.join(os.path.expanduser("~"), "faulty_inferences")
        os.makedirs(save_dir, exist_ok=True)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        msg = CompressedImage()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.format = "jpeg"
        msg.data = buffer.tobytes()
        blackboard["inference_image_publisher"].publish(msg)

    def _get_square_crops(self, frame, overlap=0.3):
        h, w = frame.shape[:2]
        size = min(h, w)
        step = int(size * (1 - overlap))
        crops = []
        if w >= h:
            x = 0
            while x + size <= w:
                crops.append((frame[0:size, x:x + size], x, 0))
                x += step
            if not crops or crops[-1][1] + size < w:
                crops.append((frame[0:size, w - size:w], w - size, 0))
        else:
            y = 0
            while y + size <= h:
                crops.append((frame[y:y + size, 0:size], 0, y))
                y += step
            if not crops or crops[-1][2] + size < h:
                crops.append((frame[h - size:h, 0:size], 0, h - size))
        return crops

    def _best_detection_in_frame(self, frame, model, threshold):
        best_det, best_conf, best_x_off, best_y_off, best_ann = None, 0.0, 0, 0, None
        for crop, x_off, y_off in self._get_square_crops(frame):
            result = model.detect(crop, conf=threshold)
            detections = result if isinstance(result, list) else result.detections
            for det in detections:
                if det.confidence > best_conf:
                    best_det, best_conf = det, det.confidence
                    best_x_off, best_y_off = x_off, y_off
                    best_ann = model.draw_detections(crop, result)
        return best_det, best_x_off, best_y_off, best_ann

    def _adaptive_kp(self, frame_w, frame_h, x1, y1, x2, y2):
        bbox_area_norm = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)
        kp = PID_KP_BASE * (PID_KP_BBOX_SCALE / max(bbox_area_norm, 1e-4))
        return max(PID_KP_MIN, min(PID_KP_MAX, kp)), bbox_area_norm

    def _adaptive_threshold(self, frame_w, frame_h, x1, y1, x2, y2) -> float:
        bbox_area_norm = ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)
        threshold = CENTERED_THRESHOLD_BASE * (PID_KP_BBOX_SCALE / max(bbox_area_norm, 1e-4))
        return max(CENTERED_THRESHOLD_MIN, min(CENTERED_THRESHOLD_MAX, threshold))

    def _pid_step(self, error: float, kp: float) -> float:
        vel = kp * error
        return max(-PID_MAX_VEL, min(PID_MAX_VEL, vel))

    def _center_on_manometer(self, blackboard: Blackboard) -> bool:
        timeout = 60.0

        self.node.get_logger().info("[GaugeReading] Phase 1: centering XY...")
        start = time.time()

        while (time.time() - start) < timeout:
            ok, frame = self._read_frame(blackboard=blackboard)
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            det, x_off, y_off, ann = self._best_detection_in_frame(
                frame, self.coarse_detector, self.coarse_confidence_threshold
            )
            self._publish_frame(blackboard, ann if det is not None else frame)

            if det is None:
                self.node.get_logger().warn("[DATA_LOG] Phase 1 | NO DETECTION - Hovering")
                self.drone.set_velocity_body(0.0, 0.0, 0.0)
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = det.xyxy
            cx = ((x1 + x2) / 2.0 + x_off - w / 2.0) / w
            cy = ((y1 + y2) / 2.0 + y_off - h / 2.0) / h

            kp, area = self._adaptive_kp(w, h, x1, y1, x2, y2)
            threshold = self._adaptive_threshold(w, h, x1, y1, x2, y2)

            vx = self._pid_step(-cy, kp)
            vy = self._pid_step(cx, kp)
            
            self.node.get_logger().info(
                f"[DATA_LOG] Phase 1 | Area: {area:.6f} | cx: {cx:.3f} | cy: {cy:.3f} | "
                f"KP: {kp:.3f} | Thr: {threshold:.3f} | vx: {vx:.3f} | vy: {vy:.3f}"
            )
            
            self.drone.set_velocity_body(vx, vy, 0.0)

            if abs(cx) < threshold and abs(cy) < threshold:
                self.drone.set_velocity_body(0.0, 0.0, 0.0)
                self.node.get_logger().info("[GaugeReading] XY centered. Starting altitude correction...")
                break
        else:
            return False

        self.node.get_logger().info("[GaugeReading] Phase 2: correcting altitude + XY...")
        start = time.time()

        while (time.time() - start) < timeout:
            ok, frame = self._read_frame(blackboard=blackboard)
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            alt_error = abs(self.drone.distance_sensor.current_distance) - APPROACH_ALTITUDE_M

            det, x_off, y_off, ann = self._best_detection_in_frame(
                frame, self.coarse_detector, self.coarse_confidence_threshold
            )
            self._publish_frame(blackboard, ann if det is not None else frame)

            if det is None:
                vz = self._pid_step(alt_error, PID_KP_Z)
                self.node.get_logger().warn(
                    f"[DATA_LOG] Phase 2 | NO DETECTION | alt_err: {alt_error:.3f} | vz: {vz:.3f}"
                )
                self.drone.set_velocity_body(0.0, 0.0, vz)
                time.sleep(0.1)
                continue

            h, w = frame.shape[:2]
            x1, y1, x2, y2 = det.xyxy
            cx = ((x1 + x2) / 2.0 + x_off - w / 2.0) / w
            cy = ((y1 + y2) / 2.0 + y_off - h / 2.0) / h

            kp, area = self._adaptive_kp(w, h, x1, y1, x2, y2)
            threshold = self._adaptive_threshold(w, h, x1, y1, x2, y2)

            vx = self._pid_step(-cy, kp)
            vy = self._pid_step(cx, kp)
            vz = self._pid_step(alt_error, PID_KP_Z)
            
            self.node.get_logger().info(
                f"[DATA_LOG] Phase 2 | Area: {area:.6f} | alt_err: {alt_error:.3f} | cx: {cx:.3f} | cy: {cy:.3f} | "
                f"KP: {kp:.3f} | Thr: {threshold:.3f} | vx: {vx:.3f} | vy: {vy:.3f} | vz: {vz:.3f}"
            )
            
            self.drone.set_velocity_body(vx, vy, vz)

            if abs(cx) < threshold and abs(cy) < threshold and abs(alt_error) < 0.1:
                self.drone.set_velocity_body(0.0, 0.0, 0.0)
                return True

        return False

    def execute(self, blackboard: Blackboard):

        if self.detector is None or self.coarse_detector is None:
            self.node.get_logger().error("One or more models failed to load.")
            return ABORT

        if "inference_image_publisher" not in blackboard:
            blackboard["inference_image_publisher"] = self.node.create_publisher(
                CompressedImage, "/gauge/compressed", 10
            )

        if "drone" in blackboard:
            self.drone = blackboard["drone"]

        self.drone._set_param("WP_YAW_BEHAVIOR", 0).wait(timeout=3.0)

        self.cam = blackboard["cam"]
        if self.cam is None:
            index = 0
            try:
                os.system(f"v4l2-ctl -d /dev/video{index} -c auto_exposure=1")
                time.sleep(0.5)
                os.system(f"v4l2-ctl -d /dev/video{index} -c backlight_compensation=0")
                os.system(f"v4l2-ctl -d /dev/video{index} -c exposure_dynamic_framerate=0")
                os.system(f"v4l2-ctl -d /dev/video{index} -c exposure_time_absolute=1")
                os.system(f"v4l2-ctl -d /dev/video{index} --set-fmt-video=width=640,height=480,pixelformat=YUYV")
            except:
                pass
            
            self.cam = cv2.VideoCapture(index, cv2.CAP_V4L2)

        self.node.get_logger().info("[GaugeReading] Centering on manometer...")
        centered = self._center_on_manometer(blackboard=blackboard)

        if not centered:
            self.node.get_logger().warn("[GaugeReading] Failed to center on manometer, proceeding anyway.")

        self.node.get_logger().info("[GaugeReading] Starting classification...")

        start_time = time.time()
        consecutive_count = 0
        last_class_id = None
        best_detection_overall = None
        best_confidence_overall = 0.0
        best_ann_overall = None
        last_ann = None
        frame_count = 0

        while (time.time() - start_time) < MAX_DURATION_S:
            ok, frame = self._read_frame(blackboard=blackboard)
            if not ok or frame is None:
                continue

            frame_count += 1

            # !! INÍCIO DAS ALTERAÇÕES !!
            det, _, _, ann = self._best_detection_in_frame(
                frame, self.detector, self.confidence_threshold
            )

            self._publish_frame(blackboard, ann if det is not None else frame)

            if det is None:
                self.node.get_logger().info("[GaugeReading] No detection found.")
                self.drone.set_velocity_body(0.0, 0.0, 0.0)
                continue

            alt_error = abs(self.drone.distance_sensor.current_distance) - APPROACH_ALTITUDE_M

            vz = self._pid_step(alt_error, PID_KP_Z)

            self.drone.set_velocity_body(0, 0, vz)

            self.node.get_logger().info(
                f"[DATA_LOG] Classification | Frame {frame_count} | "
                f"alt_err: {alt_error:.3f} | "
                f"vz: {vz:.3f}"
            )
            # !! FIM DAS ALTERAÇÕES !!

            last_ann = ann
            class_id = det.class_id
            conf = det.confidence

            if class_id == last_class_id:
                consecutive_count += 1
            else:
                consecutive_count = 1
                last_class_id = class_id

            if conf > best_confidence_overall:
                best_detection_overall = det
                best_confidence_overall = conf
                best_ann_overall = ann

            self.node.get_logger().info(
                f"[GaugeReading] Frame {frame_count}: class={class_id} "
                f"conf={conf:.3f} consecutive={consecutive_count}/{CONSECUTIVE_LIMIT}"
            )

            if consecutive_count >= CONSECUTIVE_LIMIT:
                break

        try:
            self.cam.close()
        except Exception:
            pass

        if last_class_id is not None:
            blackboard["gauge_reading"] = last_class_id
            blackboard["inference_image_cv"] = last_ann
            return SUCCEED

        if best_detection_overall is not None:
            blackboard["gauge_reading"] = best_detection_overall.class_id
            blackboard["inference_image_cv"] = best_ann_overall
            return SUCCEED

        self.node.get_logger().warn("[GaugeReading] No valid detection. Continuing mission...")
        blackboard["gauge_reading"] = -1
        blackboard["inference_image_cv"] = None
        return SUCCEED