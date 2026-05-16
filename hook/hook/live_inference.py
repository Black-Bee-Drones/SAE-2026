"""Live YOLO-seg inference on the down camera.

Default mode publishes annotated frames on ``/hook/inference/compressed``.
Enable ``scaling_eval`` to also overlay per-detection body-frame estimates
from each :class:`CameraScaling` implementation (FOV / EFL / intrinsic),
driven by the rangefinder. ``scaling_evaluator`` is an alias entry-point
that flips that default to ``True``.

Usage::

    ros2 run hook live_inference
    ros2 run hook live_inference --ros-args -p camera_device:=2
    ros2 run hook live_inference --ros-args -p scaling_eval:=true -p target_height_m:=1.7
    ros2 run hook scaling_evaluator                                            # same as above
    ros2 run rqt_image_view rqt_image_view /hook/inference/compressed
    ros2 run rqt_image_view rqt_image_view /hook/scaling_eval/compressed       # scaling_eval=true
"""

from __future__ import annotations

import math
import time
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage, Range

from nectar.ai.detection import PerClassConfidenceFilter
from nectar.ai.segmentation import Segmentor
from nectar.vision import ImageHandler, OpenCVConfig
from nectar.vision.camera import ROSConfig

from hook.core import overlay
from hook.core.camera_scaling import (
    CameraScaling,
    image_center_for,
    make_all_camera_scalings,
)
from hook.core.constants import (
    CAMERA_SCALING_METHOD,
    HOSE_CLASS,
    HOSE_CONF,
    IMAGE_HEIGHT,
    IMAGE_SOURCE,
    IMAGE_WIDTH,
    SEG_MODEL_PATH,
    SEG_PREDICT_CONF,
    SIM_IMAGE_COMPRESSED,
    SIM_MODE,
    SPHERE_CLASS,
    SPHERE_CONF,
)
from hook.core.perception import run_seg


_DEFAULT_INFERENCE_TOPIC = "/hook/inference/compressed"
_DEFAULT_SCALING_TOPIC = "/hook/scaling_eval/compressed"
_DEFAULT_RANGEFINDER_TOPIC = "/mavros/rangefinder/rangefinder"
_METHOD_COLORS = {
    "fov":       (255, 255,   0),  # cyan  (BGR)
    "efl":       (255,   0, 255),  # magenta
    "intrinsic": (  0, 255, 255),  # yellow
}
_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX


class LiveInference(Node):
    """Publish annotated camera frames; optionally overlay scaling estimates."""

    def __init__(self, scaling_eval_default: bool = False) -> None:
        super().__init__("hook_live_inference")

        self.declare_parameter("scaling_eval", scaling_eval_default)
        self._scaling_eval = bool(
            self.get_parameter("scaling_eval").get_parameter_value().bool_value
        )

        default_topic = (
            _DEFAULT_SCALING_TOPIC if self._scaling_eval else _DEFAULT_INFERENCE_TOPIC
        )
        self.declare_parameter("topic", default_topic)
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("rate_hz", 15.0)
        self.declare_parameter("camera_device", 0)
        self.declare_parameter("rangefinder_topic", _DEFAULT_RANGEFINDER_TOPIC)
        self.declare_parameter("target_height_m", 0.0)

        topic = self.get_parameter("topic").get_parameter_value().string_value
        self._jpeg_quality = int(
            self.get_parameter("jpeg_quality").get_parameter_value().integer_value
        )
        rate_hz = float(
            self.get_parameter("rate_hz").get_parameter_value().double_value
        )
        camera_device = int(
            self.get_parameter("camera_device").get_parameter_value().integer_value
        )
        rangefinder_topic = (
            self.get_parameter("rangefinder_topic")
            .get_parameter_value()
            .string_value
        )
        self._target_height_m = float(
            self.get_parameter("target_height_m").get_parameter_value().double_value
        )

        self.pub = self.create_publisher(CompressedImage, topic, 1)

        if SIM_MODE:
            cam_config = ROSConfig(topic=IMAGE_SOURCE, compressed=SIM_IMAGE_COMPRESSED)
        else:
            cam_config = OpenCVConfig(
                device_index=camera_device,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
            )
        self.camera = ImageHandler(
            node=self, image_source=IMAGE_SOURCE, config=cam_config
        )
        self.camera.open()

        self.get_logger().info(f"Loading model: {SEG_MODEL_PATH}")
        self.segmentor = Segmentor(
            SEG_MODEL_PATH, confidence_threshold=SEG_PREDICT_CONF
        )
        self.segmentor.load()
        self.segmentor.segment(np.zeros((960, 960, 3), dtype=np.uint8))

        name_to_id = {v: k for k, v in self.segmentor.class_names.items()}
        if SPHERE_CLASS not in name_to_id or HOSE_CLASS not in name_to_id:
            raise RuntimeError(
                f"Model classes mismatch: have {list(name_to_id)}, "
                f"need {SPHERE_CLASS!r} and {HOSE_CLASS!r}"
            )
        self.class_filter = PerClassConfidenceFilter(
            threshold_mapping={
                name_to_id[SPHERE_CLASS]: SPHERE_CONF,
                name_to_id[HOSE_CLASS]: HOSE_CONF,
            },
            default_threshold=1.1,
        )

        self._scalings: List[CameraScaling] = []
        self._altitude: Optional[float] = None
        if self._scaling_eval:
            self._scalings = make_all_camera_scalings()
            sensor_qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                durability=QoSDurabilityPolicy.VOLATILE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
            )
            self._range_sub = self.create_subscription(
                Range, rangefinder_topic, self._on_range, sensor_qos
            )
            self.get_logger().info(
                f"Scaling eval ON: rangefinder={rangefinder_topic}; "
                f"target_height_m={self._target_height_m:.2f}m; "
                f"active mission method={CAMERA_SCALING_METHOD!r}"
            )

        self._frames = 0
        self._t_window = time.time()
        self.get_logger().info(
            f"Publishing annotated frames on {topic} (device={camera_device})"
        )
        self.create_timer(1.0 / max(rate_hz, 1.0), self._tick)

    def _on_range(self, msg: Range) -> None:
        if math.isfinite(msg.range):
            self._altitude = float(msg.range)

    def _tick(self) -> None:
        try:
            frame, result = run_seg(self.camera, self.segmentor, self.class_filter)
        except Exception as e:
            self.get_logger().error(
                f"Inference error: {e}", throttle_duration_sec=2.0
            )
            return
        if frame is None:
            return

        annotated = overlay.annotate_seg(frame, result)
        detections = list(result.segmentations) if result is not None else []
        n_det = len(detections)
        infer_ms = (result.inference_time * 1000.0) if result is not None else 0.0

        if self._scaling_eval:
            self._draw_scaling_hud(annotated, self._altitude, n_det, infer_ms)
            for det in detections:
                self._draw_detection_estimates(annotated, det, self._altitude)
        else:
            cv2.putText(
                annotated,
                f"det={n_det} infer={infer_ms:.1f}ms",
                (12, 28),
                _HUD_FONT,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        self._publish(annotated)

        self._frames += 1
        now = time.time()
        if now - self._t_window >= 2.0:
            fps = self._frames / (now - self._t_window)
            extra = ""
            if self._scaling_eval:
                alt_str = (
                    f"{self._altitude:.2f}m" if self._altitude is not None else "n/a"
                )
                extra = f" alt={alt_str}"
            self.get_logger().info(
                f"fps={fps:.1f} det={n_det} infer={infer_ms:.1f}ms{extra}"
            )
            self._frames = 0
            self._t_window = now

    def _publish(self, img: np.ndarray) -> None:
        ok, buf = cv2.imencode(
            ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        )
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "down_camera"
        msg.format = "jpeg"
        msg.data = buf.tobytes()
        self.pub.publish(msg)

    def _draw_scaling_hud(
        self,
        img: np.ndarray,
        altitude: Optional[float],
        n_det: int,
        infer_ms: float,
    ) -> None:
        alt_str = f"{altitude:.2f}m" if altitude is not None else "n/a"
        depth_str = (
            f"{altitude - self._target_height_m:.2f}m"
            if altitude is not None
            else "n/a"
        )
        lines = [
            f"SCALING EVALUATOR  active_mission={CAMERA_SCALING_METHOD}",
            f"alt={alt_str}  target_h={self._target_height_m:.2f}m  d={depth_str}",
            f"det={n_det}  infer={infer_ms:.1f}ms",
            "method     ppm_x   ppm_y",
        ]
        for s in self._scalings:
            ppx = s.ppm_x(altitude, self._target_height_m)
            ppy = s.ppm_y(altitude, self._target_height_m)
            lines.append(f"  {s.name:<9s} {ppx:6.1f}  {ppy:6.1f}")
        overlay.draw_hud(img, lines, anchor="tl")

    def _draw_detection_estimates(
        self,
        img: np.ndarray,
        det,
        altitude: Optional[float],
    ) -> None:
        cx, cy = det.center
        cv2.circle(img, (int(cx), int(cy)), 6, (255, 255, 255), 2, cv2.LINE_AA)
        lines = [f"{det.class_name} {det.confidence:.2f}"]
        colors: List[Tuple[int, int, int]] = [(255, 255, 255)]
        for s in self._scalings:
            colors.append(_METHOD_COLORS.get(s.name, (255, 255, 255)))
            lines.append(self._estimate_line(s, cx, cy, altitude))
        _draw_label_box(img, (int(cx), int(cy)), lines, colors)

    def _estimate_line(
        self,
        scaling: CameraScaling,
        cx: float,
        cy: float,
        altitude: Optional[float],
    ) -> str:
        if altitude is None:
            return f"{scaling.name:<9s} alt n/a"
        ppm_x = scaling.ppm_x(altitude, self._target_height_m)
        ppm_y = scaling.ppm_y(altitude, self._target_height_m)
        if ppm_x <= 0.0 or ppm_y <= 0.0:
            return f"{scaling.name:<9s} depth<=0"
        pp_x, pp_y = image_center_for(scaling)
        dx = cx - pp_x
        dy = cy - pp_y
        bx, by = scaling.image_offset_to_body(
            dx, dy, altitude, self._target_height_m
        )
        return (
            f"{scaling.name:<9s} body=({bx:+.2f},{by:+.2f})m  "
            f"px=({dx:+5.0f},{dy:+5.0f})"
        )

    def destroy_node(self) -> bool:
        try:
            self.camera.cleanup()
        except Exception:
            pass
        return super().destroy_node()


def _draw_label_box(
    img: np.ndarray,
    anchor_xy: Tuple[int, int],
    lines: Iterable[str],
    colors: Iterable[Tuple[int, int, int]],
    font_scale: float = 0.55,
    thickness: int = 1,
    pad: int = 6,
    line_height: int = 22,
) -> None:
    lines = list(lines)
    colors = list(colors)
    if not lines:
        return
    sizes = [
        cv2.getTextSize(line, _HUD_FONT, font_scale, thickness)[0] for line in lines
    ]
    box_w = max(s[0] for s in sizes) + 2 * pad
    box_h = line_height * len(lines) + 2 * pad

    H, W = img.shape[:2]
    x = anchor_xy[0] + 14
    y = anchor_xy[1] - box_h // 2
    if x + box_w > W:
        x = anchor_xy[0] - box_w - 14
    x = max(0, min(W - box_w, x))
    y = max(0, min(H - box_h, y))

    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (0, 0, 0), -1)
    cv2.rectangle(img, (x, y), (x + box_w, y + box_h), (200, 200, 200), 1)
    for i, (line, color) in enumerate(zip(lines, colors)):
        ty = y + pad + (i + 1) * line_height - 6
        cv2.putText(
            img,
            line,
            (x + pad, ty),
            _HUD_FONT,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _run(scaling_eval_default: bool = False) -> None:
    rclpy.init()
    node = LiveInference(scaling_eval_default=scaling_eval_default)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main() -> None:
    _run(scaling_eval_default=False)


def main_scaling() -> None:
    _run(scaling_eval_default=True)


if __name__ == "__main__":
    main()
