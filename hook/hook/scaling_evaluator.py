"""Live evaluator for the three :class:`CameraScaling` implementations.

Hover the drone over a known target (sphere on the hose at 1.7 m, marker
on the floor, etc.), launch this node, and watch the published topic in
``rqt_image_view``. For every detection the overlay shows the body-frame
``(body_x, body_y)`` estimated by all three methods so you can read which
one matches the tape-measured ground truth.

Usage::

    ros2 run hook scaling_evaluator
    ros2 run hook scaling_evaluator --ros-args -p target_height_m:=1.7
    ros2 run rqt_image_view rqt_image_view /hook/scaling_eval/compressed
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
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


_METHOD_COLORS = {
    "fov":       (255, 255,   0),  # cyan  (BGR)
    "efl":       (255,   0, 255),  # magenta
    "intrinsic": (  0, 255, 255),  # yellow
}
_HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
_DEFAULT_RANGEFINDER_TOPIC = "/mavros/rangefinder/rangefinder"
_DEFAULT_OUTPUT_TOPIC = "/hook/scaling_eval/compressed"


class ScalingEvaluator(Node):
    def __init__(self) -> None:
        super().__init__("hook_scaling_evaluator")

        self.declare_parameter("topic", _DEFAULT_OUTPUT_TOPIC)
        self.declare_parameter("rangefinder_topic", _DEFAULT_RANGEFINDER_TOPIC)
        self.declare_parameter("target_height_m", 0.0)
        self.declare_parameter("rate_hz", 15.0)
        self.declare_parameter("jpeg_quality", 80)

        topic = self.get_parameter("topic").get_parameter_value().string_value
        rangefinder_topic = (
            self.get_parameter("rangefinder_topic").get_parameter_value().string_value
        )
        self._target_height_m = float(
            self.get_parameter("target_height_m").get_parameter_value().double_value
        )
        rate_hz = float(
            self.get_parameter("rate_hz").get_parameter_value().double_value
        )
        self._jpeg_quality = int(
            self.get_parameter("jpeg_quality").get_parameter_value().integer_value
        )

        self._scalings: List[CameraScaling] = make_all_camera_scalings()
        self._altitude: Optional[float] = None

        self._pub = self.create_publisher(CompressedImage, topic, 1)
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._range_sub = self.create_subscription(
            Range, rangefinder_topic, self._on_range, sensor_qos
        )

        if SIM_MODE:
            cam_config = ROSConfig(topic=IMAGE_SOURCE, compressed=SIM_IMAGE_COMPRESSED)
        else:
            cam_config = OpenCVConfig(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        self.camera = ImageHandler(node=self, image_source=IMAGE_SOURCE, config=cam_config)
        self.camera.open()

        self.get_logger().info(f"Loading model: {SEG_MODEL_PATH}")
        self.segmentor = Segmentor(SEG_MODEL_PATH, confidence_threshold=SEG_PREDICT_CONF)
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

        self._frames = 0
        self._t_window = time.time()
        self.get_logger().info(
            f"Publishing scaling overlay on {topic}; rangefinder={rangefinder_topic}; "
            f"target_height_m={self._target_height_m:.2f}m; "
            f"active mission method={CAMERA_SCALING_METHOD!r}"
        )
        self.create_timer(1.0 / max(rate_hz, 1.0), self._tick)

    def _on_range(self, msg: Range) -> None:
        if math_isfinite(msg.range):
            self._altitude = float(msg.range)

    def _tick(self) -> None:
        try:
            frame, result = run_seg(self.camera, self.segmentor, self.class_filter)
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}", throttle_duration_sec=2.0)
            return
        if frame is None:
            return

        annotated = overlay.annotate_seg(frame, result)
        detections = list(result.segmentations) if result is not None else []
        n_det = len(detections)
        infer_ms = (result.inference_time * 1000.0) if result is not None else 0.0
        altitude = self._altitude

        self._draw_hud(annotated, altitude, n_det, infer_ms)
        for det in detections:
            self._draw_detection_estimates(annotated, det, altitude)

        self._publish(annotated)

        self._frames += 1
        now = time.time()
        if now - self._t_window >= 2.0:
            fps = self._frames / (now - self._t_window)
            alt_str = f"{altitude:.2f}m" if altitude is not None else "n/a"
            self.get_logger().info(
                f"fps={fps:.1f} det={n_det} infer={infer_ms:.1f}ms alt={alt_str}"
            )
            self._frames = 0
            self._t_window = now

    def _draw_hud(
        self, img: np.ndarray, altitude: Optional[float], n_det: int, infer_ms: float
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
        line_colors: List[Tuple[int, int, int]] = [(255, 255, 255)]

        for s in self._scalings:
            color = _METHOD_COLORS.get(s.name, (255, 255, 255))
            text = self._estimate_line(s, cx, cy, altitude)
            lines.append(text)
            line_colors.append(color)

        _draw_label_box(img, (int(cx), int(cy)), lines, line_colors)

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
        self._pub.publish(msg)

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
            img, line, (x + pad, ty), _HUD_FONT, font_scale, color, thickness, cv2.LINE_AA
        )


def math_isfinite(x: float) -> bool:
    return x == x and x not in (float("inf"), float("-inf"))


def main() -> None:
    rclpy.init()
    node = ScalingEvaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
