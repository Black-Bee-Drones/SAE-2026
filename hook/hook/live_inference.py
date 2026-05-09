"""Live YOLO-seg inference on the down camera.

Usage:
    ros2 run hook live_inference
    ros2 run hook live_inference --ros-args -p topic:=/hook/inference/compressed
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from nectar.ai.detection import PerClassConfidenceFilter
from nectar.ai.segmentation import Segmentor
from nectar.vision import ImageHandler, OpenCVConfig
from nectar.vision.camera import ROSConfig

from hook.core import overlay
from hook.core.constants import (
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


class LiveInference(Node):
    def __init__(self) -> None:
        super().__init__("hook_live_inference")

        self.declare_parameter("topic", "/hook/inference/compressed")
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("rate_hz", 15.0)

        topic = self.get_parameter("topic").get_parameter_value().string_value
        self._jpeg_quality = int(
            self.get_parameter("jpeg_quality").get_parameter_value().integer_value
        )
        rate_hz = float(self.get_parameter("rate_hz").get_parameter_value().double_value)

        self.pub = self.create_publisher(CompressedImage, topic, 1)

        if SIM_MODE:
            cam_config = ROSConfig(topic=IMAGE_SOURCE, compressed=SIM_IMAGE_COMPRESSED)
        else:
            cam_config = OpenCVConfig(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        self.camera = ImageHandler(node=self, image_source=IMAGE_SOURCE, config=cam_config)
        self.camera.open()

        self.get_logger().info(f"Loading model: {SEG_MODEL_PATH}")
        self.segmentor = Segmentor(SEG_MODEL_PATH, confidence_threshold=SEG_PREDICT_CONF)
        self.segmentor.load()
        self.segmentor.segment(np.zeros((640, 640, 3), dtype=np.uint8))

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
        self.get_logger().info(f"Publishing annotated frames on {topic}")
        self.create_timer(1.0 / max(rate_hz, 1.0), self._tick)

    def _tick(self) -> None:
        try:
            frame, result = run_seg(self.camera, self.segmentor, self.class_filter)
        except Exception as e:
            self.get_logger().error(f"Inference error: {e}", throttle_duration_sec=2.0)
            return
        if frame is None:
            return

        annotated = overlay.annotate_seg(frame, result)
        n_det = len(result.segmentations) if result is not None else 0
        infer_ms = (result.inference_time * 1000.0) if result is not None else 0.0
        cv2.putText(
            annotated,
            f"det={n_det} infer={infer_ms:.1f}ms",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        ok, buf = cv2.imencode(
            ".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        )
        if not ok:
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "down_camera"
        msg.format = "jpeg"
        msg.data = buf.tobytes()
        self.pub.publish(msg)

        self._frames += 1
        now = time.time()
        if now - self._t_window >= 2.0:
            fps = self._frames / (now - self._t_window)
            self.get_logger().info(
                f"fps={fps:.1f} det={n_det} infer={infer_ms:.1f}ms"
            )
            self._frames = 0
            self._t_window = now

    def destroy_node(self) -> bool:
        try:
            self.camera.cleanup()
        except Exception:
            pass
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = LiveInference()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
