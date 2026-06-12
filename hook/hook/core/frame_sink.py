"""Annotated mission-frame output: disk save + live ROS publish.

Two collaborators:

- :class:`FramePublisher` — owns a single ``sensor_msgs/CompressedImage``
  publisher. JPEG-encodes lazily (only when at least one subscriber is
  connected), so the live feed is free when nobody's listening.
- :class:`FrameSink` — per-state output handle. ``emit(frame)`` does both
  ``cv2.imwrite`` (if a save root was provided) and
  ``publisher.publish`` (if a publisher was provided). One instance per
  state, the publisher is shared across all of them.

Together they replace the duplicated
``self.save_dir`` / ``self.frame_count`` / ``cv2.imwrite`` boilerplate
that used to live in every state, and add live-monitor publishing
without each state having to know anything about ROS topics.

Image-frame convention is unchanged from the rest of the package
(down camera, FLU body); the sink doesn't care — it just writes/publishes
whatever annotated frame it's handed.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CompressedImage

from hook.core.constants import (
    DETECTION_SAVE_PATH,
    MISSION_FRAME_JPEG_QUALITY,
    MISSION_FRAME_TOPIC,
    SAVE_DETECTIONS,
)


class FramePublisher:
    """Publish annotated mission frames on a single compressed-image topic.

    QoS is best-effort / depth=1 — the live feed is for monitoring, not
    a reliable record (the disk save is). When no subscriber is
    connected, :meth:`publish` short-circuits before the JPEG encode so
    the publisher costs near zero CPU during normal flight.
    """

    def __init__(
        self,
        node: Node,
        topic: str = MISSION_FRAME_TOPIC,
        jpeg_quality: int = MISSION_FRAME_JPEG_QUALITY,
        frame_id: str = "down_camera",
    ) -> None:
        self._node = node
        self._jpeg_quality = int(jpeg_quality)
        self._frame_id = frame_id
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._pub = node.create_publisher(CompressedImage, topic, qos)
        self._topic = topic

    @property
    def topic(self) -> str:
        return self._topic

    def has_subscribers(self) -> bool:
        return self._pub.get_subscription_count() > 0

    def publish(self, frame: np.ndarray) -> None:
        """Encode + publish if any subscriber is listening; otherwise no-op."""
        if frame is None or not self.has_subscribers():
            return
        ok, buf = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        )
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.format = "jpeg"
        msg.data = buf.tobytes()
        self._pub.publish(msg)


class FrameSink:
    """Per-state sink: save annotated frames to disk and / or publish live.

    Each state owns one ``FrameSink`` configured with its own ``prefix``
    (used as the subdirectory name and the file-name stem). The
    ``publisher`` is shared across all states so subscribers see one
    coherent feed across the whole mission.

    Either output is independently optional: pass ``save_root=None`` to
    disable disk writes, or ``publisher=None`` to disable publishing.
    """

    def __init__(
        self,
        *,
        prefix: str,
        save_root: Optional[Path] = None,
        publisher: Optional[FramePublisher] = None,
    ) -> None:
        self._prefix = prefix
        self._publisher = publisher
        self._counter = 0
        if save_root is not None:
            self._save_dir: Optional[Path] = Path(save_root) / prefix
            self._save_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._save_dir = None

    @property
    def save_dir(self) -> Optional[Path]:
        return self._save_dir

    @property
    def prefix(self) -> str:
        return self._prefix

    def emit(self, frame: Optional[np.ndarray]) -> None:
        """Save (if save_root given) and publish (if publisher given)."""
        if frame is None:
            return
        self._counter += 1
        if self._save_dir is not None:
            cv2.imwrite(
                str(self._save_dir / f"{self._prefix}_{self._counter:04d}.jpg"),
                frame,
            )
        if self._publisher is not None:
            self._publisher.publish(frame)


def build_state_sink(blackboard, prefix: str) -> FrameSink:
    """Factory: build a :class:`FrameSink` for a state.

    Reads the (optional) ``frame_publisher`` and ``mission_timestamp``
    from the blackboard. If ``mission_timestamp`` is missing it is
    initialized to ``now`` and written back so subsequent states share
    the same run directory. Disk save is gated on ``SAVE_DETECTIONS``.
    """
    save_root: Optional[Path] = None
    if SAVE_DETECTIONS:
        if "mission_timestamp" in blackboard:
            ts = blackboard["mission_timestamp"]
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            blackboard["mission_timestamp"] = ts
        save_root = Path(DETECTION_SAVE_PATH) / ts

    publisher = (
        blackboard["frame_publisher"]
        if "frame_publisher" in blackboard
        else None
    )
    return FrameSink(prefix=prefix, save_root=save_root, publisher=publisher)
