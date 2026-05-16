"""Decide which side of the sphere the drone should fly along.

Decision-only state: no movement, no reacquire. After this state both
``hose_side_image_unit`` and ``anchor_sign`` are on the blackboard, then
:class:`OrientToHook` may rotate the drone (refreshing both values) and
:class:`LowerAndAlign` drives the drone to the stand-off pose and on
through the LIDAR descent.

Algorithm:
  1. Sample SIDE_SAMPLE_FRAMES frames; on the first usable frame, lock
     the rope long-axis from the most-confident hose -> ``u_axis``.
  2. For every hose instance, project its centroid offset from the
     sphere onto ``u_axis``. Sign = which side of the sphere.
  3. Aggregate total long-axis length per side.
  4. Pick the side with greater length. If long/short < SIDE_LENGTH_RATIO,
     fall back to the sign anti-aligned with ``approach_image_offset``.
"""

import time
from typing import Optional, Tuple

import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.ai.detection import PerClassConfidenceFilter
from nectar.ai.segmentation import Segmentor
from nectar.control import MavrosDrone
from nectar.vision import ImageHandler

from hook.core import overlay
from hook.core.constants import (
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    SIDE_LENGTH_RATIO,
    SIDE_SAMPLE_FRAMES,
    SIDE_TIMEOUT,
)
from hook.core.frame_sink import FrameSink, build_state_sink
from hook.core.perception import (
    anchor_sign_for_side,
    best_sphere,
    hose_pose,
    hose_segments,
    run_seg,
)


class SelectHoseSide(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.sink: FrameSink = None

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]
        camera: ImageHandler = blackboard["camera"]
        segmentor: Segmentor = blackboard["segmentor"]
        class_filter: PerClassConfidenceFilter = blackboard["class_filter"]
        approach_offset = (
            blackboard["approach_image_offset"]
            if "approach_image_offset" in blackboard
            else None
        )

        self.sink = build_state_sink(blackboard, "select_side")

        side_unit = self._decide_side(
            drone, camera, segmentor, class_filter, approach_offset
        )
        if side_unit is None:
            yasmin.YASMIN_LOG_ERROR("Could not decide hose side: no usable samples.")
            return ABORT

        anchor_sign = anchor_sign_for_side(side_unit)
        blackboard["hose_side_image_unit"] = side_unit
        blackboard["anchor_sign"] = anchor_sign
        yasmin.YASMIN_LOG_INFO(
            f"Side decided: image-unit=({side_unit[0]:+.2f},{side_unit[1]:+.2f}) "
            f"anchor_sign={anchor_sign:+d}"
        )
        return SUCCEED

    def _decide_side(
        self,
        drone: MavrosDrone,
        camera,
        segmentor,
        class_filter,
        approach_offset: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        u_axis: Optional[Tuple[float, float]] = None
        length_plus = 0.0
        length_minus = 0.0
        collected = 0
        start = time.time()

        while collected < SIDE_SAMPLE_FRAMES and time.time() - start < SIDE_TIMEOUT:
            drone.delay(0.05)
            frame, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)
            hoses = hose_segments(result)
            if sphere is None or not hoses:
                continue

            if u_axis is None:
                pose = hose_pose(hoses[0])
                if pose is None:
                    continue
                u_axis = pose[4]

            sx, sy = sphere.center
            for h in hoses:
                pose = hose_pose(h)
                if pose is None:
                    continue
                hx, hy, _, length, _ = pose
                proj = (hx - sx) * u_axis[0] + (hy - sy) * u_axis[1]
                if proj >= 0:
                    length_plus += length
                else:
                    length_minus += length

            collected += 1
            plus_is_long = length_plus >= length_minus
            sign = 1.0 if plus_is_long else -1.0
            running_side = (sign * u_axis[0], sign * u_axis[1])
            self._save_frame(
                frame, result,
                sphere_center=sphere.center,
                u_axis=u_axis,
                side_unit=running_side,
                length_plus=length_plus,
                length_minus=length_minus,
                plus_is_long=plus_is_long,
                sample_idx=collected,
            )

        if u_axis is None:
            return None

        long_len = max(length_plus, length_minus)
        short_len = min(length_plus, length_minus)
        plus_is_long = length_plus >= length_minus
        ambiguous = short_len > 0 and (long_len / short_len) < SIDE_LENGTH_RATIO

        if ambiguous and approach_offset is not None:
            ax, ay = approach_offset
            dot_plus = u_axis[0] * ax + u_axis[1] * ay
            plus_is_long = dot_plus < 0
            yasmin.YASMIN_LOG_INFO(
                f"Side ambiguous (ratio={long_len/max(short_len, 1e-3):.2f}); "
                f"using approach-direction fallback."
            )

        sign = 1.0 if plus_is_long else -1.0
        return (sign * u_axis[0], sign * u_axis[1])

    def _save_frame(self, frame, result, **kwargs) -> None:
        if frame is None:
            return
        annotated = overlay.annotate_seg(frame, result)
        overlay.draw_select_side(
            annotated,
            sample_total=SIDE_SAMPLE_FRAMES,
            image_center=(IMAGE_CENTER_X, IMAGE_CENTER_Y),
            **kwargs,
        )
        self.sink.emit(annotated)
