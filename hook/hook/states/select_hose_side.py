"""Pick which segment of the hose the drone should fly along.

After ApproachSphere the drone hovers ~35 cm horizontally from the sphere at
WORK_ALTITUDE. The combined seg model emits each visible piece of rope as its
own `rose` instance, so up to two `rose` masks share the same physical rope
direction.

Algorithm (continuous, no cardinal sectors):
  1. Sample SIDE_SAMPLE_FRAMES frames; for each, fit `hose_pose` to the
     most-confident hose to obtain the rope's image-frame long-axis unit
     vector u_hose.
  2. For every hose instance, project its centroid offset from the sphere
     onto u_hose -> signed scalar s. Sign = which side of the sphere; |s|
     unused for the side decision (we use the segment's own long-axis length
     as the strength signal).
  3. Aggregate total length per side (sum across the window).
  4. Pick the side with greater length. If long/short < SIDE_LENGTH_RATIO,
     fall back: pick the sign whose direction is most opposite the
     `approach_image_offset` stored at end of ApproachSphere.
  5. Save `hose_side_image_unit` (continuous unit vector, image frame) on the
     blackboard, then shift the drone in body frame along that direction for
     HOSE_SHIFT_DURATION seconds.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.ai.detection import PerClassConfidenceFilter
from nectar.ai.segmentation import Segmentor
from nectar.control import MavrosDrone, MoveReference
from nectar.vision import ImageHandler

from hook.core.constants import (
    DETECTION_SAVE_PATH,
    HOSE_SHIFT_DURATION,
    HOSE_SHIFT_VELOCITY,
    SAVE_DETECTIONS,
    SIDE_LENGTH_RATIO,
    SIDE_REACQUIRE_FRAMES,
    SIDE_SAMPLE_FRAMES,
    SIDE_TIMEOUT,
)
from hook.core.perception import (
    best_sphere,
    hose_pose,
    hose_segments,
    pick_hose_by_dir,
    run_seg,
)


class SelectHoseSide(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.save_dir = None
        self.frame_count = 0

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

        if SAVE_DETECTIONS:
            ts = (
                blackboard["mission_timestamp"]
                if "mission_timestamp" in blackboard
                else datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            self.save_dir = Path(DETECTION_SAVE_PATH) / ts / "select_side"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        side_unit = self._decide_side(camera, segmentor, class_filter, approach_offset)
        if side_unit is None:
            yasmin.YASMIN_LOG_ERROR("Could not decide hose side: no usable samples.")
            return ABORT

        blackboard["hose_side_image_unit"] = side_unit
        yasmin.YASMIN_LOG_INFO(
            f"Chosen hose side image-unit: ({side_unit[0]:+.2f}, {side_unit[1]:+.2f})"
        )

        if not self._shift_along_side(drone, side_unit):
            return ABORT

        if not self._reacquire(camera, segmentor, class_filter, side_unit):
            return ABORT

        return SUCCEED

    # --- Decision --------------------------------------------------------

    def _decide_side(
        self,
        camera,
        segmentor,
        class_filter,
        approach_offset: Optional[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        """Returns the chosen image-frame unit vector (sphere -> chosen side)."""
        u_axis: Optional[Tuple[float, float]] = None
        length_plus = 0.0  # along +u_axis from sphere
        length_minus = 0.0  # along -u_axis from sphere
        collected = 0
        start = time.time()

        while collected < SIDE_SAMPLE_FRAMES and time.time() - start < SIDE_TIMEOUT:
            frame, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)
            hoses = hose_segments(result)
            if sphere is None or not hoses:
                time.sleep(0.05)
                continue

            # Lock the rope axis from the most-confident hose on the first
            # usable frame; reuse it across frames so plus/minus stay consistent.
            if u_axis is None:
                pose = hose_pose(hoses[0])
                if pose is None:
                    time.sleep(0.05)
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

            self._maybe_save(frame, result, segmentor, f"sample_{collected:03d}")
            collected += 1
            time.sleep(0.03)

        if u_axis is None:
            return None

        long_len, short_len = max(length_plus, length_minus), min(
            length_plus, length_minus
        )
        plus_is_long = length_plus >= length_minus
        ambiguous = short_len > 0 and (long_len / short_len) < SIDE_LENGTH_RATIO

        if ambiguous and approach_offset is not None:
            ax, ay = approach_offset
            # The drone shifted INTO `approach_offset`. The longer hose is
            # likely on the opposite side -> pick the sign whose unit-vec is
            # most anti-aligned with the approach offset.
            dot_plus = u_axis[0] * ax + u_axis[1] * ay
            plus_is_long = dot_plus < 0
            yasmin.YASMIN_LOG_INFO(
                f"Side ambiguous (ratio={long_len/max(short_len, 1e-3):.2f}); "
                f"using approach-direction fallback."
            )

        sign = 1.0 if plus_is_long else -1.0
        return (sign * u_axis[0], sign * u_axis[1])

    # --- Body shift along chosen side -----------------------------------

    def _shift_along_side(self, drone, side_unit):
        # image -y -> body +x, image +x -> body -y
        ux, uy = side_unit
        vx = -uy * HOSE_SHIFT_VELOCITY
        vy = -ux * HOSE_SHIFT_VELOCITY
        yasmin.YASMIN_LOG_INFO(
            f"Shifting along side: vx={vx:+.2f} vy={vy:+.2f} dur={HOSE_SHIFT_DURATION:.1f}s"
        )
        try:
            drone.move_velocity(
                vx=vx,
                vy=vy,
                vz=0.0,
                vyaw=0.0,
                duration=HOSE_SHIFT_DURATION,
                reference=MoveReference.BODY,
            )
        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Side shift failed: {e}")
            return False
        drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)
        return True

    # --- Re-acquire ------------------------------------------------------

    def _reacquire(self, camera, segmentor, class_filter, side_unit):
        ok = 0
        start = time.time()
        while time.time() - start < SIDE_TIMEOUT:
            frame, result = run_seg(camera, segmentor, class_filter)
            sphere = best_sphere(result)
            hoses = hose_segments(result)
            chosen = pick_hose_by_dir(sphere, hoses, side_unit)
            if sphere is not None and chosen is not None:
                ok += 1
                self._maybe_save(frame, result, segmentor, f"reacq_{ok:03d}")
                if ok >= SIDE_REACQUIRE_FRAMES:
                    return True
            else:
                ok = 0
            time.sleep(0.05)
        yasmin.YASMIN_LOG_ERROR("Failed to reacquire sphere + chosen hose after shift.")
        return False

    # --- Saving ----------------------------------------------------------

    def _maybe_save(self, frame, result, segmentor, tag):
        if not (SAVE_DETECTIONS and self.save_dir and frame is not None and result):
            return
        self.frame_count += 1
        annotated = segmentor.draw_segmentations(frame, result)
        cv2.imwrite(str(self.save_dir / f"{tag}.jpg"), annotated)
