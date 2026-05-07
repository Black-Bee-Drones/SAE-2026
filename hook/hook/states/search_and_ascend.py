import time
from datetime import datetime
from pathlib import Path

import cv2
import rclpy
import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
from yasmin_ros.yasmin_node import YasminNode

from nectar.ai.detection import PerClassConfidenceFilter
from nectar.ai.segmentation import Segmentor
from nectar.control import AltitudeSource, MavrosDrone, MoveReference
from nectar.vision import ImageHandler

from hook.core.constants import (
    ASCEND_VELOCITY,
    ASCENT_STOP_CONFIRMATIONS,
    ASCENT_TIMEOUT,
    DETECTION_SAVE_PATH,
    MAX_ASCEND_ALTITUDE,
    SAVE_DETECTIONS,
)
from hook.core.perception import best_sphere, run_seg


class SearchAndAscend(State):
    """Hold position and ascend until the sphere is detected with debounce."""

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.save_dir = None
        self.frame_count = 0

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]
        camera: ImageHandler = blackboard["camera"]
        segmentor: Segmentor = blackboard["segmentor"]
        class_filter: PerClassConfidenceFilter = blackboard["class_filter"]

        if SAVE_DETECTIONS:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            blackboard["mission_timestamp"] = timestamp
            self.save_dir = Path(DETECTION_SAVE_PATH) / timestamp / "search_ascend"
            self.save_dir.mkdir(parents=True, exist_ok=True)

        yasmin.YASMIN_LOG_INFO("Searching for sphere while ascending...")

        confirmations = 0
        latest = None
        start_time = time.time()

        while time.time() - start_time < ASCENT_TIMEOUT:
            rclpy.spin_once(YasminNode.get_instance(), timeout_sec=0.05)
            altitude = drone.get_altitude(AltitudeSource.LIDAR)
            if altitude is None:
                altitude = drone.get_altitude(AltitudeSource.AUTO)

            frame, result = run_seg(camera, segmentor, class_filter)
            if frame is None:
                time.sleep(0.05)
                continue

            sphere = best_sphere(result)

            if sphere is not None:
                confirmations += 1
                latest = sphere
                drone.move_velocity(0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY)

                if SAVE_DETECTIONS and self.save_dir: # and self.frame_count % 3 == 0:
                    annotated = segmentor.draw_segmentations(frame, result)
                    cv2.imwrite(
                        str(self.save_dir / f"sphere_{self.frame_count:04d}.jpg"),
                        annotated,
                    )
                self.frame_count += 1

                alt_txt = f"{altitude:.2f}m" if altitude is not None else "n/a"
                yasmin.YASMIN_LOG_INFO(
                    f"Sphere ({confirmations}/{ASCENT_STOP_CONFIRMATIONS}): "
                    f"conf={sphere.confidence:.2f} "
                    f"center=({sphere.center[0]:.0f},{sphere.center[1]:.0f}) alt={alt_txt}"
                )

                if confirmations >= ASCENT_STOP_CONFIRMATIONS:
                    blackboard["sphere_center"] = latest.center
                    blackboard["sphere_bbox"] = latest.bbox
                    blackboard["ascent_alt"] = altitude
                    yasmin.YASMIN_LOG_INFO(
                        f"Sphere confirmed at ({latest.center[0]:.0f},{latest.center[1]:.0f})."
                    )
                    return SUCCEED
                time.sleep(0.03)
                continue

            confirmations = 0

            if altitude is not None and altitude >= MAX_ASCEND_ALTITUDE:
                drone.move_velocity(
                    0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=2.0
                )
                yasmin.YASMIN_LOG_ERROR(
                    f"Reached ascent cap {MAX_ASCEND_ALTITUDE}m without sphere."
                )
                return ABORT

            drone.move_velocity(
                vx=0.0,
                vy=0.0,
                vz=ASCEND_VELOCITY,
                vyaw=0.0,
                reference=MoveReference.BODY,
            )
            time.sleep(0.05)

        drone.move_velocity(
            0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=2.0
        )
        yasmin.YASMIN_LOG_ERROR("Search-and-ascend timed out.")
        return ABORT
