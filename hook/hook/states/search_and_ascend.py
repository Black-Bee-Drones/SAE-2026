import time

import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from nectar.ai.detection import PerClassConfidenceFilter
from nectar.ai.segmentation import Segmentor
from nectar.control import AltitudeSource, MavrosDrone, MoveReference
from nectar.vision import ImageHandler

from hook.core import overlay
from hook.core.constants import (
    ASCEND_VELOCITY,
    ASCENT_STOP_CONFIRMATIONS,
    ASCENT_TIMEOUT,
    MAX_ASCEND_ALTITUDE,
)
from hook.core.frame_sink import FrameSink, build_state_sink
from hook.core.perception import best_sphere, run_seg


class SearchAndAscend(State):
    """Hold position and ascend until the sphere is detected with debounce."""

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])
        self.sink: FrameSink = None

    def execute(self, blackboard: Blackboard):
        drone: MavrosDrone = blackboard["drone"]
        camera: ImageHandler = blackboard["camera"]
        segmentor: Segmentor = blackboard["segmentor"]
        class_filter: PerClassConfidenceFilter = blackboard["class_filter"]

        self.sink = build_state_sink(blackboard, "search_ascend")

        yasmin.YASMIN_LOG_INFO("Searching for sphere while ascending...")

        confirmations = 0
        latest = None
        start_time = time.time()

        while time.time() - start_time < ASCENT_TIMEOUT:
            drone.delay(0.05)
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

                self._save_frame(
                    frame, result, altitude, confirmations,
                    sphere_center=sphere.center,
                )

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
            self._save_frame(frame, result, altitude, confirmations, sphere_center=None)

            if altitude is not None and altitude >= MAX_ASCEND_ALTITUDE:
                drone.move_velocity(
                    0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=2.0
                )
                yasmin.YASMIN_LOG_ERROR(
                    f"Reached ascent cap {MAX_ASCEND_ALTITUDE}m without sphere."
                )
                return ABORT

            drone.move_velocity(
                vx=0.0, vy=0.0, vz=ASCEND_VELOCITY, vyaw=0.0,
                reference=MoveReference.BODY,
            )
            time.sleep(0.05)

        drone.move_velocity(
            0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=2.0
        )
        yasmin.YASMIN_LOG_ERROR("Search-and-ascend timed out.")
        return ABORT

    def _save_frame(self, frame, result, altitude, confirmations, *, sphere_center):
        if frame is None:
            return
        annotated = overlay.annotate_seg(frame, result)
        overlay.draw_search_ascend(
            annotated,
            altitude=altitude,
            alt_max=MAX_ASCEND_ALTITUDE,
            confirmations=confirmations,
            target_confirmations=ASCENT_STOP_CONFIRMATIONS,
            sphere_center=sphere_center,
        )
        self.sink.emit(annotated)
