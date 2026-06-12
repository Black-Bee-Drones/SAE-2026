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
    ASCEND_VELOCITY_SLOW,
    ASCEND_YAW_RATE_RAD_S,
    ASCENT_STOP_CONFIRMATIONS,
    ASCENT_TIMEOUT,
    MAX_ASCEND_ALTITUDE,
)
from hook.core.frame_sink import FrameSink, build_state_sink
from hook.core.perception import best_sphere, run_seg


_LOG_PERIOD_S = 0.5


class SearchAndAscend(State):
    """Ascend until the sphere is debounced; yaw-search at the altitude cap.

    Two phases sharing the same debounce/save path:

    - ``ascend``: ``vz = ASCEND_VELOCITY`` upward while no sphere is in
      view; ``vz = ASCEND_VELOCITY_SLOW`` (still climbing, slower) while
      the sphere has been seen but not yet confirmed for
      ``ASCENT_STOP_CONFIRMATIONS`` consecutive frames. Exits to SUCCEED
      on confirmation OR transitions to ``yaw_search`` when the lidar
      altitude reaches ``MAX_ASCEND_ALTITUDE`` without a confirmed
      sphere.
    - ``yaw_search``: ``vz = 0``, ``vyaw = ASCEND_YAW_RATE_RAD_S`` while
      no sphere is in view; on detection the drone stops rotating
      (``vyaw = 0``) so the debounce can finish without sliding the
      sphere out of the frame. ``ASCENT_TIMEOUT`` bounds the whole state.
    """

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
        phase = "ascend"
        start_time = time.time()
        last_log = 0.0

        while time.time() - start_time < ASCENT_TIMEOUT:
            drone.delay(0.05)
            altitude = drone.get_altitude(AltitudeSource.LIDAR)
            if altitude is None:
                altitude = drone.get_altitude(AltitudeSource.AUTO)

            if (
                phase == "ascend"
                and altitude is not None
                and altitude >= MAX_ASCEND_ALTITUDE
            ):
                phase = "yaw_search"
                yasmin.YASMIN_LOG_INFO(
                    f"Reached ascent cap {MAX_ASCEND_ALTITUDE}m; "
                    f"yaw-searching at {ASCEND_YAW_RATE_RAD_S:+.2f} rad/s."
                )

            frame, result = run_seg(camera, segmentor, class_filter)
            if frame is None:
                continue

            sphere = best_sphere(result)

            # Phase-aware command: slow climb / stop rotating when the
            # sphere is in view but not yet confirmed; full climb / spin
            # otherwise.
            if sphere is not None:
                if phase == "ascend":
                    vz_cmd, vyaw_cmd = ASCEND_VELOCITY_SLOW, 0.0
                else:
                    vz_cmd, vyaw_cmd = 0.0, 0.0
            else:
                if phase == "ascend":
                    vz_cmd, vyaw_cmd = ASCEND_VELOCITY, 0.0
                else:
                    vz_cmd, vyaw_cmd = 0.0, ASCEND_YAW_RATE_RAD_S

            drone.move_velocity(
                vx=0.0, vy=0.0, vz=vz_cmd, vyaw=vyaw_cmd,
                reference=MoveReference.BODY,
            )

            if sphere is not None:
                confirmations += 1
                latest = sphere
            else:
                confirmations = 0

            self._save_frame(
                frame, result, altitude, confirmations, phase,
                sphere_center=sphere.center if sphere is not None else None,
            )

            now = time.time()
            if now - last_log > _LOG_PERIOD_S or (
                sphere is not None and confirmations == 1
            ):
                alt_txt = f"{altitude:.2f}m" if altitude is not None else "n/a"
                det_txt = (
                    f"sphere conf={sphere.confidence:.2f} "
                    f"({sphere.center[0]:.0f},{sphere.center[1]:.0f})"
                    if sphere is not None
                    else "sphere -"
                )
                yasmin.YASMIN_LOG_INFO(
                    f"SearchAndAscend[{phase:>11s}] alt={alt_txt} "
                    f"conf={confirmations}/{ASCENT_STOP_CONFIRMATIONS} | "
                    f"{det_txt} | "
                    f"cmd: vz={vz_cmd:+.2f} vyaw={vyaw_cmd:+.2f}"
                )
                last_log = now

            if sphere is not None and confirmations >= ASCENT_STOP_CONFIRMATIONS:
                drone.move_velocity(
                    0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY
                )
                blackboard["sphere_center"] = latest.center
                blackboard["sphere_bbox"] = latest.bbox
                blackboard["ascent_alt"] = altitude
                alt_txt = f"{altitude:.2f}m" if altitude is not None else "n/a"
                yasmin.YASMIN_LOG_INFO(
                    f"Sphere confirmed at "
                    f"({latest.center[0]:.0f},{latest.center[1]:.0f}) "
                    f"alt={alt_txt}."
                )
                return SUCCEED

        drone.move_velocity(
            0.0, 0.0, 0.0, 0.0, reference=MoveReference.BODY, duration=2.0
        )
        yasmin.YASMIN_LOG_ERROR(
            f"Search-and-ascend timed out in '{phase}' phase."
        )
        return ABORT

    def _save_frame(self, frame, result, altitude, confirmations, phase, *, sphere_center):
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
            phase=phase,
        )
        self.sink.emit(annotated)
