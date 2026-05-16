import threading
import time
from typing import Optional

import numpy as np

import yasmin
from yasmin import Blackboard, State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
from yasmin_ros.yasmin_node import YasminNode

from nectar.control import (
    DroneFactory,
    MavrosConfig,
    MavrosDrone,
    PoseSource,
    RTLMethod,
    SITL_GAZEBO_CONFIG,
)
from nectar.vision import ImageHandler, OpenCVConfig
from nectar.vision.camera import ROSConfig
from nectar.ai.segmentation import Segmentor
from nectar.ai.detection import PerClassConfidenceFilter

from hook.core.constants import (
    INITIAL_TAKEOFF_ALTITUDE,
    RTL_ALTITUDE,
    IMAGE_SOURCE,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    SEG_MODEL_PATH,
    SEG_PREDICT_CONF,
    SIM_IMAGE_COMPRESSED,
    SPHERE_CLASS,
    SPHERE_CONF,
    HOSE_CLASS,
    HOSE_CONF,
    SIM_MODE,
)
from hook.core.frame_sink import FramePublisher


class Initialize(State):
    """Concurrent drone + camera + segmentation-model init.

    The segmentation model load (file IO + CUDA kernel compile + warm-up
    inference) is the slowest single step here, and it is fully
    independent of MAVROS / camera bring-up. It runs on a daemon thread
    while the main thread initialises the drone and camera; we ``wait``
    on the model thread only when the result is actually needed (right
    before building the per-class filter).

    Sequential cost: ``drone_init + camera_init + model_load``.
    Parallel cost:   ``max(drone_init + camera_init, model_load)``.
    """

    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def execute(self, blackboard: Blackboard):
        try:
            node = YasminNode.get_instance()
            t0 = time.perf_counter()

            loader = _SegmentorLoader(SEG_MODEL_PATH, SEG_PREDICT_CONF)
            loader.start()

            yasmin.YASMIN_LOG_INFO("Initializing drone...")
            t_d0 = time.perf_counter()
            config = (
                SITL_GAZEBO_CONFIG
                if SIM_MODE
                else MavrosConfig(pose_source=PoseSource.GPS)
            )
            drone = DroneFactory.create("mavros", config, node)
            blackboard["drone"] = drone
            t_drone = time.perf_counter() - t_d0

            yasmin.YASMIN_LOG_INFO("Initializing camera...")
            t_c0 = time.perf_counter()
            if SIM_MODE:
                cam_config = ROSConfig(
                    topic=IMAGE_SOURCE,
                    compressed=SIM_IMAGE_COMPRESSED,
                )
            else:
                cam_config = OpenCVConfig(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
            camera = ImageHandler(
                node=node,
                image_source=IMAGE_SOURCE,
                config=cam_config,
            )
            camera.open()
            frame = camera.take_photo()
            if frame is None:
                yasmin.YASMIN_LOG_ERROR("Failed to get frame from camera.")
                return ABORT
            t_cam = time.perf_counter() - t_c0
            yasmin.YASMIN_LOG_INFO(
                f"Camera ready. Frame shape: {frame.shape} ({t_cam:.2f}s)"
            )
            blackboard["camera"] = camera

            segmentor = loader.join()
            if segmentor is None:
                yasmin.YASMIN_LOG_ERROR(
                    f"Segmentation model load failed: {loader.error}"
                )
                return ABORT
            blackboard["segmentor"] = segmentor

            name_to_id = {v: k for k, v in segmentor.class_names.items()}
            if SPHERE_CLASS not in name_to_id or HOSE_CLASS not in name_to_id:
                yasmin.YASMIN_LOG_ERROR(
                    f"Model classes mismatch: have {list(name_to_id)}"
                )
                return ABORT
            blackboard["class_filter"] = PerClassConfidenceFilter(
                threshold_mapping={
                    name_to_id[SPHERE_CLASS]: SPHERE_CONF,
                    name_to_id[HOSE_CLASS]: HOSE_CONF,
                },
                default_threshold=1.1,
            )

            frame_publisher = FramePublisher(node)
            blackboard["frame_publisher"] = frame_publisher
            yasmin.YASMIN_LOG_INFO(
                f"Mission frames will be published on {frame_publisher.topic}."
            )

            total = time.perf_counter() - t0
            saved = (t_drone + t_cam + loader.elapsed) - total
            yasmin.YASMIN_LOG_INFO(
                f"Initialization complete in {total:.2f}s "
                f"(drone={t_drone:.2f}s, camera={t_cam:.2f}s, "
                f"model={loader.elapsed:.2f}s; "
                f"parallel-saved={max(saved, 0.0):.2f}s vs sequential)."
            )
            return SUCCEED

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Initialization failed: {e}")
            return ABORT


class _SegmentorLoader:
    """Daemon-thread wrapper that loads + warms a :class:`Segmentor`.

    Independent of MAVROS / camera state, so it overlaps cleanly with
    drone bring-up and camera open. ``join()`` returns the ready
    Segmentor or ``None`` on failure (use ``.error`` for the exception).
    The Ultralytics load releases the GIL during disk read / CUDA kernel
    compile, so the main thread continues to make progress.
    """

    def __init__(self, model_path: str, confidence_threshold: float) -> None:
        self._model_path = model_path
        self._confidence_threshold = confidence_threshold
        self._done = threading.Event()
        self._segmentor: Optional[Segmentor] = None
        self.error: Optional[BaseException] = None
        self.elapsed: float = 0.0
        self._t0: float = 0.0
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="hook-seg-loader"
        )

    def start(self) -> None:
        yasmin.YASMIN_LOG_INFO(
            f"Loading segmentation model in background: {self._model_path}"
        )
        self._t0 = time.perf_counter()
        self._thread.start()

    def _run(self) -> None:
        try:
            seg = Segmentor(
                self._model_path,
                confidence_threshold=self._confidence_threshold,
            )
            seg.load()
            seg.segment(np.zeros((960, 960, 3), dtype=np.uint8))
            self._segmentor = seg
        except BaseException as e:
            self.error = e
        finally:
            self.elapsed = time.perf_counter() - self._t0
            self._done.set()

    def join(self) -> Optional[Segmentor]:
        self._done.wait()
        return self._segmentor


class Takeoff(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def execute(self, blackboard: Blackboard):
        if "drone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("Drone not available.")
            return ABORT

        drone: MavrosDrone = blackboard["drone"]

        try:
            yasmin.YASMIN_LOG_INFO(f"Taking off to {INITIAL_TAKEOFF_ALTITUDE}m...")

            reached = drone.takeoff(INITIAL_TAKEOFF_ALTITUDE, max_retries=5)
            drone.delay(2)

            if not reached:
                yasmin.YASMIN_LOG_WARN("Takeoff move_to timed out, continuing.")

            drone.delay(1)
            yasmin.YASMIN_LOG_INFO("Takeoff complete.")
            return SUCCEED

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Takeoff failed: {e}")
            return ABORT


class ReturnToLaunch(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def execute(self, blackboard: Blackboard):
        if "drone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("Drone not available.")
            return ABORT

        drone: MavrosDrone = blackboard["drone"]

        try:
            yasmin.YASMIN_LOG_INFO(f"Returning to launch at {RTL_ALTITUDE}m...")
            drone.rtl(
                altitude=RTL_ALTITUDE,
                method=RTLMethod.NAVIGATE,
                land=False,
            )
            drone.delay(2)
            return SUCCEED

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"RTL failed: {e}")
            return ABORT


class Land(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def execute(self, blackboard: Blackboard):
        if "drone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("Drone not available.")
            return ABORT

        drone: MavrosDrone = blackboard["drone"]

        try:
            drone.land()
            drone.delay(10)
            yasmin.YASMIN_LOG_INFO("Landed.")

            if "camera" in blackboard and blackboard["camera"]:
                blackboard["camera"].cleanup()

            return SUCCEED

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Landing failed: {e}")
            return ABORT
