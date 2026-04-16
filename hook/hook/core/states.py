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
    MoveReference,
    RTLMethod,
    SITL_GAZEBO_CONFIG,
)
from nectar.vision import ImageHandler, OpenCVConfig
from nectar.ai.detection import Detector
from nectar.ai.segmentation import Segmentor

from hook.core.constants import (
    SEARCH_ALTITUDE,
    RTL_ALTITUDE,
    IMAGE_SOURCE,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    SPHERE_MODEL_PATH,
    ROPE_MODEL_PATH,
    SPHERE_CONF_THRESHOLD,
    ROPE_CONF_THRESHOLD,
    SIM_MODE,
)


class Initialize(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def execute(self, blackboard: Blackboard):
        try:
            node = YasminNode.get_instance()
            yasmin.YASMIN_LOG_INFO("Initializing drone...")

            config = SITL_GAZEBO_CONFIG if SIM_MODE else MavrosConfig(pose_source=PoseSource.GPS)
            drone = DroneFactory.create("mavros", config, node)
            blackboard["drone"] = drone
            drone.delay(1)

            yasmin.YASMIN_LOG_INFO("Initializing camera...")
            cam_config = None if SIM_MODE else OpenCVConfig(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
            camera = ImageHandler(
                node=node,
                image_source=IMAGE_SOURCE,
                config=cam_config,
            )
            camera.open()
            drone.delay(1)

            frame = camera.take_photo()
            if frame is None:
                yasmin.YASMIN_LOG_ERROR("Failed to get frame from camera.")
                return ABORT
            yasmin.YASMIN_LOG_INFO(f"Camera ready. Frame shape: {frame.shape}")
            blackboard["camera"] = camera

            yasmin.YASMIN_LOG_INFO(f"Loading sphere detector: {SPHERE_MODEL_PATH}")
            sphere_detector = Detector(
                SPHERE_MODEL_PATH,
                confidence_threshold=SPHERE_CONF_THRESHOLD,
            )
            sphere_detector.load()
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            sphere_detector.detect(dummy)
            blackboard["sphere_detector"] = sphere_detector
            yasmin.YASMIN_LOG_INFO("Sphere detector ready.")

            yasmin.YASMIN_LOG_INFO(f"Loading rope segmentor: {ROPE_MODEL_PATH}")
            rope_segmentor = Segmentor(
                ROPE_MODEL_PATH,
                confidence_threshold=ROPE_CONF_THRESHOLD,
            )
            rope_segmentor.load()
            rope_segmentor.segment(dummy)
            blackboard["rope_segmentor"] = rope_segmentor
            yasmin.YASMIN_LOG_INFO("Rope segmentor ready.")

            yasmin.YASMIN_LOG_INFO("Initialization complete.")
            return SUCCEED

        except Exception as e:
            yasmin.YASMIN_LOG_ERROR(f"Initialization failed: {e}")
            return ABORT


class Takeoff(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def execute(self, blackboard: Blackboard):
        if "drone" not in blackboard:
            yasmin.YASMIN_LOG_ERROR("Drone not available.")
            return ABORT

        drone: MavrosDrone = blackboard["drone"]

        try:
            yasmin.YASMIN_LOG_INFO(f"Taking off to {SEARCH_ALTITUDE}m...")
            drone.set_home()
            drone.arm()
            drone.takeoff(SEARCH_ALTITUDE)
            drone.delay(3)

            reached = drone.move_to(
                z=SEARCH_ALTITUDE,
                reference=MoveReference.TAKEOFF,
                timeout=30.0,
                precision=0.3,
            )

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
