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
from nectar.ai.detection import Detector, PerClassConfidenceFilter

from hook.core.constants import (
    INITIAL_TAKEOFF_ALTITUDE,
    RTL_ALTITUDE,
    IMAGE_SOURCE,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    PRECISION_LAND_CONF,
    PRECISION_LAND_IMGSZ,
    PRECISION_LAND_IOU,
    PRECISION_LAND_MODEL_PATH,
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
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def execute(self, blackboard: Blackboard):
        try:
            node = YasminNode.get_instance()
            yasmin.YASMIN_LOG_INFO("Initializing drone...")

            config = (
                SITL_GAZEBO_CONFIG
                if SIM_MODE
                else MavrosConfig(pose_source=PoseSource.GPS)
            )
            drone = DroneFactory.create("mavros", config, node)
            blackboard["drone"] = drone
            drone.delay(1)

            yasmin.YASMIN_LOG_INFO("Initializing camera...")
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
            drone.delay(1)

            frame = camera.take_photo()
            if frame is None:
                yasmin.YASMIN_LOG_ERROR("Failed to get frame from camera.")
                return ABORT
            yasmin.YASMIN_LOG_INFO(f"Camera ready. Frame shape: {frame.shape}")
            blackboard["camera"] = camera

            yasmin.YASMIN_LOG_INFO(f"Loading segmentation model: {SEG_MODEL_PATH}")
            segmentor = Segmentor(
                SEG_MODEL_PATH, confidence_threshold=SEG_PREDICT_CONF
            )
            segmentor.load()
            segmentor.segment(np.zeros((960, 960, 3), dtype=np.uint8))
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
            yasmin.YASMIN_LOG_INFO("Segmentor + per-class filter ready.")

            frame_publisher = FramePublisher(node)
            blackboard["frame_publisher"] = frame_publisher
            yasmin.YASMIN_LOG_INFO(
                f"Mission frames will be published on {frame_publisher.topic}."
            )

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
