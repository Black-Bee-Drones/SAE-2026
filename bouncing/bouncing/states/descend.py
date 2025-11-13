import yasmin
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone
from mirela_sdk.image_processing.camera import ImageHandler
from mirela_sdk.ai import YOLODetector


class Descend(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

        self.DESCEND_ALTITUDE = self.node.get_parameter_or('descend_altitude', 3)
        self.DESCEND_GO_TIMEOUT = self.node.get_parameter_or('descend_go_timeout', 30)

    @property
    def node(self):
        return YasminNode.get_instance()

    def execute(self, blackboard):
        if ("mavdrone" not in blackboard) or not blackboard["mavdrone"]:
            yasmin.YASMIN_LOG_ERROR("MavDrone not available in Descend state.")
            return ABORT
        mavdrone: MavDrone = blackboard["mavdrone"]

        if ("image_handler" not in blackboard) or not blackboard["image_handler"]:
            yasmin.YASMIN_LOG_ERROR("ImageHandler not available in Descend state.")
            return ABORT
        image_handler: ImageHandler = blackboard["image_handler"]

        if ("yolo_detector" not in blackboard) or not blackboard["yolo_detector"]:
            yasmin.YASMIN_LOG_ERROR("YOLODetector not available in Descend state.")
            return ABORT
        yolo_detector: YOLODetector = blackboard["yolo_detector"]

        if ("target_base" not in blackboard) or not blackboard["target_base"]:
            yasmin.YASMIN_LOG_ERROR("target_base not available in Descend state.")
            return ABORT
        target_base: dict = blackboard["target_base"]

        yasmin.YASMIN_LOG_INFO("Start SEARCH.")
        yasmin.YASMIN_LOG_INFO(f'Go to base coordinate: {target_base}')
        mavdrone.offboard_position(
            x=target_base['x'],
            y=target_base['y'],
            z=self.DESCEND_ALTITUDE,
            yaw = 0.0,
            ground_reference=True,
            timeout_sec=self.DESCEND_GO_TIMEOUT,
            precision_radius=0.1,
            strategy="PID",
        )

        return SUCCEED
