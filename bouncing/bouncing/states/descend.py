import yasmin
from yasmin import State
from yasmin_ros.basic_outcomes import SUCCEED, ABORT

from mirela_sdk.control.mavros import MavDrone
from mirela_sdk.image_processing.camera import ImageHandler
from mirela_sdk.ai import YOLODetector


class Descend(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, ABORT])

    def execute(self, blackboard):
        if ("mavdrone" not in blackboard) or not blackboard["mavdrone"]:
            yasmin.YASMIN_LOG_ERROR("MavDrone not available in SearchID state.")
            return ABORT
        mavdrone: MavDrone = blackboard["mavdrone"]

        if ("image_handler" not in blackboard) or not blackboard["image_handler"]:
            yasmin.YASMIN_LOG_ERROR("ImageHandler not available in SearchID state.")
            return ABORT
        image_handler: ImageHandler = blackboard["image_handler"]

        if ("yolo_detector" not in blackboard) or not blackboard["yolo_detector"]:
            yasmin.YASMIN_LOG_ERROR("YOLODetector not available in SearchID state.")
            return ABORT
        yolo_detector: YOLODetector = blackboard["yolo_detector"]

        return SUCCEED
