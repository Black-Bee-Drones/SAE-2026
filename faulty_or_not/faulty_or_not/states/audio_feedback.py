from std_msgs.msg import UInt8
import cv2
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
import time

class AudioFeedback(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, "END"])
        self.node = YasminNode.get_instance()

    def execute(self, blackboard : Blackboard):
        control_index = blackboard["control_index"]

        if "gauge_publisher" not in blackboard:
            blackboard["gauge_publisher"] = self.node.create_publisher(UInt8, "/gauge/reading", 10)

        gauge_reading = blackboard["gauge_reading"]  # Class (0-5), default -1 if not found
        
        if gauge_reading != -1:

            # Pack optimizing bits:
            # 3 bits for class (0-7, sufficient for 6 classes)
            # 5 bits for control_index (0-31, maximum possible space)
            packed_data = (gauge_reading << 5) | (control_index & 0x1F)

            publisher = blackboard["gauge_publisher"]
            msg = UInt8()
            msg.data = packed_data
            publisher.publish(msg)

        if "inference_image_cv" in blackboard:
            cv2.imwrite(f"/tmp/detection_waypoint_{control_index}.jpg", blackboard["inference_image_cv"])
            

        # Wait end of communication
        time.sleep(5)

        locations = blackboard["locations"]
        if control_index == len(locations) - 1:
            return "END"
        blackboard["control_index"] = control_index + 1
        return SUCCEED