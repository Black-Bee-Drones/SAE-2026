from std_msgs.msg import UInt8
from sensor_msgs.msg import CompressedImage
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
            
        if "inference_image_publisher" not in blackboard:
            blackboard["inference_image_publisher"] = self.node.create_publisher(CompressedImage, "/gauge/inference_image/compressed", 10)

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
        
        # Publish the inferred image with bounding box (if available)
        if "inference_image_cv" in blackboard and blackboard["inference_image_cv"] is not None:
            image_publisher = blackboard["inference_image_publisher"]
            
            # Compress OpenCV image to JPEG
            _, buffer = cv2.imencode('.jpg', blackboard["inference_image_cv"], 
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])  # 85% quality
            
            # Create compressed message
            compressed_msg = CompressedImage()
            compressed_msg.header.stamp = self.node.get_clock().now().to_msg()
            compressed_msg.format = "jpeg"
            compressed_msg.data = buffer.tobytes()
            
            image_publisher.publish(compressed_msg)

        # Wait end of communication
        time.sleep(5)

        locations = blackboard["locations"]
        if control_index == len(locations) - 1:
            return "END"
        blackboard["control_index"] = control_index + 1
        return SUCCEED