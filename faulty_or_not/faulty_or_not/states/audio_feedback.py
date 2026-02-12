from std_msgs.msg import UInt8
import cv2
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
import time

HANDSHAKE_TIMEOUT = 10.0   # seconds to wait for ACK before giving up
PUBLISH_INTERVAL = 0.5     # seconds between retransmissions


class AudioFeedback(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, "END"])
        self.node = YasminNode.get_instance()
        self.acked = False

    def ack_callback(self, msg: UInt8):
        """Receive ACK from GroundMonitor confirming the reading was processed."""
        self.acked = True
        self.node.get_logger().info(f"ACK received for control_index {msg.data}")

    def execute(self, blackboard: Blackboard):
        control_index = blackboard["control_index"]

        if "gauge_publisher" not in blackboard:
            blackboard["gauge_publisher"] = self.node.create_publisher(UInt8, "/gauge/reading", 10)

        if "ack_subscription" not in blackboard:
            blackboard["ack_subscription"] = self.node.create_subscription(
                UInt8,
                "/gauge/ack",
                self.ack_callback,
                10
            )

        gauge_reading = blackboard["gauge_reading"]  # Class (0-5), default -1 if not found

        if gauge_reading != -1:
            # Pack optimizing bits:
            # 3 bits for class (0-7, sufficient for 6 classes)
            # 5 bits for control_index (0-31, maximum possible space)
            packed_data = (gauge_reading << 5) | (control_index & 0x1F)

            publisher = blackboard["gauge_publisher"]
            msg = UInt8()
            msg.data = packed_data

            # Handshake: keep publishing until ACK or timeout
            self.acked = False
            start_time = time.time()

            while not self.acked and (time.time() - start_time) < HANDSHAKE_TIMEOUT:
                publisher.publish(msg)
                self.node.get_logger().info(
                    f"Published gauge reading for waypoint {control_index} "
                    f"(attempt {int((time.time() - start_time) / PUBLISH_INTERVAL) + 1})"
                )
                # Sleep in small increments to check acked flag frequently
                sleep_end = time.time() + PUBLISH_INTERVAL
                while not self.acked and time.time() < sleep_end:
                    time.sleep(0.05)

            if self.acked:
                self.node.get_logger().info(f"Handshake complete for waypoint {control_index}")
            else:
                self.node.get_logger().warn(
                    f"Handshake TIMEOUT for waypoint {control_index} after {HANDSHAKE_TIMEOUT}s"
                )

        if "inference_image_cv" in blackboard and blackboard["inference_image_cv"] is not None:
            cv2.imwrite(f"/home/jetson/Pictures/detection_waypoint_{control_index}.jpg", blackboard["inference_image_cv"])

        locations = blackboard["locations"]
        if control_index == len(locations) - 1:
            return "END"
        blackboard["control_index"] = control_index + 1
        return SUCCEED