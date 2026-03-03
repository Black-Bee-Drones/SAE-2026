from std_msgs.msg import UInt8
import cv2
import random
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
        nav = blackboard["navigation_state"]
        if nav is not None:
            nav.stop_circle()

        control_index = blackboard["control_index"]
        locations = blackboard["locations"]

        if "gauge_publisher" not in blackboard:
            blackboard["gauge_publisher"] = self.node.create_publisher(UInt8, "/gauge/reading", 10)

        if "ack_subscription" not in blackboard:
            blackboard["ack_subscription"] = self.node.create_subscription(
                UInt8,
                "/gauge/ack",
                self.ack_callback,
                10
            )
        if "gauge_reading" not in blackboard:
            if control_index == len(locations) - 1:
                return "END"
            blackboard["control_index"] = control_index + 1
            return SUCCEED
            
        gauge_reading = blackboard["gauge_reading"]  # Class (0-5), default -1 if not found

        if gauge_reading is not None:
            # Pack optimizing bits:
            # 3 bits for class (0-7, sufficient for 6 classes)
            # 5 bits for control_index (0-31, maximum possible space)
            if gauge_reading != 1:
                packed_data = (gauge_reading << 5) | (control_index & 0x1F)
            else:
                print("[MODE] Kakegurui Mashou")
                packed_data = ((random.randint(0, 5)) << 5) | (control_index & 0x1F) | 0b00010000  # Set bit 4 for "uncertain" class
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

        if control_index == len(locations) - 1:
            return "END"
        blackboard["control_index"] = control_index + 1
        return SUCCEED