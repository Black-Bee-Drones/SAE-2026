from std_msgs.msg import UInt8
import cv2
import os
from yasmin import Blackboard
from yasmin import State
from yasmin_ros.yasmin_node import YasminNode
from yasmin_ros.basic_outcomes import SUCCEED, ABORT
import time

HANDSHAKE_TIMEOUT = 25.0   # seconds to wait for ACK before giving up
PUBLISH_INTERVAL = 0.5     # seconds between retransmissions
SUBSCRIBER_WAIT_TIMEOUT = 20.0  # seconds to wait for GroundMonitor subscriber to appear


class AudioFeedback(State):
    def __init__(self):
        super().__init__(outcomes=[SUCCEED, "END"])
        self.node = YasminNode.get_instance()
        self.acked = False

        # Create publisher and subscriber at init time so DDS discovery
        # happens as early as possible, before the mission loop begins.
        self.gauge_publisher = self.node.create_publisher(UInt8, "/gauge/reading", 10)
        self.ack_subscription = self.node.create_subscription(
            UInt8,
            "/gauge/ack",
            self.ack_callback,
            10
        )

    def ack_callback(self, msg: UInt8):
        """Receive ACK from GroundMonitor confirming the reading was processed."""
        self.acked = True
        self.node.get_logger().info(f"ACK received for control_index {msg.data}")

    def _wait_for_ground_monitor(self):
        """Block until GroundMonitor's subscriber is visible or timeout expires."""
        self.node.get_logger().info("Waiting for GroundMonitor subscriber on /gauge/reading...")
        start = time.time()
        while self.gauge_publisher.get_subscription_count() == 0:
            if time.time() - start > SUBSCRIBER_WAIT_TIMEOUT:
                self.node.get_logger().warn(
                    f"GroundMonitor subscriber not found after {SUBSCRIBER_WAIT_TIMEOUT}s. "
                    "Proceeding anyway — messages may be lost."
                )
                return
            time.sleep(0.2)
        self.node.get_logger().info(
            f"GroundMonitor subscriber detected "
            f"({self.gauge_publisher.get_subscription_count()} subscriber(s)). "
            "Ready to publish."
        )

    def execute(self, blackboard: Blackboard):
        control_index = blackboard["control_index"]
        locations = blackboard["locations"]

        # On the very first waypoint, wait for the remote subscriber to be ready.
        if control_index == 0:
            self._wait_for_ground_monitor()

        gauge_reading = blackboard["gauge_reading"]  # Class (0-5), or -1 if not found

        if gauge_reading is not None:
            # Pack bits:
            # upper 3 bits -> gauge class (0-7)
            # lower 5 bits -> control_index (0-31)
            if gauge_reading != -1:
                packed_data = (gauge_reading << 5) | (control_index & 0x1F)

                msg = UInt8()
                msg.data = packed_data

                # Handshake: keep publishing until ACK or timeout
                self.acked = False
                start_time = time.time()

                while not self.acked and (time.time() - start_time) < HANDSHAKE_TIMEOUT:
                    self.gauge_publisher.publish(msg)
                    self.node.get_logger().info(
                        f"Published gauge reading for waypoint {control_index} "
                        f"(attempt {int((time.time() - start_time) / PUBLISH_INTERVAL) + 1})"
                    )
                    # Sleep in small increments so ack_callback can flip self.acked
                    sleep_end = time.time() + PUBLISH_INTERVAL
                    while not self.acked and time.time() < sleep_end:
                        time.sleep(0.05)

                if self.acked:
                    self.node.get_logger().info(f"Handshake complete for waypoint {control_index}")
                else:
                    self.node.get_logger().warn(
                        f"Handshake TIMEOUT for waypoint {control_index} after {HANDSHAKE_TIMEOUT}s"
                    )

            else:
                # gauge_reading == -1: detection failed, skip this waypoint
                if control_index == len(locations) - 1:
                    self.node.get_logger().info(
                        f"Final waypoint {control_index} processed (no detection), ending mission."
                    )
                    return "END"

                blackboard["control_index"] = control_index + 1
                self.node.get_logger().info(
                    f"No detection at waypoint {control_index}, "
                    f"incrementing control_index: {control_index} -> {blackboard['control_index']}"
                )
                return SUCCEED

        if "inference_image_cv" in blackboard and blackboard["inference_image_cv"] is not None:
            save_dir = os.path.join(os.path.expanduser("~"), "faulty_images")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"detection_waypoint_{control_index}.jpg")
            cv2.imwrite(save_path, blackboard["inference_image_cv"])
            self.node.get_logger().info(f"Inference image saved to: {save_path}")

        if control_index == len(locations) - 1:
            self.node.get_logger().info(f"Final waypoint {control_index} processed, ending mission.")
            return "END"

        blackboard["control_index"] = control_index + 1
        self.node.get_logger().info(
            f"Incremented control_index: {control_index} -> {blackboard['control_index']}"
        )
        return SUCCEED