#!/usr/bin/env python3
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
import rclpy
from rclpy.node import Node

import cv2
import numpy as np
from cv_bridge import CvBridge

from sensor_msgs.msg import Image, CompressedImage


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')

        # Parâmetros
        self.declare_parameter("use_compression", True)
        self.use_compression = self.get_parameter("use_compression").value

        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE,
        )

        if self.use_compression:
            self.subscription = self.create_subscription(
                CompressedImage,
                'image_raw/compressed',
                self.compressed_callback,
                qos_profile
            )
            self.get_logger().info("Subscribed to compressed image")
        else:
            self.subscription = self.create_subscription(
                Image,
                'image_raw',
                self.image_callback,
                qos_profile
            )
            self.get_logger().info("Subscribed to raw image")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.show_image(frame)

    def compressed_callback(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.show_image(frame)

    def show_image(self, frame):
        if frame is None:
            return

        cv2.imshow("Camera Viewer", frame)
        key = cv2.waitKey(1)

        # Fecha com 'q'
        if key == ord('q'):
            self.get_logger().info("Saindo...")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
