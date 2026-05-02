import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

class TestCam(Node):
    def __init__(self):
        super().__init__('test_cam')
        self.pub = self.create_publisher(CompressedImage, '/gauge/compressed', 10)
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.camera = cv2.VideoCapture(0)  
        self.camera.start()

    def timer_callback(self):
        ret, self.frame = self.camera.read()
        if not ret:
            return
        
        height, width = self.frame.shape[:2]
        if height > width:
            diff = height - width
            self.frame = self.frame[diff // 2:diff // 2 + width, :]
        elif width > height:
            diff = width - height
            self.frame = self.frame[:, diff // 2:diff // 2 + height]

        # Encode to JPEG
        success, buffer = cv2.imencode('.jpg', self.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not success:
            self.get_logger().warn('Failed to encode test image')
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        msg.data = buffer.tobytes()

        self.get_logger().info('Publishing synthetic compressed image on /gauge/compressed')
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TestCam()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
