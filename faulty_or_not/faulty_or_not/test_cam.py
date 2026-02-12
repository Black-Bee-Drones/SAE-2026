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
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        # Create a synthetic image similar in size/format used by GaugeReading
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(img, f'TestCam {timestamp}', (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Encode to JPEG
        success, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
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
