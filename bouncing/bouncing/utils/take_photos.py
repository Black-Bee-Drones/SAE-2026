import cv2
import os
import time

import rclpy
from rclpy.node import Node

from nectar.vision import ImageHandler

from bouncing.constants import (
    CAMERA_IMAGE_SOURCE,
    CAMERA_CONFIG,
)


class TakePhotos(Node):
    def __init__(self):
        super().__init__("take_photos")


    def execute(self):
        self.node.get_logger().info('Start.')

        self.node.get_logger().info('Initializing ImageHandler...')
        self.image_handler = ImageHandler(
            node=self.node,
            image_source=CAMERA_IMAGE_SOURCE,
            config=CAMERA_CONFIG,
            image_processing_callback=self.callback_detector,
        )

        self.node.get_logger().info('Open camera...')
        self.image_handler.open()

        while rclpy.ok():
            self.image_handler.take_photo()
            time.sleep(0.2)


    def callback_detector(self, image):
        timestamp = self.node.get_clock().now().nanoseconds
        os.makedirs('photos', exist_ok=True)

        raw_path = os.path.join('photos', f'photos-{timestamp}.png')
        cv2.imwrite(raw_path, image)
        self.node.get_logger().info(f'Photo saved: {raw_path}')


def main(args=None):
    rclpy.init(args=args)
    node = TakePhotos()

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
