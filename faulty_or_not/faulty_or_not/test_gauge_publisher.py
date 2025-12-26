import rclpy
from rclpy.node import Node

from std_msgs.msg import UInt8
import random
class TestGaugePublisher(Node):
    def __init__(self):
        super().__init__('test_gauge_publisher')
        self.detection_pub = self.create_publisher(UInt8, '/gauge/reading', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)

    def timer_callback(self):
        # Simulate random gauge readings and control indices
        gauge_class = random.randint(0, 5)  # Classes 0-5 (3 bits: 0-7 max)
        control_idx = random.randint(0, 2)  # Control index 0-31 (5 bits max)
        
        # Pack: 3 bits (gauge) + 5 bits (control_index)
        packed_data = (gauge_class << 5) | (control_idx & 0x1F)
        
        msg = UInt8()
        msg.data = packed_data
        
        self.get_logger().info(
            f'Publishing: gauge_class={gauge_class}, control_idx={control_idx}, '
            f'packed={packed_data} (0b{packed_data:08b})'
        )
        
        self.detection_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = TestGaugePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
