#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.exceptions import InvalidParameterValueException

from zaxis.telemetry import MavlinkConnection
from zaxis.sensors.rngfnd_filter import RangefinderFilter

from zaxis.sensors.benewake.tfluna import TFLuna

class RangefinderNode(Node):
    """
    ROS2 Node that manages Mavlink connection and Rangefinder sensor filtering.
    Stores these instances for access by other nodes in the system.
    """

    def __init__(self):
        super().__init__('rangefinder_node')
        self.connection_name = "drone"
        self.connection_str = "127.0.0.1:14551"
        self.baud_rate = 921600
        
        self.get_logger().info(f"Initializing Rangefinder Node with connection: {self.connection_name}")
        
        try:
            self.get_logger().info(f"Creating MavlinkConnection: {self.connection_name}")
            self.connection = MavlinkConnection(self.connection_name)
            self.connection.connect(self.connection_str, self.baud_rate)
            
            self.get_logger().info("Creating RangefinderFilter")
            self.rangefinder_filter = RangefinderFilter(self.connection, sensor=TFLuna("/dev/ttyUSB0")) 

            self.get_logger().info("Starting RangefinderFilter loop")
            self.rangefinder_filter.start()

        except Exception as e:
            self.get_logger().error(f"Failed to initialize Rangefinder Node: {e}")
            raise
    
    def get_connection(self):
        """Get the Mavlink connection instance"""
        return self.connection
    
    def get_rangefinder_filter(self):
        """Get the RangefinderFilter instance"""
        return self.rangefinder_filter
    
    def destroy_node(self):
        """Stop the rangefinder filter before destroying the node"""
        self.get_logger().info("Stopping RangefinderFilter...")
        if hasattr(self, 'rangefinder_filter') and self.rangefinder_filter:
            self.rangefinder_filter.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    rangefinder_node = None
    try:
        rangefinder_node = RangefinderNode()
        rclpy.spin(rangefinder_node)

    except KeyboardInterrupt:
        print("\nRangefinder Node shutting down...")
    except Exception as e:
        print(f"Error in Rangefinder Node: {e}")
    finally: 
        if rangefinder_node:
            rangefinder_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
