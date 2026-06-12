import rclpy
import threading
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CameraSubscriber(Node):
    def __init__(self):
        super().__init__('camera_sub')
        self.bridge = CvBridge()
        self.frame = None
        self.create_subscription(Image, '/downward_camera/image_raw', self.cb, 10)

    def cb(self, msg):
        self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

def main():
    rclpy.init()                          
    cam = CameraSubscriber()             
    
    thread = threading.Thread(target=rclpy.spin, args=(cam,), daemon=True)
    thread.start()
    
    return cam  