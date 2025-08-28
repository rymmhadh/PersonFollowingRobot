#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class RecorderNode(Node):
    def __init__(self):
        super().__init__('recorder_node')
        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.recording = False
        self.out = None

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        # Démarrer l'enregistrement si pas déjà
        if not self.recording:
            self.out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame.shape[1], frame.shape[0]))
            self.recording = True
        self.out.write(frame)

def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

