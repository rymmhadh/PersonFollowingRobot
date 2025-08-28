#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        # Paramètres
        self.declare_parameter('video_device', '/dev/video0')
        self.declare_parameter('frame_rate', 30)
        video_device = self.get_parameter('video_device').value
        frame_rate = self.get_parameter('frame_rate').value

        # Publisher ROS
        self.pub_image = self.create_publisher(Image, '/image_raw', 10)
        self.bridge = CvBridge()

        # Ouvrir la caméra
        self.cap = cv2.VideoCapture(video_device)
        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open camera {video_device}")
            exit(1)
        self.cap.set(cv2.CAP_PROP_FPS, frame_rate)

        # Timer pour la capture
        self.timer = self.create_timer(1.0 / frame_rate, self.capture_frame)

    def capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to capture frame")
            return
        # Publier l'image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        self.pub_image.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

