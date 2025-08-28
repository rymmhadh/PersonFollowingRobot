#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class PersonFollowNode(Node):
    def __init__(self):
        super().__init__('person_follow_node')
        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        # Charger un détecteur simple de personne
        self.person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        people = self.person_cascade.detectMultiScale(gray, 1.1, 3)

        twist = Twist()
        if len(people) > 0:
            x, y, w, h = people[0]
            cx = x + w // 2
            cy = y + h // 2
            # Déplacement en face
            error_x = cx - frame.shape[1] // 2
            twist.linear.x = 0.2 if abs(error_x) < 50 else 0.0
            twist.angular.z = -0.002 * error_x
        self.pub_cmd.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = PersonFollowNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

