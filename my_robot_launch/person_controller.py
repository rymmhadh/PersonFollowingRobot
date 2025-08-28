#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import math

class PersonController(Node):
    def __init__(self):
        super().__init__('person_controller')
        self.pub = self.create_publisher(Twist, '/person/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.move_actor)  # 10 Hz

        # Trajectoire rectangulaire
        self.path = [
            (0, 0), (0, 5), (5, 5), (5, 0), (5, -5),
            (0, -5), (-5, -5), (-5, 0), (-5, 5), (0, 5), (0, 0)
        ]
        self.current_wp = 0
        self.speed = 0.2           # vitesse linéaire lente
        self.angular_speed = 0.3   # vitesse angulaire lente
        self.pos = [0.0, 0.0]      # position actuelle simulée
        self.yaw = 0.0             # orientation actuelle simulée

    def move_actor(self):
        twist = Twist()
        target = self.path[self.current_wp]

        # Calcul de l’angle vers le waypoint
        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        target_angle = math.atan2(dy, dx)
        angle_diff = target_angle - self.yaw

        # Normaliser angle entre -pi et pi
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # Tourner d’abord si l’angle est grand
        if abs(angle_diff) > 0.05:
            twist.linear.x = 0.0
            twist.angular.z = self.angular_speed if angle_diff > 0 else -self.angular_speed
            # Met à jour yaw simulé
            self.yaw += twist.angular.z * 0.1
        else:
            twist.linear.x = self.speed
            twist.angular.z = 0.0
            # Met à jour position simulée
            self.pos[0] += twist.linear.x * math.cos(self.yaw) * 0.1
            self.pos[1] += twist.linear.x * math.sin(self.yaw) * 0.1

            # Vérifie si le waypoint est atteint
            if math.hypot(target[0]-self.pos[0], target[1]-self.pos[1]) < 0.1:
                self.current_wp = (self.current_wp + 1) % len(self.path)

        self.pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = PersonController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
