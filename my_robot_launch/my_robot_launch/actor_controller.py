#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Twist
import math
import time


class ActorController(Node):
    def __init__(self):
        super().__init__('actor_controller')

        # Client pour SetEntityState
        self.cli = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("En attente du service /gazebo/set_entity_state...")
        self.get_logger().info("Service disponible.")
        self.req = SetEntityState.Request()

    def yaw_to_quat(self, yaw):
        """Convertit un yaw en quaternion (x, y, z, w)."""
        return (0.0, 0.0, math.sin(yaw/2.0), math.cos(yaw/2.0))

    def set_pose(self, name, x, y, z=0.0, yaw=0.0, vx=0.0, vy=0.0, vz=0.0):
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        qx, qy, qz, qw = self.yaw_to_quat(yaw)
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw

        twist = Twist()
        twist.linear.x = vx
        twist.linear.y = vy
        twist.linear.z = vz

        state = EntityState()
        state.name = name
        state.pose = pose
        state.twist = twist
        state.reference_frame = "world"

        self.req.state = state
        future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, future)

    def lerp(self, start, end, t):
        return start + (end - start) * t

    def move_smooth(self, start, end, duration, steps=50):
        """Déplace l’acteur de start à end en steps étapes sur duration secondes."""
        x0, y0, yaw0 = start
        x1, y1, yaw1 = end
        for i in range(steps):
            t = (i+1)/steps
            x = self.lerp(x0, x1, t)
            y = self.lerp(y0, y1, t)
            yaw = self.lerp(yaw0, yaw1, t)
            self.set_pose('actor1', x, y, 0.0, yaw)
            time.sleep(duration/steps)

    def run_scenario(self):
        actor_name = "actor1"

        # 0–30s : acteur immobile
        self.get_logger().info("0–30s : acteur immobile")
        self.set_pose(actor_name, 0.0, 0.0, 0.0, 0.0)
        time.sleep(30)

        # 30–40s : tourner progressivement en face du robot puis dos
        self.get_logger().info("30–40s : rotation progressive de l'acteur")
        self.move_smooth((0.0, 0.0, 0.0), (0.0, 0.0, math.pi), duration=10.0)

        # 40–100s : trajectoire rectangulaire
        self.get_logger().info("40–100s : trajectoire rectangulaire")
        trajectory = [
            (0.0, 0.0, math.pi),      # départ
            (2.0, 0.0, math.pi/2),
            (2.0, 2.0, 0.0),
            (0.0, 2.0, -math.pi/2),
            (0.0, 0.0, math.pi)
        ]

        for i in range(len(trajectory)-1):
            self.move_smooth(trajectory[i], trajectory[i+1], duration=15.0)

        self.get_logger().info("Trajectoire terminée, acteur continue à marcher en place")

        # Si tu veux que l’acteur continue à marcher
        for i in range(20):
            self.set_pose(actor_name, 0.0, 0.0, 0.0, math.pi)
            time.sleep(1)


def main(args=None):
    rclpy.init(args=args)
    node = ActorController()
    try:
        node.run_scenario()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
