#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
from gz.transport13 import Node as GzNode
from gz.msgs10.pose_v_pb2 import Pose_V

# Objects we care about
TRACKED_OBJECTS = {"red_cube", "blue_cube", "green_cube"}

# Named places for mobile robot navigation
TRACKED_LOCATIONS = {"drop_zone", "red_zone", "blue_zone", "green_zone", "yellow_zone", "handoff_point"}

TRACKED_ROBOTS = {"turtlebot3_waffle"}

class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.publisher = self.create_publisher(String, '/object_map', 10)
        self.gz_node = GzNode()
        self.gz_node.subscribe(
            Pose_V,
            '/world/panda_world/pose/info',
            self.pose_callback
        )
        self.get_logger().info('Perception node started, listening to Gazebo...')

    def pose_callback(self, msg):
        object_map = {}
        for pose in msg.pose:
            if pose.name in TRACKED_OBJECTS or pose.name in TRACKED_LOCATIONS or pose.name in TRACKED_ROBOTS:
                object_map[pose.name] = {
                    "x": round(pose.position.x, 4),
                    "y": round(pose.position.y, 4),
                    "z": round(pose.position.z, 4)
                }
        if object_map:
            out = String()
            out.data = json.dumps(object_map)
            self.publisher.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()