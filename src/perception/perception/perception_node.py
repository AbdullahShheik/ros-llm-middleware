#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
from ament_index_python.packages import get_package_share_directory
from gz.transport13 import Node as GzNode
from gz.msgs10.pose_v_pb2 import Pose_V


from world_model.scene_tracking import get_tracked_names

# Resolved through the ament index rather than relative to __file__: this
# node runs from the install space (lib/perception/), which has no fixed
# relative path back to the source tree. This is also the exact same file
# world.launch.py hands to Gazebo, so the entity classification here is
# always derived from the world actually being simulated.
SDF_PATH = os.path.join(
    get_package_share_directory("world"), "worlds", "panda_world.sdf"
)


# Gazebo's pose/info topic is unthrottled and reports every LINK in the
# world (the Panda alone has 19), not just top-level objects/zones -- far
# too high a rate and too noisy to republish 1:1. pose_callback only
# updates in-memory state; this timer publishes the latest snapshot at a
# bounded rate instead, decoupling /object_map's publish rate from
# whatever rate Gazebo happens to stream at.
PUBLISH_PERIOD_SEC = 0.2


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.publisher = self.create_publisher(String, '/object_map', 10)

        self.tracked_objects, self.tracked_zones, self.tracked_robots = get_tracked_names(SDF_PATH)
        self.get_logger().info(
            f"Tracking {len(self.tracked_objects)} object(s) "
            f"{sorted(self.tracked_objects)}, {len(self.tracked_zones)} "
            f"zone(s) {sorted(self.tracked_zones)}, and {len(self.tracked_robots)} "
            f"robot(s) {sorted(self.tracked_robots)} (derived from {SDF_PATH})"
        )

        # Latest known pose per tracked name. Overwritten on every Gazebo
        # pose update (pose_callback) but only actually published on the
        # timer below (publish_object_map) -- see PUBLISH_PERIOD_SEC.
        self.latest_state = {}
        # Fires once: lets you confirm on a real run that "panda" and
        # "turtlebot3_waffle" actually appear as their own pose.name entries
        # in this topic (separate from their child links), rather than
        # assuming it based on how single-link objects like red_cube work.
        self._logged_first_pose_names = False

        self.gz_node = GzNode()
        self.gz_node.subscribe(
            Pose_V,
            '/world/panda_world/pose/info',
            self.pose_callback
        )
        self.create_timer(PUBLISH_PERIOD_SEC, self.publish_object_map)
        self.get_logger().info('Perception node started, listening to Gazebo...')

    def pose_callback(self, msg):
        """Gazebo callback -- fires at Gazebo's own (high, unthrottled)
        update rate. Only updates in-memory state; never publishes directly."""
        if not self._logged_first_pose_names:
            self._logged_first_pose_names = True
            self.get_logger().info(
                f"First Gazebo pose message has {len(msg.pose)} entries: "
                f"{sorted(p.name for p in msg.pose)}"
            )

        for pose in msg.pose:
            if (pose.name in self.tracked_objects
                    or pose.name in self.tracked_zones
                    or pose.name in self.tracked_robots):
                self.latest_state[pose.name] = {
                    "x": round(pose.position.x, 4),
                    "y": round(pose.position.y, 4),
                    "z": round(pose.position.z, 4),
                }

    def publish_object_map(self):
        """Timer callback -- publishes whatever the latest known state is,
        at a bounded rate independent of Gazebo's pose-update frequency."""
        if not self.latest_state:
            return
        out = String()
        out.data = json.dumps(self.latest_state)
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