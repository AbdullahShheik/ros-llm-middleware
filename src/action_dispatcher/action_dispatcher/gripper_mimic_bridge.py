#!/usr/bin/env python3
"""
gripper_mimic_bridge.py

Mirrors robotiq_85_left_knuckle_joint's position onto
robotiq_85_right_knuckle_joint in software, via right_knuckle_controller
(a ForwardCommandController). This replaces the physics-engine-level
SDF <mimic> constraint, which dartsim does not support (and which
gz_ros2_control's own <param name="mimic"> just wires through to
anyway, so it doesn't help under dartsim either).

multiplier = -1 matches the URDF's own mimic multiplier for
robotiq_85_right_knuckle_joint (see robotiq_2f_85_macro.urdf.xacro).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

LEFT_JOINT = 'robotiq_85_left_knuckle_joint'
MULTIPLIER = -1.0
COMMAND_TOPIC = '/right_knuckle_controller/commands'


class GripperMimicBridge(Node):
    def __init__(self):
        super().__init__('gripper_mimic_bridge')
        self.pub = self.create_publisher(Float64MultiArray, COMMAND_TOPIC, 10)
        self.sub = self.create_subscription(
            JointState, '/joint_states', self.on_joint_states, 10
        )
        self.get_logger().info(
            f'Mirroring {LEFT_JOINT} -> {COMMAND_TOPIC} (multiplier={MULTIPLIER})'
        )

    def on_joint_states(self, msg: JointState):
        if LEFT_JOINT not in msg.name:
            return
        idx = msg.name.index(LEFT_JOINT)
        left_position = msg.position[idx]

        out = Float64MultiArray()
        out.data = [left_position * MULTIPLIER]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = GripperMimicBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()