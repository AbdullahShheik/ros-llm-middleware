#!/usr/bin/env python3
"""
gripper_mimic_bridge.py

Mirrors a gripper's left knuckle joint position onto its right knuckle
joint in software, via a ForwardCommandController. This replaces the
physics-engine-level SDF <mimic> constraint, which dartsim does not
support (and which gz_ros2_control's own <param name="mimic"> just wires
through to anyway, so it doesn't help under dartsim either).

Parametrized (node params, not hardcoded) so the same script serves
either arm -- launched twice, once per arm, with different values --
rather than needing a near-duplicate second file. Defaults match the
first arm's names exactly, so an unparametrized launch is unchanged
from before this was parametrized.

multiplier = -1 matches the URDF's own mimic multiplier for the right
knuckle joint (see robotiq_2f_85_macro.urdf.xacro).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

MULTIPLIER = -1.0


class GripperMimicBridge(Node):
    def __init__(self):
        super().__init__('gripper_mimic_bridge')
        self.declare_parameter('left_joint', 'robotiq_85_left_knuckle_joint')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('command_topic', '/right_knuckle_controller/commands')

        self.left_joint = self.get_parameter('left_joint').value
        joint_states_topic = self.get_parameter('joint_states_topic').value
        command_topic = self.get_parameter('command_topic').value

        self.pub = self.create_publisher(Float64MultiArray, command_topic, 10)
        self.sub = self.create_subscription(
            JointState, joint_states_topic, self.on_joint_states, 10
        )
        self.get_logger().info(
            f'Mirroring {self.left_joint} ({joint_states_topic}) -> '
            f'{command_topic} (multiplier={MULTIPLIER})'
        )

    def on_joint_states(self, msg: JointState):
        if self.left_joint not in msg.name:
            return
        idx = msg.name.index(self.left_joint)
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