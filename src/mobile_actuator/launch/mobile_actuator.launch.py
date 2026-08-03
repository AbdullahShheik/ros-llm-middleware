from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="mobile_actuator",
            executable="mobile_actuator_node.py",
            name="mobile_actuator_node",
            output="screen",
            emulate_tty=True,
        ),
    ])