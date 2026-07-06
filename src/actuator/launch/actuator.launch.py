"""
Launches the actuator node with the Franka Panda MoveIt configuration.
"""
import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder

def load_yaml_file(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def generate_launch_description():
    custom_moveit_controllers = os.path.join(
        get_package_share_directory("world"), "config", "panda_moveit_controllers.yaml"
    )

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(file_path="config/panda.urdf.xacro")
        .robot_description_semantic(file_path="config/panda.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path=custom_moveit_controllers)
        .planning_pipelines("ompl", ["ompl"])
        .moveit_cpp(
            file_path=os.path.join(
                get_package_share_directory("actuator"), "config", "moveit_cpp.yaml"
            )
        )
        .to_moveit_configs()
    )

    controllers_config = load_yaml_file(custom_moveit_controllers)

    actuator_node = Node(
        package="actuator",
        executable="actuator_node.py",
        name="actuator_node",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            controllers_config,
            {"use_sim_time": True},
        ],
        ros_arguments=['--ros-args', '-p', 'use_sim_time:=true'],
    )

    return LaunchDescription([actuator_node])