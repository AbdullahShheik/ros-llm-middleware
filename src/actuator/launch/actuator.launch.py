"""
Launches the actuator node with the Franka Panda MoveIt configuration,
using the local panda.srdf that defines the Robotiq 2F-85 hand group.
"""
import os
import re
import subprocess
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
    local_srdf = os.path.join(
        get_package_share_directory("world"), "config", "panda.srdf"
    )

    local_urdf_xacro = os.path.join(
        get_package_share_directory("world"), "urdf", "panda_gz.urdf.xacro"
    )

    # moveit_resources_panda_description ships no arm-only xacro, so the
    # Franka panda_finger_joint1/2 still get emitted even though the
    # fingers are physically replaced by the Robotiq gripper and are not
    # present in Gazebo's model.sdf. Freeze them as fixed joints so
    # CurrentStateMonitor doesn't wait forever for a /joint_states value
    # that will never arrive ("Missing panda_finger_joint1"). Pre-process
    # into a plain URDF since MoveItConfigsBuilder.robot_description() only
    # takes a file path (it runs xacro itself, so we can't hand it the
    # patched string directly).
    robot_description_xml = subprocess.check_output(['xacro', local_urdf_xacro]).decode('utf-8')
    robot_description_xml = robot_description_xml.replace(
        '<joint name="panda_finger_joint1" type="prismatic">',
        '<joint name="panda_finger_joint1" type="fixed">',
    )
    robot_description_xml = robot_description_xml.replace(
        '<joint name="panda_finger_joint2" type="prismatic">',
        '<joint name="panda_finger_joint2" type="fixed">',
    )
    robot_description_xml = re.sub(
        r'<mimic joint="panda_finger_joint1"\s*/>', '', robot_description_xml
    )
    patched_urdf_path = '/tmp/ros_llm_middleware_panda_patched.urdf'
    with open(patched_urdf_path, 'w') as f:
        f.write(robot_description_xml)

    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(file_path=patched_urdf_path)
        .robot_description_semantic(file_path=local_srdf)
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