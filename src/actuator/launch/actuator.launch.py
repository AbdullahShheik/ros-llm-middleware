"""
Launches the actuator node with the Franka Panda MoveIt configuration,
using the local panda.srdf that defines the Robotiq 2F-85 hand group.

Parametrized by the `arm_id` launch argument ("1" default, or "2") so this
same file launches either arm's actuator_node.py instance -- arm 2's
robot_description/robot_description_semantic are the SAME xacro/SRDF
sources as arm 1, text-renamed panda_ -> panda2_ (see
_rename_for_second_arm, duplicated from world/launch/moveit2.launch.py:
that file combines both arms into one shared-scene move_group for a
different, non-actuator consumer, so it can't be imported directly from
this package -- these are two independent single-arm MoveItPy configs, one
per actuator_node.py process, coordinated instead via the explicit
shared-workspace lock in actuator_node.py).
"""
import os
import re
import subprocess
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def load_yaml_file(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _rename_for_second_arm(text: str) -> str:
    """See world/launch/moveit2.launch.py's function of the same name --
    kept identical and duplicated here rather than shared, since launch
    files aren't importable across packages without a library target."""
    text = text.replace('end_effector_frame_fixed_joint', 'panda2_end_effector_frame_fixed_joint')
    text = text.replace('"hand"', '"hand2"')
    text = text.replace('name="virtual_joint"', 'name="virtual_joint2"')
    text = text.replace('panda_', 'panda2_')
    text = text.replace('robotiq_85_', 'robotiq2_85_')
    return text


def _generate(context, *args, **kwargs):
    arm_id = LaunchConfiguration('arm_id').perform(context)
    is_arm2 = (arm_id == "2")

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

    with open(local_srdf, 'r') as f:
        srdf_text = f.read()

    node_name = "actuator_node"
    patched_urdf_path = '/tmp/ros_llm_middleware_panda_patched.urdf'
    patched_srdf_path = local_srdf

    if is_arm2:
        node_name = "actuator_node2"
        robot_description_xml = _rename_for_second_arm(robot_description_xml)
        srdf_text = _rename_for_second_arm(srdf_text)
        patched_urdf_path = '/tmp/ros_llm_middleware_panda2_patched.urdf'
        patched_srdf_path = '/tmp/ros_llm_middleware_panda2_patched.srdf'
        with open(patched_srdf_path, 'w') as f:
            f.write(srdf_text)

    with open(patched_urdf_path, 'w') as f:
        f.write(robot_description_xml)

    moveit_config_builder = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(file_path=patched_urdf_path)
        .robot_description_semantic(file_path=patched_srdf_path)
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
    moveit_config = moveit_config_builder.to_dict()

    if is_arm2:
        # kinematics.yaml/ompl_planning.yaml above are the stock
        # moveit_resources_panda_moveit_config files, which only define the
        # "panda_arm" group -- the SRDF's rename above means arm 2's group
        # is "panda2_arm" (see actuator_node.py's configure_for_arm), so a
        # matching entry has to be added under each. Same reuse-not-retune
        # reasoning as moveit2.launch.py's identical injection: same solver
        # tuning/planner list, just keyed by the renamed group.
        moveit_config['robot_description_kinematics']['panda2_arm'] = dict(
            moveit_config['robot_description_kinematics']['panda_arm']
        )
        moveit_config['ompl']['panda2_arm'] = {
            'planner_configs': list(moveit_config['ompl']['panda_arm']['planner_configs'])
        }
        # moveit_cpp.yaml's default joint_state_topic ("/joint_states") is
        # arm 1's own joint_state_broadcaster's topic -- gz_ros2_control's
        # namespaced controller_manager for arm 2 (SDF's
        # <ros><namespace>panda2</namespace>) publishes its joint states to
        # /panda2/joint_states instead, so MoveItPy's planning scene monitor
        # has to be pointed there or it never sees a real joint state and
        # refuses to plan (same reasoning as actuator_node.py's own
        # JOINT_STATES_TOPIC, a separate subscription in the same process).
        moveit_config['planning_scene_monitor_options']['joint_state_topic'] = (
            '/panda2/joint_states'
        )

    controllers_config = load_yaml_file(custom_moveit_controllers)

    actuator_node = Node(
        package="actuator",
        executable="actuator_node.py",
        name=node_name,
        output="screen",
        parameters=[
            moveit_config,
            controllers_config,
            {"use_sim_time": True},
        ],
        ros_arguments=['--ros-args', '-p', 'use_sim_time:=true'],
        additional_env={'ACTUATOR_ARM_ID': arm_id},
    )

    return [actuator_node]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('arm_id', default_value='1'),
        OpaqueFunction(function=_generate),
    ])
