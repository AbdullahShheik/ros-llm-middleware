from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def load_file(package_path, file_path):
    with open(os.path.join(package_path, file_path), 'r') as f:
        return f.read()

def load_yaml(package_path, file_path):
    with open(os.path.join(package_path, file_path), 'r') as f:
        return yaml.safe_load(f)

def generate_launch_description():
    moveit_config_path = get_package_share_directory('moveit_resources_panda_moveit_config')
    panda_description_path = get_package_share_directory('moveit_resources_panda_description')

    # Load robot description
    robot_description = load_file(panda_description_path, 'urdf/panda.urdf')

    # Load SRDF
    robot_description_semantic = load_file(moveit_config_path, 'config/panda.srdf')

    # Load kinematics
    kinematics_yaml = load_yaml(moveit_config_path, 'config/kinematics.yaml')

    # Load planning config
    ompl_yaml = load_yaml(moveit_config_path, 'config/ompl_planning.yaml')

    # Load controllers -- our own copy (world/config/panda_moveit_controllers.yaml)
    # rather than the upstream moveit_resources one, since it adds a
    # panda_hand_controller entry matching panda_ros2_controllers.yaml
    world_pkg_path = get_package_share_directory('world')
    moveit_controllers = load_yaml(world_pkg_path, 'config/panda_moveit_controllers.yaml')

    return LaunchDescription([
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}, {'use_sim_time': True}],
            output='screen'
        ),

        # Static transform
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'panda_link0'],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        # NOTE: /joint_states now comes directly from ros2_control's
        # joint_state_broadcaster (spawned in world.launch.py), which is a
        # real ROS2 publisher running inside the gz_ros2_control-hosted
        # controller_manager. The previous ros_gz_bridge relay of Gazebo's
        # native joint_state topic has been removed since that topic no
        # longer exists (the per-joint JointPositionController plugins and
        # native joint_state_publisher plugin were replaced by gz_ros2_control
        # in models/panda/model.sdf).

        # MoveIt2 move_group
        Node(
            package='moveit_ros_move_group',
            executable='move_group',
            name='move_group',
            parameters=[
                {'robot_description': robot_description},
                {'robot_description_semantic': robot_description_semantic},
                {'robot_description_kinematics': kinematics_yaml},
                {'planning_pipelines': ['ompl']},
                {'ompl': ompl_yaml},
                moveit_controllers,
                {'publish_robot_description_semantic': True},
                {'use_sim_time': True},
            ],
            output='screen'
        ),
    ])