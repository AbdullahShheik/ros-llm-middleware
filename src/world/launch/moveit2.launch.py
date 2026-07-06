from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import yaml
import subprocess

def load_yaml(package_path, file_path):
    with open(os.path.join(package_path, file_path), 'r') as f:
        return yaml.safe_load(f)

def generate_launch_description():
    moveit_config_path = get_package_share_directory('moveit_resources_panda_moveit_config')
    world_pkg_path = get_package_share_directory('world')

    # Process xacro to get URDF with ros2_control tags included
    xacro_file = os.path.join(get_package_share_directory('world'), 'urdf', 'panda_gz.urdf.xacro')
    robot_description = subprocess.check_output(
        ['xacro', xacro_file]
    ).decode('utf-8')

    # Load SRDF
    robot_description_semantic = load_yaml.__class__  # placeholder
    with open(os.path.join(moveit_config_path, 'config', 'panda.srdf'), 'r') as f:
        robot_description_semantic = f.read()

    # Load kinematics
    kinematics_yaml = load_yaml(moveit_config_path, 'config/kinematics.yaml')

    # Load planning config
    ompl_yaml = load_yaml(moveit_config_path, 'config/ompl_planning.yaml')

    # Load controllers
    moveit_controllers = load_yaml(world_pkg_path, 'config/panda_moveit_controllers.yaml')

    return LaunchDescription([
        # Robot state publisher with xacro-processed URDF
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[
                {'robot_description': robot_description},
                {'use_sim_time': True}
            ],
            output='screen'
        ),

        # Static transform world -> panda_link0
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'panda_link0'],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

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