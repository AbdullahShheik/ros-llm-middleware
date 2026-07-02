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

    # Load controllers
    moveit_controllers = load_yaml(moveit_config_path, 'config/moveit_controllers.yaml')

    return LaunchDescription([
        # Robot state publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            parameters=[{'robot_description': robot_description}],
            output='screen'
        ),

        # Static transform
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'panda_link0'],
            output='screen'
        ),

        # Relay Gazebo joint states to /joint_states
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='joint_state_relay',
            arguments=['/world/panda_world/model/panda/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model'],
            remappings=[
                ('/world/panda_world/model/panda/joint_state', '/joint_states')
            ],
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
            ],
            output='screen'
        ),
    ])