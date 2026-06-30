from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('world')
    world_path = os.path.join(
        get_package_share_directory('world'),
        'worlds',
        'panda_world.sdf'
    )
    models_path = os.path.join(pkg_share, 'models')

    return LaunchDescription([
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', models_path),

        ExecuteProcess(
            cmd=['gz', 'sim', world_path],
            output='screen'
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='clock_bridge',
            arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
            output='screen'
        ),

        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='joint_state_bridge',
            arguments=['/world/panda_world/model/panda/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model'],
            output='screen'
        )
    ])