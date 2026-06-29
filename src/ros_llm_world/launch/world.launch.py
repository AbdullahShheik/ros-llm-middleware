from launch import LaunchDescription
from launch.actions import ExecuteProcess
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    world_path = os.path.join(
        get_package_share_directory('ros_llm_world'),
        'worlds',
        'panda_world.sdf'
    )

    return LaunchDescription([
        ExecuteProcess(
            cmd=['gz', 'sim', world_path],
            output='screen'
        )
    ])