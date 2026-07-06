import os
import shutil

from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction, OpaqueFunction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _prepare_and_launch(context, *args, **kwargs):
    pkg_share = get_package_share_directory('world')

    world_path = os.path.join(pkg_share, 'worlds', 'panda_world.sdf')
    controllers_yaml = os.path.join(pkg_share, 'config', 'panda_ros2_controllers.yaml')

    # SDF has no notion of a ROS package path, so models/panda/model.sdf ships
    # with a literal placeholder in its gz_ros2_control <parameters> tag.
    # Substitute the real absolute path into a scratch copy of the model
    # rather than editing the installed share directory in place.
    tmp_models_root = '/tmp/ros_llm_middleware_gz_models'
    if os.path.exists(tmp_models_root):
        shutil.rmtree(tmp_models_root)
    shutil.copytree(os.path.join(pkg_share, 'models'), tmp_models_root)

    patched_model_sdf = os.path.join(tmp_models_root, 'panda', 'model.sdf')
    with open(patched_model_sdf, 'r') as f:
        content = f.read()
    content = content.replace('__PANDA_ROS2_CONTROLLERS_YAML__', controllers_yaml)
    with open(patched_model_sdf, 'w') as f:
        f.write(content)

    return [
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', tmp_models_root),
        SetEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', '/opt/ros/jazzy/lib'),

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

        # joint_states now comes directly from ros2_control's
        # joint_state_broadcaster (a real ROS2 publisher inside the
        # gz_ros2_control-hosted controller_manager), so the old
        # ros_gz_bridge relay of Gazebo's native joint_state topic is gone --
        # keeping both would double-publish /joint_states.

        # Give gz_ros2_control's embedded controller_manager a few seconds to
        # come up inside the gz sim process before spawning controllers
        # against it.
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['joint_state_broadcaster'],
                    output='screen',
                ),
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['panda_arm_controller'],
                    output='screen',
                ),
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['panda_hand_controller'],
                    output='screen',
                ),
            ]
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=_prepare_and_launch)
    ])