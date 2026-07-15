import os
import shutil
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction, OpaqueFunction, IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import PythonLaunchDescriptionSource

def _prepare_and_launch(context, *args, **kwargs):
    pkg_share = get_package_share_directory('world')
    world_path = os.path.join(pkg_share, 'worlds', 'panda_world.sdf')
    controllers_yaml = os.path.join(pkg_share, 'config', 'panda_ros2_controllers.yaml')

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
        # Start moveit2 (robot_state_publisher) first
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('world'), 'launch', 'moveit2.launch.py')
            )
        ),

        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', tmp_models_root),
        SetEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH', '/opt/ros/jazzy/lib'),

        # Clock bridge starts immediately
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='clock_bridge',
            arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
            output='screen'
        ),

        # Delay Gazebo by 3 seconds to give robot_state_publisher time to publish /robot_description
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=['gz', 'sim', world_path],
                    output='screen'
                ),
            ]
        ),

        # Spawn controllers 5s after Gazebo starts. Each spawner polls/retries
        # for controller_manager's services via --controller-manager-timeout,
        # so this outer delay only needs to avoid spawning before the gz sim
        # process itself has started, not to guess full model-load time.
        TimerAction(
        period=5.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster', '--controller-manager-timeout', '60', '--switch-timeout', '60', '--service-call-timeout', '60'],
                output='screen',
            ),
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['panda_arm_controller', '--controller-manager-timeout', '60', '--switch-timeout', '60', '--service-call-timeout', '60'],
                output='screen',
            ),
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['robotiq_gripper_controller', '--controller-manager-timeout', '60', '--switch-timeout', '60', '--service-call-timeout', '60'],
                output='screen',
            ),
        ]
    ),
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=_prepare_and_launch)
    ])