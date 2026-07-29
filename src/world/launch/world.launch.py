import os
import shutil
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction, OpaqueFunction, IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, get_package_prefix
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

    gz_sim_vendor_prefix = get_package_prefix('gz_sim_vendor')

    # gz-sim's DetachableJoint system (panda_world.sdf's per-cube grasp-
    # stabilization plugin) has no "start detached" option -- every cube
    # starts rigidly joined to the gripper the instant the world loads.
    # panda_world.sdf now starts the world paused (<start_paused>true</...>
    # below) precisely so nothing -- including the arm, which has no
    # active ros2_control controller until the spawners in the 15s
    # TimerAction -- can move before we're ready; without that, unpaused
    # physics would let the arm sag under gravity into an arbitrary
    # configuration and drag every still-attached cube along with it,
    # long before anything released them. Release the cubes here anyway
    # (while still paused, so nothing has had a chance to move yet) rather
    # than waiting for actuator_node.py's own defensive release, which
    # doesn't run until the 20s TimerAction. Repeated over ~1s (each
    # `gz topic` invocation is a fresh process that already waits briefly
    # for its own subscriber discovery) for reliability.
    release_cubes_cmd = ' '.join(
        [
            'for i in 1 2 3 4 5;', 'do',
        ] + [
            f"gz topic -t /model/{cube}/detachable_joint/detach -m gz.msgs.Empty -p '' ;"
            for cube in ('red_cube', 'blue_cube', 'green_cube')
        ] + [
            'sleep 0.2;', 'done'
        ]
    )

    # Unpauses the world once the arm controllers below are actually
    # active, so physics only ever starts stepping once something is
    # already holding the arm in place -- see release_cubes_cmd above and
    # panda_world.sdf's <start_paused> for why it must not run free before
    # this point.
    unpause_world_cmd = [
        'gz', 'service', '-s', '/world/panda_world/control',
        '--reqtype', 'gz.msgs.WorldControl',
        '--reptype', 'gz.msgs.Boolean',
        '--timeout', '2000',
        '--req', 'pause: false',
    ]

    return [
        # Start moveit2 (robot_state_publisher) first
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory('world'), 'launch', 'moveit2.launch.py')
            )
        ),

        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH',
            tmp_models_root + ':' + os.path.join(get_package_share_directory('turtlebot3_gazebo'), 'models')),
        SetEnvironmentVariable('GZ_SIM_SYSTEM_PLUGIN_PATH',
            os.path.join(gz_sim_vendor_prefix, 'lib') + ':' + os.path.join(gz_sim_vendor_prefix, 'lib', 'gz-sim-8', 'plugins')),

        # Clock bridge starts immediately
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='clock_bridge',
            arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
            output='screen'
        ),

        # Delay Gazebo by 3 seconds to give robot_state_publisher time to
        # publish /robot_description. No -r: panda_world.sdf's
        # <start_paused>true</start_paused> keeps physics from stepping at
        # all until unpause_world_cmd runs below, once the arm controllers
        # are actually active.
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=['gz', 'sim', world_path],
                    output='screen'
                ),
            ]
        ),

        # Release every cube's DetachableJoint at 6s (3s after Gazebo
        # starts at 3s) -- see release_cubes_cmd above for why this has to
        # happen this early, well before the 15s controller spawn.
        TimerAction(
            period=6.0,
            actions=[
                ExecuteProcess(
                    cmd=['bash', '-c', release_cubes_cmd],
                    output='screen',
                ),
            ]
        ),

        # Spawn controllers + start perception and dispatcher at 15s
        TimerAction(
            period=15.0,
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
                    arguments=['robotiq_gripper_controller'],
                    output='screen',
                ),
                # Perception: reads Gazebo poses, publishes /object_map
                Node(
                    package='perception',
                    executable='perception_node.py',
                    name='perception_node',
                    output='screen',
                ),
                # IK feasibility service
                Node(
                    package='action_dispatcher',
                    executable='ik_feasibility_service.py',
                    name='ik_feasibility_service',
                    output='screen',
                ),
                # Action dispatcher
                Node(
                    package='action_dispatcher',
                    executable='dispatcher_node.py',
                    name='action_dispatcher',
                    output='screen',
                ),
            ]
        ),

        # Unpause at 16.5s: 1.5s after the controller spawners kick off at
        # 15s, giving them time to actually load/configure/activate before
        # physics starts stepping (see unpause_world_cmd above).
        TimerAction(
            period=16.5,
            actions=[
                ExecuteProcess(
                    cmd=unpause_world_cmd,
                    output='screen',
                ),
            ]
        ),

        # Actuator at 20s: needs controllers active first
        TimerAction(
            period=20.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(get_package_share_directory('actuator'), 'launch', 'actuator.launch.py')
                    )
                ),
            ]
        ),

        # Bridge TurtleBot3 cmd_vel and odom between ROS2 and Gazebo
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='turtlebot3_bridge',
            arguments=[
                '/cmd_vel@geometry_msgs/msg/TwistStamped]gz.msgs.Twist',
                '/model/turtlebot3_waffle/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
                '/model/turtlebot3_waffle/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            ],
            output='screen',
        ),
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=_prepare_and_launch)
    ])