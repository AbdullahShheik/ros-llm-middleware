import os
import shutil
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable, TimerAction, OpaqueFunction, IncludeLaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch.launch_description_sources import PythonLaunchDescriptionSource

launch_dir = os.path.dirname(os.path.realpath(__file__))
map_file = os.path.join(launch_dir, '..', 'maps', 'panda_world_map.yaml')

def _prepare_and_launch(context, *args, **kwargs):
    pkg_share = get_package_share_directory('world')
    world_path = os.path.join(pkg_share, 'worlds', 'panda_world.sdf')
    controllers_yaml = os.path.join(pkg_share, 'config', 'panda_ros2_controllers.yaml')
    nav2_params_file = os.path.join(pkg_share, 'config', 'nav2_params.yaml')
    bt_xml_path = os.path.join(
        get_package_share_directory('nav2_bt_navigator'),
        'behavior_trees',
        'navigate_to_pose_w_replanning_and_recovery.xml'
    )

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
    #
    # Each cube also has a second DetachableJoint to the TurtleBot (for the
    # mobile-robot pickup feature), which starts attached for the same
    # reason and previously was only released by attach_detach_node.py --
    # a node that is not part of this launch tree and had to be started
    # separately (run_demo.sh's extra tmux window). If it wasn't running,
    # the cube stayed permanently welded to the parked TurtleBot, fighting
    # the arm's own grasp joint. Released here too so both joints start
    # detached regardless of whether attach_detach_node.py is running;
    # that node still attaches/detaches this joint on demand afterwards.
    release_cubes_cmd = ' '.join(
        [
            'for i in 1 2 3 4 5;', 'do',
        ] + [
            f"gz topic -t /model/{cube}/detachable_joint/detach -m gz.msgs.Empty -p '' ;"
            for cube in ('red_cube', 'blue_cube', 'green_cube', 'yellow_cube')
        ] + [
            f"gz topic -t /{cube}/detach -m gz.msgs.Empty -p '' ;"
            for cube in ('red_cube', 'blue_cube', 'green_cube', 'yellow_cube')
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

        # Bridge TurtleBot3 topics between ROS2 and Gazebo
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='turtlebot3_bridge',
            parameters=[{
                'config_file': os.path.join(pkg_share, 'config', 'turtlebot3_bridge.yaml'),
                'qos_overrides./tf_static.publisher.durability': 'transient_local',
            }],
            output='screen',
        ),

        # Static TF: base_footprint → base_link → base_scan
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_footprint_to_base_link',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0.010',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'base_footprint',
                '--child-frame-id', 'base_link'
            ],
            output='screen'
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_base_scan',
            arguments=[
                '--x', '-0.064', '--y', '0', '--z', '0.121',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'base_scan'
            ],
            output='screen'
        ),

        # Delay Gazebo by 3s to give robot_state_publisher time to publish /robot_description
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
                    output='screen',)
            ]
        ),
        # Nav2 stack at 15s: needs Gazebo world loaded + /tf, /scan flowing
        TimerAction(
            period=15.0,
            actions=[
                Node(
                    package='nav2_map_server',
                    executable='map_server',
                    name='map_server',
                    output='screen',
                    parameters=[
                        nav2_params_file,
                        {'yaml_filename': map_file},
                        {'use_sim_time': True},
                    ],
                ),
                Node(
                    package='nav2_amcl',
                    executable='amcl',
                    name='amcl',
                    output='screen',
                    parameters=[nav2_params_file, {'use_sim_time': True}],
                ),
                Node(
                    package='nav2_planner',
                    executable='planner_server',
                    name='planner_server',
                    output='screen',
                    parameters=[nav2_params_file, {'use_sim_time': True}],
                ),
                Node(
                    package='nav2_controller',
                    executable='controller_server',
                    name='controller_server',
                    output='screen',
                    parameters=[nav2_params_file, {'use_sim_time': True}],
                    remappings=[('cmd_vel_nav', 'cmd_vel')],  # internal → bridge topic
                ),
                Node(
                    package='nav2_behaviors',
                    executable='behavior_server',
                    name='behavior_server',
                    output='screen',
                    parameters=[nav2_params_file, {'use_sim_time': True}],
                ),
                Node(
                    package='nav2_bt_navigator',
                    executable='bt_navigator',
                    name='bt_navigator',
                    output='screen',
                    parameters=[
                        nav2_params_file,
                        {
                            'use_sim_time': True,
                            'default_nav_to_pose_bt_xml': bt_xml_path,
                            'default_nav_through_poses_bt_xml': bt_xml_path,
                        },
                    ],
                ),
                Node(
                    package='nav2_waypoint_follower',
                    executable='waypoint_follower',
                    name='waypoint_follower',
                    output='screen',
                    parameters=[nav2_params_file, {'use_sim_time': True}],
                ),
                Node(
                    package='nav2_lifecycle_manager',
                    executable='lifecycle_manager',
                    name='lifecycle_manager_navigation',
                    output='screen',
                    parameters=[nav2_params_file, {'use_sim_time': True}],
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
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['right_knuckle_controller'],
                    output='screen',
                ),
                Node(
                    package='action_dispatcher',
                    executable='gripper_mimic_bridge.py',
                    name='gripper_mimic_bridge',
                    output='screen',
                ),
                Node(
                    package='perception',
                    executable='perception_node.py',
                    name='perception_node',
                    output='screen',
                ),
                Node(
                    package='action_dispatcher',
                    executable='ik_feasibility_service.py',
                    name='ik_feasibility_service',
                    output='screen',
                ),
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
            period=14.5,
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
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(
                        os.path.join(get_package_share_directory('mobile_actuator'), 'launch', 'mobile_actuator.launch.py')
                    )
                ),
            ]
        ),
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=_prepare_and_launch)
    ])