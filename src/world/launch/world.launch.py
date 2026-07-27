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

        # Nav2 stack at 8s: needs Gazebo world loaded + /tf, /scan flowing from turtlebot3_bridge
        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package='nav2_map_server',
                    executable='map_server',
                    name='map_server',
                    output='screen',
                    parameters=[
                        nav2_params_file,
                        {'yaml_filename': os.path.join(pkg_share, 'maps', 'panda_world_map.yaml')},
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
                    remappings=[('cmd_vel', 'cmd_vel_nav')],
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
                    package='nav2_velocity_smoother',
                    executable='velocity_smoother',
                    name='velocity_smoother',
                    output='screen',
                    parameters=[nav2_params_file, {'use_sim_time': True}],
                    remappings=[('cmd_vel', 'cmd_vel_nav')],
                ),
                Node(
                    package='nav2_collision_monitor',
                    executable='collision_monitor',
                    name='collision_monitor',
                    output='screen',
                    parameters=[nav2_params_file, {'use_sim_time': True}],
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

        # Delay Gazebo by 3 seconds to give robot_state_publisher time to publish /robot_description
        TimerAction(
            period=3.0,
            actions=[
                ExecuteProcess(
                    cmd=['gz', 'sim', '-r', '-v', '4', world_path],
                    output='screen'
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
            parameters=[{
                'config_file': os.path.join(pkg_share, 'config', 'turtlebot3_bridge.yaml'),
                'qos_overrides./tf_static.publisher.durability': 'transient_local',
            }],
            output='screen',
        ),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_footprint_to_base_link',
            arguments=['0', '0', '0.010', '0', '0', '0', 'base_footprint', 'base_link'],
            output='screen'
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_base_scan',
            arguments=['-0.064', '0', '0.121', '0', '0', '0', 'base_link', 'base_scan'],
            output='screen'
        ),
    ]

def generate_launch_description():
    return LaunchDescription([
        OpaqueFunction(function=_prepare_and_launch)
    ])