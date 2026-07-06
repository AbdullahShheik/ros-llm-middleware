"""
Launches the actuator node with the Franka Panda MoveIt configuration.

moveit_py needs the full MoveIt config (URDF, SRDF, kinematics, planning
pipeline, controllers) available as ROS parameters on its own node. We
build that config here with MoveItConfigsBuilder -- pointed at the same
moveit_resources_panda_moveit_config / moveit_resources_panda_description
packages used in world/launch/moveit2.launch.py -- and pass it to the
actuator node as parameters.

Run alongside world.launch.py and moveit2.launch.py (move_group is still
needed for the /compute_ik service used by ik_feasibility_service.py):

  ros2 launch world world.launch.py
  ros2 launch world moveit2.launch.py
  ros2 launch actuator actuator.launch.py
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("moveit_resources_panda")
        .robot_description(file_path="config/panda.urdf.xacro")
        .robot_description_semantic(file_path="config/panda.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .trajectory_execution(file_path="config/moveit_controllers.yaml")
        .planning_pipelines("ompl", ["ompl"])
        .moveit_cpp(
            file_path=os.path.join(
                get_package_share_directory("actuator"), "config", "moveit_cpp.yaml"
            )
        )
        .to_moveit_configs()
    )

    actuator_node = Node(
        package="actuator",
        executable="actuator_node.py",
        name="actuator_node",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": True},
        ],
    )

    return LaunchDescription([actuator_node])