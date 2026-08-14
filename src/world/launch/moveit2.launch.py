from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import re
import yaml
import subprocess

def load_yaml(package_path, file_path):
    with open(os.path.join(package_path, file_path), 'r') as f:
        return yaml.safe_load(f)


def _rename_for_second_arm(text: str) -> str:
    """Applies the SAME panda_ -> panda2_ / robotiq_85_ -> robotiq2_85_
    rename used in models/panda2/model.sdf to URDF/SRDF text derived from
    the exact same source as the first arm, so MoveIt sees two distinctly
    named groups/links/joints instead of two copies of the same names.
    Order matters: the bare-word special cases (which don't start with
    "panda_"/"robotiq_85_" and would otherwise collide with unrelated
    substrings) must be handled before the blanket prefix renames.
    """
    text = text.replace('end_effector_frame_fixed_joint', 'panda2_end_effector_frame_fixed_joint')
    # "hand"/"virtual_joint" are the only bare (unprefixed) semantic names
    # in panda.srdf -- every other name already starts with panda_ or
    # robotiq_85_, covered by the blanket renames below. Targeted so they
    # don't touch unrelated substrings (there are no other quoted "hand"
    # attribute values in the file).
    text = text.replace('"hand"', '"hand2"')
    text = text.replace('name="virtual_joint"', 'name="virtual_joint2"')
    text = text.replace('panda_', 'panda2_')
    text = text.replace('robotiq_85_', 'robotiq2_85_')
    return text


def _strip_robot_wrapper(text: str) -> str:
    """Removes the outer <robot ...>...</robot> wrapper, returning just the
    inner body, so two single-arm descriptions can be merged under one
    combined <robot> root (one shared planning scene, not two disconnected
    ones -- see this file's header comment for why)."""
    start = text.index('>', text.index('<robot')) + 1
    end = text.rindex('</robot>')
    return text[start:end]


def generate_launch_description():
    # Both arms are combined into ONE robot_description/robot_description_semantic
    # served by a single move_group, rather than two fully separate MoveIt
    # stacks. This is deliberate: it makes MoveIt's own collision checking
    # treat each arm's current live pose as an obstacle when planning a move
    # for the OTHER arm, for free -- the "perfect coordination" the second
    # arm was added for, without a hand-rolled mutual-exclusion mechanism.
    moveit_config_path = get_package_share_directory('moveit_resources_panda_moveit_config')
    world_pkg_path = get_package_share_directory('world')

    # Process xacro to get URDF with ros2_control tags included
    xacro_file = os.path.join(get_package_share_directory('world'), 'urdf', 'panda_gz.urdf.xacro')
    robot_description_arm1 = subprocess.check_output(
        ['xacro', xacro_file]
    ).decode('utf-8')

    # Debug: verify ros2_control is present
    import sys
    count = robot_description_arm1.count('ros2_control')
    print(f'[DEBUG] robot_description (arm1) has {count} ros2_control tags', file=sys.stderr)

    # moveit_resources_panda_description ships no arm-only xacro, so the
    # Franka panda_finger_joint1/2 still get emitted even though the
    # fingers are physically replaced by the Robotiq gripper and are not
    # present in Gazebo's model.sdf. Freeze them as fixed joints so
    # CurrentStateMonitor doesn't wait forever for a /joint_states value
    # that will never arrive ("Missing panda_finger_joint1").
    robot_description_arm1 = robot_description_arm1.replace(
        '<joint name="panda_finger_joint1" type="prismatic">',
        '<joint name="panda_finger_joint1" type="fixed">',
    )
    robot_description_arm1 = robot_description_arm1.replace(
        '<joint name="panda_finger_joint2" type="prismatic">',
        '<joint name="panda_finger_joint2" type="fixed">',
    )
    robot_description_arm1 = re.sub(r'<mimic joint="panda_finger_joint1"\s*/>', '', robot_description_arm1)

    # Second arm: same xacro source, renamed -- not a second xacro process
    # (the upstream panda.urdf.xacro has no prefix/macro support to call
    # twice; confirmed by inspecting it directly, zero xacro:macro/xacro:arg
    # tags in 288 lines). Text-level rename is lower-risk than hand-authoring
    # a second static URDF fragment.
    robot_description_arm2 = _rename_for_second_arm(robot_description_arm1)

    # Concatenating both arms' bodies under one <robot> root is not enough
    # on its own: robot_state_publisher/move_group parse robot_description
    # into a single kinematic TREE, and with no joint connecting them,
    # panda_link0 and panda2_link0 are each their own disconnected root --
    # confirmed live ("Failed to find root link: Two root links found:
    # [panda2_link0] and [panda_link0]", both robot_state_publisher and
    # move_group crash on startup). A fixed joint below welds panda2_link0
    # under panda_link0 at their true relative offset (0, 0.6, 0) -- same
    # number as the world->panda2_link0 TF transform this file used to
    # publish separately; that static_transform_publisher is removed below
    # since robot_state_publisher now derives (and publishes) that same
    # transform from this joint instead.
    mount_joint = (
        '<joint name="panda2_mount_joint" type="fixed">'
        '<parent link="panda_link0"/><child link="panda2_link0"/>'
        '<origin xyz="0 0.6 0" rpy="0 0 0"/>'
        '</joint>'
    )

    robot_description = (
        '<?xml version="1.0"?>\n<robot name="panda_dual">'
        + _strip_robot_wrapper(robot_description_arm1)
        + _strip_robot_wrapper(robot_description_arm2)
        + mount_joint
        + '</robot>'
    )

    # Load SRDF -- use the world package's Robotiq-adapted SRDF (defines the
    # Robotiq 2F-85 "hand" group), not the stock moveit_resources SRDF which
    # references the Franka hand's panda_finger_joint1/2. Those joints don't
    # exist in this robot's URDF (replaced by the Robotiq gripper), so
    # loading the stock SRDF here left move_group's CurrentStateMonitor
    # waiting forever for a /joint_states value that never arrives
    # ("Missing panda_finger_joint1"), while actuator_node.py (which already
    # loads the correct SRDF from world_pkg_path) worked fine.
    with open(os.path.join(world_pkg_path, 'config', 'panda.srdf'), 'r') as f:
        srdf_arm1 = f.read()
    srdf_arm2 = _rename_for_second_arm(srdf_arm1)
    robot_description_semantic = (
        '<?xml version="1.0" encoding="utf-8"?>\n<robot name="panda_dual">'
        + _strip_robot_wrapper(srdf_arm1)
        + _strip_robot_wrapper(srdf_arm2)
        + '</robot>'
    )

    # Load kinematics -- clone panda_arm's solver config under panda2_arm
    # rather than maintaining a second static override file (same solver,
    # same tuning, just a different group name to key it by).
    kinematics_yaml = load_yaml(moveit_config_path, 'config/kinematics.yaml')
    kinematics_yaml['panda2_arm'] = dict(kinematics_yaml['panda_arm'])

    # Load planning config -- same idea: panda2_arm reuses panda_arm's
    # planner list (the shared planner_configs: definitions above are
    # already group-agnostic, only the per-group planner list needs a
    # panda2_arm entry).
    ompl_yaml = load_yaml(moveit_config_path, 'config/ompl_planning.yaml')
    ompl_yaml['panda2_arm'] = {'planner_configs': list(ompl_yaml['panda_arm']['planner_configs'])}

    # Load controllers -- panda_moveit_controllers.yaml (a repo file, not an
    # installed package default) already has both arms' entries; see that
    # file's comments for why panda2_arm_controller's name is written as
    # the fully-namespaced /panda2/panda2_arm_controller.
    moveit_controllers = load_yaml(world_pkg_path, 'config/panda_moveit_controllers.yaml')

    return LaunchDescription([
        # Robot state publisher with the combined (both arms) URDF
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

        # Static transform world -> panda_link0: identity, unchanged from
        # before this file supported two arms. MoveIt's "world" frame is
        # defined to coincide with panda_link0 (not Gazebo's true world
        # origin, which is 0.2m away) -- actuator_node.py's own FRAME_OFFSET
        # already handles that conversion for arm 1 and continues to.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_transform_publisher',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'panda_link0'],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        # No separate panda_link0 -> panda2_link0 static transform here:
        # robot_state_publisher now derives and publishes that transform
        # itself from robot_description's panda2_mount_joint above (same
        # 0.6m Y offset, matching panda2's true spawn offset from panda in
        # panda_world.sdf). A second broadcaster of the same frame would
        # just be a redundant, conflicting publisher.

        # MoveIt2 move_group -- one instance, both groups (panda_arm,
        # panda2_arm, hand, hand2) in one planning scene.
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
