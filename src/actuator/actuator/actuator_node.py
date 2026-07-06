#!/usr/bin/env python3
"""
Layer 3: Actuator node.

Subscribes to /execution_command (std_msgs/String, JSON), e.g.:
  {
    "task_id": "T1",
    "action": "pick",
    "robot_type": "arm",
    "pose": {"x": 0.5, "y": 0.0, "z": 1.025}
  }

Plans and executes a Cartesian pose goal for the Franka Panda arm with
MoveIt2 (moveit_py), then (for pick/place) actuates the "hand" gripper
group to its "close"/"open" named state.

Current robot state:
  MoveItPy's internal planning scene monitor already subscribes to
  /joint_states on its own to track the live robot state, so
  set_start_state_to_current_state() always plans from wherever the
  arm actually is right now. This node ALSO subscribes to /joint_states
  itself, purely as a readiness gate -- it refuses to plan until at
  least one joint state has been observed, so we never race MoveIt's
  own monitor on startup.

Publishes feedback to /execution_feedback (std_msgs/String, JSON):
  {
    "task_id": "T1",
    "status": "success" | "failed",
    "stage": "planning" | "execution" | "gripper" | "complete" | "dispatch",
    "detail": "<human readable>"
  }

Requires MoveItPy to be initialized with the panda MoveIt configuration.
See launch/actuator.launch.py, which loads the same
moveit_resources_panda_moveit_config package used by moveit2.launch.py
and passes it to this node as ROS parameters.
"""

import json
import threading

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

from moveit.planning import MoveItPy

# Same world -> panda_link0 frame offset used in
# action_dispatcher/ik_feasibility_service.py. Keep these in sync --
# the IK check already validated feasibility using this transform,
# so the actuator must target the exact same point.
FRAME_OFFSET = {"x": -0.2, "y": 0.0, "z": -1.025}

ARM_GROUP = "panda_arm"
HAND_GROUP = "hand"
BASE_FRAME = "panda_link0"
EEF_LINK = "panda_link8"  # verify against your SRDF if this differs

# Skills that should also actuate the gripper once the arm reaches pose.
# Maps action -> named group_state on the "hand" group (see panda.srdf).
GRIPPER_ACTION_FOR_SKILL = {
    "pick": "close",
    "place": "open",
}


class ActuatorNode(Node):
    def __init__(self):
        super().__init__("actuator_node")
        self.callback_group = ReentrantCallbackGroup()

        self._have_joint_state = False
        self._js_lock = threading.Lock()

        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
            callback_group=self.callback_group,
        )

        self.create_subscription(
            String,
            "/execution_command",
            self._execution_command_callback,
            10,
            callback_group=self.callback_group,
        )

        self.feedback_pub = self.create_publisher(String, "/execution_feedback", 10)

        self.get_logger().info("Initializing MoveItPy (this can take a few seconds)...")
        self.moveit = MoveItPy(node_name="actuator_moveit_py")
        self.arm = self.moveit.get_planning_component(ARM_GROUP)

        try:
            self.hand = self.moveit.get_planning_component(HAND_GROUP)
        except Exception as e:
            self.hand = None
            self.get_logger().warn(
                f"No '{HAND_GROUP}' planning component available ({e}); "
                f"gripper actuation will be skipped."
            )

        self.get_logger().info("Actuator ready, listening on /execution_command")

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _joint_state_callback(self, msg: JointState):
        with self._js_lock:
            self._have_joint_state = True

    def _execution_command_callback(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Bad /execution_command payload: {e}")
            return

        task_id = cmd.get("task_id", "UNKNOWN")
        action = cmd.get("action")
        robot_type = cmd.get("robot_type")
        pose = cmd.get("pose")

        if robot_type != "arm":
            self._publish_feedback(
                task_id, "failed", "dispatch",
                f"Actuator only handles robot_type='arm', got '{robot_type}'"
            )
            return

        if not pose:
            self._publish_feedback(task_id, "failed", "dispatch", "No pose in execution command")
            return

        with self._js_lock:
            ready = self._have_joint_state

        if not ready:
            self._publish_feedback(
                task_id, "failed", "planning",
                "No /joint_states received yet, refusing to plan blind"
            )
            return

        if not self._plan_and_execute_arm(task_id, pose):
            return  # failure feedback already published inside

        gripper_state = GRIPPER_ACTION_FOR_SKILL.get(action)
        if gripper_state:
            if self.hand is not None:
                if not self._plan_and_execute_gripper(task_id, gripper_state):
                    return
            else:
                self.get_logger().warn(
                    f"[{task_id}] Skipping gripper '{gripper_state}': no hand group configured"
                )

        self._publish_feedback(task_id, "success", "complete", f"Task {task_id} ('{action}') finished")

    # ------------------------------------------------------------------
    # Planning / execution
    # ------------------------------------------------------------------

    def _target_pose_stamped(self, pose: dict) -> PoseStamped:
        target = PoseStamped()
        target.header.frame_id = BASE_FRAME
        target.pose.position.x = pose["x"] + FRAME_OFFSET["x"]
        target.pose.position.y = pose["y"] + FRAME_OFFSET["y"]
        target.pose.position.z = pose["z"] + FRAME_OFFSET["z"]
        target.pose.orientation.w = 1.0
        return target

    def _plan_and_execute_arm(self, task_id, pose: dict) -> bool:
        self.arm.set_start_state_to_current_state()
        target = self._target_pose_stamped(pose)
        self.arm.set_goal_state(pose_stamped_msg=target, pose_link=EEF_LINK)

        plan_result = self.arm.plan()
        if not plan_result:
            self._publish_feedback(task_id, "failed", "planning", f"MoveIt2 failed to find a plan to {pose}")
            return False

        exec_ok = self.moveit.execute(plan_result.trajectory, controllers=[])
        if not exec_ok:
            self._publish_feedback(task_id, "failed", "execution", "Trajectory execution failed")
            return False

        self._publish_feedback(task_id, "success", "arm_motion", f"Arm reached {pose}")
        return True

    def _plan_and_execute_gripper(self, task_id, named_state: str) -> bool:
        self.hand.set_start_state_to_current_state()
        self.hand.set_goal_state(configuration_name=named_state)

        plan_result = self.hand.plan()
        if not plan_result:
            self._publish_feedback(task_id, "failed", "gripper", f"Failed to plan gripper '{named_state}'")
            return False

        exec_ok = self.moveit.execute(plan_result.trajectory, controllers=[])
        if not exec_ok:
            self._publish_feedback(task_id, "failed", "gripper", f"Failed to execute gripper '{named_state}'")
            return False

        self._publish_feedback(task_id, "success", "gripper", f"Gripper set to '{named_state}'")
        return True

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def _publish_feedback(self, task_id, status, stage, detail):
        fb = {"task_id": task_id, "status": status, "stage": stage, "detail": detail}
        out = String()
        out.data = json.dumps(fb)
        self.feedback_pub.publish(out)
        log = self.get_logger().info if status == "success" else self.get_logger().warn
        log(f"[{task_id}] {status} @ {stage}: {detail}")


def main(args=None):
    rclpy.init(args=args)
    node = ActuatorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
