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
MoveIt2 (moveit_py), then (for pick/place) sends a GripperCommand action
directly to the Robotiq 2F-85 gripper controller.

Gripper is no longer driven through MoveIt's hand group — the Robotiq
has a single actuated joint (robotiq_85_left_knuckle_joint) controlled
via control_msgs/GripperCommand:
  position 0.0 = fully open
  position 0.8 = fully closed

Publishes feedback to /execution_feedback (std_msgs/String, JSON).
"""

import json
import threading

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from control_msgs.action import ParallelGripperCommand
from sensor_msgs.msg import JointState
from moveit.planning import MoveItPy

# World -> panda_link0 frame offset — must match ik_feasibility_service.py
FRAME_OFFSET = {"x": -0.2, "y": 0.0, "z": -1.025}

ARM_GROUP  = "panda_arm"
BASE_FRAME = "panda_link0"
EEF_LINK   = "panda_link8"

GRIPPER_ACTION = "/robotiq_gripper_controller/gripper_cmd"

# Robotiq 2F-85: 0.0 = open, 0.8 = closed
GRIPPER_POSITION = {
    "pick":  0.8,   # close to grasp
    "place": 0.0,   # open to release
}


class ActuatorNode(Node):
    def __init__(self):
        super().__init__("actuator_node")
        self.callback_group = ReentrantCallbackGroup()

        self._have_joint_state = False
        self._js_lock = threading.Lock()

        # Joint state readiness gate
        self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
            callback_group=self.callback_group,
        )

        # Execution command input
        self.create_subscription(
            String,
            "/execution_command",
            self._execution_command_callback,
            10,
            callback_group=self.callback_group,
        )

        self.feedback_pub = self.create_publisher(String, "/execution_feedback", 10)

        # Robotiq gripper action client
        self._gripper_client = ActionClient(
            self,
            ParallelGripperCommand,
            GRIPPER_ACTION,
            callback_group=self.callback_group,
        )

        self.get_logger().info("Initializing MoveItPy (this can take a few seconds)...")
        self.moveit = MoveItPy(node_name="actuator_moveit_py")
        self.arm = self.moveit.get_planning_component(ARM_GROUP)

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

        task_id    = cmd.get("task_id", "UNKNOWN")
        action     = cmd.get("action")
        robot_type = cmd.get("robot_type")
        pose       = cmd.get("pose")

        if robot_type != "arm":
            self._publish_feedback(
                task_id, "failed", "dispatch",
                f"Actuator only handles robot_type='arm', got '{robot_type}'"
            )
            return

        if not pose:
            self._publish_feedback(
                task_id, "failed", "dispatch",
                "No pose in execution command"
            )
            return

        with self._js_lock:
            ready = self._have_joint_state

        if not ready:
            self._publish_feedback(
                task_id, "failed", "planning",
                "No /joint_states received yet, refusing to plan blind"
            )
            return

        # Move arm to target pose
        if not self._plan_and_execute_arm(task_id, pose):
            return

        # Actuate gripper if this action requires it
        gripper_position = GRIPPER_POSITION.get(action)
        if gripper_position is not None:
            if not self._send_gripper_command(task_id, gripper_position):
                return

        self._publish_feedback(
            task_id, "success", "complete",
            f"Task {task_id} ('{action}') finished"
        )

    # ------------------------------------------------------------------
    # Arm planning / execution (unchanged)
    # ------------------------------------------------------------------

    def _target_pose_stamped(self, pose: dict) -> PoseStamped:
        target = PoseStamped()
        target.header.frame_id = BASE_FRAME
        target.pose.position.x = pose["x"] + FRAME_OFFSET["x"]
        target.pose.position.y = pose["y"] + FRAME_OFFSET["y"]
        target.pose.position.z = pose["z"] + FRAME_OFFSET["z"] + 0.15
        target.pose.orientation.w = 1.0
        return target

    def _plan_and_execute_arm(self, task_id: str, pose: dict) -> bool:
        self.arm.set_start_state_to_current_state()
        target = self._target_pose_stamped(pose)
        self.arm.set_goal_state(pose_stamped_msg=target, pose_link=EEF_LINK)

        plan_result = self.arm.plan()
        if not plan_result:
            self._publish_feedback(
                task_id, "failed", "planning",
                f"MoveIt2 failed to find a plan to {pose}"
            )
            return False

        exec_ok = self.moveit.execute(plan_result.trajectory, controllers=[])
        if not exec_ok:
            self._publish_feedback(
                task_id, "failed", "execution",
                "Trajectory execution failed"
            )
            return False

        self._publish_feedback(
            task_id, "success", "arm_motion",
            f"Arm reached {pose}"
        )
        return True

    # ------------------------------------------------------------------
    # Gripper — direct GripperCommand action (no MoveIt)
    # ------------------------------------------------------------------

    def _send_gripper_command(self, task_id: str, position: float) -> bool:
        label = "close" if position > 0.0 else "open"

        if not self._gripper_client.wait_for_server(timeout_sec=5.0):
            self._publish_feedback(
                task_id, "failed", "gripper",
                f"Gripper action server not available ({GRIPPER_ACTION})"
            )
            return False

        goal = ParallelGripperCommand.Goal()
        goal.command.name = ["robotiq_85_left_knuckle_joint"]
        goal.command.position = [position]
        goal.command.effort = [50.0]

        future = self._gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if not goal_handle or not goal_handle.accepted:
            self._publish_feedback(
                task_id, "failed", "gripper",
                f"Gripper goal rejected for '{label}'"
            )
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        self._publish_feedback(
            task_id, "success", "gripper",
            f"Gripper set to '{label}' (position={position})"
        )
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