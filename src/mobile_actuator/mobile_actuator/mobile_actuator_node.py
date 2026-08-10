#!/usr/bin/env python3
"""
Layer 3: Mobile Actuator node.

Subscribes to /execution_command (std_msgs/String, JSON), e.g.:
  {
    "task_id": "T2",
    "plan_id": "P1",
    "action": "navigate",
    "robot_type": "wheeled",
    "pose": {"x": 1.5, "y": -0.8, "z": 0.0}
  }

Sends a NavigateToPose action goal to Nav2, waits for the result, then
publishes to /execution_feedback (std_msgs/String, JSON) in the same
schema as actuator_node.py:
  {
    "plan_id":  "P1",
    "task_id":  "T2",
    "status":   "success" | "failed",
    "stage":    "navigation" | "dispatch" | ...,
    "detail":   "<human-readable string>"
  }
"""

import json
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

# Nav2's NavigateToPose operates in the map frame.
NAV_FRAME = "map"

# How long to wait for the Nav2 action server to come up.
# BT navigator waits for map_server, AMCL, costmaps, 30s gives real room.
NAV_SERVER_TIMEOUT_SEC = 30.0

# How long to wait for goal acceptance. If bt_navigator is alive,
# acceptance is near-instant.
GOAL_ACCEPT_TIMEOUT_SEC = 10.0

# Navigation result timeout, None means wait indefinitely.
# Nav2 handles its own recovery behaviours internally; we don't impose a
# wall-clock cap here so a legitimately slow path (crowded costmap,
# recovery spin) is not killed by us.
NAV_RESULT_TIMEOUT_SEC = None

# All action values from the dispatcher that physically drive NavigateToPose.
# "survey" and "transport" are semantically distinct but use the same Nav2
# primitive as their physical first step.
WHEELED_NAV_ACTIONS = {"navigate", "navigate_to", "survey", "transport"}


class MobileActuatorNode(Node):
    def __init__(self):
        super().__init__("mobile_actuator_node")
        self.callback_group = ReentrantCallbackGroup()

        # Serialise execution exactly like actuator_node.py: Nav2 is a
        # shared, stateful resource (active goal, costmaps, BT state).
        # Sending a second NavigateToPose while one is active cancels or
        # races the first. The lock guarantees at most one active Nav2
        # goal at a time regardless of how fast /execution_command fires.
        self._execution_lock = threading.Lock()

        self.create_subscription(
            String,
            "/execution_command",
            self._execution_command_callback,
            10,
            callback_group=self.callback_group,
        )

        self.feedback_pub = self.create_publisher(String, "/execution_feedback", 10)

        self._nav_client = ActionClient(
            self,
            NavigateToPose,
            "navigate_to_pose",
            callback_group=self.callback_group,
        )

        self.get_logger().info("MobileActuator ready, listening on /execution_command")

    # ------------------------------------------------------------------
    # Subscription callback
    # ------------------------------------------------------------------

    def _execution_command_callback(self, msg: String):
        try:
            cmd = json.loads(msg.data)
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Bad /execution_command payload: {e}")
            return

        task_id    = cmd.get("task_id", "UNKNOWN")
        plan_id    = cmd.get("plan_id", "")
        action     = cmd.get("action")
        robot_type = cmd.get("robot_type")
        pose       = cmd.get("pose")

        # Both actuators share /execution_command and filter by robot_type,
        # mirroring actuator_node.py's pattern. Silent return for arm
        # commands, no feedback needed, the arm actuator handles those.
        if robot_type not in ('wheeled', 'mobile_robot'):
            return

        if not pose:
            self._publish_feedback(
                task_id, "failed", "dispatch",
                "No pose in execution command",
                plan_id,
            )
            return

        if action not in WHEELED_NAV_ACTIONS:
            self._publish_feedback(
                task_id, "failed", "dispatch",
                f"Unsupported action '{action}' for wheeled robot "
                f"(supported: {sorted(WHEELED_NAV_ACTIONS)})",
                plan_id,
            )
            return

        with self._execution_lock:
            self._run_navigation(task_id, plan_id, action, pose)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _run_navigation(self, task_id: str, plan_id: str, action: str, pose: dict):
        """Send a NavigateToPose goal and block until terminal result.

        Intermediate Nav2 feedback (distance_remaining, ETA) is logged at
        DEBUG but not forwarded to /execution_feedback — Layer1 only cares
        about terminal results, consistent with how actuator_node.py only
        publishes at phase boundaries rather than on every MoveIt tick."""

        self.get_logger().info(f"[{task_id}] '{action}' → NavigateToPose to {pose}")

        if not self._nav_client.wait_for_server(timeout_sec=NAV_SERVER_TIMEOUT_SEC):
            self._publish_feedback(
                task_id, "failed", "navigation",
                f"Nav2 action server not available after {NAV_SERVER_TIMEOUT_SEC}s",
                plan_id,
            )
            return

        goal = self._build_goal(pose)

        # send_goal_async is non-blocking; we wait on a threading.Event for
        # the GoalHandle, then again for the result. Same pattern as
        # actuator_node._wait_for_future() and dispatcher_node.check_ik():
        # a busy-spin on future.done() starves the MultiThreadedExecutor
        # threads that need to process the very response we're waiting for.
        goal_handle = self._wait_for_future(
            self._nav_client.send_goal_async(
                goal,
                feedback_callback=self._nav_feedback_callback,
            ),
            timeout_sec=GOAL_ACCEPT_TIMEOUT_SEC,
        )

        if goal_handle is None:
            self._publish_feedback(
                task_id, "failed", "navigation",
                "Timed out waiting for Nav2 to accept the goal",
                plan_id,
            )
            return

        if not goal_handle.accepted:
            self._publish_feedback(
                task_id, "failed", "navigation",
                "Nav2 rejected the NavigateToPose goal",
                plan_id,
            )
            return

        self.get_logger().info(f"[{task_id}] Goal accepted, waiting for result…")

        result_response = self._wait_for_future(
            goal_handle.get_result_async(),
            timeout_sec=NAV_RESULT_TIMEOUT_SEC,
        )

        if result_response is None:
            # Only reachable if NAV_RESULT_TIMEOUT_SEC is set to a finite
            # value. Kept for safety if the constant is tightened later.
            self._publish_feedback(
                task_id, "failed", "navigation",
                "Timed out waiting for navigation result",
                plan_id,
            )
            return

        status = result_response.status

        if status == GoalStatus.STATUS_SUCCEEDED:
            self._publish_feedback(
                task_id, "success", "complete",
                f"Task {task_id} ('{action}') reached {pose}",
                plan_id,
            )
        elif status == GoalStatus.STATUS_CANCELED:
            self._publish_feedback(
                task_id, "failed", "navigation",
                f"Navigation goal was cancelled (status={status})",
                plan_id,
            )
        else:
            self._publish_feedback(
                task_id, "failed", "navigation",
                f"Navigation failed (GoalStatus={status})",
                plan_id,
            )

    def _build_goal(self, pose: dict) -> NavigateToPose.Goal:
        goal = NavigateToPose.Goal()
        ps = PoseStamped()
        ps.header.frame_id = NAV_FRAME
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = float(pose.get("x", 0.0))
        ps.pose.position.y = float(pose.get("y", 0.0))
        ps.pose.position.z = 0.0  # Nav2 navigates on the 2-D plane
        # Dispatcher currently only gives x/y/z. Accept optional qz/qw for
        # future extension (e.g. docking orientation); default is identity
        # quaternion (yaw = 0). Nav2's goal checker only checks position by
        # default, so orientation only affects final heading.
        ps.pose.orientation.z = float(pose.get("qz", 0.0))
        ps.pose.orientation.w = float(pose.get("qw", 1.0))
        goal.pose = ps
        return goal

    def _nav_feedback_callback(self, feedback_msg):
        """Log Nav2's periodic distance-remaining ticks at DEBUG only."""
        fb = feedback_msg.feedback
        self.get_logger().debug(
            f"Nav2: distance_remaining={fb.distance_remaining:.2f}m"
        )

    # ------------------------------------------------------------------
    # Shared async helper — mirrors actuator_node._wait_for_future()
    # ------------------------------------------------------------------

    def _wait_for_future(self, future, timeout_sec):
        """Block on `future` via threading.Event (no busy-spin).
        Returns future.result() on completion, None on timeout.
        Pass timeout_sec=None to wait indefinitely."""
        done_event = threading.Event()
        future.add_done_callback(lambda _f: done_event.set())
        if not done_event.wait(timeout=timeout_sec):
            return None
        return future.result()

    # ------------------------------------------------------------------
    # Feedback — exact schema as actuator_node._publish_feedback()
    # ------------------------------------------------------------------

    def _publish_feedback(
        self, task_id: str, status: str, stage: str, detail: str, plan_id: str = ""
    ):
        fb = {
            "plan_id": plan_id,
            "task_id": task_id,
            "status":  status,
            "stage":   stage,
            "detail":  detail,
        }
        out = String()
        out.data = json.dumps(fb)
        self.feedback_pub.publish(out)
        self.get_logger().info(f"[{task_id}] {status} @ {stage}: {detail}")


def main(args=None):
    rclpy.init(args=args)
    node = MobileActuatorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()