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
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
from moveit.planning import MoveItPy
import math

# World -> panda_link0 frame offset — must match ik_feasibility_service.py
FRAME_OFFSET = {"x": -0.2, "y": 0.0, "z": 0.0}

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
        # Serializes _run_pick_or_place() so only one execution_command
        # drives MoveItPy/the gripper at a time (see _execution_command_callback).
        self._execution_lock = threading.Lock()

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

        # Robotiq gripper action client. The gripper is driven by
        # position_controllers/GripperActionController (see
        # world/config/panda_ros2_controllers.yaml), which serves
        # control_msgs/action/GripperCommand -- NOT ParallelGripperCommand,
        # which is a different action type served only by
        # parallel_gripper_action_controller (not installed / not used
        # here). Sending the wrong action type silently hangs forever in
        # wait_for_server(), since action client/server matching requires
        # identical types.
        self._gripper_client = ActionClient(
            self,
            GripperCommand,
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
        plan_id    = cmd.get("plan_id", "")
        action     = cmd.get("action")
        robot_type = cmd.get("robot_type")
        pose       = cmd.get("pose")

        if robot_type != "arm":
            self._publish_feedback(
                task_id, "failed", "dispatch",
                f"Actuator only handles robot_type='arm', got '{robot_type}'",
                plan_id
            )
            return

        if not pose:
            self._publish_feedback(
                task_id, "failed", "dispatch",
                "No pose in execution command",
                plan_id
            )
            return

        with self._js_lock:
            ready = self._have_joint_state

        if not ready:
            self._publish_feedback(
                task_id, "failed", "planning",
                "No /joint_states received yet, refusing to plan blind",
                plan_id
            )
            return

        # Serialize execution: MoveItPy's planning component and the
        # gripper are shared, stateful resources that are not safe to
        # drive from two execution_commands at once. Without this lock, a
        # task_id's *first* feedback message (e.g. "arm_motion" after
        # phase 1) could reach Layer1 while later phases of that same
        # task are still running, and Layer1 would advance to the next
        # wave/instruction and dispatch a new execution_command that runs
        # concurrently against the same MoveItPy state.
        with self._execution_lock:
            self._run_pick_or_place(task_id, plan_id, action, pose)

    def _run_pick_or_place(self, task_id: str, plan_id: str, action: str, pose: dict):
        # Ensure the gripper is open before approaching a pick target,
        # regardless of what state a previous task left it in. Must NOT
        # apply to "place": the gripper is holding the object at that
        # point, and opening it before approach would drop it mid-air.
        if action == "pick":
            if not self._send_gripper_command(task_id, GRIPPER_POSITION["place"], plan_id):
                return

        # Phase 1: move to pre-grasp / pre-place (above target)
        pre_pose = dict(pose)
        pre_pose["z"] = pose["z"] + 0.20  # 20cm above target
        if not self._plan_and_execute_arm(task_id, pre_pose, plan_id):
            return

        # Phase 2: descend straight down to grasp/place height
        if not self._plan_and_execute_arm(task_id, pose, plan_id):
            return

        # Actuate gripper: close to grasp (pick) or open to release (place)
        gripper_position = GRIPPER_POSITION.get(action)
        if gripper_position is not None:
            if not self._send_gripper_command(task_id, gripper_position, plan_id):
                return

        # Retreat to a safe height — transport height after a grasp,
        # clearance after a release — before reporting the task complete.
        if not self._retreat(task_id, pose, plan_id, return_to_ready=(action != "pick")):
            return

        self._publish_feedback(
            task_id, "success", "complete",
            f"Task {task_id} ('{action}') finished",
            plan_id
        )

    def _retreat(self, task_id: str, pose: dict, plan_id: str, return_to_ready: bool) -> bool:
        cleared = False
        for z_offset in (0.25, 0.35):
            lift_pose = {"x": pose["x"], "y": pose["y"], "z": pose["z"] + z_offset}
            self.arm.set_start_state_to_current_state()
            target = self._target_pose_stamped(lift_pose)
            self.arm.set_goal_state(pose_stamped_msg=target, pose_link=EEF_LINK)

            plan_result = self.arm.plan()
            if plan_result and self.moveit.execute(plan_result.trajectory, controllers=[]):
                cleared = True
                break
            self.get_logger().warn(
                f"[{task_id}] Retreat planning failed at +{z_offset}m, retrying higher"
            )

        if not cleared:
            self._publish_feedback(
                task_id, "failed", "retreat", "Failed to retreat to a safe height", plan_id
            )
            return False

        if not return_to_ready:
            # Skip after "pick": the object is actively held by a
            # position-controlled friction grip with no explicit
            # attachment, and the "ready" configuration can be a large,
            # fast joint-space move far from the current pose -- enough to
            # shake the object loose mid-transport. Only return to "ready"
            # once the gripper is empty again (see below).
            return True

        # Beyond clearing the object, return to the SRDF's "ready" joint
        # configuration rather than leaving the arm at whatever Cartesian
        # pose the lift above happened to land on. There is no "home"
        # skill wired up in the dispatcher (SKILL_TO_ROBOT has no entry
        # for it), so without this the arm was left in an arbitrary --
        # sometimes tightly folded -- configuration between tasks. That
        # caused CheckStartStateCollision to reject the *next* task's very
        # first plan ("N contact(s) detected: panda_link1/2 - robotiq_...")
        # before it could even move, since gripper links folded back near
        # the arm's base links in that leftover pose. Best-effort: if this
        # fails, the task itself already succeeded (object was placed/
        # lifted), so don't fail the task over it.
        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name="ready")
        ready_plan = self.arm.plan()
        if ready_plan:
            self.moveit.execute(ready_plan.trajectory, controllers=[])
        else:
            self.get_logger().warn(f"[{task_id}] Could not plan back to 'ready' configuration")

        return True

    # ------------------------------------------------------------------
    # Arm planning / execution (unchanged)
    # ------------------------------------------------------------------

    def _target_pose_stamped(self, pose: dict) -> PoseStamped:
        target = PoseStamped()
        target.header.frame_id = BASE_FRAME
        target.pose.position.x = pose["x"] + FRAME_OFFSET["x"] - 0.045
        target.pose.position.y = pose["y"] + FRAME_OFFSET["y"] - 0.01
        # Approach from above: cube top (z=0) + gripper clearance
        # Cube is 0.06m tall, gripper fingers need ~0.05m clearance above cube top
        target.pose.position.z = pose["z"] + FRAME_OFFSET["z"] + 0.27
        # Point gripper straight down (180deg around X axis)
        target.pose.orientation.x = 0.924
        target.pose.orientation.y = -0.382
        target.pose.orientation.z = 0.0
        target.pose.orientation.w = 0.0
        return target

    def _plan_and_execute_arm(self, task_id: str, pose: dict, plan_id: str = "") -> bool:
        self.arm.set_start_state_to_current_state()
        target = self._target_pose_stamped(pose)
        self.arm.set_goal_state(pose_stamped_msg=target, pose_link=EEF_LINK)

        plan_result = self.arm.plan()
        if not plan_result:
            self._publish_feedback(
                task_id, "failed", "planning",
                f"MoveIt2 failed to find a plan to {pose}",
                plan_id
            )
            return False

        exec_ok = self.moveit.execute(plan_result.trajectory, controllers=[])
        if not exec_ok:
            self._publish_feedback(
                task_id, "failed", "execution",
                "Trajectory execution failed",
                plan_id
            )
            return False

        self._publish_feedback(
            task_id, "success", "arm_motion",
            f"Arm reached {pose}",
            plan_id
        )
        return True

    # ------------------------------------------------------------------
    # Gripper — direct GripperCommand action (no MoveIt)
    # ------------------------------------------------------------------

    def _wait_for_future(self, future, timeout_sec: float):
        """Block this (already-locked, sequential) thread for a Future
        without busy-spinning -- see check_ik in dispatcher_node.py for
        the same pattern and its rationale."""
        done_event = threading.Event()
        future.add_done_callback(lambda _f: done_event.set())
        if not done_event.wait(timeout=timeout_sec):
            return None
        return future.result()

    def _send_gripper_command(self, task_id: str, position: float, plan_id: str = "") -> bool:
        label = "close" if position > 0.0 else "open"

        if not self._gripper_client.wait_for_server(timeout_sec=10.0):
            self._publish_feedback(
                task_id, "failed", "gripper", "Gripper action server not available", plan_id
            )
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 50.0

        goal_handle = self._wait_for_future(
            self._gripper_client.send_goal_async(goal), timeout_sec=10.0
        )
        if goal_handle is None or not goal_handle.accepted:
            self._publish_feedback(
                task_id, "failed", "gripper", f"Gripper goal to '{label}' rejected or timed out", plan_id
            )
            return False

        result = self._wait_for_future(goal_handle.get_result_async(), timeout_sec=10.0)
        if result is None:
            self._publish_feedback(
                task_id, "failed", "gripper", f"Gripper '{label}' result timed out", plan_id
            )
            return False

        self._publish_feedback(task_id, "success", "gripper", f"Gripper set to '{label}'", plan_id)
        return True

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def _publish_feedback(self, task_id, status, stage, detail, plan_id=""):
        fb = {"plan_id": plan_id, "task_id": task_id, "status": status, "stage": stage, "detail": detail}
        out = String()
        out.data = json.dumps(fb)
        self.feedback_pub.publish(out)
        self.get_logger().info(f"[{task_id}] {status} @ {stage}: {detail}")


def main(args=None):
    rclpy.init(args=args)
    node = ActuatorNode()
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