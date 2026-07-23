#!/usr/bin/env python3

import re

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from ros_llm_interfaces.srv import CheckIKFeasibility
import json
import threading

# Object name mapping from LLM team naming to /object_map naming
OBJECT_NAME_MAP = {
    "red_block": "red_cube",
    "blue_block": "blue_cube",
    "green_block": "green_cube"
}

# Fallback surface height for "place" targets that only specify x/y (e.g.
# "place it at 0.3, 0.4") -- every tracked object rests at this height once
# settled on the floor (see perception_node.py TRACKED_OBJECTS).
PLACE_SURFACE_Z = 0.04

_NUMBER_RE = re.compile(r'-?\d+(?:\.\d+)?')

# Skills that don't involve a specific object (navigate to a place, survey
# an area) -- object_name resolution is skipped for these.
NO_OBJECT_SKILLS = {"navigate", "survey"}

VALID_ROBOTS = {"arm", "wheeled"}

class ActionDispatcher(Node):
    def __init__(self):
        super().__init__('action_dispatcher')
        self.callback_group = ReentrantCallbackGroup()

        self.object_map = {}

        # Subscribe to object map from perception
        self.create_subscription(
            String,
            '/object_map',
            self.object_map_callback,
            10,
            callback_group=self.callback_group
        )

        # Subscribe to subtask commands from LLM team
        self.create_subscription(
            String,
            '/layer1/taskplan',
            self.subtask_callback,
            10,
            callback_group=self.callback_group
        )

        # Publisher for execution command
        self.execution_command_pub = self.create_publisher(
            String,
            '/execution_command',
            10
        )

        # Feedback publisher: every subtask this dispatcher rejects
        # (unknown skill, unknown object, IK-infeasible, missing pose)
        # must still be reported here. Layer1 tracks one plan "active" at
        # a time and blocks all further instructions until every dispatched
        # task's feedback arrives -- a silently dropped subtask (log +
        # return, no feedback) leaves Layer1 waiting forever and wedges
        # the whole session, not just that one task.
        self.feedback_pub = self.create_publisher(
            String,
            '/execution_feedback',
            10
        )

        # IK feasibility service client
        self.ik_client = self.create_client(
            CheckIKFeasibility,
            '/check_ik_feasibility',
            callback_group=self.callback_group
        )

        self.get_logger().info('Action Dispatcher ready, waiting for subtasks on /layer1/taskplan')

    def object_map_callback(self, msg):
        self.object_map = json.loads(msg.data)

    def _resolve_place_pose(self, target_location):
        """Resolve a place skill's target_location into a world pose.

        target_location may be another known object's name (e.g.
        "blue_cube" / "blue_block"), or literal coordinates such as
        "0.3, 0.4" / "(0.3, 0.4, 0.05)". Returns None if it can't be
        resolved to either."""
        if not target_location:
            return None

        mapped_name = OBJECT_NAME_MAP.get(target_location, target_location)
        if mapped_name in self.object_map:
            return dict(self.object_map[mapped_name])

        numbers = [float(n) for n in _NUMBER_RE.findall(target_location)]
        if len(numbers) >= 2:
            return {
                "x": numbers[0],
                "y": numbers[1],
                "z": numbers[2] if len(numbers) >= 3 else PLACE_SURFACE_Z,
            }

        return None

    def _resolve_location(self, target_location):
        """Resolve a named location (zone or object name) to a pose via
        /object_map. Used for wheeled-robot skills (navigate, survey,
        transport)."""
        if not target_location:
            return None
        mapped_name = OBJECT_NAME_MAP.get(target_location, target_location)
        if mapped_name in self.object_map:
            return dict(self.object_map[mapped_name])
        return None

    def subtask_callback(self, msg):
        subtask = json.loads(msg.data)
        self.get_logger().info(f'Received subtask: {subtask}')

        task_id = subtask.get('id')
        plan_id = subtask.get('plan_id', '')
        required_skills = subtask.get('required_skills', [])
        args = subtask.get('args', {})
        llm_object_name = args.get('object_name')
        skill = required_skills[0] if required_skills else None

        # "home" takes no object_name/target -- the actuator already
        # returns the arm to its "ready" configuration as the last step of
        # every pick/place (see actuator_node.py's _retreat()), so this is
        # already satisfied by the time it's requested. Ack it immediately
        # instead of routing it through the object_name lookup below (which
        # would otherwise reject every plan's closing "home" subtask, since
        # it never carries an object_name, aborting an otherwise-successful
        # plan in Layer1).
        if skill == 'home':
            self._ack(task_id, plan_id, 'complete', 'Arm already at ready position')
            return

        # Robot assignment comes from Layer1, not derived from skill.
        robot_type = subtask.get('robot')
        if robot_type not in VALID_ROBOTS:
            self.get_logger().error(f'Unknown/missing robot field: {robot_type}')
            self._reject(task_id, plan_id, 'dispatch', f'Unknown/missing robot field: {robot_type}')
            return

        if robot_type == 'wheeled':
            target_location = args.get('target_location')
            pose = self._resolve_location(target_location)
            if pose is None:
                self.get_logger().error(f"Could not resolve target_location '{target_location}'")
                self._reject(
                    task_id, plan_id, 'dispatch',
                    f"Could not resolve target_location '{target_location}' to a pose"
                )
                return

            # TODO: replace with real nav feasibility service call once
            # Nav2 is integrated. Stubbed to always succeed for now.
            feasible = self.check_nav_feasibility(pose)
            if not feasible:
                self.get_logger().warn(f'Nav check failed for {target_location}, task {task_id} rejected')
                self._reject(task_id, plan_id, 'nav_check', f'Nav check failed for {target_location}')
                return

            execution_cmd = json.dumps({
                'task_id': task_id,
                'plan_id': plan_id,
                'action': skill,
                'robot_type': robot_type,
                'pose': pose
            })
            msg_out = String()
            msg_out.data = execution_cmd
            self.execution_command_pub.publish(msg_out)
            self.get_logger().info(f'Task {task_id} dispatched to wheeled executor with pose {pose}')
            return

        #map object name
        object_name = OBJECT_NAME_MAP.get(llm_object_name)
        if object_name is None and skill not in NO_OBJECT_SKILLS:
            self.get_logger().error(f'Unknown object name: {llm_object_name}')
            self._reject(task_id, plan_id, 'dispatch', f'Unknown object name: {llm_object_name}')
            return

        # "place" moves the held object to target_location, NOT back to
        # object_name's own (now in-gripper) position -- resolve that
        # separately from the pick/inspect/push path below, which targets
        # the object itself.
        if skill == 'place':
            target_location = args.get('target_location')
            pose = self._resolve_place_pose(target_location)
            if pose is None:
                self.get_logger().error(f"Could not resolve target_location '{target_location}'")
                self._reject(
                    task_id, plan_id, 'dispatch',
                    f"Could not resolve target_location '{target_location}' to a pose"
                )
                return

            # Fast-fail IK check only when the target resolved to a known
            # tracked object (the feasibility service can only look up
            # poses by name); raw-coordinate targets fall through to the
            # arm planner itself, which already reports a clear
            # 'planning failed' feedback if unreachable.
            resolved_name = OBJECT_NAME_MAP.get(target_location, target_location)
            if resolved_name in self.object_map and not self.check_ik(resolved_name):
                self.get_logger().warn(f'IK check failed for {resolved_name}, task {task_id} rejected')
                self._reject(task_id, plan_id, 'ik_check', f'IK check failed for {resolved_name}')
                return
        else:
            #for arm tasks, check IK feasibility
            feasible = self.check_ik(object_name)
            if not feasible:
                self.get_logger().warn(f'IK check failed for {object_name}, task {task_id} rejected')
                self._reject(task_id, plan_id, 'ik_check', f'IK check failed for {object_name}')
                return

            #look up pose from object map
            if object_name not in self.object_map:
                self.get_logger().error(f'Object {object_name} not in object map')
                self._reject(task_id, plan_id, 'dispatch', f'Object {object_name} not in object map')
                return

            pose = self.object_map[object_name]

        #dispatch to executor
        execution_cmd = json.dumps({
            'task_id': task_id,
            'plan_id': plan_id,
            'action': skill,
            'robot_type': robot_type,
            'pose': pose
        })
        msg_out = String()
        msg_out.data = execution_cmd
        self.execution_command_pub.publish(msg_out)
        self.get_logger().info(f'Task {task_id} dispatched to {robot_type} executor with pose {pose}')

    def _reject(self, task_id, plan_id, stage, detail):
        """Report a subtask this dispatcher cannot route so Layer1's
        feedback tracking sees a terminal (failed) result instead of
        waiting forever for a task that will never produce one."""
        self._publish_feedback(task_id, plan_id, stage, 'failed', detail)

    def _ack(self, task_id, plan_id, stage, detail):
        """Report a subtask the dispatcher resolves itself without routing
        it to /execution_command (e.g. 'home', already satisfied by the
        actuator's own post-task retreat)."""
        self._publish_feedback(task_id, plan_id, stage, 'success', detail)

    def _publish_feedback(self, task_id, plan_id, stage, status, detail):
        fb = {
            'plan_id': plan_id,
            'task_id': task_id,
            'status': status,
            'stage': stage,
            'detail': detail,
        }
        out = String()
        out.data = json.dumps(fb)
        self.feedback_pub.publish(out)

    def check_ik(self, object_name):
        if not self.ik_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn('IK feasibility service not available')
            return False

        request = CheckIKFeasibility.Request()
        request.object_name = object_name

        future = self.ik_client.call_async(request)

        # Block this callback's thread on an Event instead of busy-spinning
        # on future.done() -- a tight spin loop here starves the
        # MultiThreadedExecutor's other worker threads (including the ones
        # needed to process the very service response we're waiting for
        # when the pool is small), and burns a full CPU core doing nothing.
        done_event = threading.Event()
        future.add_done_callback(lambda _f: done_event.set())
        if not done_event.wait(timeout=10.0):
            self.get_logger().warn('IK feasibility check timed out')
            return False

        if future.result() is not None:
            return future.result().feasible
        return False

    def check_nav_feasibility(self, pose):
        """Stub -- always returns True until the real nav feasibility
        service (backed by Nav2's compute_path_to_pose) is wired in."""
        return True

def main(args=None):
    rclpy.init(args=args)
    node = ActionDispatcher()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()