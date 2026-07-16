#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from ros_llm_interfaces.srv import CheckIKFeasibility
import json

# Object name mapping from LLM team naming to /object_map naming
OBJECT_NAME_MAP = {
    "red_block": "red_cube",
    "blue_block": "blue_cube",
    "green_block": "green_cube"
}

# Skill to robot type mapping
SKILL_TO_ROBOT = {
    "pick": "arm",
    "place": "arm",
    "transport": "wheeled",
    "survey": "wheeled",
    "navigate": "wheeled"
}

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

        # IK feasibility service client
        self.ik_client = self.create_client(
            CheckIKFeasibility,
            '/check_ik_feasibility',
            callback_group=self.callback_group
        )

        self.get_logger().info('Action Dispatcher ready, waiting for subtasks on /layer1/taskplan')

    def object_map_callback(self, msg):
        self.object_map = json.loads(msg.data)

    def subtask_callback(self, msg):
        subtask = json.loads(msg.data)
        self.get_logger().info(f'Received subtask: {subtask}')

        task_id = subtask.get('id')
        required_skills = subtask.get('required_skills', [])
        args = subtask.get('args', {})
        llm_object_name = args.get('object_name')

        #map object name
        object_name = OBJECT_NAME_MAP.get(llm_object_name)
        if object_name is None:
            self.get_logger().error(f'Unknown object name: {llm_object_name}')
            return

        #determine robot type from skill
        skill = required_skills[0] if required_skills else None
        robot_type = SKILL_TO_ROBOT.get(skill)
        if robot_type is None:
            self.get_logger().error(f'Unknown skill: {skill}')
            return

        #for arm tasks, check IK feasibility
        if robot_type == 'arm':
            feasible = self.check_ik(object_name)
            if not feasible:
                self.get_logger().warn(f'IK check failed for {object_name}, task {task_id} rejected')
                return

        #look up pose from object map
        if object_name not in self.object_map:
            self.get_logger().error(f'Object {object_name} not in object map')
            return

        pose = self.object_map[object_name]

        #dispatch to executor
        execution_cmd = json.dumps({
            'task_id': task_id,
            'action': skill,
            'robot_type': robot_type,
            'pose': pose
        })
        msg_out = String()
        msg_out.data = execution_cmd
        self.execution_command_pub.publish(msg_out)
        self.get_logger().info(f'Task {task_id} dispatched to {robot_type} executor with pose {pose}')

    def check_ik(self, object_name):
        if not self.ik_client.wait_for_service(timeout_sec=3.0):
            self.get_logger().warn('IK feasibility service not available')
            return False

        request = CheckIKFeasibility.Request()
        request.object_name = object_name

        future = self.ik_client.call_async(request)

        while not future.done():
            pass

        if future.result() is not None:
            return future.result().feasible
        return False

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