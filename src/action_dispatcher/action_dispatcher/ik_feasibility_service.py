#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from ros_llm_interfaces.srv import CheckIKFeasibility
import json

class IKFeasibilityService(Node):
    def __init__(self):
        super().__init__('ik_feasibility_service')
        self.callback_group = ReentrantCallbackGroup()
        self.object_map = {}

        self.create_subscription(
            String,
            '/object_map',
            self.object_map_callback,
            10,
            callback_group=self.callback_group
        )

        self.create_service(
            CheckIKFeasibility,
            '/check_ik_feasibility',
            self.handle_request,
            callback_group=self.callback_group
        )

        self.ik_client = self.create_client(
            GetPositionIK,
            '/compute_ik',
            callback_group=self.callback_group
        )

        self.get_logger().info('IK Feasibility Service ready at /check_ik_feasibility')

    def object_map_callback(self, msg):
        self.object_map = json.loads(msg.data)

    def handle_request(self, request, response):
        object_name = request.object_name
        self.get_logger().info(f'Checking IK feasibility for: {object_name}')

        if object_name not in self.object_map:
            response.feasible = False
            response.reason = f'Object {object_name} not found in /object_map'
            return response

        pose = self.object_map[object_name]

        if not self.ik_client.wait_for_service(timeout_sec=3.0):
            response.feasible = False
            response.reason = 'MoveIt2 /compute_ik service not available'
            return response

        ik_request = GetPositionIK.Request()
        ik_request.ik_request.group_name = 'panda_arm'
        ik_request.ik_request.pose_stamped = PoseStamped()
        ik_request.ik_request.pose_stamped.header.frame_id = 'panda_link0'
        ik_request.ik_request.pose_stamped.pose.position.x = pose['x'] - 0.2
        ik_request.ik_request.pose_stamped.pose.position.y = pose['y']
        ik_request.ik_request.pose_stamped.pose.position.z = pose['z'] - 1.025
        ik_request.ik_request.pose_stamped.pose.orientation.w = 1.0
        ik_request.ik_request.timeout.sec = 3
        ik_request.ik_request.robot_state.is_diff = True

        future = self.ik_client.call_async(ik_request)

        while not future.done():
            pass

        if future.result() is not None:
            error_code = future.result().error_code.val
            if error_code == 1:
                response.feasible = True
                response.reason = f'IK solution found for {object_name} at {pose}'
            else:
                response.feasible = False
                response.reason = f'No IK solution for {object_name} at {pose}, error code: {error_code}'
        else:
            response.feasible = False
            response.reason = 'IK service call timed out'

        self.get_logger().info(f'Result: {response.feasible} — {response.reason}')
        return response

def main(args=None):
    rclpy.init(args=args)
    node = IKFeasibilityService()
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