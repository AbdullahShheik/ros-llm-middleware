#!/usr/bin/env python3
"""
arm_workspace_lock_node.py

Mutual-exclusion lock over the shared workspace region both arms can
physically reach (see panda_world.sdf's panda2 <include><pose> and
actuator_node.py's SHARED_ZONE_BOUNDS). Each arm's actuator_node.py calls
/shared_workspace/acquire before moving to a target inside that region and
/shared_workspace/release once clear of it again -- targets outside the
shared region need no lock at all, so the two arms can still work fully
concurrently as long as they're not both reaching into the same space at
once. This is a deliberate alternative to relying on MoveIt's own
collision checking to know the other arm's live pose: actuator_node.py's
MoveItPy plans in its own in-process planning scene (moveit_cpp.yaml's
PlanningSceneMonitor, not the external move_group node), one instance per
arm, with no built-in cross-instance awareness -- a small explicit lock is
simpler and more certain than depending on unverified multi-source
joint_states behavior.

A single Trigger-based acquire blocks (server-side) until the lock is
free rather than returning "busy" immediately, mirroring
attach_detach_node.py's _is_pickup_point_clear wait-loop shape -- the
caller's blocking service call just waits, no client-side retry loop
needed.
"""

import time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Trigger

ACQUIRE_TIMEOUT_S = 60.0
POLL_INTERVAL_S = 0.2


class ArmWorkspaceLockNode(Node):
    def __init__(self):
        super().__init__('arm_workspace_lock_node')
        # Reentrant + MultiThreadedExecutor so a blocked acquire() call
        # doesn't stall the release() call that would free it -- same
        # reasoning as attach_detach_node.py's pickup-point wait loop.
        self.callback_group = ReentrantCallbackGroup()
        self._locked = False
        self._holder = None

        self.acquire_service = self.create_service(
            Trigger, '/shared_workspace/acquire', self.acquire_callback,
            callback_group=self.callback_group,
        )
        self.release_service = self.create_service(
            Trigger, '/shared_workspace/release', self.release_callback,
            callback_group=self.callback_group,
        )
        self.get_logger().info(
            'Arm workspace lock ready: /shared_workspace/acquire, /shared_workspace/release'
        )

    def acquire_callback(self, request, response):
        waited = 0.0
        while self._locked:
            if waited >= ACQUIRE_TIMEOUT_S:
                response.success = False
                response.message = (
                    f'Shared workspace still held after {ACQUIRE_TIMEOUT_S}s'
                )
                self.get_logger().error(response.message)
                return response
            time.sleep(POLL_INTERVAL_S)
            waited += POLL_INTERVAL_S

        self._locked = True
        response.success = True
        response.message = 'Shared workspace acquired'
        return response

    def release_callback(self, request, response):
        # Idempotent on purpose: a caller that times out waiting for
        # something else downstream but still calls release() in its
        # cleanup path shouldn't fail just because it never actually
        # acquired the lock.
        self._locked = False
        response.success = True
        response.message = 'Shared workspace released'
        return response


def main(args=None):
    rclpy.init(args=args)
    node = ArmWorkspaceLockNode()
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
