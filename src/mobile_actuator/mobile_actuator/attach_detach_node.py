#!/usr/bin/env python3

import math
import time
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import String
from std_srvs.srv import Trigger
from nav_msgs.msg import Odometry

from gz.transport13 import Node as GzNode
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.boolean_pb2 import Boolean
import json

CUBE_NAMES = {"red_cube", "blue_cube", "green_cube"}

CUBE_ATTACH_Z = 0.15
# Four earlier attempts at this point all shared the same flaw: they kept
# both arms facing the same +X direction (or only rotated arm 2 by a
# fixed -90deg) and tried to tune X/Y to find SOME point both could
# reach. Three X values at y=0.5 (0.7, 0.55, 0.62) all cleared Nav2 fine
# and always succeeded at the initial descend-and-grasp, but every one
# failed the RETREAT phase right after (lifting straight up with the
# object now rigidly attached), same error every time ("Unable to sample
# any valid states for goal tree") -- an orientation/joint-limit dead
# zone from reaching SIDEWAYS with the fixed straight-down grasp
# orientation, not a distance problem (the whole single-arm system was
# only ever proven at straight-AHEAD reach). Rotating only arm 2 by a
# fixed -90deg (the next attempt) just moved which axis was lateral for
# arm 2; it didn't remove the lateral component for either arm.
#
# Both arms are now rotated 45deg INWARD, mirrored (see panda_world.sdf),
# so they angle toward each other. This point, (0.70, 0.50), sits exactly
# on the two arms' shared aim line (their perpendicular bisector) -- a
# genuinely zero-lateral, straight-ahead reach for BOTH arms
# simultaneously (confirmed by the frame math: arm1_local=(0.707,~0),
# arm2_local=(0.707,~0), ~83% of nominal reach for both), not just a
# smaller lateral offset like every value tried before this one.
#
# X also clears build_map.py's merged both-arms obstacle block (its real
# edge, after both arms' 45deg rotation, is at x~0.318) by ~0.38m --
# comfortably past the mobile robot's own footprint (robot_radius 0.22,
# see nav2_params.yaml), so Nav2 accepts "handoff_point" as a valid,
# reachable goal with real margin, not just barely.
PICKUP_POINT_X = 0.70
PICKUP_POINT_Y = 0.50
CUBE_PICKUP_Z  = 0.04

# How close another cube has to be sitting to the pickup point to count as
# "occupying" it -- generous over the ~0.02-0.1mm settle precision a clean
# teleport lands at, but tight enough not to false-flag a cube resting
# somewhere else entirely.
PICKUP_CLEAR_TOLERANCE_M = 0.05
# How long detach_callback will wait (parked at the handoff point, cube
# still attached) for an earlier cube to be picked up off the pickup point
# before giving up and reporting failure, rather than teleporting a second
# cube on top of the first.
PICKUP_CLEAR_TIMEOUT_S = 30.0
PICKUP_CLEAR_POLL_INTERVAL_S = 0.5

# set_pose is fire-and-forget -- Gazebo acks the service call before the
# pose is actually applied in a physics step, same as documented at length
# in actuator_node.py (which this mirrors, minus the retry-cycle: that
# exists there for a precision arm grasp, not needed for "is this cube
# now sitting near the target x/y"). Only x/y are checked, deliberately
# not z: for detach the cube settles at a fixed resting height so x/y
# arrival implies z arrival anyway, but for attach the cube is teleported
# UP into the air pre-weld and immediately starts falling under gravity
# until the attach message welds it -- a z check there would race against
# that fall instead of confirming anything useful.
TELEPORT_CONFIRM_TOLERANCE_M = 0.01
TELEPORT_CONFIRM_TIMEOUT_S = 5.0
TELEPORT_CONFIRM_POLL_INTERVAL_S = 0.1


class AttachDetachNode(Node):
    def __init__(self):
        super().__init__("attach_detach_node")

        # Reentrant so detach_callback can block/poll waiting for the
        # pickup point to clear without stalling object_map_callback --
        # with the default mutually-exclusive group + single-threaded
        # spin, a blocking wait here would also block the very callback
        # that could ever report the pickup point clear (see main()'s
        # MultiThreadedExecutor below). Same pattern as ActionDispatcher
        # in dispatcher_node.py.
        self.callback_group = ReentrantCallbackGroup()

        self.gz_node = GzNode()

        self.gz_attach_pubs = {}
        self.gz_detach_pubs = {}
        for cube in CUBE_NAMES:
            self.gz_attach_pubs[cube] = self.gz_node.advertise(
                f"/{cube}/attach", Empty
            )
            self.gz_detach_pubs[cube] = self.gz_node.advertise(
                f"/{cube}/detach", Empty
            )
            self.get_logger().info(
                f"Gz publishers ready: /{cube}/attach, /{cube}/detach"
            )

        self.startup_timer = self.create_timer(3.0, self._startup_detach_all)

        self.attach_service = self.create_service(
            Trigger, "/attach_cube", self.attach_callback,
            callback_group=self.callback_group
        )
        self.detach_service = self.create_service(
            Trigger, "/detach_cube", self.detach_callback,
            callback_group=self.callback_group
        )

        self.active_cube_sub = self.create_subscription(
            String, "/active_cube", self.active_cube_callback, 10,
            callback_group=self.callback_group
        )

        self.turtlebot_x = -2.0
        self.turtlebot_y = 0.0
        # Latest live cube positions from /object_map -- {"red_cube": {"x":.., "y":.., "z":..}, ...}.
        # Used by _is_pickup_point_clear() to check whether an earlier
        # cube is still sitting at the pickup point.
        self.latest_object_map = {}
        self.object_map_sub = self.create_subscription(
            String, "/object_map", self.object_map_callback, 10,
            callback_group=self.callback_group
        )

        self.attached_cube = None

        self.get_logger().info(
            "AttachDetach node ready. "
            "Publish cube name to /active_cube, "
            "then call /attach_cube or /detach_cube."
        )

    def object_map_callback(self, msg: String):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.latest_object_map = data
        if "turtlebot3_waffle" in data:
            self.turtlebot_x = data["turtlebot3_waffle"]["x"]
            self.turtlebot_y = data["turtlebot3_waffle"]["y"]

    def _is_pickup_point_clear(self) -> bool:
        """True if no OTHER known cube is currently RESTING at the pickup
        point. Checked before detaching -- if an earlier cube hasn't been
        picked up by the arm yet, teleporting a new one onto the same spot
        would place two cubes on top of each other.

        Checks x/y/z together (3D distance to the actual resting pose, not
        just x/y) -- deliberately, not an oversight: once the arm has
        picked a cube up, it retreats straight up before doing anything
        else (same x/y, only z changes), so an x/y-only check would see it
        as "still occupying" the pickup point indefinitely -- observed
        live as a second cube's detach timing out at 30s even though the
        first cube was safely held 0.4m up in the gripper, nowhere near
        actually blocking the spot."""
        target = (PICKUP_POINT_X, PICKUP_POINT_Y, CUBE_PICKUP_Z)
        for cube in CUBE_NAMES:
            if cube == self.attached_cube:
                continue
            pos = self.latest_object_map.get(cube)
            if pos is None:
                continue
            if math.dist((pos["x"], pos["y"], pos["z"]), target) \
                    <= PICKUP_CLEAR_TOLERANCE_M:
                return False
        return True

    def _startup_detach_all(self):
        """Detach all cubes at startup — DetachableJoint starts attached by default."""
        for cube in CUBE_NAMES:
            self.gz_detach_pubs[cube].publish(Empty())
            self.get_logger().info(f"Startup detach: {cube}")
        self.startup_timer.cancel()

    def active_cube_callback(self, msg: String):
        cube = msg.data.strip()
        if cube not in CUBE_NAMES:
            self.get_logger().warn(
                f"Unknown cube: '{cube}'. Valid: {CUBE_NAMES}"
            )
            return
        self.attached_cube = cube
        self.get_logger().info(f"Active cube set to: {cube}")

    def _teleport_cube(self, cube_name: str, x: float, y: float, z: float) -> bool:
        """
        Teleport a cube to (x, y, z) using the Gazebo set_pose service.
        Returns True on success.
        """
        req = Pose()
        req.name = cube_name
        req.position.x = x
        req.position.y = y
        req.position.z = z
        req.orientation.x = 0.0
        req.orientation.y = 0.0
        req.orientation.z = 0.0
        req.orientation.w = 1.0

        result, success = self.gz_node.request(
            "/world/panda_world/set_pose",
            req,
            Pose,
            Boolean,
            2000
        )

        if not success:
            self.get_logger().error(
                f"set_pose service call failed for {cube_name}"
            )
            return False

        self.get_logger().info(
            f"Teleported {cube_name} to ({x:.2f}, {y:.2f}, {z:.2f})"
        )
        return True

    def _wait_for_object_at(self, cube_name: str, x: float, y: float) -> bool:
        """Block until /object_map reports `cube_name` within
        TELEPORT_CONFIRM_TOLERANCE_M (x/y only, see that constant's
        comment) of (x, y), or the timeout expires. The confirmation
        set_pose itself doesn't provide -- its ack means "request
        accepted", not "pose applied"."""
        deadline = time.monotonic() + TELEPORT_CONFIRM_TIMEOUT_S
        while True:
            pos = self.latest_object_map.get(cube_name)
            if pos is not None and math.hypot(
                pos["x"] - x, pos["y"] - y
            ) <= TELEPORT_CONFIRM_TOLERANCE_M:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(TELEPORT_CONFIRM_POLL_INTERVAL_S)

    def attach_callback(self, request, response):
        if self.attached_cube is None:
            response.success = False
            response.message = "No active cube. Publish to /active_cube first."
            self.get_logger().warn("attach called but no active cube set.")
            return response

        ok = self._teleport_cube(
            self.attached_cube,
            self.turtlebot_x,
            self.turtlebot_y,
            CUBE_ATTACH_Z
        )
        if not ok:
            response.success = False
            response.message = f"Failed to teleport {self.attached_cube}"
            return response

        if not self._wait_for_object_at(
            self.attached_cube, self.turtlebot_x, self.turtlebot_y
        ):
            response.success = False
            response.message = (
                f"Teleport for {self.attached_cube} to the TurtleBot did not "
                f"land within {TELEPORT_CONFIRM_TIMEOUT_S}s"
            )
            self.get_logger().error(response.message)
            return response

        self.gz_attach_pubs[self.attached_cube].publish(Empty())
        self.get_logger().info(
            f"Attached {self.attached_cube} to TurtleBot at "
            f"({self.turtlebot_x:.2f}, {self.turtlebot_y:.2f}, {CUBE_ATTACH_Z})"
        )

        response.success = True
        response.message = f"Attached {self.attached_cube} to TurtleBot"
        return response

    def detach_callback(self, request, response):
        if self.attached_cube is None:
            response.success = False
            response.message = "No cube currently attached."
            self.get_logger().warn("detach called but no cube is attached.")
            return response

        # Wait here -- still attached, physically parked at the handoff
        # point -- until the pickup point is clear of any earlier cube the
        # arm hasn't picked up yet. Only meant to run into its timeout in
        # unusual cases: fleet-capacity wave-splitting (layer1_pipeline.py)
        # already serializes different cubes' mobile-robot subtasks
        # against each other, but that alone doesn't guarantee the arm has
        # finished picking up cube A by the time cube B's transport
        # finishes -- this closes that remaining gap.
        waited = 0.0
        while not self._is_pickup_point_clear():
            if waited >= PICKUP_CLEAR_TIMEOUT_S:
                response.success = False
                response.message = (
                    f"Pickup point still occupied after {PICKUP_CLEAR_TIMEOUT_S}s; "
                    "not detaching to avoid stacking cubes."
                )
                self.get_logger().error(response.message)
                return response
            time.sleep(PICKUP_CLEAR_POLL_INTERVAL_S)
            waited += PICKUP_CLEAR_POLL_INTERVAL_S

        self.gz_detach_pubs[self.attached_cube].publish(Empty())
        ok = self._teleport_cube(
            self.attached_cube,
            PICKUP_POINT_X,
            PICKUP_POINT_Y,
            CUBE_PICKUP_Z
        )
        if not ok or not self._wait_for_object_at(
            self.attached_cube, PICKUP_POINT_X, PICKUP_POINT_Y
        ):
            # Previously this fell through to response.success = True
            # unconditionally -- set_pose's own ack means "request
            # accepted", not "pose applied" (same distinction actuator_node
            # documents at length), so a caller (the arm's next pick) could
            # be told the cube is at the pickup point when it never
            # actually landed there, and fail its own IK check downstream
            # with no clear reason why.
            response.success = False
            response.message = (
                f"Teleport for {self.attached_cube} to the pickup point did "
                f"not land within {TELEPORT_CONFIRM_TIMEOUT_S}s"
            )
            self.get_logger().error(response.message)
            return response

        self.get_logger().info(
            f"Detached {self.attached_cube} at "
            f"({self.turtlebot_x:.2f}, {self.turtlebot_y:.2f})"
        )

        prev = self.attached_cube
        self.attached_cube = None
        response.success = True
        response.message = f"Detached {prev} at handoff point"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = AttachDetachNode()
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
