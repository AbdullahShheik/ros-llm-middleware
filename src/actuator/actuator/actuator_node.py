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

import functools
import json
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionClient
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose
from control_msgs.action import GripperCommand
from sensor_msgs.msg import JointState
from moveit.planning import MoveItPy
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive
from gz.transport13 import Node as GzNode
from gz.msgs10.empty_pb2 import Empty as GzEmpty
from gz.msgs10.stringmsg_pb2 import StringMsg as GzStringMsg
import math

# World -> panda_link0 frame offset — must match ik_feasibility_service.py
FRAME_OFFSET = {"x": -0.2, "y": 0.0, "z": 0.0}

ARM_GROUP  = "panda_arm"
BASE_FRAME = "panda_link0"
EEF_LINK   = "panda_link8"
GRIPPER_BASE_LINK = "robotiq_85_base_link"
GRIPPER_TOUCH_LINKS = [
    "robotiq_85_base_link",
    "robotiq_85_left_knuckle_link", "robotiq_85_right_knuckle_link",
    "robotiq_85_left_finger_link", "robotiq_85_right_finger_link",
    "robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link",
    "robotiq_85_left_inner_knuckle_link", "robotiq_85_right_inner_knuckle_link",
]

GRIPPER_ACTION = "/robotiq_gripper_controller/gripper_cmd"

# Robotiq 2F-85: 0.0 = open, 0.8 = closed
GRIPPER_POSITION = {
    "pick":  0.8,   # close to grasp
    "place": 0.0,   # open to release
}

# True size of the tracked cubes' collision boxes in panda_world.sdf
# (<box><size>0.06 0.06 0.06</size></box>) -- used once an object is
# actually attached to the gripper, where accuracy matters for checking
# it against *other* scene geometry while being carried.
TRACKED_OBJECT_SIZE = 0.06

# Deliberately smaller than TRACKED_OBJECT_SIZE: used as the world
# collision-avoidance proxy for an object *before* it's grasped. The open
# gripper's fingertips only clear the true 0.06m box by ~1.25cm per side
# at the correct, precisely-centered grasp pose (0.085m open aperture,
# 0.06m box) -- tight enough that ordinary calibration/mesh tolerance
# pushes the "correct" pose into being flagged as colliding with a
# full-size box, which OMPL then can't sample a valid goal state for at
# all ("Unable to sample any valid states for goal tree"). A smaller
# proxy still meaningfully blocks any gross approach path from swinging
# through the object -- which is what was actually pushing it out of
# place -- while leaving real margin around the correct grasp pose.
COLLISION_PROXY_SIZE = 0.04

# Tracked cube model names in panda_world.sdf -- each has a
# gz-sim DetachableJoint system whose <parent_link> is the cube's own
# "link" and <child_link> is GRIPPER_BASE_LINK on the "panda" model, used
# to physically pin a grasped cube to the gripper (see module docstring /
# _attach_grasped_object). Topic names here must match the ones declared
# on those plugins.
DETACHABLE_CUBES = ("red_cube", "blue_cube", "green_cube")


def _detachable_joint_topics(cube_name: str) -> dict:
    base = f"/model/{cube_name}/detachable_joint"
    return {"attach": f"{base}/attach", "detach": f"{base}/detach", "state": f"{base}/state"}


DETACHABLE_JOINT_TOPICS = {name: _detachable_joint_topics(name) for name in DETACHABLE_CUBES}

# Grasp-detection tolerance (fallback proximity check -- no contact sensor
# is configured on the Robotiq fingers): how far the cube's live
# /object_map position may be from the pose the gripper just closed at and
# still count as "actually inside the gripper" rather than knocked away or
# missed entirely. Half the 0.06m cube plus margin for perception noise.
GRASP_XY_TOLERANCE = 0.035
GRASP_Z_TOLERANCE = 0.04

# How long to wait for a DetachableJoint's <output_topic> to confirm a
# requested attach/detach actually took effect, rather than assuming a
# published gz-transport message was received and acted on.
JOINT_STATE_CONFIRM_TIMEOUT = 3.0


class ActuatorNode(Node):
    def __init__(self):
        super().__init__("actuator_node")
        self.callback_group = ReentrantCallbackGroup()

        self._have_joint_state = False
        self._js_lock = threading.Lock()
        # Serializes _run_pick_or_place() so only one execution_command
        # drives MoveItPy/the gripper at a time (see _execution_command_callback).
        self._execution_lock = threading.Lock()

        # Live object positions from perception. The pose in
        # /execution_command is a snapshot taken back when layer1/the
        # dispatcher first resolved it; the multi-second approach+descend
        # sequence below can nudge the object a few cm off that snapshot
        # (residual contact even along a controlled descent), so pick
        # re-reads the object's current position from here right before
        # the final descend-and-close, instead of closing on wherever it
        # used to be. See _refresh_pick_target().
        self._object_map = {}
        self._object_map_lock = threading.Lock()
        self.create_subscription(
            String,
            "/object_map",
            self._object_map_callback,
            10,
            callback_group=self.callback_group,
        )

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

        # gz-transport link to each cube's DetachableJoint system: one
        # persistent attach/detach Publisher per cube (created once here
        # rather than per-call, since a freshly-advertised gz-transport
        # publisher needs a brief discovery window before a subscriber on
        # another process actually receives its first message), plus a
        # subscription to each <output_topic> so attach/detach can be
        # confirmed as physically real instead of assumed from the
        # gz-transport publish call succeeding.
        self._gz_node = GzNode()
        self._detach_state_lock = threading.Lock()
        self._detach_state = {}  # cube_name -> "attached" | "detached"
        self._detach_state_event = threading.Event()
        self._attach_pub = {}
        self._detach_pub = {}
        for name, topics in DETACHABLE_JOINT_TOPICS.items():
            self._attach_pub[name] = self._gz_node.advertise(topics["attach"], GzEmpty)
            self._detach_pub[name] = self._gz_node.advertise(topics["detach"], GzEmpty)
            self._gz_node.subscribe(
                GzStringMsg, topics["state"], functools.partial(self._on_detachable_state, name)
            )

        # gz-sim's DetachableJoint system starts every cube already
        # rigidly joined to the gripper at world load -- there is no
        # "start detached" SDF option. world.launch.py already releases
        # every cube shortly after Gazebo comes up (long before this node
        # even starts, since the arm free-falls under gravity with no
        # active controller until 15s -- see that launch file for why it
        # can't wait for this node). This is just a defensive backstop for
        # e.g. re-running this node on its own against an already-loaded
        # world. Repeated with short gaps to ride out the publisher/
        # subscriber discovery window.
        for _ in range(5):
            for name in DETACHABLE_CUBES:
                self._detach_pub[name].publish(GzEmpty())
            time.sleep(0.1)

        self.get_logger().info("Initializing MoveItPy (this can take a few seconds)...")
        self.moveit = MoveItPy(node_name="actuator_moveit_py")
        self.arm = self.moveit.get_planning_component(ARM_GROUP)
        self._scene_monitor = self.moveit.get_planning_scene_monitor()
        # Name of the object currently attached to the gripper (None if
        # empty-handed). Tracked so retreat/place know what they're
        # carrying without re-deriving it from a possibly-stale pose.
        self._attached_object_name = None

        self.get_logger().info("Actuator ready, listening on /execution_command")

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------

    def _joint_state_callback(self, msg: JointState):
        with self._js_lock:
            self._have_joint_state = True

    def _object_map_callback(self, msg: String):
        try:
            with self._object_map_lock:
                self._object_map = json.loads(msg.data)
        except json.JSONDecodeError:
            pass

    def _find_tracked_object(self, pose: dict):
        """Return (name, live_pose) for whichever tracked object is
        closest to `pose` (the snapshot from /execution_command), or
        (None, None) if nothing is close enough to plausibly be it. 0.06m
        cube half-diagonal plus generous margin for how far it can have
        drifted mid-approach; beyond that this is almost certainly a
        different object (or a "place" pose over empty space)."""
        with self._object_map_lock:
            object_map = dict(self._object_map)

        best_name, best_dist = None, None
        for name, live_pose in object_map.items():
            dist = math.dist(
                (pose["x"], pose["y"]), (live_pose["x"], live_pose["y"])
            )
            if best_dist is None or dist < best_dist:
                best_name, best_dist = name, dist

        if best_name is not None and best_dist <= 0.15:
            return best_name, dict(object_map[best_name])
        return None, None

    def _refresh_pick_target(self, pose: dict) -> dict:
        """Return the live position of whichever tracked object is
        closest to `pose`, so the final descend-and-close targets where
        the object actually is now rather than where it was when the task
        was dispatched. Falls back to the original pose if no tracked
        object is close enough."""
        name, live_pose = self._find_tracked_object(pose)
        if live_pose is None:
            return pose
        if (live_pose["x"], live_pose["y"]) != (pose["x"], pose["y"]):
            self.get_logger().info(
                f"Pick target drifted from dispatch snapshot -- retargeting to {live_pose} ({name})"
            )
        return live_pose

    # ------------------------------------------------------------------
    # Planning scene: collision objects + attach/detach
    # ------------------------------------------------------------------

    def _object_pose_stamped(self, pose: dict) -> Pose:
        p = Pose()
        p.position.x = pose["x"] + FRAME_OFFSET["x"]
        p.position.y = pose["y"] + FRAME_OFFSET["y"]
        p.position.z = pose["z"] + FRAME_OFFSET["z"]
        p.orientation.w = 1.0
        return p

    def _sync_collision_object(self, name: str, pose: dict):
        """Add/move a box collision object in the planning scene so MoveIt
        actually knows this object exists and routes the arm around it
        during transit, instead of planning in a world it believes is
        empty (there was, before this, no collision-object or octomap
        awareness of the tracked cubes at all -- confirmed empty via
        `ros2 service call /get_planning_scene`)."""
        obj = CollisionObject()
        obj.header.frame_id = BASE_FRAME
        obj.id = name
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [COLLISION_PROXY_SIZE] * 3
        obj.primitives = [primitive]
        obj.primitive_poses = [self._object_pose_stamped(pose)]
        obj.operation = CollisionObject.ADD
        self._scene_monitor.process_collision_object(obj)

    def _remove_collision_object(self, name: str):
        obj = CollisionObject()
        obj.header.frame_id = BASE_FRAME
        obj.id = name
        obj.operation = CollisionObject.REMOVE
        self._scene_monitor.process_collision_object(obj)

    def _on_detachable_state(self, name: str, msg: GzStringMsg):
        with self._detach_state_lock:
            self._detach_state[name] = msg.data
        self._detach_state_event.set()

    def _wait_for_detach_state(self, name: str, expected: str, timeout_sec: float) -> bool:
        deadline = time.monotonic() + timeout_sec
        while True:
            with self._detach_state_lock:
                if self._detach_state.get(name) == expected:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._detach_state_event.wait(timeout=remaining)
            self._detach_state_event.clear()

    def _detect_grasp(self, name: str, close_pose: dict) -> bool:
        """Proximity fallback for grasp detection (no contact sensor is
        configured on the Robotiq fingers): the grasp only counts as real
        if the cube's live /object_map position is still essentially where
        the gripper just closed, i.e. it wasn't knocked away or missed."""
        with self._object_map_lock:
            live = self._object_map.get(name)
        if live is None:
            return False
        xy_dist = math.dist((live["x"], live["y"]), (close_pose["x"], close_pose["y"]))
        z_dist = abs(live["z"] - close_pose["z"])
        return xy_dist <= GRASP_XY_TOLERANCE and z_dist <= GRASP_Z_TOLERANCE

    def _attach_grasped_object(self, task_id: str, name: str, pose: dict, plan_id: str) -> bool:
        """Physically pin `name` to the gripper via its DetachableJoint the
        moment a grasp is detected, so it moves rigidly with the arm
        through lift/transport/retreat instead of relying on
        bullet-featherstone's contact solving to hold it there. Only folds
        the object into MoveIt's planning scene (_attach_object) once that
        real attach is confirmed over <output_topic> -- so a task only
        reports "success" if the grasp actually physically held, not
        because the gripper happened to report itself closed."""
        if not self._detect_grasp(name, pose):
            self._publish_feedback(
                task_id, "failed", "grasp",
                f"'{name}' not detected within the gripper after closing (proximity check failed)",
                plan_id
            )
            return False

        self._attach_pub[name].publish(GzEmpty())
        if not self._wait_for_detach_state(name, "attached", JOINT_STATE_CONFIRM_TIMEOUT):
            self._publish_feedback(
                task_id, "failed", "grasp",
                f"DetachableJoint for '{name}' did not confirm attach", plan_id
            )
            return False

        self._attach_object(name, pose)
        return True

    def _release_grasped_object(self, task_id: str, pose: dict, plan_id: str):
        """Detach whatever is currently grasped, both physically (gz-sim's
        DetachableJoint) and in MoveIt's planning scene. Best-effort on the
        gz confirmation -- unlike attach, a detach that fails to confirm
        still has to clear the planning scene, since leaving MoveIt
        believing the arm is still holding the object is worse than a
        possibly-stale physical joint."""
        name = self._attached_object_name
        if name is None:
            return
        self._detach_pub[name].publish(GzEmpty())
        if not self._wait_for_detach_state(name, "detached", JOINT_STATE_CONFIRM_TIMEOUT):
            self.get_logger().warn(f"[{task_id}] DetachableJoint for '{name}' did not confirm detach")
        self._detach_object(pose)

    def _attach_object(self, name: str, pose: dict):
        """Move the collision object from the world into the gripper's own
        collision body, so subsequent planning (retreat, transport) knows
        it now moves rigidly with the gripper instead of either treating
        it as a static obstacle to route around (which the grasp pose
        itself would already violate) or -- absent this entirely, as
        before -- not knowing about it at all and never noticing a
        collision between the held object and anything else in the
        scene."""
        attached = AttachedCollisionObject()
        attached.link_name = GRIPPER_BASE_LINK
        attached.object.header.frame_id = BASE_FRAME
        attached.object.id = name
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [TRACKED_OBJECT_SIZE] * 3
        attached.object.primitives = [primitive]
        attached.object.primitive_poses = [self._object_pose_stamped(pose)]
        attached.object.operation = CollisionObject.ADD
        attached.touch_links = GRIPPER_TOUCH_LINKS
        self._scene_monitor.process_attached_collision_object(attached)
        self._attached_object_name = name

    def _detach_object(self, pose: dict):
        """Release whatever is currently attached back into the world as a
        plain (no longer robot-carried) collision object at its drop
        location."""
        name = self._attached_object_name
        if name is None:
            return
        detached = AttachedCollisionObject()
        detached.link_name = GRIPPER_BASE_LINK
        detached.object.header.frame_id = BASE_FRAME
        detached.object.id = name
        detached.object.operation = CollisionObject.REMOVE
        self._scene_monitor.process_attached_collision_object(detached)
        self._attached_object_name = None
        self._sync_collision_object(name, pose)

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
        # Which tracked object (if any) this task concerns, and register
        # it as a real collision object in the planning scene *before*
        # planning anything. Previously MoveIt had zero knowledge of any
        # object in the world -- confirmed via `ros2 service call
        # /get_planning_scene`, which returned an empty world.collision_objects
        # and an empty octomap -- so every "collision-free" plan the arm
        # ever computed was free purely by luck, with nothing stopping it
        # from routing through the very object it was about to grasp.
        object_name = None
        if action == "pick":
            object_name, live_pose = self._find_tracked_object(pose)
            if live_pose is not None:
                pose = live_pose
                self._sync_collision_object(object_name, pose)

        # Ensure the gripper is open before approaching a pick target,
        # regardless of what state a previous task left it in. Must NOT
        # apply to "place": the gripper is holding the object at that
        # point, and opening it before approach would drop it mid-air.
        if action == "pick":
            if not self._send_gripper_command(task_id, GRIPPER_POSITION["place"], plan_id):
                return self._recover_after_failure(task_id, plan_id)

        # Phase 1: move to pre-grasp / pre-place (above target). With the
        # object now a real collision object, this free-space move is
        # collision-checked against it like any other obstacle instead of
        # being able to swing straight through where it sits.
        pre_pose = dict(pose)
        pre_pose["z"] = pose["z"] + 0.20  # 20cm above target
        if not self._plan_and_execute_arm(task_id, pre_pose, plan_id):
            return self._recover_after_failure(task_id, plan_id)

        # Phase 2: descend straight down to grasp/place height, in small
        # hops rather than one long plan() -- see
        # _plan_and_execute_arm_straight() for why. Re-read the object's
        # live position first (phase 1 above can take several seconds).
        #
        # Collision-checking against the *target* object is turned off
        # for this phase, not just at the final pose: even a waypoint 1cm
        # above the top of its true 0.06m box was rejected
        # ("Unable to sample any valid states for goal tree") against a
        # deliberately conservative 0.04m proxy, meaning the open
        # gripper's own linkage occupies more of the space directly above
        # a grasp target than a simple fingertip-clearance calculation
        # accounts for. Collision-checking the *descent itself* isn't what
        # was preventing the object from being pushed anyway -- that
        # requires actually reaching the commanded pose, which this
        # enables -- while phase 1's transit (a much longer, genuinely
        # obstacle-relevant move from wherever the arm currently is) stays
        # collision-aware.
        if action == "pick":
            pose = self._refresh_pick_target(pose)
            if object_name is not None:
                self._remove_collision_object(object_name)
        if not self._plan_and_execute_arm_straight(task_id, pre_pose, pose, plan_id):
            return self._recover_after_failure(task_id, plan_id)

        # One more live re-check right before closing: the descent above
        # can itself take several seconds (each hop is its own MoveIt plan
        # + trajectory execution), long enough for the object to drift
        # further after the refresh above already ran. If it moved more
        # than a trivial amount since then, do one last small corrective
        # horizontal move at the same height before clamping down on
        # (what would otherwise be) the wrong spot.
        # (Collision-checking against this object is already off at this
        # point -- see the near_pose approach above -- so this corrective
        # move doesn't need to touch the collision object either.)
        if action == "pick":
            corrected = self._refresh_pick_target(pose)
            if math.dist((corrected["x"], corrected["y"]), (pose["x"], pose["y"])) > 0.005:
                if not self._plan_and_execute_arm_straight(task_id, pose, corrected, plan_id, steps=1):
                    return self._recover_after_failure(task_id, plan_id)
                pose = corrected

        # Actuate gripper: close to grasp (pick) or open to release (place)
        gripper_position = GRIPPER_POSITION.get(action)
        if gripper_position is not None:
            if not self._send_gripper_command(task_id, gripper_position, plan_id):
                return self._recover_after_failure(task_id, plan_id)

        # Now that the gripper has actually closed, check whether the
        # object is really between the fingers and (if so) physically pin
        # it to the gripper via its DetachableJoint -- see
        # _attach_grasped_object for why (bullet-featherstone's own
        # contact solving isn't reliable enough to hold a grasp through
        # lift/transport on its own). This also folds the object into
        # MoveIt's own collision body so retreat doesn't treat it as a
        # static obstacle to route around and it moves rigidly with the
        # gripper in planning too, instead of MoveIt still believing the
        # world is empty-handed.
        if action == "pick" and object_name is not None:
            if not self._attach_grasped_object(task_id, object_name, pose, plan_id):
                return self._recover_after_failure(task_id, plan_id)
        elif action == "place":
            self._release_grasped_object(task_id, pose, plan_id)

        # Retreat to a safe height — transport height after a grasp,
        # clearance after a release — before reporting the task complete.
        if not self._retreat(task_id, pose, plan_id, return_to_ready=(action != "pick")):
            return self._recover_after_failure(task_id, plan_id)

        self._publish_feedback(
            task_id, "success", "complete",
            f"Task {task_id} ('{action}') finished",
            plan_id
        )

    def _recover_after_failure(self, task_id: str, plan_id: str):
        """Best-effort cleanup after ANY failed phase of a pick/place.
        Whichever phase failed already published its own "failed"
        feedback for this task -- this does not touch or repeat that --
        but without this, a failed task left the arm wherever it stalled
        (sometimes mid-descent, gripper half-closed on nothing) and never
        returned to "ready". Layer1 aborts the plan on task failure
        already, so the *session* survives; the problem was purely that
        the *robot* didn't, silently carrying a bad state into whatever
        the next instruction tried to do -- CheckStartStateCollision
        rejecting the next plan outright because the previous failure
        left the gripper folded back into the arm's own links, or a
        stale attached/collision object contradicting where the object
        actually is.
        """
        self.get_logger().warn(f"[{task_id}] Recovering arm/gripper state after failure")

        # Release the gripper regardless of what it was doing -- a
        # partially-closed grip on nothing, or on an object at an unknown
        # angle, is worse to leave for the next task than an open one.
        self._send_gripper_command(task_id, GRIPPER_POSITION["place"], plan_id)

        # If something was attached (grasped successfully, then a later
        # phase -- e.g. retreat -- failed), detach it back into the world
        # at its last known live position rather than leaving MoveIt
        # believing the robot is still holding an object it may no longer
        # have a working grip on.
        if self._attached_object_name is not None:
            name = self._attached_object_name
            with self._object_map_lock:
                live_pose = self._object_map.get(name)
            self._release_grasped_object(
                task_id, dict(live_pose) if live_pose else {"x": 0.0, "y": 0.0, "z": 0.0}, plan_id
            )

        self.arm.set_start_state_to_current_state()
        self.arm.set_goal_state(configuration_name="ready")
        ready_plan = self.arm.plan()
        if ready_plan:
            self.moveit.execute(ready_plan.trajectory, controllers=[])
        else:
            self.get_logger().warn(f"[{task_id}] Could not plan back to 'ready' configuration during recovery")

    def _retreat(self, task_id: str, pose: dict, plan_id: str, return_to_ready: bool) -> bool:
        cleared = False
        for z_offset in (0.4, 0.5):
            lift_pose = {"x": pose["x"], "y": pose["y"], "z": pose["z"] + z_offset}
            # Straight (in small hops) lift for the same reason as the
            # descent in _run_pick_or_place: an OMPL swing here can drag a
            # just-grasped object sideways into something, or (after a
            # place) swipe back through the object just released.
            if self._plan_and_execute_arm_straight(task_id, pose, lift_pose, plan_id, steps=3):
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
        # No X/Y correction needed: tf2_echo panda_link0 -> the two
        # fingertip links shows their midpoint's X/Y exactly matches
        # panda_link8's (only Z differs, handled below) at this grasp
        # orientation, so panda_link8's own X/Y is already the grasp center.
        target.pose.position.x = pose["x"] + FRAME_OFFSET["x"]
        target.pose.position.y = pose["y"] + FRAME_OFFSET["y"]
        # panda_link8 (EEF_LINK, what MoveIt actually plans to) sits 0.098m
        # above the gripper fingertips' actual contact point at this fixed
        # grasp orientation -- confirmed via tf2_echo panda_link0 ->
        # robotiq_85_*_finger_tip_link (fingertip midpoint X/Y match
        # panda_link8 exactly; only Z differs, by 0.098m). Commanding
        # panda_link8 to "pose[z]" would leave the fingertips 0.098m above
        # the target height, so add that fixed offset here.
        target.pose.position.z = pose["z"] + FRAME_OFFSET["z"] + 0.098
        # Point gripper straight down: pure 180deg about X, no yaw twist.
        # The cube collision boxes in panda_world.sdf are axis-aligned with
        # no rotation, so closing the gripper along a world-axis-aligned
        # line (rather than the previous ~45deg-yawed line) lets the
        # fingers close flush against opposite faces instead of catching
        # them at an angle and sliding/pushing the object.
        target.pose.orientation.x = 1.0
        target.pose.orientation.y = 0.0
        target.pose.orientation.z = 0.0
        target.pose.orientation.w = 0.0
        return target

    def _plan_and_execute_arm(self, task_id: str, pose: dict, plan_id: str = "", _silent_success: bool = False) -> bool:
        """`_silent_success` only suppresses the *success* feedback for
        this step (used for intermediate hops in
        _plan_and_execute_arm_straight, to avoid spamming Layer1 with one
        "arm_motion" message per hop). Failure feedback is never
        suppressed: an earlier version silenced both here, so a failing
        *intermediate* hop returned False without publishing anything at
        all -- Layer1 never got a "failed" for the task and just hung
        waiting forever for a "complete" that was never coming, instead of
        aborting the plan and staying ready for the next instruction."""
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

        if not _silent_success:
            self._publish_feedback(
                task_id, "success", "arm_motion",
                f"Arm reached {pose}",
                plan_id
            )
        return True

    def _plan_and_execute_arm_straight(
        self, task_id: str, start_pose: dict, end_pose: dict, plan_id: str = "", steps: int = 5
    ) -> bool:
        """Move from start_pose to end_pose in several small OMPL hops
        instead of one long plan().

        OMPL (RRTConnect) only guarantees *a* collision-free path between
        two poses, not a direct one -- for the ~20cm pre-grasp-to-grasp
        descent this was observed swinging sideways mid-trajectory and
        clipping the object well before "reaching" the commanded pose, so
        by the time the gripper closed the object had already been
        knocked out of place. A true Cartesian planner (Pilz "LIN") was
        tried instead, but its trajectory generation rejected this arm's
        approach orientation outright ("Joint acceleration limit of
        panda_jointN violated") regardless of velocity/acceleration
        scaling -- consistent with the straight-line path passing close to
        a wrist singularity for this pose, not a tunable speed problem.
        Chopping the same distance into short (~4cm) hops sidesteps both
        problems: each individual OMPL solve has too little room between
        its own start/goal to usefully swing sideways, and it can still
        use non-straight joint trajectories to route around any
        near-singular configuration.
        """
        for i in range(1, steps + 1):
            frac = i / steps
            waypoint = {
                "x": start_pose["x"] + (end_pose["x"] - start_pose["x"]) * frac,
                "y": start_pose["y"] + (end_pose["y"] - start_pose["y"]) * frac,
                "z": start_pose["z"] + (end_pose["z"] - start_pose["z"]) * frac,
            }
            is_last = i == steps
            if not self._plan_and_execute_arm(task_id, waypoint, plan_id, _silent_success=not is_last):
                return False
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

        # The controller's own stall_timeout (panda_ros2_controllers.yaml)
        # is measured against sim time, but this wait is real wall-clock
        # time (threading.Event) -- under a slow/oversubscribed sim (low
        # real-time factor), 1 sim-second of stall detection can take much
        # longer than 1 wall-second, so this needs real margin beyond that.
        # Closing against an actual grasped object (real mechanical
        # resistance through the now-functional mimic linkage, see
        # panda_world.sdf's physics engine + model.sdf's
        # position_proportional_gain) has been observed taking as long as
        # ~27s even under moderate load, right up against the previous
        # 30s cap -- 60s gives real margin instead of a coin flip.
        result = self._wait_for_future(goal_handle.get_result_async(), timeout_sec=60.0)
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