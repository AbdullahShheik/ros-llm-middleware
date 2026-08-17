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
from controller_manager_msgs.srv import ListControllers
from sensor_msgs.msg import JointState
from moveit.planning import MoveItPy
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject
from shape_msgs.msg import SolidPrimitive
from gz.transport13 import Node as GzNode
from gz.msgs10.empty_pb2 import Empty as GzEmpty
from gz.msgs10.stringmsg_pb2 import StringMsg as GzStringMsg
from gz.msgs10.pose_pb2 import Pose as GzPose
from gz.msgs10.boolean_pb2 import Boolean as GzBoolean
from gz.msgs10.world_control_pb2 import WorldControl as GzWorldControl
from gz.msgs10.pose_v_pb2 import Pose_V as GzPoseV
from gz.msgs10.world_stats_pb2 import WorldStatistics as GzWorldStats
from gz.msgs10.stringmsg_v_pb2 import StringMsg_V as GzStringMsgV
import math
import numpy as np

# World -> panda_link0 frame offset — must match ik_feasibility_service.py
FRAME_OFFSET = {"x": -0.2, "y": 0.0, "z": 0.0}

ARM_GROUP  = "panda_arm"
BASE_FRAME = "panda_link0"
# NOT panda_link8: the URDF and model.sdf disagree about where that link is
# (URDF has it at the flange, coincident with panda_hand; the SDF has it
# 0.103m beyond panda_hand), so it names a different physical point in each
# and planning to it put the real gripper 0.103m from where MoveIt believed
# it was. robotiq_85_base_link is unambiguous in both. Must stay in sync with
# panda.srdf's panda_arm tip_link so the goal constraint and the KDL solver
# tip are the same link.
EEF_LINK   = "robotiq_85_base_link"
GRIPPER_BASE_LINK = "robotiq_85_base_link"
GRIPPER_TOUCH_LINKS = [
    # world/urdf/panda_gz.urdf.xacro includes BOTH the stock Panda hand
    # (moveit_resources_panda_description) and the Robotiq macro, so MoveIt's
    # collision model still carries panda_leftfinger/panda_rightfinger even
    # though Gazebo's model.sdf only has the Robotiq. Those phantom fingers
    # sit exactly where a grasped object is held, so leaving them out of
    # touch_links makes every post-grasp plan abort with
    # "CheckStartStateCollision ... panda_leftfinger - <cube>".
    "panda_hand", "panda_leftfinger", "panda_rightfinger",
    "robotiq_85_base_link",
    "robotiq_85_left_knuckle_link", "robotiq_85_right_knuckle_link",
    "robotiq_85_left_finger_link", "robotiq_85_right_finger_link",
    "robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link",
    "robotiq_85_left_inner_knuckle_link", "robotiq_85_right_inner_knuckle_link",
]

GRIPPER_ACTION = "/robotiq_gripper_controller/gripper_cmd"
GRIPPER_JOINT_NAME = "robotiq_85_left_knuckle_joint"

# Robotiq 2F-85: 0.0 = open, 0.8 = closed. The grasping close is NOT 0.8 --
# see GRIPPER_POSITION below the grasp-geometry block, which derives it from
# the object's width.
GRIPPER_FULLY_OPEN = 0.0

# How far off "fully open" (0.0) the gripper joint must be for a timed-out
# *close* command to count as "stalled against something" rather than
# "never actually moved" -- see _send_gripper_command's timeout fallback.
# bullet-featherstone's contact solving can make the close goal never
# report "reached" at all when it's genuinely pressed against the cube
# (the position controller keeps straining but never converges/stalls out
# cleanly), which is a harsher version of the same known pinch-grasp
# instability this whole DetachableJoint workaround exists for -- the
# gripper physically being stuck against resistance well past "open" is
# itself evidence of a real grip, arguably stronger evidence than reaching
# any specific commanded angle.
#
# NOTE: currently unreachable. The only close this node issues is the
# grasping one, which no longer waits for a result at all (see
# _send_gripper_command's wait_for_result), so no close can reach the
# timeout branch this guards. Kept because _send_gripper_command is still
# general over both directions and a future blocking close would need it
# again -- delete both together if that stops being true.
GRIPPER_STALL_MIN_POSITION = 0.15

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
DETACHABLE_CUBES = ("red_cube", "blue_cube", "green_cube", "yellow_cube")


def _detachable_joint_topics(cube_name: str) -> dict:
    base = f"/model/{cube_name}/detachable_joint"
    return {"attach": f"{base}/attach", "detach": f"{base}/detach", "state": f"{base}/state"}


DETACHABLE_JOINT_TOPICS = {name: _detachable_joint_topics(name) for name in DETACHABLE_CUBES}

# ---------------------------------------------------------------------------
# Grasp geometry, derived from world/models/panda/model.sdf
# ---------------------------------------------------------------------------
# Where the cube's *center* must sit, expressed in robotiq_85_base_link's own
# frame, to be exactly centered between the closed fingertip pads. This is
# also EEF_LINK -- the frame MoveIt plans to -- so the offset below doubles as
# the object-pose -> planning-target conversion with nothing in between. Its
# +z points distally (toward the fingers -- i.e. straight down in world at
# this grasp orientation).
#
# Derivation. Every Robotiq joint uses axis <xyz>0 -1 0</xyz>, so a joint
# angle q is a rotation Ry(-q). For the left fingertip the chain is:
#   robotiq_85_left_knuckle_joint    @ (0.03060114, 0, 0.05490452) rel. base    [model.sdf:816], rotates by q
#   robotiq_85_left_finger_joint     @ (0.03152616, 0, -0.00376347) rel. knuckle [model.sdf:841], FIXED
#   robotiq_85_left_finger_tip_joint @ (0.00563134, 0,  0.04718515) rel. finger  [model.sdf:883], mimics q with multiplier -1 [model.sdf:889]
# The tip joint's -1 mimic makes the tip link's orientation Ry(-q)Ry(+q) = I at
# every aperture -- the parallel-jaw property -- so tip pads stay axis-aligned
# with this frame. Tip-link origin is then K + Ry(-q)(F+T):
#   q=0.0 (open):   (+0.067759, 0, 0.098326)   -> tip-origin separation 0.135517
#   q=0.8 (closed): (+0.025340, 0, 0.111812)   -> tip-origin separation 0.050680
# Cross-check against the real 2F-85 stroke (0.085m open, 0m closed) puts the
# pad plane 0.025259m (open) / 0.025340m (closed) inboard of the tip-link
# origin -- agreeing to 0.08mm over the full stroke, and confirming the pads
# meet exactly on x=0 when closed. Independently, left_finger_tip.stl's
# bounding box places its inner face at x=-0.025259 -- an exact match to the
# derived open-stroke figure, which validates both the chain above and the
# mesh alignment.
#
# So laterally the grasp center is x=0, y=0 exactly, by symmetry -- and note
# what that means at q=0.8: the pads MEET (gap 0.16mm). Teleporting a 60mm
# cube to be centred there would bury each pad ~30mm inside it; dartsim
# resolves that penetration explosively (observed launching a cube to
# (-73, 38, 223)m). So the grasping close must stop at the aperture that
# brackets the object, not at 0.8 -- hence _grip_geometry_for() below.
#
# Vertically, that same STL shows the *pad face* (verts within 1mm of the
# inner face) spans z in [+0.013015, +0.051019] in tip-link coordinates,
# centred at +0.032017 -- and since the tip origin's own height varies with
# q, so does the pad centre. Both fall out of the same solve.
_KNUCKLE = (0.03060114, 0.05490452)   # left_knuckle_joint (x, z) rel. base  [model.sdf:816]
_TIP_CHAIN = (0.0371575, 0.04342168)  # fixed knuckle->finger->tip sum       [model.sdf:841,883]
_PAD_INSET = 0.025259                 # left_finger_tip.stl inner face
_PAD_CENTRE_Z = 0.032017              # left_finger_tip.stl pad-face midpoint

# Leave the pads a hair off the object rather than flush against it. The grip
# carries no load (the DetachableJoint does), so a sub-millimetre gap is
# invisible while removing the contact forces that make dartsim's pinch-grasp
# unstable in the first place.
GRIP_CLEARANCE = 0.002


def _grip_geometry_for(object_width: float):
    """(knuckle angle, pad-centre height in base-link frame) for gripping an
    object `object_width` across. Solves K_x + (FTx cos q - FTz sin q) =
    object_width/2 + clearance/2 + pad_inset as R*cos(q + phi)."""
    target_x = (object_width + GRIP_CLEARANCE) / 2.0 + _PAD_INSET
    r = math.hypot(*_TIP_CHAIN)
    phi = math.atan2(_TIP_CHAIN[1], _TIP_CHAIN[0])
    q = math.acos((target_x - _KNUCKLE[0]) / r) - phi
    z = (_KNUCKLE[1] + _TIP_CHAIN[0] * math.sin(q)
         + _TIP_CHAIN[1] * math.cos(q) + _PAD_CENTRE_Z)
    return q, z


GRIPPER_GRASP_POSITION, GRASP_OFFSET_Z = _grip_geometry_for(TRACKED_OBJECT_SIZE)

GRIPPER_POSITION = {
    "pick":  GRIPPER_GRASP_POSITION,   # close until the pads bracket the cube
    "place": GRIPPER_FULLY_OPEN,       # open to release
}

# Where the object's centre must sit in the gripper's own frame.
GRASP_OFFSET_IN_GRIPPER = (0.0, 0.0, GRASP_OFFSET_Z)

# EEF_LINK's z offset above the grasp point, used to turn a commanded object
# pose into a planning target. Identical to GRASP_OFFSET_IN_GRIPPER's z --
# EEF_LINK *is* robotiq_85_base_link, so it is the same measurement, not a
# coincidence -- so the arm descends to exactly the height at which the pads
# straddle the object's mid-height.
#
# This was previously a hardcoded 0.098, measured by tf2_echo to the fingertip
# *link origin* rather than to the fingertip *pad*. The pad sits 0.013-0.051m
# further distal, so the smaller number drove the arm ~4cm too far DOWN: with
# a cube resting at world z=0.03 the pads ended up punched through the ground
# plane while catching only the cube's bottom few mm. That is the mechanical
# root of both the multi-second gripper stall (fingers jamming into the floor,
# not the cube) and the cube being shoved out of place during the descent.
EEF_TO_GRASP_Z = GRASP_OFFSET_Z

# gz-sim's UserCommands system (ignition-gazebo-user-commands-system, loaded
# in panda_world.sdf) serves this; it takes a gz.msgs.Pose naming the model to
# move and replies gz.msgs.Boolean.
WORLD_NAME = "panda_world"
SET_POSE_SERVICE = f"/world/{WORLD_NAME}/set_pose"
# TODO(set_pose reliability): this service frequently just does not answer.
# Measured across clean runs: ~7 no-reply requests and ~4 requests whose pose
# never landed at all, per 3-cube run, with transport_ok=False every time and
# not one service-level rejection ever observed. The retry loop below plus the
# arrival poll hide most of it, but it is not free: it is the single largest
# contributor to pick time (picks run 40-95s, much of it burning 2s ack
# timeouts and 5s arrival polls), and it still occasionally costs a whole cube
# when all TELEPORT_MAX_ATTEMPTS are consumed. It also compounds with
# GRASP_VERIFY_MAX_CYCLES, since every verify cycle issues a fresh teleport --
# worst case ~9 set_pose calls for one pick.
# Not chased in Phase 1. Worth investigating whether it correlates with low
# real-time factor / physics-step scheduling, and whether the DetachableJoint
# could take an explicitly-specified transform instead of capturing whatever
# pose exists at attach time, which would remove the need to pre-position the
# cube via set_pose at all.
#
# NOT a success criterion -- just how long to block for a courtesy ack before
# giving up on the reply and going to measure the result directly. Deliberately
# left short rather than raised above the observed >2s service latency:
# raising it only makes this block longer waiting for a signal that has been
# shown to mean nothing (an ack precedes the pose being applied by up to
# ~120ms, and a timeout does not imply the pose was dropped). Getting to the
# arrival poll sooner is strictly better than waiting longer for the ack.
SET_POSE_TIMEOUT_MS = 2000

# set_pose is effectively fire-and-forget: Gazebo acks it before the pose is
# applied in a physics step (measured 39-156ms of lag against gz's own pose
# feed), and it sometimes returns an outright failure instead. Publishing the
# DetachableJoint attach straight off that ack is therefore a race, and when
# the joint wins it freezes the cube at its PRE-teleport pose -- which is
# exactly the 14.6-21.6mm residual that showed up on blue/green. So the attach
# now waits for Gazebo to confirm the cube actually arrived.
#
# Tolerance: when the teleport does land, the cube settles 0.02-0.10mm from
# target; the smallest "not yet arrived" delta observed was 14.58mm. 2mm sits
# ~20x above the landed error and ~7x below the smallest not-arrived error, so
# it separates the two cleanly without being tight enough to trip on pose-feed
# quantisation or a cube still micro-settling on the table.
TELEPORT_ARRIVAL_TOLERANCE = 0.002
# The sole success criterion, so this has to cover the slow path rather than
# just the fast one: arrivals were measured at ~40-156ms when the service
# replied promptly, but the service itself has been seen taking longer than
# its own 2.0s request deadline, and the pose can land any time in there. 5s
# gives that real room while keeping the worst case bounded (each attempt is
# at most SET_POSE_TIMEOUT_MS + this, so 3 attempts is ~21s against a 600s
# task budget).
TELEPORT_ARRIVAL_TIMEOUT = 5.0
TELEPORT_POLL_INTERVAL = 0.01
# Two distinct failure modes were observed: an outright service rejection, and
# a success that never lands. Three attempts covers a one-off rejection
# followed by a slow apply without looping unboundedly (worst case ~4.5s).
TELEPORT_MAX_ATTEMPTS = 3

# Confirming the cube ARRIVED is not the same as confirming it is still there
# once the DetachableJoint has actually formed. The joint freezes whatever
# transform holds at the instant it is created, and the cube is unconstrained
# until then, so it can free-fall out of position during the attach handshake:
# measured as blue holding -10.07mm and green 6.87mm off target, both well
# outside the 2mm the arrival poll had just verified. So the held offset is
# re-checked after the joint confirms, and the whole teleport+attach cycle is
# repeated (detaching first, so the cube can be repositioned cleanly) if it
# drifted.
#
# Originally 3 (today 2 of 3 cubes lost this race on the first attempt, so a
# single retry is not obviously enough; at roughly coin-flip odds three cycles
# gives two further chances while keeping the worst case bounded). Raised to 6
# after runs under heavier background load (extra always-on nodes competing
# with Gazebo's physics/service thread) needed more than 3 tries to converge
# under TELEPORT_ARRIVAL_TOLERANCE even though each retry's held offset was
# trending down (e.g. 16.13mm -> 9.30mm -> 5.70mm across 3 cycles). Raised
# again to 9 after a run still failed at 6 cycles with a held offset of
# 2.55mm -- just over TELEPORT_ARRIVAL_TOLERANCE, consistent with the same
# converging-but-out-of-tries pattern, not a new failure mode. This does not
# address the underlying set_pose reliability issue (still "not chased in
# Phase 1" per the TODO above); it only widens the margin the retry loop
# already exists to paper over.
GRASP_VERIFY_MAX_CYCLES = 9
# Brief settle before measuring, so the reading is the post-attach resting
# pose rather than a pose-feed sample published just before the joint formed.
GRASP_VERIFY_SETTLE_SEC = 0.15

# Set True to re-enable the per-pick teleport/attach checkpoint logging that
# found this race. Left in but gated off rather than deleted: the bug was
# invisible without ground-truth checkpoints, it is timing-dependent so it can
# resurface under different load, and gating means it costs literally nothing
# when off (the FK read is skipped entirely, not just the log call).
GRASP_DEBUG = False

# TEMPORARY DIAGNOSTIC (set back to False when done): measure and log the true
# wall-clock latency of every set_pose call together with the sim's real-time
# factor at that moment, plus a control call to an unrelated gz service issued
# back-to-back. Uses a deliberately huge request timeout so slow replies are
# measured rather than censored at SET_POSE_TIMEOUT_MS.
SETPOSE_LATENCY_PROBE = False

# A gz-transport keepalive was tried here and REMOVED: it did not help. With a
# cheap read-only request every 4s keeping the connection warm, this node still
# lost 8 of 17 of those very requests (47%), statistically the same as without
# it. That falsifies "the connection goes stale during the idle gaps between
# grasps" as the explanation. See the set_pose TODO above for what is and is
# not still on the table.
SETPOSE_PROBE_TIMEOUT_MS = 20000
CONTROL_SERVICE = "/gazebo/worlds"   # Empty -> StringMsg_V, read-only

# gz-sim exposes no per-joint "teleport the arm back into range" service --
# `gz service -l` on this world offers only /world/<w>/control
# (gz.msgs.WorldControl), whose WorldReset submessage has all / time_only /
# model_only. So a whole-world model reset is the ONLY Gazebo-level recovery
# available when the arm's joint state diverges; there is no targeted one.
# model_only is used rather than all: it restores every model's pose and joint
# positions to their SDF values without rewinding sim time, which would break
# every node running on use_sim_time.
WORLD_CONTROL_SERVICE = f"/world/{WORLD_NAME}/control"
WORLD_CONTROL_TIMEOUT_MS = 3000
# How long to wait after a reset for /joint_states to report in-range values.
DIVERGENCE_RESET_SETTLE_SEC = 8.0

# The reset above is NOT side-effect-free: `reset: {model_only: true}` on
# WORLD_CONTROL_SERVICE was observed, in manual testing, to silently deactivate
# panda_arm_controller. Nothing reports that back -- the gz service still
# answers Boolean(data=true), and /joint_states keeps flowing because
# joint_state_broadcaster is a separate controller that stays active -- so the
# reset looks entirely successful from every signal this node used to check.
#
# An inactive arm controller means every subsequent trajectory is rejected by
# the controller manager, which surfaces as an unrelated-looking execution
# failure on the *next* task. So the controller's state is now queried
# explicitly and is a hard precondition for reporting recovery.
ARM_CONTROLLER_NAME = "panda_arm_controller"
LIST_CONTROLLERS_SERVICE = "/controller_manager/list_controllers"
CONTROLLER_QUERY_TIMEOUT = 5.0
# Queried after the joint settle loop rather than immediately after the reset:
# if the joints happen to come back in range on the first 0.25s poll, the query
# would otherwise race the controller manager actually applying the
# deactivation, and read a stale "active". A short fixed settle on top of the
# settle loop closes that gap.
CONTROLLER_CHECK_SETTLE_SEC = 1.0

# Outcomes of _reset_diverged_state. Not a bool any more: "the reset did not
# fix the joints" and "the reset worked but left the arm uncontrollable" need
# different messages and are diagnosed differently, and collapsing them was
# how the second one got reported as success.
RESET_OK = "ok"
RESET_REFUSED = "refused"
RESET_JOINTS_STILL_DIVERGED = "joints_still_diverged"
RESET_CONTROLLER_NOT_ACTIVE = "controller_not_active"

# Slack allowed on top of the robot model's own joint limits before the state
# is called diverged. Only there to absorb float noise and benign controller
# overshoot -- the real limits come from the model (see _joint_bounds), never
# from a hardcoded guess. Anything beyond a model limit blocks planning
# outright (MoveIt's CheckStartStateBounds aborts the pipeline), so a small
# value is right: the observed divergence overshot by tens of radians.
JOINT_BOUND_TOLERANCE = 0.05

# How long to let the gripper's closing motion play out before teleport+attach.
# The close is purely visual now (see _send_gripper_command), so this only has
# to cover "the fingers have visibly begun to shut", not "the fingers have
# converged against real contact resistance".
GRIPPER_VISUAL_CLOSE_SETTLE_SEC = 0.4

# How long _gripper_transform() will wait for the planning scene monitor's
# robot state to catch up to now before running FK for the grasp teleport.
ROBOT_STATE_WAIT_SEC = 2.0

# How long to wait for a DetachableJoint's <output_topic> to confirm a
# requested attach/detach actually took effect, rather than assuming a
# published gz-transport message was received and acted on.
JOINT_STATE_CONFIRM_TIMEOUT = 3.0

# Release (place) side of the same confirm-and-retry discipline the attach
# side above already uses. This exists because the detach used to be treated
# as best-effort: publish, warn if unconfirmed, clear the tracking state
# anyway. With a measured ~50% gz-transport message loss rate on this setup
# that silently desynced the node from physics roughly every other place --
# the node believed the cube released and reported success, while the
# DetachableJoint still held it. Perception then reported the cube at gripper
# height instead of floor height, and the next task planned against that,
# which is how a cube ended up dropped from height during failure recovery.
#
# Three attempts against a ~50% per-message loss rate leaves ~12% odds of all
# three being lost, and each costs at most JOINT_STATE_CONFIRM_TIMEOUT, so the
# worst case is ~9s against a 600s task budget.
RELEASE_DETACH_MAX_ATTEMPTS = 3

# Deliberately longer than GRASP_VERIFY_SETTLE_SEC: a grasped cube is held
# rigidly by the joint the instant it forms, but a released one is falling the
# last millimetres onto the surface under gravity, so the resting pose needs a
# moment to actually become resting.
PLACE_VERIFY_SETTLE_SEC = 0.3

# Deliberately looser than TELEPORT_ARRIVAL_TOLERANCE (2mm) for the same
# reason: the arrival poll measures a cube that physics is not moving, whereas
# this measures one that has just settled onto a surface. 5mm still catches
# every failure mode that matters here (never released, released somewhere
# else, rolled off the target) without failing an otherwise-correct place over
# a millimetre of settle.
PLACE_VERIFY_TOLERANCE = 0.005


class ActuatorNode(Node):
    def __init__(self):
        super().__init__("actuator_node")
        self.callback_group = ReentrantCallbackGroup()

        self._have_joint_state = False
        self._last_joint_state = None
        # ARM_CONTROLLER_NAME's state as of the last divergence reset, so
        # _divergence_feedback can name it. Only written/read inside a single
        # _reset_diverged_state -> _divergence_feedback sequence, which runs
        # sequentially on one command.
        self._last_controller_state = None
        # Why the last model reset was not confirmed (no reply vs. an explicit
        # refusal), for the same reason and with the same lifetime.
        self._last_reset_refusal = None
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

        # Used only on the physics-divergence recovery path, to verify the arm
        # controller survived the world reset (see ARM_CONTROLLER_NAME).
        # Reentrant callback group so the call can be awaited from inside the
        # /execution_command callback without deadlocking the executor.
        self._list_controllers_client = self.create_client(
            ListControllers,
            LIST_CONTROLLERS_SERVICE,
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

        # DIAGNOSTIC (GRASP_DEBUG): Gazebo's own view of where the cubes are,
        # so the grasp checkpoints can compare against physics ground truth
        # rather than against /object_map (perception's derived view) or
        # against the pose this node just commanded. The cubes are top-level
        # models, so their pose on this feed is already world-frame.
        self._gz_pose_lock = threading.Lock()
        self._gz_model_pose = {}
        self._grasp_dbg_t0 = time.monotonic()
        self._grasp_dbg_target = None
        for _t in (f"/world/{WORLD_NAME}/pose/info", f"/world/{WORLD_NAME}/dynamic_pose/info"):
            self._gz_node.subscribe(GzPoseV, _t, self._on_gz_pose)

        self._rtf = None
        if SETPOSE_LATENCY_PROBE:
            self._gz_node.subscribe(
                GzWorldStats, f"/world/{WORLD_NAME}/stats",
                lambda m: setattr(self, "_rtf", m.real_time_factor)
            )

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

        # Real position limits for the arm's actuated joints, taken from the
        # robot model MoveIt itself planned against rather than duplicated
        # from the URDF/SRDF by hand -- so this can never drift out of sync
        # with the limits CheckStartStateBounds enforces.
        self._joint_bounds = {}
        _jmg = self.moveit.get_robot_model().get_joint_model_group(ARM_GROUP)
        for _name, _b in zip(_jmg.active_joint_model_names, _jmg.active_joint_model_bounds):
            _vb = _b[0] if isinstance(_b, (list, tuple)) else _b
            if getattr(_vb, "position_bounded", False):
                self._joint_bounds[_name] = (_vb.min_position, _vb.max_position)
        self.get_logger().info(
            f"Loaded position limits for {len(self._joint_bounds)} arm joints "
            "(used to detect physics divergence)"
        )
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
            self._last_joint_state = msg

    def _get_joint_position(self, joint_name: str):
        with self._js_lock:
            msg = self._last_joint_state
        if msg is None or joint_name not in msg.name:
            return None
        return msg.position[msg.name.index(joint_name)]

    # ------------------------------------------------------------------
    # Physics divergence: detection + Gazebo-level reset
    # ------------------------------------------------------------------

    def _diverged_joints(self):
        """Arm joints whose live position is outside the robot model's own
        limits, as [(name, value, lower, upper)].

        This is NOT an "unreachable pose" check. A pose can be perfectly
        valid and simply unplannable; this catches the different, worse case
        where the simulation itself has come apart and /joint_states reports
        physically impossible values (observed: panda_joint7 at tens of
        radians after a grasp+lift destabilised dartsim). In that state
        MoveIt's CheckStartStateBounds aborts the planning pipeline before it
        ever looks at the goal, so *every* subsequent request fails
        instantly -- including the recovery plan back to "ready", and
        including the next task's first motion. Distinguishing the two is the
        difference between "this task failed" and "the robot is wedged"."""
        bad = []
        for name, (lo, hi) in self._joint_bounds.items():
            v = self._get_joint_position(name)
            if v is None:
                continue
            if v < lo - JOINT_BOUND_TOLERANCE or v > hi + JOINT_BOUND_TOLERANCE:
                bad.append((name, v, lo, hi))
        return bad

    @staticmethod
    def _format_divergence(bad) -> str:
        return "; ".join(
            f"{n}={v:.3f} outside [{lo:.4f}, {hi:.4f}]" for n, v, lo, hi in bad
        )

    def _arm_controller_state(self, task_id: str):
        """The controller manager's own view of ARM_CONTROLLER_NAME: its state
        string ("active", "inactive", ...), the sentinel "absent" if the
        controller is not loaded at all, or None if the question could not be
        answered (service missing, or no reply in time).

        None and "absent" are deliberately distinct from "inactive": all three
        block a success report, but they point at different problems (a dead
        controller manager, an unloaded controller, and a deactivated one)."""
        if not self._list_controllers_client.wait_for_service(
            timeout_sec=CONTROLLER_QUERY_TIMEOUT
        ):
            self.get_logger().error(
                f"[{task_id}] {LIST_CONTROLLERS_SERVICE} is not available -- cannot "
                f"confirm whether '{ARM_CONTROLLER_NAME}' survived the reset"
            )
            return None

        resp = self._wait_for_future(
            self._list_controllers_client.call_async(ListControllers.Request()),
            timeout_sec=CONTROLLER_QUERY_TIMEOUT,
        )
        if resp is None:
            self.get_logger().error(
                f"[{task_id}] {LIST_CONTROLLERS_SERVICE} did not reply within "
                f"{CONTROLLER_QUERY_TIMEOUT}s"
            )
            return None

        for controller in resp.controller:
            if controller.name == ARM_CONTROLLER_NAME:
                return controller.state
        return "absent"

    def _reset_diverged_state(self, task_id: str) -> str:
        """Try to bring a diverged simulation back to a plannable state, and
        report precisely how far that got (one of the RESET_* constants).

        gz-sim offers no way to set individual joint positions, so the only
        lever available is a whole-world model reset (see
        WORLD_CONTROL_SERVICE). That is deliberately heavy-handed -- it also
        returns every cube to its spawn pose -- but the alternative is an arm
        that rejects every future request for the rest of the session.

        Crucially, the reset is not free: it has been observed to silently
        deactivate ARM_CONTROLLER_NAME. So "did the joints come back in range"
        is necessary but NOT sufficient to call this a recovery, and this
        returns RESET_OK only when the arm is both plannable AND still
        controllable."""
        self.get_logger().warn(
            f"[{task_id}] Requesting gz-sim model reset via {WORLD_CONTROL_SERVICE} "
            "(this also returns every cube to its spawn pose)"
        )
        req = GzWorldControl()
        req.reset.model_only = True
        ok, rep = self._gz_node.request(
            WORLD_CONTROL_SERVICE, req, GzWorldControl, GzBoolean, WORLD_CONTROL_TIMEOUT_MS
        )
        if not ok or not rep.data:
            # Distinguish the two, because they mean different things and this
            # world produces both: transport_ok=False is the same gz-transport
            # no-reply flakiness documented at SET_POSE_SERVICE (observed here
            # too, timing out at exactly WORLD_CONTROL_TIMEOUT_MS), and unlike
            # a service-level refusal it does NOT prove the reset was skipped.
            # Either way this reports failure rather than guessing -- a reset
            # that may or may not have happened is not a recovery -- but the
            # operator needs to know which one they are looking at.
            self._last_reset_refusal = (
                f"no reply within {WORLD_CONTROL_TIMEOUT_MS}ms (gz-transport did not "
                "answer; the reset may or may not have been applied)"
                if not ok else
                "gz-sim explicitly refused the request"
            )
            self.get_logger().error(
                f"[{task_id}] Model reset not confirmed: {self._last_reset_refusal}"
            )
            return RESET_REFUSED

        # A reset does not clear the DetachableJoints, and it invalidates any
        # object this node believed it was carrying.
        for name in DETACHABLE_CUBES:
            self._detach_pub[name].publish(GzEmpty())
        # pose=None, not a placeholder: the reset returned every cube to its
        # spawn pose, so this node has no idea where the one it was carrying
        # now is until perception reports again.
        if self._attached_object_name is not None:
            self._detach_object(None)

        joints_ok = False
        deadline = time.monotonic() + DIVERGENCE_RESET_SETTLE_SEC
        while time.monotonic() < deadline:
            if not self._diverged_joints():
                self.get_logger().warn(f"[{task_id}] Joint state back within limits after reset")
                joints_ok = True
                break
            time.sleep(0.25)

        # Checked on every path, including the one where the joints never came
        # back: an inactive controller is very likely *why* they did not (with
        # nothing driving them the arm just sags), and it is the more
        # actionable finding either way, so it is reported in preference to
        # the joint-range result below.
        time.sleep(CONTROLLER_CHECK_SETTLE_SEC)
        state = self._arm_controller_state(task_id)
        if state != "active":
            self.get_logger().error(
                f"[{task_id}] '{ARM_CONTROLLER_NAME}' is "
                f"{'unknown' if state is None else repr(state)} after the model reset "
                "-- the arm is not commandable. NOT reporting recovery."
            )
            self._last_controller_state = state
            return RESET_CONTROLLER_NOT_ACTIVE

        return RESET_OK if joints_ok else RESET_JOINTS_STILL_DIVERGED

    def _divergence_feedback(self, outcome: str, diverged) -> tuple:
        """(status, stage, detail) for a divergence-recovery outcome.

        Shared by both call sites so they cannot drift apart -- the bug this
        replaces was precisely one call site's success/failure mapping being
        wrong, and there is no reason for two copies of it to exist."""
        divergence = self._format_divergence(diverged)

        if outcome == RESET_OK:
            return (
                "success", "physics_divergence",
                "Recovered from a diverged joint state via gz-sim model reset; "
                f"'{ARM_CONTROLLER_NAME}' confirmed still active afterwards. Cube "
                "positions were reset too, so this task's target may have moved.",
            )

        if outcome == RESET_CONTROLLER_NOT_ACTIVE:
            # Deliberately a different stage from physics_divergence: this is
            # not "the sim blew up", it is "the recovery itself broke the
            # robot", and it needs a human, not a retry.
            state = self._last_controller_state
            return (
                "failed", "divergence_recovery_controller",
                f"gz-sim model reset left '{ARM_CONTROLLER_NAME}' in state "
                f"{'unknown (could not query the controller manager)' if state is None else repr(state)}, "
                "not 'active'. The arm will reject every trajectory until it is "
                "reactivated, so this is NOT a recovery and must not be reported as "
                f"one. Original divergence: {divergence}. Reactivate with: "
                f"ros2 control set_controller_state {ARM_CONTROLLER_NAME} active "
                "(verify the arm is physically safe to re-energise first), or restart "
                "the sim.",
            )

        if outcome == RESET_REFUSED:
            return (
                "failed", "physics_divergence",
                "Arm joint state is physically invalid and the gz-sim model reset was "
                f"not confirmed: {self._last_reset_refusal} ({divergence}). This is a "
                "simulation blow-up, not a planning failure -- the sim needs "
                "restarting.",
            )

        return (
            "failed", "physics_divergence",
            "Arm joint state is physically invalid and a gz-sim model reset did not "
            f"recover it ({divergence}). This is a simulation blow-up, not a planning "
            "failure -- the sim needs restarting.",
        )

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

    def _on_gz_pose(self, msg: GzPoseV):
        with self._gz_pose_lock:
            for p in msg.pose:
                self._gz_model_pose[p.name] = (p.position.x, p.position.y, p.position.z)

    def _gz_pose_of(self, name: str):
        with self._gz_pose_lock:
            return self._gz_model_pose.get(name)

    def _grasp_checkpoint(self, tag: str, name: str, t0: float, target=None):
        """DIAGNOSTIC (GRASP_DEBUG): one line per checkpoint through the
        teleport+attach window, so cube drift can be separated from a
        mis-computed teleport target.

        Deliberately reads FK with wait=False: the whole point is to measure
        what happens inside this window, and blocking on
        wait_for_current_robot_state() here would inflate the very timings
        being recorded."""
        if not GRASP_DEBUG:
            return
        cube = self._gz_pose_of(name)
        g = self._gripper_transform(wait=False)[:3, 3]
        gw = (float(g[0]) - FRAME_OFFSET["x"],
              float(g[1]) - FRAME_OFFSET["y"],
              float(g[2]) - FRAME_OFFSET["z"])
        with self._object_map_lock:
            om = self._object_map.get(name)
        bits = [f"[GRASP_DEBUG] {name:11s} {tag:<16s} t=+{(time.monotonic()-t0)*1000.0:8.1f}ms"]
        bits.append(f"cube_gz=({cube[0]:.5f},{cube[1]:.5f},{cube[2]:.5f})" if cube
                    else "cube_gz=<none>")
        bits.append(f"grip_fk_world=({gw[0]:.5f},{gw[1]:.5f},{gw[2]:.5f})")
        if om is not None:
            bits.append(f"cube_objmap=({om['x']:.5f},{om['y']:.5f},{om['z']:.5f})")
        if target is not None:
            bits.append(f"target=({target['x']:.5f},{target['y']:.5f},{target['z']:.5f})")
            if cube is not None:
                bits.append("cube-target=(%+.2f,%+.2f,%+.2f)mm" % (
                    (cube[0] - target["x"]) * 1000.0,
                    (cube[1] - target["y"]) * 1000.0,
                    (cube[2] - target["z"]) * 1000.0))
        self.get_logger().debug("  ".join(bits))

    def _gripper_transform(self, wait: bool = True):
        """Live 4x4 pose of GRIPPER_BASE_LINK in the BASE_FRAME
        (panda_link0), straight out of MoveIt's own state monitor.

        No new TF machinery is needed for this: the planning scene monitor
        this node already holds tracks the current robot state, and
        RobotState.get_global_link_transform() runs FK on it for any link
        in the model. That keeps the transform used for the grasp teleport
        consistent with the state MoveIt planned against, rather than
        introducing a second, independently-timed source of truth.

        The wait is load-bearing, not defensive: the monitor's current_state
        lags the controllers (this node logs "No state update received within
        100ms" routinely), and reading it unguarded straight after the descent
        was measured returning a gripper pose ~0.10m stale -- i.e. from
        partway down the approach. The teleport then placed the cube against
        that stale pose while the DetachableJoint froze it against the real
        one, leaving exactly the vertical offset this whole change exists to
        remove."""
        if wait:
            self._scene_monitor.wait_for_current_robot_state(
                self.get_clock().now(), ROBOT_STATE_WAIT_SEC
            )
        with self._scene_monitor.read_only() as scene:
            return np.asarray(scene.current_state.get_global_link_transform(GRIPPER_BASE_LINK))

    @staticmethod
    def _quat_from_matrix(m):
        """Rotation matrix -> (x, y, z, w), via Shepperd's method (pick the
        largest diagonal term so the square root never loses precision)."""
        t = m[0][0] + m[1][1] + m[2][2]
        if t > 0.0:
            s = math.sqrt(t + 1.0) * 2.0
            return ((m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s,
                    (m[1][0] - m[0][1]) / s, 0.25 * s)
        if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
            s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
            return (0.25 * s, (m[0][1] + m[1][0]) / s,
                    (m[0][2] + m[2][0]) / s, (m[2][1] - m[1][2]) / s)
        if m[1][1] > m[2][2]:
            s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
            return ((m[0][1] + m[1][0]) / s, 0.25 * s,
                    (m[1][2] + m[2][1]) / s, (m[0][2] - m[2][0]) / s)
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        return ((m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s,
                0.25 * s, (m[1][0] - m[0][1]) / s)

    def _teleport_to_grasp(self, name: str):
        """Move `name` to exactly the pose it would occupy if it were
        perfectly grasped -- GRASP_OFFSET_IN_GRIPPER in the gripper's own
        frame -- and return that pose in world coordinates (or None on
        failure).

        This is what replaces waiting for a real friction grip to
        establish itself. gz-sim's DetachableJoint freezes whatever
        relative transform happens to hold at the instant it attaches, so
        the only way to guarantee a clean, centered grasp is to *put* the
        object in the right place first rather than hope physics and
        perception have already converged on it.

        The object also adopts the gripper's own orientation, so its faces
        stay flush against the pads for any approach yaw rather than only
        the axis-aligned one _target_pose_stamped currently commands."""
        t = self._gripper_transform()
        rot, origin = t[:3, :3], t[:3, 3]
        grasp = origin + rot @ np.asarray(GRASP_OFFSET_IN_GRIPPER)

        # BASE_FRAME -> world is the inverse of the world -> BASE_FRAME shift
        # every other pose in this node goes through (_object_pose_stamped);
        # the panda is placed axis-aligned in panda_world.sdf, so it is a pure
        # translation with no rotation to undo.
        world = {
            "x": float(grasp[0]) - FRAME_OFFSET["x"],
            "y": float(grasp[1]) - FRAME_OFFSET["y"],
            "z": float(grasp[2]) - FRAME_OFFSET["z"],
        }
        quat = self._quat_from_matrix(rot)

        self._grasp_dbg_t0 = time.monotonic()
        self._grasp_dbg_target = world
        self._grasp_checkpoint("1-pre-set_pose", name, self._grasp_dbg_t0, world)

        # The target is deliberately NOT recomputed between _teleport_object's
        # internal attempts: the arm is stationary through this whole window
        # (checkpoint FK was byte-identical across all three points on every
        # cube), so a fresh FK read would add a second source of truth for no
        # benefit.
        if not self._teleport_object(name, world, quat):
            return None

        self._grasp_checkpoint("2-post-set_pose", name, self._grasp_dbg_t0, world)
        return world

    def _teleport_object(self, name: str, world: dict, quat) -> bool:
        """Ask Gazebo to put `name` at world-frame position `world` with
        orientation `quat` (x, y, z, w), and return whether it actually
        arrived there.

        Shared by both ends of a task: pick teleports the cube into the
        gripper's grasp pose before attaching (_teleport_to_grasp), and place
        teleports it onto the commanded target after detaching
        (_release_grasped_object). Both need the same thing from set_pose --
        not "was the request accepted" but "did the cube move" -- so both go
        through this."""
        req = GzPose()
        req.name = name
        req.position.x, req.position.y, req.position.z = world["x"], world["y"], world["z"]
        req.orientation.x, req.orientation.y = quat[0], quat[1]
        req.orientation.z, req.orientation.w = quat[2], quat[3]

        # Ask, then let the cube's measured position be the ONLY arbiter of
        # success. The ack carries no usable information in either direction:
        # a success ack arrives up to ~120ms before the pose is actually
        # applied, and a transport timeout (measured at exactly the 2.0s
        # request deadline, three times over, on two separate cubes) means
        # only that the reply was slow -- Gazebo may well have applied the
        # pose regardless. Treating a timeout as failure is what threw away
        # teleports that had very likely landed. So both paths fall through to
        # the same arrival poll.
        for attempt in range(1, TELEPORT_MAX_ATTEMPTS + 1):
            _probe_t0 = time.monotonic()
            ok, rep = self._gz_node.request(
                SET_POSE_SERVICE, req, GzPose, GzBoolean,
                SETPOSE_PROBE_TIMEOUT_MS if SETPOSE_LATENCY_PROBE else SET_POSE_TIMEOUT_MS
            )
            if SETPOSE_LATENCY_PROBE:
                _sp_ms = (time.monotonic() - _probe_t0) * 1000.0
                _c0 = time.monotonic()
                _cok, _ = self._gz_node.request(
                    CONTROL_SERVICE, GzEmpty(), GzEmpty, GzStringMsgV,
                    SETPOSE_PROBE_TIMEOUT_MS
                )
                _ctl_ms = (time.monotonic() - _c0) * 1000.0
                self.get_logger().warn(
                    f"[SETPOSE_PROBE] {name} attempt={attempt} "
                    f"set_pose={_sp_ms:9.1f}ms ok={ok} svc={getattr(rep, 'data', None)} | "
                    f"control({CONTROL_SERVICE})={_ctl_ms:9.1f}ms ok={_cok} | "
                    f"rtf={self._rtf if self._rtf is None else round(self._rtf, 3)}"
                )
            if not (ok and getattr(rep, "data", False)):
                svc = getattr(rep, "data", None) if ok else "n/a (no reply)"
                self.get_logger().warn(
                    f"{SET_POSE_SERVICE} gave no usable ack for '{name}' "
                    f"(transport_ok={ok}, service_ok={svc}) -- checking whether the "
                    f"pose landed anyway (attempt {attempt}/{TELEPORT_MAX_ATTEMPTS})"
                )

            if self._wait_for_object_at(name, world):
                return True

            self.get_logger().warn(
                f"'{name}' still not within {TELEPORT_ARRIVAL_TOLERANCE * 1000:.0f}mm of "
                f"({world['x']:.4f}, {world['y']:.4f}, {world['z']:.4f}) after "
                f"{TELEPORT_ARRIVAL_TIMEOUT}s (attempt {attempt}/{TELEPORT_MAX_ATTEMPTS})"
            )

        return False

    def _object_offset_from(self, name: str, target: dict):
        """Distance (m) between `name`'s ground-truth position and `target`,
        or None if Gazebo has not reported the object yet."""
        live = self._gz_pose_of(name)
        if live is None:
            return None
        return math.dist(live, (target["x"], target["y"], target["z"]))

    def _wait_for_object_at(self, name: str, target: dict) -> bool:
        """Block until Gazebo's pose feed shows `name` within
        TELEPORT_ARRIVAL_TOLERANCE of `target`, or the timeout expires.

        This is the confirmation set_pose itself does not provide -- its ack
        means "request accepted", not "pose applied". Without it the attach
        below can pin the cube at wherever it still happens to be."""
        deadline = time.monotonic() + TELEPORT_ARRIVAL_TIMEOUT
        goal = (target["x"], target["y"], target["z"])
        while True:
            live = self._gz_pose_of(name)
            if live is not None and math.dist(live, goal) <= TELEPORT_ARRIVAL_TOLERANCE:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(TELEPORT_POLL_INTERVAL)

    def _attach_grasped_object(self, task_id: str, name: str, pose: dict, plan_id: str) -> bool:
        """Place `name` exactly where a clean grasp would hold it, then pin
        it there via its DetachableJoint so it moves rigidly with the arm
        through lift/transport/retreat.

        The teleport replaces what used to be a proximity gate
        (_detect_grasp): that check asked whether physics and perception
        had *already* converged on the cube sitting roughly between the
        fingers, and if so let DetachableJoint freeze whatever residual
        error was present -- which is precisely how a visible cube/gripper
        offset got baked in for the rest of the task. Re-checking proximity
        after the teleport would only be re-measuring a pose this node just
        commanded, so it is dropped rather than replaced; the meaningful
        failure modes that remain (set_pose rejected, attach never
        confirmed) are both checked below.

        Only folds the object into MoveIt's planning scene (_attach_object)
        once the real attach is confirmed over <output_topic>, so a task
        reports "success" only if the grasp actually physically held."""
        held_err = None
        for cycle in range(1, GRASP_VERIFY_MAX_CYCLES + 1):
            grasp_pose = self._teleport_to_grasp(name)
            if grasp_pose is None:
                # Distinct from a "grasp" failure on purpose: nothing was
                # attempted with the gripper here. Gazebo either refused to
                # move the cube or never confirmed it arrived, across
                # TELEPORT_MAX_ATTEMPTS tries -- attaching anyway would pin a
                # cube that is somewhere other than between the pads, which is
                # the silent-misplacement failure this whole path exists to
                # prevent.
                self._publish_feedback(
                    task_id, "failed", "teleport",
                    f"Could not place '{name}' at the grasp pose after "
                    f"{TELEPORT_MAX_ATTEMPTS} attempts: {SET_POSE_SERVICE} either "
                    "refused the request or the cube never arrived within "
                    f"{TELEPORT_ARRIVAL_TOLERANCE * 1000:.0f}mm of target. Refusing "
                    "to attach a possibly-misplaced object.",
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

            # DIAGNOSTIC (GRASP_DEBUG): checkpoint 3, the moment the joint is
            # confirmed and the relative transform is frozen for the rest of
            # the task.
            self._grasp_checkpoint(
                "3-post-attach", name, self._grasp_dbg_t0, self._grasp_dbg_target
            )

            # The joint is formed, so whatever offset exists now is the one
            # the arm will carry for the whole task. Measure it rather than
            # assume the arrival check still holds.
            time.sleep(GRASP_VERIFY_SETTLE_SEC)
            held_err = self._object_offset_from(name, grasp_pose)
            if held_err is not None and held_err <= TELEPORT_ARRIVAL_TOLERANCE:
                if cycle > 1:
                    self.get_logger().info(
                        f"[{task_id}] '{name}' held within "
                        f"{held_err * 1000:.2f}mm after {cycle} teleport-attach cycles"
                    )
                self._attach_object(name, grasp_pose)
                return True

            self.get_logger().warn(
                f"[{task_id}] '{name}' drifted during the attach handshake -- held "
                f"{'unknown' if held_err is None else f'{held_err * 1000:.2f}mm'} off "
                f"target (tolerance {TELEPORT_ARRIVAL_TOLERANCE * 1000:.0f}mm), "
                f"detaching and retrying (cycle {cycle}/{GRASP_VERIFY_MAX_CYCLES})"
            )
            # Detach before retrying: set_pose cannot cleanly reposition a body
            # that is currently pinned to the gripper by the DetachableJoint.
            self._detach_pub[name].publish(GzEmpty())
            self._wait_for_detach_state(name, "detached", JOINT_STATE_CONFIRM_TIMEOUT)

        self._publish_feedback(
            task_id, "failed", "grasp_verify",
            f"'{name}' would not stay within {TELEPORT_ARRIVAL_TOLERANCE * 1000:.0f}mm "
            f"of the grasp pose through the attach handshake after "
            f"{GRASP_VERIFY_MAX_CYCLES} full teleport-attach cycles (last held offset "
            f"{'unknown' if held_err is None else f'{held_err * 1000:.2f}mm'}). "
            "Refusing to report a grasp that is out of tolerance.",
            plan_id
        )
        return False

    def _detach_with_confirm(self, task_id: str, name: str) -> bool:
        """Publish detach for `name` and wait for its <output_topic> to say it
        actually happened, retrying up to RELEASE_DETACH_MAX_ATTEMPTS.

        The publish call succeeding proves nothing -- gz-transport drops
        roughly half these messages here, and treating "published" as
        "released" is exactly the desync this exists to close. Callers must
        not clear any attachment state unless this returns True.

        Publishes no feedback and does no repositioning, so both the normal
        place path and failure recovery can use it; each reports failure in
        whatever way suits its context."""
        for attempt in range(1, RELEASE_DETACH_MAX_ATTEMPTS + 1):
            self._detach_pub[name].publish(GzEmpty())
            if self._wait_for_detach_state(name, "detached", JOINT_STATE_CONFIRM_TIMEOUT):
                if attempt > 1:
                    self.get_logger().info(
                        f"[{task_id}] DetachableJoint for '{name}' confirmed detached "
                        f"after {attempt} attempts"
                    )
                return True
            self.get_logger().warn(
                f"[{task_id}] DetachableJoint for '{name}' did not confirm detach within "
                f"{JOINT_STATE_CONFIRM_TIMEOUT}s (attempt {attempt}/"
                f"{RELEASE_DETACH_MAX_ATTEMPTS})"
            )
        return False

    def _release_grasped_object(self, task_id: str, pose: dict, plan_id: str) -> bool:
        """Put whatever is currently grasped down at exactly `pose` and let go
        of it -- physically (gz-sim's DetachableJoint) and in MoveIt's
        planning scene -- confirming each half actually happened.

        This is the mirror of _attach_grasped_object, and for the same reason.
        It used to be best-effort: publish detach, warn if unconfirmed, clear
        the tracking state regardless, on the theory that leaving MoveIt
        believing the arm is holding something is worse than a possibly-stale
        physical joint. That reasoning was backwards. With ~50% gz-transport
        message loss here, "clear the state regardless" meant the node
        routinely reported a release that never physically happened, and every
        downstream consumer -- perception, the next task's target -- then
        worked from a world model contradicting physics. A stale attachment
        the node KNOWS about is recoverable; one it has convinced itself is
        gone is not.

        So the internal tracking state is only ever cleared once the detach is
        confirmed over <output_topic>. If it is never confirmed, this fails
        loudly and leaves _attached_object_name and the planning-scene
        attachment exactly as they were, because that is the truth.

        Ordering note: the cube is teleported onto the target AFTER the joint
        is released, not before. set_pose cannot cleanly reposition a body
        that is currently pinned by the DetachableJoint (see the retry path in
        _attach_grasped_object), and the cube's link is the joint's
        <parent_link>, so moving it while joined pulls on the arm. By this
        point the arm has already descended to the place pose with the gripper
        open, so the cube is at the target to within a settle -- the teleport
        is a fine correction, not a drop.

        Returns True only if the object is confirmed released AND resting
        within PLACE_VERIFY_TOLERANCE of `pose`."""
        name = self._attached_object_name
        if name is None:
            return True

        # Phase A: detach, and require Gazebo to confirm it.
        if not self._detach_with_confirm(task_id, name):
            # Deliberately leaves _attached_object_name set and the object
            # attached in the planning scene. The cube is, as far as anything
            # can tell, still physically joined to the gripper; recording
            # otherwise is what turned a lost message into a silent desync.
            self._publish_feedback(
                task_id, "failed", "release_detach",
                f"DetachableJoint for '{name}' never confirmed detach after "
                f"{RELEASE_DETACH_MAX_ATTEMPTS} attempts -- the object is still held. "
                "Keeping it marked as attached rather than reporting a release that "
                "did not happen.",
                plan_id
            )
            return False

        # Past this point the object IS physically released, so the tracking
        # state must be cleared no matter how the rest of this goes -- the
        # same rule as above, just pointing the other way.
        placed = self._teleport_object(name, pose, (0.0, 0.0, 0.0, 1.0))

        # Phase C: confirm where it actually came to rest, rather than
        # trusting the pose this node just commanded.
        time.sleep(PLACE_VERIFY_SETTLE_SEC)
        rest = self._gz_pose_of(name)
        rest_err = (None if rest is None
                    else math.dist(rest, (pose["x"], pose["y"], pose["z"])))

        # Hand the planning scene the cube's ACTUAL resting position, not the
        # one it was asked for. Those differ exactly when the check below is
        # about to fail, and seeding MoveIt with the commanded pose in that
        # case would rebuild the same scene-contradicts-physics problem this
        # method exists to prevent, one layer up.
        self._detach_object(
            {"x": rest[0], "y": rest[1], "z": rest[2]} if rest is not None else pose
        )

        if not placed or rest_err is None or rest_err > PLACE_VERIFY_TOLERANCE:
            if not placed:
                why = (f"{SET_POSE_SERVICE} never got it within "
                       f"{TELEPORT_ARRIVAL_TOLERANCE * 1000:.0f}mm of the target across "
                       f"{TELEPORT_MAX_ATTEMPTS} attempts")
            elif rest_err is None:
                why = "Gazebo has not reported its pose, so where it landed is unknown"
            else:
                why = (f"it settled {rest_err * 1000:.2f}mm off, outside the "
                       f"{PLACE_VERIFY_TOLERANCE * 1000:.0f}mm tolerance")
            self._publish_feedback(
                task_id, "failed", "place_verify",
                f"'{name}' was released, but is not resting at the commanded target: "
                f"{why}. The gripper is empty and the object is no longer attached, "
                "but it is not where the task asked for it.",
                plan_id
            )
            return False

        return True

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

    def _detach_object(self, pose):
        """Release whatever is currently attached back into the world as a
        plain (no longer robot-carried) collision object at its drop
        location.

        `pose` may be None when the caller genuinely does not know where the
        object ended up (failure recovery, or after a world reset moved every
        cube). In that case the attachment is still cleared but no world
        collision object is re-added, because the alternative callers used to
        pass -- a placeholder {0, 0, 0} -- put a phantom obstacle at the
        robot's own base and told MoveIt to route around it."""
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
        if pose is None:
            self._remove_collision_object(name)
        else:
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

        if robot_type not in ('arm', 'robotic_arm'):
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

        # Divergence backstop, deliberately sited next to the /joint_states
        # readiness gate: both answer "is it meaningful to plan at all right
        # now?", and both must run before the execution lock is taken. This
        # is what stops one task's blow-up cascading -- without it a diverged
        # arm silently reports every following task as a routine planning
        # failure, which is exactly how a failed retreat on one cube turned
        # into an unrelated-looking failure on the next.
        diverged = self._diverged_joints()
        if diverged:
            self.get_logger().error(
                f"[{task_id}] PHYSICS DIVERGENCE: {self._format_divergence(diverged)}"
            )
            outcome = self._reset_diverged_state(task_id)
            status, stage, detail = self._divergence_feedback(outcome, diverged)
            self._publish_feedback(task_id, status, stage, detail, plan_id)
            # Any outcome short of RESET_OK -- including a reset that fixed the
            # joints but left the arm controller inactive -- means this command
            # must not proceed to planning.
            if outcome != RESET_OK:
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
        # Gripper-occupancy precondition, checked before ANY arm motion or
        # gripper actuation so a mismatch costs nothing and disturbs nothing.
        # Read here rather than in _execution_command_callback because
        # _attached_object_name is only mutated under _execution_lock, which
        # the caller holds around this.
        #
        # Picking while already holding something used to be unguarded, and it
        # does not fail gracefully: the pick path teleports the new cube into
        # the grasp pose and attaches it, i.e. into space the held cube
        # already occupies, and the physics engine resolves that
        # interpenetration by ejecting one of them violently. Refusing up
        # front is the only safe answer -- there is no "put the current one
        # down first" the actuator can invent, since it has no target to put
        # it down at.
        if action == "pick" and self._attached_object_name is not None:
            self._publish_feedback(
                task_id, "failed", "precondition",
                f"Cannot pick: the gripper is already holding "
                f"'{self._attached_object_name}'. It must be placed before another "
                "pick. Refusing before any motion so the held object is left "
                "undisturbed.",
                plan_id
            )
            # Plain return, NOT _recover_after_failure: recovery opens the
            # gripper and releases whatever is attached, which is precisely the
            # object this guard exists to protect.
            return

        # The mirror case, and a direct symptom of the desync this node's
        # release path now prevents: a place dispatched with nothing attached
        # means someone's world model is wrong. Reporting success for moving
        # air would hide that.
        if action == "place" and self._attached_object_name is None:
            self._publish_feedback(
                task_id, "failed", "precondition",
                "Cannot place: the gripper is not holding anything. Either the pick "
                "never completed or this node's attachment state has desynced from "
                "physics.",
                plan_id
            )
            return

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

        # Actuate gripper: close to grasp (pick) or open to release (place).
        # The grasping close is fire-and-forget -- what actually holds the
        # object is the DetachableJoint attached just below, so the close only
        # has to *look* right and there is nothing to be gained by blocking
        # until the position loop converges against contact resistance.
        gripper_position = GRIPPER_POSITION.get(action)
        if gripper_position is not None:
            if not self._send_gripper_command(
                task_id, gripper_position, plan_id, wait_for_result=(action != "pick")
            ):
                return self._recover_after_failure(task_id, plan_id)

        # Now that the fingers have visibly closed, put the object exactly
        # where a clean grasp would hold it and pin it there via its
        # DetachableJoint -- see _attach_grasped_object for why (bullet-
        # featherstone's own contact solving isn't reliable enough to hold a
        # grasp through lift/transport on its own). This also folds the object into
        # MoveIt's own collision body so retreat doesn't treat it as a
        # static obstacle to route around and it moves rigidly with the
        # gripper in planning too, instead of MoveIt still believing the
        # world is empty-handed.
        if action == "pick" and object_name is not None:
            if not self._attach_grasped_object(task_id, object_name, pose, plan_id):
                return self._recover_after_failure(task_id, plan_id)
        elif action == "place":
            # Unlike the old best-effort call this replaces, a release that
            # cannot be confirmed fails the task. It has already published its
            # own distinct feedback (release_detach / place_verify).
            if not self._release_grasped_object(task_id, pose, plan_id):
                return self._recover_after_failure(task_id, plan_id)

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

        # Check for divergence first: if the joint state is physically
        # invalid then nothing below can work -- the gripper command still
        # goes through, but every plan (including the one back to "ready")
        # aborts at CheckStartStateBounds before a goal is even sampled, so
        # the old code's "Could not plan back to 'ready'" warning was really
        # reporting a wedged simulation as if it were an awkward pose.
        # Handling it here rather than only at the next command's entry gate
        # means the arm is recovered at the point of failure.
        diverged = self._diverged_joints()
        if diverged:
            self.get_logger().error(
                f"[{task_id}] PHYSICS DIVERGENCE during recovery: "
                f"{self._format_divergence(diverged)}"
            )
            outcome = self._reset_diverged_state(task_id)
            status, stage, detail = self._divergence_feedback(outcome, diverged)
            self._publish_feedback(task_id, status, stage, detail, plan_id)
            # A successful reset already put the arm at its SDF home
            # configuration, so the rest of this routine has nothing to do.
            if outcome == RESET_OK:
                return
            # Everything below drives the arm and gripper. With the controller
            # inactive that is not merely futile, it would bury the real
            # finding under a pile of rejected-trajectory noise -- and the
            # "plan back to ready" at the end would report a controller
            # problem as an awkward-pose problem, the same conflation this
            # whole path exists to undo. Stop here and leave it for a human.
            if outcome == RESET_CONTROLLER_NOT_ACTIVE:
                self.get_logger().error(
                    f"[{task_id}] Skipping arm/gripper recovery: "
                    f"'{ARM_CONTROLLER_NAME}' is not active, so no trajectory can run."
                )
                return

        # Release the gripper regardless of what it was doing -- a
        # partially-closed grip on nothing, or on an object at an unknown
        # angle, is worse to leave for the next task than an open one.
        self._send_gripper_command(task_id, GRIPPER_POSITION["place"], plan_id)

        # If something was attached (grasped successfully, then a later
        # phase -- e.g. retreat -- failed), detach it back into the world
        # rather than leaving MoveIt believing the robot is still holding an
        # object it may no longer have a working grip on.
        #
        # Deliberately NOT _release_grasped_object: that is the place path,
        # and it would do two things wrong here. It teleports the object onto
        # a commanded target, and a recovery has no commanded target -- the
        # object just falls wherever the gripper happens to be. And it
        # publishes its own task feedback, which would emit a second "failed"
        # for a task whose real failure was already reported (see this
        # method's docstring). So recovery uses the shared confirm step
        # directly and reports through the log instead.
        if self._attached_object_name is not None:
            name = self._attached_object_name
            if self._detach_with_confirm(task_id, name):
                # Where it landed is whatever physics decided; prefer Gazebo's
                # ground truth over perception, and accept not knowing.
                gz = self._gz_pose_of(name)
                if gz is not None:
                    rest = {"x": gz[0], "y": gz[1], "z": gz[2]}
                else:
                    with self._object_map_lock:
                        live_pose = self._object_map.get(name)
                    rest = dict(live_pose) if live_pose else None
                self._detach_object(rest)
            else:
                # Same rule as the place path: an unconfirmed detach leaves
                # the tracking state alone, because the object is still held.
                # The next command's precondition guard will refuse a pick on
                # that basis, which is the correct outcome.
                self.get_logger().error(
                    f"[{task_id}] Could not confirm detach of '{name}' during recovery -- "
                    "it is still attached to the gripper. Leaving it marked as held so "
                    "this node's state keeps matching physics."
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
        # No X/Y correction needed: the fingertip pads are symmetric about
        # EEF_LINK's own x=0/y=0 (see GRASP_OFFSET_IN_GRIPPER's derivation --
        # the pads meet exactly on the centreline), so EEF_LINK's X/Y is
        # already the grasp centre at this orientation.
        target.pose.position.x = pose["x"] + FRAME_OFFSET["x"]
        target.pose.position.y = pose["y"] + FRAME_OFFSET["y"]
        # EEF_LINK (robotiq_85_base_link, what MoveIt actually plans to) sits
        # EEF_TO_GRASP_Z above the fingertip pads' contact patch at this fixed
        # grasp orientation, so commanding it to "pose[z]" alone would leave
        # the pads that far above the target. See EEF_TO_GRASP_Z for the full
        # derivation from model.sdf plus the fingertip collision mesh.
        target.pose.position.z = pose["z"] + FRAME_OFFSET["z"] + EEF_TO_GRASP_Z
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

    def _send_gripper_command(
        self, task_id: str, position: float, plan_id: str = "", wait_for_result: bool = True
    ) -> bool:
        """`wait_for_result=False` returns as soon as the controller has
        *accepted* the goal (plus a short settle for the motion to visibly
        start), instead of blocking until it reports the commanded position
        reached. Used for the grasping close, where the grip is no longer
        load-bearing -- the object is pinned by its DetachableJoint, not by
        friction -- so waiting for the position loop to converge against
        real contact resistance bought nothing but the multi-second stall
        it was observed causing."""
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

        if not wait_for_result:
            time.sleep(GRIPPER_VISUAL_CLOSE_SETTLE_SEC)
            self._publish_feedback(
                task_id, "success", "gripper",
                f"Gripper '{label}' goal accepted (not waiting for convergence)", plan_id
            )
            return True

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
            # A close that stalled partway (not still ~open) means the
            # gripper is physically pressed against something -- bullet-
            # featherstone's contact solving can prevent that from ever
            # cleanly reporting "reached", but the resistance itself is
            # real evidence of a grip. Treat it as a (stalled) success so
            # the pick can still proceed into grasp confirmation instead
            # of failing outright. "open" timeouts get no such leniency --
            # there's nothing it could be legitimately stuck against.
            if label == "close":
                stalled_position = self._get_joint_position(GRIPPER_JOINT_NAME)
                if stalled_position is not None and stalled_position >= GRIPPER_STALL_MIN_POSITION:
                    self._publish_feedback(
                        task_id, "success", "gripper",
                        f"Gripper 'close' timed out but stalled at position {stalled_position:.3f} "
                        "-- treating as gripping an object",
                        plan_id
                    )
                    return True

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