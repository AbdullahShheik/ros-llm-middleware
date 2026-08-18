"""
Placement-accuracy instrumentation for Layer 1's eval log (see
layer1_pipeline.py's _record_placement_error / _finalize_run).

Computes the GEOMETRIC TRUTH target for a "place" subtask: the pose that
subtask's own args (landmark, or relative_to+direction+distance) deterministically
resolve to under the pattern-geometry rules in SYSTEM_PROMPT rule 15 --
independent of whatever action_dispatcher.dispatcher_node actually resolved
and sent to the actuator for that same subtask. Comparing the two is what
isolates a Layer 2 resolution bug (stale relative_to reference, wrong
landmark lookup, rounding) from a Layer 3 execution bug (physics settle,
gripper release, teleport drift).

The geometry function itself already exists and is not duplicated here: it
is resolve_landmark()/resolve_relative() in
action_dispatcher/action_dispatcher/spatial_placement.py -- the exact
function dispatcher_node.py's own 'place' branch calls to get ITS target
pose. Loaded here by file path rather than package import: action_dispatcher
is a separate ROS2 package with no ament_python packaging (dispatcher_node.py
itself only becomes importable via the sys.path[0]-is-the-script's-own-
directory trick -- see the ModuleNotFoundError this caused under
`colcon build --symlink-install`), so it is not reachable on a generic
PYTHONPATH the way an installed package would be.
"""

import importlib.util
import math
from pathlib import Path

_ACTION_DISPATCHER_PKG = Path(__file__).resolve().parents[1] / "action_dispatcher" / "action_dispatcher"


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ACTION_DISPATCHER_PKG / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spatial_placement = _load_module("action_dispatcher_spatial_placement", "spatial_placement.py")

# The same OBJECT_NAME_MAP dispatcher_node.py's 'place' branch uses to
# resolve a relative_to reference's name against object_map -- loaded by
# file path, same as spatial_placement.py above, rather than imported from
# dispatcher_node.py itself: that module pulls in rclpy at module level,
# which this eval-only helper has no other reason to depend on.
OBJECT_NAME_ALIASES = _load_module(
    "action_dispatcher_object_name_map", "object_name_map.py"
).OBJECT_NAME_MAP


def compute_geometric_truth(args: dict, object_map: dict):
    """The pose a 'place' subtask's own args resolve to under the pure
    landmark/relative_to geometry rules, as a {"x","y","z"} dict, or None if:
      - the subtask used target_location instead (a direct object/coordinate
        reference, not a pattern-geometry placement -- see rule 15), or
      - a relative_to reference isn't (yet) in object_map, or
      - the args fail to resolve at all (bad landmark name, bad direction).

    Mirrors dispatcher_node.py's own 'place' branch exactly (same
    resolve_relative/resolve_landmark calls, same relative_to-wins-over-
    landmark precedence), but reads object_map as Layer 1 currently sees it
    rather than as Layer 2 saw it at dispatch time. Any divergence between
    the two is itself a meaningful signal for planning_error_m, not a bug in
    this function.
    """
    relative_to = args.get("relative_to")
    landmark = args.get("landmark")

    if relative_to:
        ref_name = OBJECT_NAME_ALIASES.get(relative_to, relative_to)
        ref_pose = object_map.get(ref_name)
        if ref_pose is None:
            return None
        try:
            return spatial_placement.resolve_relative(
                ref_pose, args.get("direction"), float(args.get("distance", 1)),
            )
        except (ValueError, TypeError):
            return None

    if landmark:
        try:
            return spatial_placement.resolve_landmark(landmark)
        except ValueError:
            return None

    return None


def euclidean(a, b):
    """3D distance between two poses, each either a {"x","y","z"} dict or an
    (x, y, z) sequence -- or None if either side is missing/unresolved."""
    if a is None or b is None:
        return None
    ax, ay, az = (a["x"], a["y"], a["z"]) if isinstance(a, dict) else a
    bx, by, bz = (b["x"], b["y"], b["z"]) if isinstance(b, dict) else b
    return math.dist((ax, ay, az), (bx, by, bz))


def xyz_list(pose):
    """{"x","y","z"} dict or (x,y,z) sequence -> [x, y, z] for JSON logging,
    or None."""
    if pose is None:
        return None
    return [pose["x"], pose["y"], pose["z"]] if isinstance(pose, dict) else list(pose)
