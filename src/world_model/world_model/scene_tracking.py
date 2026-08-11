"""
scene_tracking.py
------------------
Derives which world entities are trackable -- "objects" (things the arm can
manipulate), "zones" (named navigation targets for the mobile robot), and
"robots" (the arm and mobile base themselves) -- by parsing the Gazebo world
SDF directly, instead of a hand-maintained allowlist.

Objects/zones are declared as top-level <model name="..."> under <world>,
and already follow a naming convention: object models end in "_cube", zone
models end in "_zone". Robots are declared differently -- via <include>,
not <model> -- which is exactly the same distinction get_robot_base_pose()
in build_environment.py already relies on to find the arm's spawn pose, so
this just generalizes that to "every <include>'d model is a robot."
Everything else (ground_plane, floor_mat, wall_*) matches neither shape and
is excluded automatically.

A model's own internal links (e.g. panda_link0..8, the gripper links) are
nested INSIDE that <model>/<include>, not siblings of it under <world> --
world.findall(...) only sees direct children, so link-level noise is never
a candidate here regardless of naming.

Deliberately dependency-light (stdlib XML parsing only, nothing else) so
both perception_node.py (a ROS2/Gazebo node) and build_environment.py (a
plain Layer 1 helper with no ROS dependency of its own) can import this
without either one dragging in the other's dependencies (rclpy,
gz.transport13) just to get this classification.
"""

from functools import lru_cache
import xml.etree.ElementTree as ET

OBJECT_SUFFIX = "_cube"
ZONE_SUFFIX = "_zone"


@lru_cache(maxsize=None)
def get_tracked_names(sdf_path: str) -> tuple[frozenset, frozenset, frozenset]:
    """
    Parse a Gazebo world SDF and return (object_names, zone_names,
    robot_names):
      - object_names/zone_names: top-level <model name="..."> matching the
        world's own _cube/_zone naming convention.
      - robot_names: every <include><name>...</name></include> at the top
        level -- how robots are spawned in this world, as opposed to props.

    Cached per sdf_path -- the world file doesn't change at runtime, and
    both callers (perception_node.py at startup, build_environment_prompt
    once per instruction) would otherwise re-parse the same small XML file
    repeatedly for no reason.
    """
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    world = root.find("world")
    if world is None:
        raise ValueError(f"No <world> element found in {sdf_path}")

    objects, zones = set(), set()
    for model in world.findall("model"):
        name = model.get("name")
        if not name:
            continue
        if name.endswith(OBJECT_SUFFIX):
            objects.add(name)
        elif name.endswith(ZONE_SUFFIX):
            zones.add(name)

    robots = set()
    for include in world.findall("include"):
        name_el = include.find("name")
        if name_el is not None and name_el.text:
            robots.add(name_el.text.strip())

    return frozenset(objects), frozenset(zones), frozenset(robots)
