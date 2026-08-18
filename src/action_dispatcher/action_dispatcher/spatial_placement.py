#!/usr/bin/env python3
"""Deterministic placement geometry for the `place` skill's `landmark` and
`relative_to` args.

Neither the LLM nor this module ever hardcodes coordinates for a named
shape ("triangle", "square", ...). Instead the LLM composes two general
primitives -- an anchor (`resolve_landmark`) and an offset from another
object's live position (`resolve_relative`) -- and the caller (typically
`dispatcher_node.py`) chains them per subtask. Every named point below is
derived from the single `WORKSPACE_BOUNDS` rectangle, not a per-shape table.
"""

import math

# One cube edge (0.06m, see panda_world.sdf) plus clearance margin for the
# gripper. The unit "1" in `distance` always means this many meters.
DEFAULT_SPACING_M = 0.12

# Centered in the SHARED region both arm bases can reach, not just arm1's
# own reach as this box used to be. Confirmed live as a real bug, not just
# a leftover: "landmark: center" (the default anchor most multi-object
# arrangements start from) resolved to a point in front of arm1 alone, so
# a two-arm build never actually happened in the space between the arms
# the way the feature was meant to work.
#
# Both arms are now rotated inward, mirrored, so they angle toward each
# other instead of both facing the same +X direction (see panda_world.sdf
# and actuator_node.py's BASE_YAW) -- see attach_detach_node.py's
# PICKUP_POINT_X/Y for the full story of why 50deg, not the 45deg (or the
# unrotated/single-arm-rotated schemes) tried before it. This box sits
# near pickup_point (0.62, 0.50, the point both arms' straight-ahead
# reach converges on) but offset enough that a cube waiting to be picked
# up doesn't overlap one already placed (~0.18-0.25m clearance). Every
# corner keeps each arm's own lateral offset under ~0.24m and total reach
# under 75% of the Panda's ~0.855m nominal max. The only hardcoded
# geometry here -- every named landmark below is derived from this one
# rectangle.
WORKSPACE_BOUNDS = {"x_min": 0.38, "x_max": 0.46, "y_min": 0.42, "y_max": 0.58}
_WORKSPACE_Z = 0.04

# Horizontal directions offset x/y in the workspace plane; `above`/`below`
# offset z instead (same x/y as the reference), for stacking. One fixed,
# documented convention: "front"/away_from_arm is +x (deeper into the
# workspace, away from the Panda's base), "left" is +y.
DIRECTION_VECTORS = {
    "front": (1, 0, 0),
    "away_from_arm": (1, 0, 0),
    "behind": (-1, 0, 0),
    "toward_arm": (-1, 0, 0),
    "left": (0, 1, 0),
    "right": (0, -1, 0),
    "above": (0, 0, 1),
    "below": (0, 0, -1),
}


def resolve_landmark(name: str | None) -> dict:
    """Resolve a named workspace landmark to a world pose.

    `None` or an unrecognized name falls back to "center" -- the single
    default anchor used whenever an instruction doesn't specify where to
    start. Recognized names are derived from WORKSPACE_BOUNDS, not looked
    up in a per-shape table.
    """
    b = WORKSPACE_BOUNDS
    landmarks = {
        "top_left": (b["x_min"], b["y_max"]),
        "top_right": (b["x_max"], b["y_max"]),
        "bottom_left": (b["x_min"], b["y_min"]),
        "bottom_right": (b["x_max"], b["y_min"]),
        "center": ((b["x_min"] + b["x_max"]) / 2, (b["y_min"] + b["y_max"]) / 2),
    }
    if name is None:
        name = "center"
    if name not in landmarks:
        raise ValueError(
            f"Unknown landmark '{name}'. Valid landmarks: {sorted(landmarks)}"
        )
    x, y = landmarks[name]
    return {"x": x, "y": y, "z": _WORKSPACE_Z}


def resolve_relative(
    reference_pose: dict,
    direction: str,
    distance: float = 1,
    spacing: float = DEFAULT_SPACING_M,
) -> dict:
    """Offset from a reference object's (live) pose by `distance` units of
    `spacing` along `direction`. Horizontal directions move x/y and keep
    the reference's z; `above`/`below` move z and keep x/y -- so stacking
    ("place this on top of that") and beside-placement ("to the left of
    that") share one mechanism instead of two.
    """
    if direction not in DIRECTION_VECTORS:
        raise ValueError(
            f"Unknown direction '{direction}'. Valid directions: "
            f"{sorted(DIRECTION_VECTORS)}"
        )
    dx, dy, dz = DIRECTION_VECTORS[direction]
    offset = distance * spacing
    return {
        "x": reference_pose["x"] + dx * offset,
        "y": reference_pose["y"] + dy * offset,
        "z": reference_pose["z"] + dz * offset,
    }
