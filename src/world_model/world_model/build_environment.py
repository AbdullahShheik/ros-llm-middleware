#!/usr/bin/env python3

import yaml
import numpy as np
from PIL import Image
import os
import math
import xml.etree.ElementTree as ET

from world_model.scene_tracking import get_tracked_names


PANDA_REACH_RADIUS_M = 0.855


# ---------------------------------------------------------------------------
# SDF parsing
# ---------------------------------------------------------------------------

def get_robot_base_pose(sdf_path: str, model_name: str) -> dict:
    """
    Parse a Gazebo SDF world file and return the pose of a named model.

    Args:
        sdf_path:   absolute path to the .sdf world file
        model_name: the model name as defined in the SDF <include><name> tag

    Returns:
        dict with keys x, y, z (floats)
    """
    tree = ET.parse(sdf_path)
    root = tree.getroot()

    world = root.find("world")
    if world is None:
        raise ValueError(f"No <world> element found in {sdf_path}")

    # Search <include> blocks — panda is spawned via <include><name>panda</name>
    for include in world.findall("include"):
        name_el = include.find("name")
        if name_el is not None and name_el.text.strip() == model_name:
            pose_el = include.find("pose")
            if pose_el is None:
                raise ValueError(f"Model '{model_name}' found but has no <pose>")
            parts = pose_el.text.strip().split()
            return {
                "x": float(parts[0]),
                "y": float(parts[1]),
                "z": float(parts[2]),
            }

    # Also search direct <model> blocks just in case
    for model in world.findall("model"):
        if model.get("name") == model_name:
            pose_el = model.find("pose")
            if pose_el is None:
                raise ValueError(f"Model '{model_name}' found but has no <pose>")
            parts = pose_el.text.strip().split()
            return {
                "x": float(parts[0]),
                "y": float(parts[1]),
                "z": float(parts[2]),
            }

    raise ValueError(f"Model '{model_name}' not found in {sdf_path}")


def resolve_robot_poses(sdf_path: str, tracked_robots: frozenset, object_map: dict) -> dict:
    """
    For each tracked robot, prefer its LIVE position from object_map
    (published by perception_node.py off the same Gazebo pose stream that
    already gives us live object/zone positions). Fall back to the static
    SDF spawn pose only if no live reading has arrived yet -- e.g. before
    perception_node's first publish, or if it isn't running.

    Returns {name: {"x", "y", "z", "live": bool}} -- "live" is surfaced in
    the rendered prompt so a stale fallback is never silently mistaken for
    a fresh reading.
    """
    poses = {}
    for name in tracked_robots:
        if name in object_map:
            poses[name] = {**object_map[name], "live": True}
        else:
            poses[name] = {**get_robot_base_pose(sdf_path, name), "live": False}
    return poses


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------

def load_map(map_yaml_path: str) -> dict:
    """Load map.yaml and the associated .pgm file."""
    with open(map_yaml_path, "r") as f:
        meta = yaml.safe_load(f)

    pgm_path = os.path.join(os.path.dirname(map_yaml_path), meta["image"])
    img = Image.open(pgm_path).convert("L")
    grid = np.array(img)

    image_height, image_width = grid.shape
    resolution = float(meta["resolution"])

    cell_size_m = 0.5
    grid_cols = max(1, int((image_width * resolution) / cell_size_m))
    grid_rows = max(1, int((image_height * resolution) / cell_size_m))

    return {
        "grid": grid,
        "resolution": resolution,
        "origin": meta["origin"],
        "occupied_thresh": meta.get("occupied_thresh"),
        "free_thresh": meta.get("free_thresh"),
        "negate": meta.get("negate", 0),
        "image_width": image_width,
        "image_height": image_height,
        "grid_cols": grid_cols,
        "grid_rows": grid_rows,
    }


# ---------------------------------------------------------------------------
# Coordinate utilities
# ---------------------------------------------------------------------------

def world_to_pixel(x: float, y: float, map_meta: dict) -> tuple:
    """Convert world (x, y) in meters to pixel (row, col) in the map image."""
    ox, oy = map_meta["origin"][0], map_meta["origin"][1]
    res = map_meta["resolution"]
    col = int((x - ox) / res)
    row = int((y - oy) / res)
    row = map_meta["image_height"] - 1 - row
    row = max(0, min(map_meta["image_height"] - 1, row))
    col = max(0, min(map_meta["image_width"] - 1, col))
    return row, col


# ---------------------------------------------------------------------------
# ASCII grid builder
# ---------------------------------------------------------------------------

def build_ascii_grid(map_meta: dict, object_map: dict, arm_center: tuple,
                      tracked_objects: frozenset, tracked_zones: frozenset,
                      robot_poses: dict = None) -> str:
    """
    Downsample the occupancy grid to grid_rows x grid_cols ASCII
    where each cell = 0.5m x 0.5m.

    Symbols:
      # = occupied / wall
      . = free space
      A = arm workspace (do not navigate through)
      O = tracked object (cube)
      Z = named zone
      R = robot (live position if available, else spawn -- see resolve_robot_poses)
    """
    raw = map_meta["grid"]
    img_h = map_meta["image_height"]
    img_w = map_meta["image_width"]
    res = map_meta["resolution"]
    origin = map_meta["origin"]
    negate = map_meta.get("negate", 0)
    occ_thresh = map_meta.get("occupied_thresh")
    free_thresh = map_meta.get("free_thresh")

    cols = map_meta["grid_cols"]
    rows = map_meta["grid_rows"]

    px_per_cell_x = img_w / cols
    px_per_cell_y = img_h / rows

    ascii_rows = [["." for _ in range(cols)] for _ in range(rows)]

    for r in range(rows):
        for c in range(cols):
            x0 = int(math.floor(c * px_per_cell_x))
            x1 = int(math.floor((c + 1) * px_per_cell_x))
            y0 = int(math.floor(r * px_per_cell_y))
            y1 = int(math.floor((r + 1) * px_per_cell_y))

            x0 = max(0, min(img_w - 1, x0))
            x1 = max(1, min(img_w, x1))
            y0 = max(0, min(img_h - 1, y0))
            y1 = max(1, min(img_h, y1))

            block = raw[y0:y1, x0:x1].astype(float) / 255.0
            if negate:
                block = 1.0 - block
            mean_val = float(block.mean()) if block.size > 0 else 1.0

            if free_thresh is not None and mean_val < free_thresh:
                symbol = "#"
            elif occ_thresh is not None and mean_val > occ_thresh:
                symbol = "."
            else:
                symbol = "?"

            # World coords of this cell center
            px_col = x0 + (x1 - x0) // 2
            px_row = y0 + (y1 - y0) // 2
            world_x = origin[0] + px_col * res
            world_y = origin[1] + (img_h - 1 - px_row) * res

            # Mark arm workspace — only overwrite free cells, never walls
            if symbol == "." and arm_center is not None:
                dx = world_x - arm_center[0]
                dy = world_y - arm_center[1]
                if math.hypot(dx, dy) <= PANDA_REACH_RADIUS_M:
                    symbol = "A"

            ascii_rows[r][c] = symbol

    # Overlay live objects and zones on top
    if object_map:
        for name, pos in object_map.items():
            row_px, col_px = world_to_pixel(pos["x"], pos["y"], map_meta)
            ascii_col = min(cols - 1, max(0, int(col_px // max(1, px_per_cell_x))))
            ascii_row = min(rows - 1, max(0, int((img_h - 1 - row_px) // max(1, px_per_cell_y))))

            if name in tracked_objects:
                ascii_rows[ascii_row][ascii_col] = "O"
            elif name in tracked_zones:
                ascii_rows[ascii_row][ascii_col] = "Z"

    # Overlay robots (live position if available, else spawn fallback)
    if robot_poses:
        for pos in robot_poses.values():
            row_px, col_px = world_to_pixel(pos["x"], pos["y"], map_meta)
            ascii_col = min(cols - 1, max(0, int(col_px // max(1, px_per_cell_x))))
            ascii_row = min(rows - 1, max(0, int((img_h - 1 - row_px) // max(1, px_per_cell_y))))
            ascii_rows[ascii_row][ascii_col] = "R"

    return "\n".join(" ".join(row) for row in ascii_rows)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_environment_prompt(
    map_yaml_path: str,
    sdf_path: str,
    object_map: dict,
    robot_model_name: str = "panda",
) -> str:
    """
    Main entry point called by layer1_pipeline.py.

    Args:
        map_yaml_path:    absolute path to map.yaml
        sdf_path:         absolute path to the Gazebo world .sdf file
        object_map:       latest dict from /object_map topic (parsed JSON)
        robot_model_name: model name of the arm in the SDF (default: "panda")

    Returns:
        A formatted string to prepend to the Layer 1 LLM prompt.
    """
    map_meta = load_map(map_yaml_path)

    tracked_objects, tracked_zones, tracked_robots = get_tracked_names(sdf_path)
    objects = {k: v for k, v in object_map.items() if k in tracked_objects}
    zones   = {k: v for k, v in object_map.items() if k in tracked_zones}
    robots  = resolve_robot_poses(sdf_path, tracked_robots, object_map)

    arm_pose = robots.get(robot_model_name) or get_robot_base_pose(sdf_path, robot_model_name)
    arm_center = (arm_pose["x"], arm_pose["y"])

    ascii_grid = build_ascii_grid(map_meta, object_map, arm_center, tracked_objects, tracked_zones, robots)

    lines = []
    lines.append("=== ENVIRONMENT CONTEXT ===")
    lines.append("")
    lines.append(
        f"Map grid ({map_meta['grid_rows']}x{map_meta['grid_cols']}, "
        f"cell=0.5m, resolution={map_meta['resolution']}m/pixel, "
        f"origin={map_meta['origin']}):"
    )
    lines.append(
        "  # = wall/obstacle  . = free  "
        "A = arm workspace (no wheeled navigation)  "
        "O = object  Z = zone  R = robot"
    )
    lines.append("")
    for row in ascii_grid.split("\n"):
        lines.append("  " + row)

    lines.append("")
    lines.append(f"Arm reach radius: {PANDA_REACH_RADIUS_M}m (centered on '{robot_model_name}' below)")

    # Compact, structured (not prose) -- an LLM doing spatial reasoning needs
    # name + rounded coordinates, not a natural-language sentence per object.
    # Each robot is tagged [live] (from perception, this instant) or
    # [spawn] (static SDF fallback -- no /object_map reading for it yet) so
    # a stale position is never silently mistaken for a fresh one.
    lines.append("")
    lines.append("ROBOTS:")
    for name, pos in sorted(robots.items()):
        tag = "live" if pos["live"] else "spawn"
        lines.append(f"  {name} ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}) [{tag}]")

    lines.append("")
    lines.append("OBJECTS:")
    if objects:
        for name, pos in sorted(objects.items()):
            dx = pos["x"] - arm_center[0]
            dy = pos["y"] - arm_center[1]
            reachable = math.hypot(dx, dy) <= PANDA_REACH_RADIUS_M
            lines.append(
                f"  {name} ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f}) "
                f"[reachable_by_arm: {'true' if reachable else 'false'}]"
            )
    else:
        lines.append("  none")

    lines.append("")
    lines.append("ZONES:")
    if zones:
        for name, pos in sorted(zones.items()):
            lines.append(f"  {name} ({pos['x']:.2f}, {pos['y']:.2f})")
    else:
        lines.append("  none")

    lines.append("")
    lines.append("Constraints:")
    lines.append("  - Center area marked A is reserved for arm workspace; do not route wheeled navigation through it")
    lines.append("  - Arm can only reach objects within its workspace (A cells)")
    lines.append("")
    lines.append("=== END ENVIRONMENT CONTEXT ===")

    return "\n".join(lines)