#!/usr/bin/env python3
import numpy as np
from PIL import Image

# Map settings
resolution = 0.05  # meters/pixel, matches SLAM output convention
map_size_m = 10.0  # 10m x 10m canvas (room is 8x8, centered at origin)
pixels = int(map_size_m / resolution)  # 200x200

# Values: 254 = free (white), 0 = occupied (black), 205 = unknown (gray)
FREE = 254
OCCUPIED = 0
UNKNOWN = 205

# Start with unknown everywhere
grid = np.full((pixels, pixels), UNKNOWN, dtype=np.uint8)

def world_to_pixel(x, y):
    # Map origin (world 0,0) is at the center of the canvas.
    # PGM row 0 is the TOP of the image = max Y in world coords.
    col = int((x + map_size_m / 2) / resolution)
    row = int((map_size_m / 2 - y) / resolution)
    return row, col

def fill_rect(cx, cy, size_x, size_y, value):
    x0, y0 = cx - size_x / 2, cy - size_y / 2
    x1, y1 = cx + size_x / 2, cy + size_y / 2
    r0, c0 = world_to_pixel(x0, y1)
    r1, c1 = world_to_pixel(x1, y0)
    r0, r1 = max(0, min(r0, r1)), min(pixels, max(r0, r1))
    c0, c1 = max(0, min(c0, c1)), min(pixels, max(c0, c1))
    grid[r0:r1, c0:c1] = value

# Interior of the room is free space (walls are at exactly +-4)
fill_rect(0, 0, 7.8, 7.8, FREE)

# Walls (from panda_world.sdf): pose (x, y), size (size_x, size_y)
fill_rect(0, 4, 8, 0.2, OCCUPIED)    # wall_north
fill_rect(0, -4, 8, 0.2, OCCUPIED)   # wall_south
fill_rect(4, 0, 0.2, 8, OCCUPIED)    # wall_east
fill_rect(-4, 0, 0.2, 8, OCCUPIED)   # wall_west
# Both arms as ONE merged block rather than two separate squares -- panda
# is at (0.2,0), panda2 at (0.2,1.0) (panda_world.sdf), 1.0m apart, and two
# separate squares left only a ~0.10m sliver of "free" space between them
# (confirmed live: with robot_radius 0.22 + inflation_radius 0.2, nowhere
# near enough for the mobile robot to actually pass through, but the
# costmap didn't know that and would still let a path get planned into the
# sliver, into a dead end). One block spanning both arms' full extent means
# the planner never considers routing through there at all.
#
# 0.35x0.35 per arm (was 0.9x0.9, then 0.5x0.5): measured directly off the
# real collision mesh (link0.stl, panda/model.sdf) rather than guessed --
# its footprint is only ~0.23m x 0.19m, so even 0.5m (confirmed live: the
# retreat phase of a pick at the shared pickup point, right after a
# successful grasp, failed to plan -- "Unable to sample any valid states
# for goal tree" -- at the reach margin 0.5m's edge forced) was still too
# tight a corner between Nav2 clearance and arm reach. 0.35m is still
# ~1.5-1.8x the real base for a real pad, and this time frees up enough
# room that the pickup point (attach_detach_node.py's PICKUP_POINT_X/Y)
# can sit at close to the same ~76% reach margin the original single-arm
# WORKSPACE_BOUNDS box was tuned to and reliably worked at. Y spans from
# panda's back edge (0-0.175) to panda2's front edge (1.0+0.175); X
# unchanged in shape, just the smaller size.
fill_rect(0.2, 0.5, 0.35, 1.35, OCCUPIED)

img = Image.fromarray(grid, mode='L')
out_path = '/home/abdullah/HU/STRP/ros-llm-middleware/src/world/maps/panda_world_map.pgm'
img.save(out_path)

# Write the matching YAML
yaml_content = f"""image: panda_world_map.pgm
resolution: {resolution}
origin: [{-map_size_m/2}, {-map_size_m/2}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
"""
yaml_path = '/home/abdullah/HU/STRP/ros-llm-middleware/src/world/maps/panda_world_map.yaml'
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"Saved map to {out_path} and {yaml_path}")
