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
# in an earlier version of this file (with robot_radius 0.22 +
# inflation_radius 0.2, nowhere near enough for the mobile robot to
# actually pass through, but the costmap didn't know that and would still
# let a path get planned into the sliver, into a dead end). One block
# spanning both arms' full extent means the planner never considers
# routing through there at all.
#
# Both arms are now rotated inward (mirrored, see panda_world.sdf) instead
# of both facing +X, so their footprints are no longer simple axis-aligned
# squares -- this block is the actual axis-aligned bounding box of each
# arm's real collision mesh (link0.stl, panda/model.sdf, measured
# directly: x:[-0.154,0.072] y:[-0.095,0.095] in link-local coordinates)
# after rotating it +/-50deg (see panda_world.sdf for why 50deg, not the
# 45deg tried first) and placing it at each arm's real spawn pose, union'd
# together. Deliberately computed from the real mesh rather than a padded
# guess -- multiple earlier guesses (0.9m, then 0.5m, then 0.35m per arm,
# all symmetric squares centered on the base) were each too large in some
# direction and too small in another, which is what kept forcing an
# uncomfortable trade-off between Nav2 clearance and arm reach margin at
# the shared pickup point.
fill_rect(0.1736, 0.5, 0.2908, 1.3581, OCCUPIED)

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
