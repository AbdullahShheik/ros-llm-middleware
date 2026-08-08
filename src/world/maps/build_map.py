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
fill_rect(0.2, 0, 0.6, 0.6, OCCUPIED) 

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
