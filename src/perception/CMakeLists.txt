cmake_minimum_required(VERSION 3.8)
project(perception)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

# scene_tracking.py is no longer installed here -- it moved into the
# world_model package, which perception now depends on, so both this node
# and Layer 1 import the one shared copy instead of each keeping its own.
install(PROGRAMS
  perception/perception_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY perception
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()