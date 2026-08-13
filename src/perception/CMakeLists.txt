cmake_minimum_required(VERSION 3.8)
project(perception)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

install(PROGRAMS
  perception/perception_node.py
  perception/scene_tracking.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY perception
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()