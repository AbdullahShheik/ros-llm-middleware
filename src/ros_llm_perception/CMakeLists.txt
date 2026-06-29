cmake_minimum_required(VERSION 3.8)
project(ros_llm_perception)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

install(PROGRAMS
  ros_llm_perception/perception_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY ros_llm_perception
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()