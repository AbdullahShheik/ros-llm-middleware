cmake_minimum_required(VERSION 3.8)
project(actuator)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

install(PROGRAMS
  actuator/actuator_node.py
  actuator/arm_workspace_lock_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY actuator
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()