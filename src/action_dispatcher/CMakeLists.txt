cmake_minimum_required(VERSION 3.8)
project(action_dispatcher)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(ros_llm_interfaces REQUIRED)

install(PROGRAMS
  action_dispatcher/ik_feasibility_service.py
  action_dispatcher/dispatcher_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY action_dispatcher
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
