cmake_minimum_required(VERSION 3.8)
project(world)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)

install(DIRECTORY launch urdf
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY models
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY worlds
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY maps
  DESTINATION share/${PROJECT_NAME}
)

ament_package()