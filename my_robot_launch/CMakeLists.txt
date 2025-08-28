cmake_minimum_required(VERSION 3.5)
project(my_robot_launch)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(sensor_msgs REQUIRED)

# Installer les scripts Python dans share/my_robot_launch/scripts
install(
  PROGRAMS
  scripts/camera_node.py
  scripts/person_follow_node.py
  scripts/recorder_node.py
  DESTINATION lib/${PROJECT_NAME}  # ROS2 créera lib/my_robot_launch/
)

# Installer les fichiers launch
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()

