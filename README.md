# ros-llm-middleware

This project develops a translation middleware that bridges the gap between high-level natural language instructions and low-level parameterized robot execution in a heterogeneous multi-robot system built on ROS2.

The core problem is that LLMs can decompose a natural language command like "pick the red cube and bring it to the workstation" into logical subtasks, but they cannot natively produce the numerical parameters such as 3D poses, waypoints, and gripper configurations that robot motion planners require. This middleware handles that translation.

The system follows a three-layer architecture. The first layer uses an LLM to decompose a natural language instruction into a dependency-aware DAG of subtasks. The second layer maps each subtask to a specific robot based on a Robotics Competency Library, runs feasibility checks, and resolves object poses from the environment. The third layer executes the resulting parameterized motion goals via ROS2 action servers, with MoveIt2 handling arm trajectories and Nav2 handling mobile navigation.