# ros-llm-middleware

This project develops a translation middleware that bridges the gap between high-level natural language instructions and low-level parameterized robot execution in a heterogeneous multi-robot system built on ROS2.

The core problem is that LLMs can decompose a natural language command like "pick the red cube and bring it to the workstation" into logical subtasks, but they cannot natively produce the numerical parameters such as 3D poses, waypoints, and gripper configurations that robot motion planners require. This middleware handles that translation.

The system follows a three-layer architecture. The first layer uses an LLM to decompose a natural language instruction into a dependency-aware DAG of subtasks. The second layer maps each subtask to a specific robot based on a Robotics Competency Library, runs feasibility checks, and resolves object poses from the environment. The third layer executes the resulting parameterized motion goals via ROS2 action servers, with MoveIt2 handling arm trajectories and Nav2 handling mobile navigation.

## Running the demo

The whole stack — Gazebo world, MoveIt2 actuator, perception, action dispatcher, and the Layer 1 LLM pipeline — can be brought up with one command instead of five manual terminals.

### Prerequisites

- The workspace has been built at least once: `colcon build` from the repo root.
- [tmux](https://github.com/tmux/tmux) is installed (`sudo apt install -y tmux`).
- A Groq API key. Copy `.env.example` to `.env` and fill in `GROQ_API_KEY`, or `export GROQ_API_KEY=...` before running. `.env` is gitignored.
- `networkx` and `groq` Python packages — `run_demo.sh` installs these automatically (`pip install --break-system-packages`) only if they're missing.

### Start everything

```bash
./run_demo.sh
```

This opens a tmux session named `ros_llm_demo` with one window per component:

| # | Window        | What it runs                                                              |
|---|----------------|----------------------------------------------------------------------------|
| 0 | `world`        | `ros2 launch world world.launch.py` — Gazebo, MoveIt2 (`move_group`), controller spawners |
| 1 | `actuator`     | Activates `joint_state_broadcaster`, `robotiq_gripper_controller`, `panda_arm_controller`, then `ros2 launch actuator actuator.launch.py` |
| 2 | `perception`   | `ros2 run perception perception_node.py`                                  |
| 3 | `ik_service`   | `ros2 run action_dispatcher ik_feasibility_service.py`                    |
| 4 | `dispatcher`   | `ros2 run action_dispatcher dispatcher_node.py`                           |
| 5 | `layer1`       | `python3 layer1_pipeline.py --ros` (run from `src/layer1/` so `robot_skills.json` resolves) |
| 6 | `shell`        | A pre-sourced scratch shell for `send_instruction.sh`, `ros2 topic echo`, etc. |

Each window (after `world`) waits on real ROS graph state before starting — the `controller_manager` service list, the `/execution_command` topic, or a specific node appearing in `ros2 node list` — instead of a fixed `sleep`, so startup adapts to how long Gazebo/MoveIt actually take on your machine. If a wait exceeds its timeout the window prints a warning and proceeds anyway so you can see the underlying error rather than hanging silently.

Navigate the session with standard tmux keys: `Ctrl-b` then a window number (`0`-`6`), or `Ctrl-b w` for a picker. Detach without stopping anything with `Ctrl-b d`; re-attach later with `tmux attach -t ros_llm_demo`.

If Gazebo starts paused, click **Play** in the Gazebo GUI (window `0`).

Run detached from the start (skip the auto-attach) with:

```bash
./run_demo.sh --attach=false
```

### Send an instruction

Once the `layer1` window shows it's ready, send a natural-language command from any terminal:

```bash
./send_instruction.sh "Pick up the red block"
```

This is equivalent to:

```bash
ros2 topic pub --once /layer1/instruction std_msgs/msg/String "{data: 'Pick up the red block'}"
```

### Stop everything

```bash
./stop_demo.sh
```

Kills the `ros_llm_demo` tmux session and checks for any straggler processes (Gazebo, `move_group`, node runners) that may have detached from tmux's process tree, offering to kill those too.

### Troubleshooting

- **"tmux is not installed"** — `sudo apt update && sudo apt install -y tmux`.
- **"Workspace is not built yet"** — run `colcon build` from the repo root first.
- **A window is stuck on "Waiting for..."** — switch to the window it depends on (usually `world` or `actuator`) and check for launch errors there.
- **Wrong ROS distro** — the scripts default to `jazzy`; override with `ROS_DISTRO=<distro> ./run_demo.sh`.
- **A previous session is still around** — `./stop_demo.sh`, or `tmux kill-session -t ros_llm_demo`.