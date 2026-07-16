# ros-llm-middleware

This project develops a translation middleware that bridges the gap between high-level natural language instructions and low-level parameterized robot execution in a heterogeneous multi-robot system built on ROS2.

The core problem is that LLMs can decompose a natural language command like "pick the red cube and bring it to the workstation" into logical subtasks, but they cannot natively produce the numerical parameters such as 3D poses, waypoints, and gripper configurations that robot motion planners require. This middleware handles that translation.

The system follows a three-layer architecture. The first layer uses an LLM to decompose a natural language instruction into a dependency-aware DAG of subtasks. The second layer maps each subtask to a specific robot based on a Robotics Competency Library, runs feasibility checks, and resolves object poses from the environment. The third layer executes the resulting parameterized motion goals via ROS2 action servers, with MoveIt2 handling arm trajectories and Nav2 handling mobile navigation.

## Running the demo

`src/world/launch/world.launch.py` is a **unified launch file**: one `ros2 launch world world.launch.py` already brings up Gazebo (auto-played), MoveIt2, the ros2_control controller spawners (spawned active — no manual activation step needed), perception, the IK feasibility service, the action dispatcher, and the actuator, staged internally with timers. The only piece it doesn't cover is the standalone Layer 1 LLM pipeline (`src/layer1/layer1_pipeline.py`), which needs its own Python environment and a Groq API key.

`run_demo.sh` wraps both into a single command and one tmux session, starting Layer 1 only once the rest of the ROS graph is actually ready (no guessed delays on top of what's already in the launch file).

### Prerequisites

- The workspace has been built at least once: `colcon build` from the repo root.
- [tmux](https://github.com/tmux/tmux) is installed (`sudo apt install -y tmux`).
- A Groq API key. Copy `.env.example` to `.env` at the repo root and fill in `GROQ_API_KEY`, or `export GROQ_API_KEY=...` before running — `layer1_pipeline.py` loads `.env` from the repo root itself. `.env` is gitignored.
- `networkx` and `groq` Python packages — `run_demo.sh` installs these automatically (`pip install --break-system-packages`) only if they're missing.

### Start everything

```bash
./run_demo.sh
```

This opens a tmux session named `ros_llm_demo` with:

| # | Window   | What it runs                                                                                  |
|---|----------|------------------------------------------------------------------------------------------------|
| 0 | `world`  | `ros2 launch world world.launch.py` — Gazebo, MoveIt2, controllers, perception, IK service, dispatcher, actuator (all owned by this one launch file) |
| 1 | `status` | A live `watch` of `ros2 node list`, `ros2 control list_controllers`, and the key topics/services, refreshed every 2s — useful since `world`'s log lines from every component are interleaved together |
| 2 | `layer1` | Waits for `action_dispatcher`, `/check_ik_feasibility`, and `/execution_command` to be live, then runs `python3 layer1_pipeline.py --ros` |
| 3 | `shell`  | A pre-sourced scratch shell for `send_instruction.sh`, `ros2 topic echo`, etc.                 |

The `layer1` window polls actual ROS graph state instead of sleeping a fixed duration, so it starts as soon as the pipeline is genuinely ready to accept instructions — no earlier, no needlessly later.

Navigate the session with standard tmux keys: `Ctrl-b` then a window number (`0`-`3`), or `Ctrl-b w` for a picker. Detach without stopping anything with `Ctrl-b d`; re-attach later with `tmux attach -t ros_llm_demo`.

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

Kills the `ros_llm_demo` tmux session and checks for any straggler processes (Gazebo, `move_group`, the world-launch child nodes, Layer 1) that may have detached from tmux's process tree, offering to kill those too.

### Troubleshooting

- **"tmux is not installed"** — `sudo apt update && sudo apt install -y tmux`.
- **"Workspace is not built yet"** — run `colcon build` from the repo root first.
- **`layer1` window stuck on "Waiting for..."** — switch to the `world` window and check for launch errors there; the unified launch stages Gazebo, controllers, perception/IK/dispatcher, and the actuator on internal timers, so a slow machine may just need more time.
- **Wrong ROS distro** — the scripts default to `jazzy`; override with `ROS_DISTRO=<distro> ./run_demo.sh`.
- **A previous session is still around** — `./stop_demo.sh`, or `tmux kill-session -t ros_llm_demo`.