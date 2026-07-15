#!/usr/bin/env bash
#
# One-command demo launcher for ros-llm-middleware.
#
# Brings up, in order, inside a single tmux session with one window per
# component:
#   1. world        -> ros2 launch world world.launch.py   (Gazebo + MoveIt2 + controller spawners)
#   2. actuator      -> activates controllers, then ros2 launch actuator actuator.launch.py
#   3. perception    -> ros2 run perception perception_node.py
#   4. dispatcher    -> ik_feasibility_service.py, then dispatcher_node.py
#   5. layer1        -> python3 layer1_pipeline.py --ros
#
# Usage:
#   ./run_demo.sh
#   ./run_demo.sh --attach=false   # start everything but leave the session detached
#
# See README.md for details.

set -euo pipefail

# ---------------------------------------------------------------------------
# Config / paths
# ---------------------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
WS_SETUP="${REPO_ROOT}/install/setup.bash"
LAYER1_DIR="${REPO_ROOT}/src/layer1"
LAYER1_SCRIPT="${LAYER1_DIR}/layer1_pipeline.py"
SESSION="ros_llm_demo"
ATTACH=true

for arg in "$@"; do
  case "$arg" in
    --attach=false) ATTACH=false ;;
    --attach=true)  ATTACH=true ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Colors / logging helpers
# ---------------------------------------------------------------------------
c_info()  { printf '\033[1;34m[demo]\033[0m %s\n' "$*"; }
c_ok()    { printf '\033[1;32m[demo]\033[0m %s\n' "$*"; }
c_warn()  { printf '\033[1;33m[demo]\033[0m %s\n' "$*"; }
c_err()   { printf '\033[1;31m[demo]\033[0m %s\n' "$*" >&2; }

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if ! command -v tmux >/dev/null 2>&1; then
  c_err "tmux is not installed."
  c_err "Install it with: sudo apt update && sudo apt install -y tmux"
  exit 1
fi

if [[ ! -f "$ROS_SETUP" ]]; then
  c_err "Could not find ROS 2 setup file at $ROS_SETUP"
  c_err "Set ROS_DISTRO to the correct distro name and re-run, e.g.:"
  c_err "  ROS_DISTRO=humble ./run_demo.sh"
  exit 1
fi

if [[ ! -f "$WS_SETUP" ]]; then
  c_err "Workspace is not built yet: $WS_SETUP not found."
  c_err "Build it first with:  cd \"$REPO_ROOT\" && colcon build"
  exit 1
fi

if [[ ! -f "$LAYER1_SCRIPT" ]]; then
  c_err "Could not find layer1_pipeline.py at $LAYER1_SCRIPT"
  exit 1
fi

# Load a .env file for GROQ_API_KEY if present (not committed to git).
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/.env"
  set +a
fi

if [[ -z "${GROQ_API_KEY:-}" ]]; then
  c_warn "GROQ_API_KEY is not set. Layer1 will fail to call the LLM."
  c_warn "Create a .env file (see .env.example) or export GROQ_API_KEY before running."
fi

# Check/install the python deps layer1 needs, only if actually missing.
missing_py_deps=()
python3 -c "import networkx" 2>/dev/null || missing_py_deps+=("networkx")
python3 -c "import groq" 2>/dev/null || missing_py_deps+=("groq")
if [[ ${#missing_py_deps[@]} -gt 0 ]]; then
  c_info "Installing missing Python dependencies: ${missing_py_deps[*]}"
  python3 -m pip install --break-system-packages "${missing_py_deps[@]}"
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  c_err "A tmux session named '$SESSION' already exists."
  c_err "Attach with:  tmux attach -t $SESSION"
  c_err "Or tear it down first with:  ./stop_demo.sh"
  exit 1
fi

# ---------------------------------------------------------------------------
# Shared setup sourced at the start of every pane
# ---------------------------------------------------------------------------
SOURCE_ENV="source '$ROS_SETUP' && source '$WS_SETUP' && cd '$REPO_ROOT'"

# Small helper (re-defined inside every pane) that polls a shell condition
# until it's true or a timeout elapses, instead of a fixed sleep.
WAIT_FN='
wait_until() {
  local desc="$1"; shift
  local timeout="$1"; shift
  local waited=0
  echo "[demo] Waiting for: $desc"
  until "$@" >/dev/null 2>&1; do
    sleep 1
    waited=$((waited + 1))
    if [[ $waited -ge $timeout ]]; then
      echo "[demo] TIMEOUT after ${timeout}s waiting for: $desc" >&2
      echo "[demo] Continuing anyway; check earlier panes for errors." >&2
      return 1
    fi
  done
  echo "[demo] Ready: $desc"
}
'

c_info "Starting tmux session '$SESSION' in $REPO_ROOT"

# ---------------------------------------------------------------------------
# Window 0: world (Gazebo + MoveIt2 + controller spawners)
# ---------------------------------------------------------------------------
tmux new-session -d -s "$SESSION" -n world -c "$REPO_ROOT"
tmux send-keys -t "${SESSION}:world" "$SOURCE_ENV && echo '[demo] Launching world (Gazebo + MoveIt2)...' && ros2 launch world world.launch.py" C-m

# ---------------------------------------------------------------------------
# Window 1: actuator (waits for controller_manager, activates controllers,
# then launches the MoveIt2-backed actuator node)
# ---------------------------------------------------------------------------
tmux new-window -t "$SESSION" -n actuator -c "$REPO_ROOT"
tmux send-keys -t "${SESSION}:actuator" "$SOURCE_ENV && $WAIT_FN
wait_until 'controller_manager services' 120 bash -c 'ros2 service list | grep -q /controller_manager/list_controllers'
echo '[demo] Activating controllers...'
ros2 control set_controller_state joint_state_broadcaster active
ros2 control set_controller_state robotiq_gripper_controller active
ros2 control set_controller_state panda_arm_controller active
echo '[demo] Controllers active. Launching actuator node (MoveItPy init can take a while)...'
ros2 launch actuator actuator.launch.py" C-m

# ---------------------------------------------------------------------------
# Window 2: perception
# ---------------------------------------------------------------------------
tmux new-window -t "$SESSION" -n perception -c "$REPO_ROOT"
tmux send-keys -t "${SESSION}:perception" "$SOURCE_ENV && $WAIT_FN
wait_until '/execution_command topic (actuator ready)' 180 bash -c 'ros2 topic list | grep -q /execution_command'
echo '[demo] Launching perception_node...'
ros2 run perception perception_node.py" C-m

# ---------------------------------------------------------------------------
# Window 3: ik_feasibility_service
# ---------------------------------------------------------------------------
tmux new-window -t "$SESSION" -n ik_service -c "$REPO_ROOT"
tmux send-keys -t "${SESSION}:ik_service" "$SOURCE_ENV && $WAIT_FN
wait_until '/execution_command topic (actuator ready)' 180 bash -c 'ros2 topic list | grep -q /execution_command'
echo '[demo] Launching ik_feasibility_service...'
ros2 run action_dispatcher ik_feasibility_service.py" C-m

# ---------------------------------------------------------------------------
# Window 4: dispatcher_node
# ---------------------------------------------------------------------------
tmux new-window -t "$SESSION" -n dispatcher -c "$REPO_ROOT"
tmux send-keys -t "${SESSION}:dispatcher" "$SOURCE_ENV && $WAIT_FN
wait_until 'ik_feasibility_service node' 180 bash -c 'ros2 node list | grep -q ik_feasibility_service'
echo '[demo] Launching dispatcher_node...'
ros2 run action_dispatcher dispatcher_node.py" C-m

# ---------------------------------------------------------------------------
# Window 5: layer1 pipeline (ROS bridge mode)
# ---------------------------------------------------------------------------
tmux new-window -t "$SESSION" -n layer1 -c "$LAYER1_DIR"
tmux send-keys -t "${SESSION}:layer1" "$SOURCE_ENV && cd '$LAYER1_DIR' && $WAIT_FN
wait_until 'dispatcher_node (ROS node name: action_dispatcher)' 180 bash -c 'ros2 node list | grep -q action_dispatcher'
echo '[demo] Launching layer1_pipeline.py --ros...'
python3 '$LAYER1_SCRIPT' --ros" C-m

# ---------------------------------------------------------------------------
# Window 6: scratch shell for sending instructions / inspecting topics
# ---------------------------------------------------------------------------
tmux new-window -t "$SESSION" -n shell -c "$REPO_ROOT"
tmux send-keys -t "${SESSION}:shell" "$SOURCE_ENV && echo '[demo] Scratch shell ready. Try: ./send_instruction.sh \"Pick up the red block\"' && echo '[demo] or: ros2 topic pub --once /layer1/instruction std_msgs/msg/String \"{data: '\\''Pick up the red block'\\''}\"'" C-m

tmux select-window -t "${SESSION}:world"

c_ok "All components launching. Windows: world, actuator, perception, ik_service, dispatcher, layer1, shell"
c_ok "Each stage waits on the actual ROS graph state of the previous one (no fixed sleeps)."
c_warn "If Gazebo starts paused, click Play in the Gazebo GUI (window: world)."
c_info "Switch windows in tmux with: Ctrl-b then window number (0-6), or Ctrl-b w to pick from a list."
c_info "Send a command once everything is up:  ./send_instruction.sh \"Pick up the red block\""
c_info "Stop everything with:  ./stop_demo.sh"

if [[ "$ATTACH" == true ]]; then
  exec tmux attach -t "$SESSION"
else
  c_info "Started detached. Attach anytime with: tmux attach -t $SESSION"
fi
