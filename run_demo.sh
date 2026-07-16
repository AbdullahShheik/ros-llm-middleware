#!/usr/bin/env bash
#
# One-command demo launcher for ros-llm-middleware.
#
# As of the "unified launch" change (world.launch.py), a single
# `ros2 launch world world.launch.py` now brings up, in order and on its
# own internal timers: MoveIt2 (robot_state_publisher + move_group),
# Gazebo (auto-played with -r), the ros2_control controller spawners
# (joint_state_broadcaster, panda_arm_controller, robotiq_gripper_controller
# -- spawned active, no manual activation step needed), perception,
# ik_feasibility_service, action_dispatcher, and finally the actuator node.
#
# This script wraps that single launch plus the one piece it does NOT
# cover -- the standalone Layer 1 LLM pipeline -- into one tmux session,
# gated on real ROS graph state instead of guessed sleeps.
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

# layer1_pipeline.py loads GROQ_API_KEY from <repo_root>/.env itself, so we
# only need to warn here, not source it ourselves.
if [[ ! -f "${REPO_ROOT}/.env" ]] && [[ -z "${GROQ_API_KEY:-}" ]]; then
  c_warn "No .env file and GROQ_API_KEY is not set. Layer1 will fail to call the LLM."
  c_warn "Create ${REPO_ROOT}/.env from .env.example (or export GROQ_API_KEY) before sending instructions."
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
      echo "[demo] Continuing anyway; check the world window for errors." >&2
      return 1
    fi
  done
  echo "[demo] Ready: $desc"
}
'

c_info "Starting tmux session '$SESSION' in $REPO_ROOT"

# ---------------------------------------------------------------------------
# Window 0: world -- the unified launch. This single launch file now owns
# MoveIt2, Gazebo, controller spawning/activation, perception,
# ik_feasibility_service, action_dispatcher, and the actuator node, staged
# internally via TimerActions (see src/world/launch/world.launch.py).
# ---------------------------------------------------------------------------
tmux new-session -d -s "$SESSION" -n world -c "$REPO_ROOT"
tmux send-keys -t "${SESSION}:world" "$SOURCE_ENV && echo '[demo] Launching unified world pipeline (Gazebo + MoveIt2 + controllers + perception + IK service + dispatcher + actuator)...' && ros2 launch world world.launch.py" C-m

# ---------------------------------------------------------------------------
# Window 1: status -- live view of ROS graph readiness, since the unified
# launch interleaves every component's logs into one window. Useful for
# debugging without hunting through that combined output.
# ---------------------------------------------------------------------------
tmux new-window -t "$SESSION" -n status -c "$REPO_ROOT"
tmux send-keys -t "${SESSION}:status" "$SOURCE_ENV && watch -n 2 'echo \"--- ros2 node list ---\"; ros2 node list 2>/dev/null; echo; echo \"--- controllers ---\"; ros2 control list_controllers 2>/dev/null; echo; echo \"--- key topics ---\"; ros2 topic list 2>/dev/null | grep -E \"execution_command|object_map|layer1|check_ik_feasibility\" '" C-m

# ---------------------------------------------------------------------------
# Window 2: layer1 -- waits for the full pipeline to be ready (dispatcher
# node up, IK feasibility service available, actuator subscribed to
# /execution_command) before starting, instead of guessing a delay.
# ---------------------------------------------------------------------------
tmux new-window -t "$SESSION" -n layer1 -c "$LAYER1_DIR"
tmux send-keys -t "${SESSION}:layer1" "$SOURCE_ENV && cd '$LAYER1_DIR' && $WAIT_FN
wait_until 'action_dispatcher node' 120 bash -c 'ros2 node list | grep -q action_dispatcher'
wait_until '/check_ik_feasibility service' 120 bash -c 'ros2 service list | grep -q /check_ik_feasibility'
wait_until '/execution_command topic (actuator ready)' 180 bash -c 'ros2 topic list | grep -q /execution_command'
echo '[demo] Full pipeline ready. Launching layer1_pipeline.py --ros...'
python3 '$LAYER1_SCRIPT' --ros" C-m

# ---------------------------------------------------------------------------
# Window 3: scratch shell for sending instructions / inspecting topics
# ---------------------------------------------------------------------------
tmux new-window -t "$SESSION" -n shell -c "$REPO_ROOT"
tmux send-keys -t "${SESSION}:shell" "$SOURCE_ENV && echo '[demo] Scratch shell ready. Try: ./send_instruction.sh \"Pick up the red block\"' && echo '[demo] or: ros2 topic pub --once /layer1/instruction std_msgs/msg/String \"{data: '\\''Pick up the red block'\\''}\"'" C-m

tmux select-window -t "${SESSION}:world"

c_ok "All components launching. Windows: world, status, layer1, shell"
c_ok "world runs the unified world.launch.py (Gazebo starts auto-played with -r; controllers spawn active)."
c_ok "layer1 waits on real ROS graph state (no fixed sleeps) before starting the LLM pipeline."
c_info "Switch windows in tmux with: Ctrl-b then window number (0-3), or Ctrl-b w to pick from a list."
c_info "Send a command once everything is up:  ./send_instruction.sh \"Pick up the red block\""
c_info "Stop everything with:  ./stop_demo.sh"

if [[ "$ATTACH" == true ]]; then
  exec tmux attach -t "$SESSION"
else
  c_info "Started detached. Attach anytime with: tmux attach -t $SESSION"
fi
