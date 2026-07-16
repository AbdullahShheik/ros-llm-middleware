#!/usr/bin/env bash
#
# Cleanly tears down the tmux demo session started by run_demo.sh, and
# mops up any ROS/Gazebo processes it left behind. Since world.launch.py
# now spawns perception, the IK feasibility service, the dispatcher, and
# the actuator itself (as child processes of one `ros2 launch`), killing
# that launch process should take all of them down -- this is a
# belt-and-suspenders sweep for anything that detached from tmux/launch's
# process tree.
#
# Usage: ./stop_demo.sh

set -uo pipefail

SESSION="ros_llm_demo"

c_info()  { printf '\033[1;34m[demo]\033[0m %s\n' "$*"; }
c_ok()    { printf '\033[1;32m[demo]\033[0m %s\n' "$*"; }
c_warn()  { printf '\033[1;33m[demo]\033[0m %s\n' "$*"; }

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
  c_info "Killing tmux session '$SESSION'..."
  tmux kill-session -t "$SESSION"
  c_ok "tmux session stopped."
else
  c_warn "No tmux session named '$SESSION' is running."
fi

PATTERNS=(
  "ros2 launch world world.launch.py"
  "ros2 launch actuator actuator.launch.py"
  "actuator_node.py"
  "perception_node.py"
  "ik_feasibility_service.py"
  "dispatcher_node.py"
  "layer1_pipeline.py --ros"
  "gz sim"
  "move_group"
)

leftover_pids=()
for pattern in "${PATTERNS[@]}"; do
  # shellcheck disable=SC2009
  pids=$(pgrep -f -- "$pattern" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    leftover_pids+=($pids)
  fi
done

if [[ ${#leftover_pids[@]} -gt 0 ]]; then
  c_warn "Found leftover demo processes still running: ${leftover_pids[*]}"
  read -r -p "[demo] Kill them? [y/N] " reply
  if [[ "$reply" =~ ^[Yy]$ ]]; then
    kill "${leftover_pids[@]}" 2>/dev/null || true
    sleep 1
    for pid in "${leftover_pids[@]}"; do
      kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    done
    c_ok "Leftover processes killed."
  else
    c_warn "Left running. Clean them up manually with: kill ${leftover_pids[*]}"
  fi
else
  c_ok "No leftover demo processes found."
fi
