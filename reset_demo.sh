#!/usr/bin/env bash
#
# Heavier-handed reset for when stop_demo.sh + a fresh run_demo.sh isn't
# enough -- specifically the symptom where controller spawners die with
# "Failed to acquire lock in 20 seconds ... process has died [exit code 1]"
# and every downstream node (perception, dispatcher, actuator) never comes
# up, even though Gazebo and move_group look fine.
#
# Root cause observed: killing a large batch of ROS processes (e.g. after
# a launch got stuck, or a crashed session) can leave stale Fast-DDS shared
# memory segments behind in /dev/shm (fastrtps_*, sem.fastrtps_*). Those
# orphaned segments poison discovery for the *next* launch -- new spawners
# can't acquire the controller_manager lock, and even `ros2 control
# list_controllers` hangs past its own timeout. stop_demo.sh's process
# sweep does not touch /dev/shm, so it does not fix this.
#
# This script: kills the tmux session, sweeps every process from the demo
# stack (including the nav2/turtlebot pieces stop_demo.sh doesn't cover),
# restarts the ros2 daemon, and -- only once nothing ROS-related is left
# running -- deletes this user's stale fastrtps_*/fastdds_* entries from
# /dev/shm. It refuses to touch /dev/shm while anything is still alive,
# since a live process legitimately owns its own segments.
#
# Usage: ./reset_demo.sh

set -uo pipefail

SESSION="ros_llm_demo"

c_info()  { printf '\033[1;34m[reset]\033[0m %s\n' "$*"; }
c_ok()    { printf '\033[1;32m[reset]\033[0m %s\n' "$*"; }
c_warn()  { printf '\033[1;33m[reset]\033[0m %s\n' "$*"; }
c_err()   { printf '\033[1;31m[reset]\033[0m %s\n' "$*" >&2; }

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
  c_info "Killing tmux session '$SESSION'..."
  tmux kill-session -t "$SESSION"
else
  c_info "No tmux session named '$SESSION' running."
fi

# Broader than stop_demo.sh's list: also covers the nav2/turtlebot pieces
# that were seen left running as orphans after their parent launch died.
PATTERNS=(
  "ros2 launch world world.launch.py"
  "ros2 launch actuator actuator.launch.py"
  "ros2 launch mobile_actuator"
  "actuator_node.py"
  "perception_node.py"
  "ik_feasibility_service.py"
  "dispatcher_node.py"
  "layer1_pipeline.py --ros"
  "gz sim"
  "move_group"
  "controller_manager"
  "spawner "
  "bt_navigator"
  "amcl"
  "planner_server"
  "controller_server"
  "behavior_server"
  "waypoint_follower"
  "map_server"
  "lifecycle_manager"
  "costmap"
  "turtlebot3_bridge"
  "clock_bridge"
  "gripper_mimic"
  "robot_state_publisher"
  "static_transform_publisher"
  "ros_gz_bridge"
  "parameter_bridge"
)

collect_pids() {
  local pids=()
  for pattern in "${PATTERNS[@]}"; do
    # shellcheck disable=SC2009
    local found
    found=$(pgrep -f -- "$pattern" 2>/dev/null || true)
    if [[ -n "$found" ]]; then
      pids+=($found)
    fi
  done
  # Guard against `printf '%s\n'` still emitting one blank line with zero
  # arguments -- that turned "nothing found" into a phantom one-element
  # array downstream (mapfile would read that blank line as a PID).
  if [[ ${#pids[@]} -gt 0 ]]; then
    printf '%s\n' "${pids[@]}" | sort -un
  fi
}

mapfile -t pids < <(collect_pids)

if [[ ${#pids[@]} -gt 0 ]]; then
  c_info "Stopping ${#pids[@]} demo-related process(es): ${pids[*]}"
  kill "${pids[@]}" 2>/dev/null || true
  sleep 3
  mapfile -t survivors < <(collect_pids)
  if [[ ${#survivors[@]} -gt 0 ]]; then
    c_warn "SIGKILL-ing ${#survivors[@]} survivor(s): ${survivors[*]}"
    kill -9 "${survivors[@]}" 2>/dev/null || true
    sleep 1
  fi
else
  c_ok "No leftover demo processes found."
fi

if command -v ros2 >/dev/null 2>&1; then
  c_info "Restarting the ros2 daemon (clears its cached, possibly-stale discovery state)..."
  timeout 15 ros2 daemon stop >/dev/null 2>&1 || true
fi

mapfile -t remaining < <(collect_pids)
if [[ ${#remaining[@]} -gt 0 ]]; then
  c_err "Processes still alive after SIGKILL, refusing to touch /dev/shm: ${remaining[*]}"
  c_err "Investigate manually (they may not be killable, e.g. zombie/defunct) before re-running."
  exit 1
fi

c_info "Clearing this user's stale Fast-DDS shared memory (/dev/shm)..."
before=$(ls /dev/shm 2>/dev/null | wc -l)
find /dev/shm -maxdepth 1 -user "$(id -un)" \
     \( -name 'fastrtps_*' -o -name 'sem.fastrtps_*' -o -name 'fastdds_*' -o -name 'sem.fastdds_*' \) \
     -delete 2>/dev/null
after=$(ls /dev/shm 2>/dev/null | wc -l)
c_ok "/dev/shm entries: $before -> $after"

c_ok "Reset complete. Start fresh with: ./run_demo.sh"
