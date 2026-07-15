#!/usr/bin/env bash
#
# Publishes a natural-language instruction to /layer1/instruction so you
# don't have to hand-type the ros2 topic pub incantation every time.
#
# Usage:
#   ./send_instruction.sh "Pick up the red block"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
WS_SETUP="${REPO_ROOT}/install/setup.bash"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 \"<natural language instruction>\"" >&2
  echo 'Example: ./send_instruction.sh "Pick up the red block"' >&2
  exit 1
fi

INSTRUCTION="$1"

# shellcheck disable=SC1090
[[ -f "$ROS_SETUP" ]] && source "$ROS_SETUP"
# shellcheck disable=SC1090
[[ -f "$WS_SETUP" ]] && source "$WS_SETUP"

if ! command -v ros2 >/dev/null 2>&1; then
  echo "[send_instruction] ros2 not found on PATH even after sourcing $ROS_SETUP" >&2
  exit 1
fi

echo "[send_instruction] Publishing to /layer1/instruction: $INSTRUCTION"
ros2 topic pub --once /layer1/instruction std_msgs/msg/String "{data: '${INSTRUCTION//\'/\\\'}'}"
