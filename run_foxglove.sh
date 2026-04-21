#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

source_setup() {
  local setup_file="$1"
  set +u
  # shellcheck disable=SC1090
  source "$setup_file"
  set -u
}

if [[ -f /opt/ros/humble/setup.bash ]]; then
  source_setup /opt/ros/humble/setup.bash
elif [[ -f /opt/ros/foxy/setup.bash ]]; then
  source_setup /opt/ros/foxy/setup.bash
fi

if [[ -n "${ROS_WS:-}" && -f "$ROS_WS/install/setup.bash" ]]; then
  source_setup "$ROS_WS/install/setup.bash"
fi

PORT="${FOXGLOVE_PORT:-8765}"

if ! command -v ros2 >/dev/null 2>&1; then
  printf "ros2 not found. Source your ROS2 environment first.\n" >&2
  exit 1
fi

printf "Starting Foxglove Bridge on ws://localhost:%s\n" "$PORT"
ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:="$PORT"
