#!/usr/bin/env bash
set -euo pipefail

MODE="reactive"
CONFIG="configs/demo.yaml"
WORKSPACE="${ROS_WS:-}"
LAUNCH_SIM="${LAUNCH_SIM:-0}"
LAUNCH_FOXGLOVE="${LAUNCH_FOXGLOVE:-0}"
LAUNCH_RVIZ="${LAUNCH_RVIZ:-0}"
SIM_LAUNCH_CMD="${SIM_LAUNCH_CMD:-}"
FOXGLOVE_PORT="${FOXGLOVE_PORT:-8765}"
FOXGLOVE_BRIDGE_CMD="${FOXGLOVE_BRIDGE_CMD:-}"
RVIZ_CONFIG="${RVIZ_CONFIG:-rviz/demo.rviz}"

usage() {
  printf "Usage: %s [--mode reactive|rrt|mpc|rl] [--config path] [--workspace path] [--sim] [--foxglove] [--rviz]\n" "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --workspace)
      WORKSPACE="$2"
      shift 2
      ;;
    --sim)
      LAUNCH_SIM="1"
      shift
      ;;
    --foxglove)
      LAUNCH_FOXGLOVE="1"
      shift
      ;;
    --rviz)
      LAUNCH_RVIZ="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf "Unknown argument: %s\n" "$1" >&2
      usage
      exit 2
      ;;
  esac
done

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

if [[ -f /opt/ros/humble/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/humble/setup.bash
elif [[ -f /opt/ros/foxy/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /opt/ros/foxy/setup.bash
fi

if [[ -n "$WORKSPACE" ]]; then
  if [[ -f "$WORKSPACE/install/setup.bash" ]]; then
    # shellcheck disable=SC1090
    source "$WORKSPACE/install/setup.bash"
  elif [[ -f "$WORKSPACE/src/install/setup.bash" ]]; then
    # shellcheck disable=SC1090
    source "$WORKSPACE/src/install/setup.bash"
  fi
fi

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"

cleanup() {
  for job in $(jobs -p); do
    kill "$job" 2>/dev/null || true
  done
}
trap cleanup EXIT

if [[ "$LAUNCH_SIM" == "1" ]]; then
  if [[ -z "$SIM_LAUNCH_CMD" ]]; then
    printf "LAUNCH_SIM=1 but SIM_LAUNCH_CMD is empty. Set it to your F1TENTH/drone sim launch command.\n" >&2
    exit 2
  fi
  bash -lc "$SIM_LAUNCH_CMD" &
  sleep 4
fi

if [[ "$LAUNCH_FOXGLOVE" == "1" ]]; then
  if [[ -n "$FOXGLOVE_BRIDGE_CMD" ]]; then
    bash -lc "$FOXGLOVE_BRIDGE_CMD" &
  elif command -v ros2 >/dev/null 2>&1; then
    ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:="$FOXGLOVE_PORT" &
  else
    printf "ros2 not found; start Foxglove Bridge manually and connect Foxglove to ws://localhost:%s\n" "$FOXGLOVE_PORT" >&2
  fi
  printf "Foxglove: open Foxglove Studio and connect to ws://localhost:%s\n" "$FOXGLOVE_PORT"
  sleep 2
fi

if [[ "$LAUNCH_RVIZ" == "1" ]]; then
  if command -v rviz2 >/dev/null 2>&1; then
    rviz2 -d "$RVIZ_CONFIG" &
  else
    printf "rviz2 not found; continuing without RViz.\n" >&2
  fi
fi

python3 -m unified_autonomy.main_demo --config "$CONFIG" --mode "$MODE"
