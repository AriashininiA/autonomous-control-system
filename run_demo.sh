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

if [[ -z "$WORKSPACE" && -f "$PROJECT_DIR/../../install/setup.bash" ]]; then
  WORKSPACE="$(cd "$PROJECT_DIR/../.." && pwd)"
fi

source_setup() {
  local setup_file="$1"
  set +u
  # shellcheck disable=SC1090
  source "$setup_file"
  set -u
}

port_in_use() {
  python3 - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.2)
    raise SystemExit(0 if sock.connect_ex(("127.0.0.1", port)) == 0 else 1)
PY
}

if [[ -f /opt/ros/humble/setup.bash ]]; then
  source_setup /opt/ros/humble/setup.bash
elif [[ -f /opt/ros/foxy/setup.bash ]]; then
  source_setup /opt/ros/foxy/setup.bash
fi

if [[ -n "$WORKSPACE" ]]; then
  if [[ -f "$WORKSPACE/install/setup.bash" ]]; then
    source_setup "$WORKSPACE/install/setup.bash"
  elif [[ -f "$WORKSPACE/src/install/setup.bash" ]]; then
    source_setup "$WORKSPACE/src/install/setup.bash"
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
    if [[ "$LAUNCH_FOXGLOVE" == "1" ]] && port_in_use "$FOXGLOVE_PORT"; then
      printf "Foxglove port %s is already in use. Stop the old run or set FOXGLOVE_PORT to another port.\n" "$FOXGLOVE_PORT" >&2
      exit 2
    fi
    SIM_LAUNCH_CMD="ros2 launch f1tenth_gym_ros gym_bridge_launch.py foxglove_port:=$FOXGLOVE_PORT open_foxglove:=$LAUNCH_FOXGLOVE"
  fi
  bash -lc "$SIM_LAUNCH_CMD" &
  sleep 4
fi

if [[ "$LAUNCH_FOXGLOVE" == "1" && "$LAUNCH_SIM" != "1" ]]; then
  if [[ -z "$FOXGLOVE_BRIDGE_CMD" ]] && port_in_use "$FOXGLOVE_PORT"; then
    printf "Foxglove port %s is already in use. Stop the old bridge or set FOXGLOVE_PORT to another port.\n" "$FOXGLOVE_PORT" >&2
    exit 2
  fi
  if [[ -n "$FOXGLOVE_BRIDGE_CMD" ]]; then
    bash -lc "$FOXGLOVE_BRIDGE_CMD" &
  elif command -v ros2 >/dev/null 2>&1; then
    ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:="$FOXGLOVE_PORT" &
  else
    printf "ros2 not found; start Foxglove Bridge manually and connect Foxglove to ws://localhost:%s\n" "$FOXGLOVE_PORT" >&2
  fi
  printf "Foxglove: open Foxglove Studio and connect to ws://localhost:%s\n" "$FOXGLOVE_PORT"
  sleep 2
elif [[ "$LAUNCH_FOXGLOVE" == "1" ]]; then
  printf "Foxglove: connect to ws://localhost:%s after the simulator launch opens/starts Foxglove Bridge\n" "$FOXGLOVE_PORT"
fi

if [[ "$LAUNCH_RVIZ" == "1" ]]; then
  if command -v rviz2 >/dev/null 2>&1; then
    rviz2 -d "$RVIZ_CONFIG" &
  else
    printf "rviz2 not found; continuing without RViz.\n" >&2
  fi
fi

python3 -m unified_autonomy.main_demo --config "$CONFIG" --mode "$MODE"
