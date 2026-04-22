# Autonomous Control System

A modular ROS2 autonomy stack for F1TENTH-scale racing simulation. This project integrates classical planning, trajectory tracking, safety constraints, visualization, live mode switching, and metrics logging into one reusable control system.

The goal of this repository is to present autonomy work as a complete engineered system rather than separate lab assignments: common interfaces, shared configuration, repeatable demos, and clear runtime observability.

## 🧑‍💻 Authors

**Aria Shi** [GitHub](https://github.com/AriashininiA), [LinkedIn](https://www.linkedin.com/in/aria-xingni-shi)  
MSE @ Upenn | Machine Learning Engineer | BSc @ UCL | Physics & Math

## Demo

[Watch the autonomous control system demo](auto-system-demo.mp4)

## Highlights

- Unified multiple autonomy approaches under one ROS2 controller interface.
- Implemented live switching between reactive, RRT, MPC, and RL-placeholder modes.
- Built a Foxglove-ready demo pipeline for inspecting planner output, vehicle state, and drive commands.
- Added a FastAPI dashboard for mode control and runtime status.
- Logged repeatable run metrics as CSV traces and JSON summaries.
- Refactored coursework algorithms into reusable Python modules with shared configuration and safety limits.

## System Overview

```text
F1TENTH simulator
  -> /scan
  -> /ego_racecar/odom
        |
        v
unified_autonomy.main_demo
  -> selected controller mode
     - reactive follow-the-gap
     - local RRT + pure pursuit
     - kinematic MPC
     - RL policy placeholder
        |
        v
  -> /drive
  -> /unified/planned_path
  -> /unified/goal
  -> /unified/obstacles
  -> /unified/status
  -> logs/*.csv and logs/*_summary.json
```

## Environment

This project is designed to run in:

1. Ubuntu VM
2. [ROS2 Humble](https://docs.ros.org/en/humble/Installation.html)
3. [f1tenth_gym_ros](https://github.com/f1tenth/f1tenth_gym_ros.git)
4. Python dependencies from `requirements.txt`

Typical workspace layout:

```text
roboracer_ws/
  src/
    f1tenth_gym_ross/
    autonomous-control-system/
```

Install Python dependencies from this repository:

```bash
cd roboracer_ws/src/autonomous-control-system
python3 -m pip install -r requirements.txt
```

Build and source the ROS workspace:

```bash
cd roboracer_ws
colcon build --symlink-install
source install/setup.bash
```

## Quick Start

Launch the simulator, controller, and Foxglove bridge:

```bash
cd roboracer_ws/src/autonomous-control-system
./run_demo.sh --mode reactive --sim --foxglove
```

Open Foxglove Studio and connect to:

```text
ws://localhost:8765
```

Start the dashboard in a second terminal:

```bash
cd roboracer_ws/src/autonomous-control-system
./run_dashboard.sh
```

Open:

```text
http://127.0.0.1:8080
```

The dashboard can change the active autonomy mode during the demo.

## Autonomy Modes

| Mode | Status | Description |
|---|---:|---|
| `reactive` | Working | LiDAR follow-the-gap controller for fast obstacle-aware driving. |
| `rrt` | Working | Local RRT planner with pure-pursuit tracking of the generated path. |
| `mpc` | Working | Kinematic MPC tracker using waypoint assets from `data/waypoints`. |
| `rl` | Placeholder | Safe-stop adapter reserved for a future trained reinforcement-learning policy. |

## Repository Structure

| Path | Purpose |
|---|---|
| `configs/demo.yaml` | Main configuration for topics, modes, safety limits, assets, metrics, dashboard, and visualization. |
| `run_demo.sh` | Launches the controller with optional simulator and Foxglove bridge. |
| `run_dashboard.sh` | Starts the FastAPI dashboard. |
| `src/unified_autonomy/main_demo.py` | Main ROS2 orchestrator for subscriptions, mode selection, safety clipping, publishing, visualization, and metrics. |
| `src/unified_autonomy/interfaces.py` | Shared dataclasses and controller interface. |
| `src/unified_autonomy/adapters/` | Mode adapters for reactive, RRT, MPC, and RL. |
| `src/unified_autonomy/control/` | Follow-the-gap and MPC control logic. |
| `src/unified_autonomy/planning/local_rrt.py` | Local RRT planner. |
| `src/unified_autonomy/tracking/pure_pursuit.py` | Pure-pursuit path tracker. |
| `src/unified_autonomy/dashboard/` | Dashboard API and static UI. |
| `src/unified_autonomy/metrics.py` | CSV and JSON run logging. |
| `data/maps/` | Demo map assets. |
| `data/waypoints/` | Waypoint CSV files for tracking and MPC. |
| `data/vision/` | Optional vision model, calibration, and image resources. |
| `tools/vision/` | Vision training and conversion utilities. |
| `scripts/waypoints/` | Waypoint planning, smoothing, and overlay tools. |
| `docs/autonomy-notes/README.md` | Placeholder for knowledge notes on pure pursuit, gap-follow, RRT, MPC, and RL. |

## Metrics

Each run writes a CSV trace and JSON summary under `logs/`.

Tracked fields include:

- active mode
- elapsed time
- vehicle pose
- measured speed
- commanded speed
- commanded steering
- collision count
- completion status

## Implementation Notes

This project consolidates and refactors algorithms from earlier autonomy labs into a single system:

- Follow-the-gap control -> `src/unified_autonomy/control/reactive_follow_gap.py`
- RRT planning -> `src/unified_autonomy/planning/local_rrt.py`
- Pure pursuit tracking -> `src/unified_autonomy/tracking/pure_pursuit.py`
- MPC tracking -> `src/unified_autonomy/control/mpc_tracker.py`
- Map and waypoint assets -> `data/maps/` and `data/waypoints/`
- Vision runtime assets -> `data/vision/`

Adapters return a common `ControlCommand`, and the main ROS node owns safety clipping, publishing, visualization, dashboard state, and metrics. This keeps each autonomy strategy interchangeable while preserving one consistent runtime.

## Knowledge Notes

Knowledge and implementation notes for pure pursuit, gap-follow, RRT, MPC, and RL are collected separately in `docs/autonomy-notes/README.md`.
