# Unified Autonomous Systems Platform

Portfolio-ready integration of classical and learning-based autonomy for F1TENTH-scale racing and drone-control coursework.

## Project Positioning

**Unified Autonomous Systems Platform with Learning-Based and Classical Control** consolidates separate autonomy labs into one modular ROS2 system. The platform runs perception, planning, control, visualization, and evaluation behind a single demo interface, making it possible to compare MPC, RRT/reactive planning, and reinforcement-learning policies under common topics, configs, and metrics.

## Architecture

```text
Sensors / Simulator
  /scan, /ego_racecar/odom, camera topics
        |
        v
main_demo.py orchestrator
  - subscribes to state and perception topics
  - selects one autonomy mode
  - calls the mode adapter through a shared interface
  - publishes /drive
  - publishes visualization markers and planned paths
  - records metrics
        |
        +--> MPC adapter         runs migrated lab-8 MPC math
        +--> RRT/reactive adapter runs migrated lab-6 RRT + lab-4 follow-gap
        +--> RL adapter          placeholder for project 9 policy
        |
        v
Ackermann command /drive
```

## Quick Start

```bash
cd unified_autonomy_platform
./run_demo.sh --mode reactive --sim --foxglove
```

This launches the F1TENTH gym bridge, map server, robot model, Foxglove Bridge, and the autonomy controller. Foxglove should open automatically; if it does not, open Foxglove Studio and connect to `ws://localhost:8765`.

Start the product-style web dashboard in a second terminal:

```bash
pip install -r requirements.txt
./run_dashboard.sh
```

Then open `http://127.0.0.1:8080`. The dashboard writes requested mode changes to `dashboard/state.json`; the ROS orchestrator polls that file and switches between MPC, RL, RRT, and reactive modes live.

Other modes:

```bash
./run_demo.sh --mode mpc
./run_demo.sh --mode rrt
./run_demo.sh --mode rl
```

The RL adapter is intentionally a placeholder until the ninth RL project is finalized.

## Folder Responsibilities

- `run_demo.sh`: one-command demo runner; sources ROS2, optionally launches sim/Foxglove Bridge, then starts the orchestrator.
- `configs/demo.yaml`: single source of truth for topics, selected mode, safety limits, assets, metrics, and visualization.
- `src/unified_autonomy/main_demo.py`: central ROS2 orchestrator for perception -> planning -> control.
- `src/unified_autonomy/interfaces.py`: shared dataclasses and controller interface.
- `src/unified_autonomy/adapters/`: mode-specific adapters for MPC, RL, RRT, and reactive follow-the-gap.
- `src/unified_autonomy/metrics.py`: collision/speed/time/tracking metrics logger.
- `src/unified_autonomy/visualization.py`: Foxglove-compatible path, goal, obstacle, and status publishers.
- `foxglove/`: Foxglove setup notes and recommended topic layout.
- `src/unified_autonomy/dashboard/`: FastAPI dashboard and static product UI for mode switching and metrics.
- `src/unified_autonomy/dashboard_state.py`: file-backed bridge between the dashboard and ROS node.
- `data/maps`, `data/waypoints`: canonical portfolio assets copied or symlinked from the original labs.
- `logs`: generated metrics CSV and JSON summaries.

## Migration Map From Coursework

Keep the original lab repos as archived references. The reusable algorithms and demo assets now live inside this platform:

- MPC math from `lab-8-model-predictive-control-team7/mpc/mpc/mpc_node.py` is refactored into `src/unified_autonomy/control/mpc_tracker.py`.
- Follow-the-gap from `lab-4-follow-the-gap-team7/gap_follow/gap_follow/reactive_node.py` is refactored into `src/unified_autonomy/control/reactive_follow_gap.py`.
- RRT from `lab-6-motion-planning-team7/lab7_pkg/scripts/rrt_node.py` is refactored into `src/unified_autonomy/planning/local_rrt.py`.
- Pure pursuit from `lab-5-slam-and-pure-pursuit-team7/pure_pursuit/...` is refactored into `src/unified_autonomy/tracking/pure_pursuit.py`.
- Waypoint tools from `lab-5-slam-and-pure-pursuit-team7/tools/` are copied under `scripts/waypoints/`.
- Demo maps and waypoint CSVs are copied under `data/maps/` and `data/waypoints/`.
- Vision files from `lab-7-vision-lab-team7/src/` are copied under `src/unified_autonomy/perception/vision/` as optional perception utilities.
- Vision model/resource assets are copied under `data/vision/`, including `model_78.onnx`, its external `model_78.onnx.data`, `model_78.pt`, calibration images, and demo images.
- Vision training/conversion utilities are under `tools/vision/`, separate from runtime code and assets.
- Add project 9 RL policy later under `models/rl/` and wire it through `RLAdapter`.

## Metrics

Each run logs:

- mode
- elapsed time
- average speed
- max speed
- collisions
- completion status
- optional tracking error when a reference path is available

## Portfolio Framing

Use this project as one integrated autonomy stack, not as a list of assignments:

> Built a modular ROS2 autonomy platform that unifies LiDAR perception, local planning, trajectory tracking, learning-based policies, visualization, and quantitative evaluation. Implemented mode switching between MPC, RRT/reactive planning, and RL controllers under a shared interface, enabling controller comparison in simulation with consistent metrics.
