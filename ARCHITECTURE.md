# Architecture Plan

## Title

**Unified Autonomous Systems Platform with Learning-Based and Classical Control**

## GitHub / CV Description

Modular ROS2 autonomy stack unifying LiDAR perception, local planning, trajectory tracking, reinforcement-learning policy integration, visualization, and evaluation. The system supports one-command demos and mode switching between MPC, RRT/reactive planning, and RL controllers using shared interfaces and common metrics.

## Runtime Flow

```text
Simulator / vehicle
  -> /scan, /ego_racecar/odom
  -> MainDemo ROS2 node
  -> selected ControllerMode adapter
  -> /drive Ackermann command
  -> /unified/* visualization topics
  -> logs/*.csv and logs/*_summary.json
```

## File-by-File Responsibilities

- `run_demo.sh`: one command entry point. Sources ROS2/workspace setup, optionally starts simulator/Foxglove Bridge, and launches the Python orchestrator.
- `configs/demo.yaml`: all demo settings: mode, topics, safety limits, map/waypoint/policy paths, visualization, metrics.
- `src/unified_autonomy/main_demo.py`: central orchestrator. Owns ROS subscriptions, mode selection, control tick, safety clipping, `/drive`, visualization, and metrics.
- `src/unified_autonomy/interfaces.py`: stable data contracts: `VehicleState`, `PerceptionFrame`, `ControlCommand`, `ModeOutput`, and `ControllerMode`.
- `src/unified_autonomy/adapters/reactive_adapter.py`: working starter adapter based on follow-the-gap.
- `src/unified_autonomy/adapters/rrt_adapter.py`: active adapter that runs local RRT and tracks the generated path with pure pursuit.
- `src/unified_autonomy/adapters/mpc_adapter.py`: active adapter that runs the ROS-free kinematic MPC tracker.
- `src/unified_autonomy/adapters/rl_adapter.py`: project-9 RL placeholder. Load the trained policy here once stable.
- `src/unified_autonomy/metrics.py`: CSV and JSON logging for mode, speed, time, collision count, and run status.
- `src/unified_autonomy/visualization.py`: Foxglove-compatible ROS topics for path, goal, obstacle points, and status text.
- `src/unified_autonomy/dashboard/`: FastAPI app and static dashboard for live mode switching and metrics.
- `src/unified_autonomy/dashboard_state.py`: file-backed control/status bridge used by FastAPI and `main_demo.py`.
- `foxglove/README.md`: Foxglove connection notes and recommended panel/topic layout.
- `rviz/demo.rviz`: legacy fallback visualization layout.

## Migration Plan

| Existing source | Action | New home |
|---|---|---|
| `lab-4-follow-the-gap-team7/gap_follow/gap_follow/reactive_node.py` | Algorithm migrated; ROS publishing removed | `control/reactive_follow_gap.py` + `adapters/reactive_adapter.py` |
| `lab-6-motion-planning-team7/lab7_pkg/scripts/rrt_node.py` | Occupancy grid, RRT sampling, collision check migrated | `planning/local_rrt.py` + `adapters/rrt_adapter.py` |
| `lab-8-model-predictive-control-team7/mpc/mpc/mpc_node.py` | Optimizer split from ROS node; returns `ControlCommand` | `control/mpc_tracker.py` + `adapters/mpc_adapter.py` |
| `lab-5-slam-and-pure-pursuit-team7/pure_pursuit/...` | Tracker migrated for RRT path tracking | `tracking/pure_pursuit.py` |
| `lab-5-slam-and-pure-pursuit-team7/tools/` | Copied as reproducible map/waypoint utilities | `scripts/waypoints/` |
| `lab-7-vision-lab-team7/src/` | Copied as optional perception utilities | `perception/vision/` |
| `lab-7-vision-lab-team7/nn_object_detector/model_78.*`, `resource/`, `calibration/` | Copied as local vision assets | `data/vision/` |
| `lab-7-vision-lab-team7/nn_object_detector/f110_yolo_hw.ipynb`, conversion scripts | Copied as development tools | `tools/vision/` |
| project 9 drone/RL work | Placeholder now; add policy loader when stable | `adapters/rl_adapter.py`, `models/rl/` |

## Interface Rule

Adapters should never publish `/drive` directly. They should accept:

```python
update(state: VehicleState, perception: PerceptionFrame) -> ModeOutput
```

and return:

```python
ControlCommand(speed=..., steering_angle=..., mode=...)
```

This is the key change that makes the work read as one professional system instead of separate labs.

## README Content Checklist

- One-sentence project value proposition.
- Architecture diagram.
- Demo GIF or screenshots from Foxglove.
- Quick start with `./run_demo.sh --mode reactive`.
- Mode comparison table: MPC vs RRT/reactive vs RL.
- Metrics table from at least one run.
- Clear note that RL policy is a plug-in mode and project-9 integration is in progress.
- “What I built” section focused on interfaces, integration, evaluation, and ROS2 system design.
- “Future work” section: finish MPC adapter refactor, finish RL policy integration, add CI/smoke tests, add richer tracking error metrics.

## Job-Application Framing

Lead with integration and engineering judgment:

- “Unified multiple autonomy algorithms into a reusable ROS2 platform.”
- “Designed shared controller interfaces for MPC, sampling-based planning, reactive planning, and RL policies.”
- “Built one-command demos with Foxglove visualization and quantitative metrics.”
- “Refactored course implementations into production-style modules with common topics, configs, lifecycle, and evaluation.”
