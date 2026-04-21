# Foxglove Visualization

Foxglove is the primary visualization surface for this portfolio project.

## Start

Terminal 1:

```bash
./run_demo.sh --mode reactive --foxglove
```

Terminal 2, if you prefer starting the bridge separately:

```bash
./run_foxglove.sh
```

Then open Foxglove Studio and connect to:

```text
ws://localhost:8765
```

## Recommended Panels

- 3D panel with fixed frame `map`
- Plot panel for `/drive.drive.speed` and `/drive.drive.steering_angle`
- Raw Messages panel for `/unified/status`
- Diagnostics-style layout for dashboard comparison metrics

## Key Topics

- `/scan`: LiDAR scan
- `/ego_racecar/odom`: vehicle state
- `/drive`: Ackermann command
- `/unified/planned_path`: selected planner/controller path
- `/unified/goal`: current goal marker
- `/unified/obstacles`: local obstacle points
- `/unified/status`: active mode and speed text marker

