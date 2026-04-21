from __future__ import annotations

import math

import numpy as np

from unified_autonomy.interfaces import ControlCommand, ModeOutput, PerceptionFrame, VehicleState
from unified_autonomy.planning.local_rrt import LocalRRTPlanner, RRTConfig
from unified_autonomy.tracking.pure_pursuit import PurePursuitConfig, PurePursuitTracker


class RRTReactiveAdapter:
    """Adapter for lab-6 local RRT plus pure-pursuit tracking."""

    name = "rrt"

    def __init__(self, config: dict, safety: dict):
        self.planner = LocalRRTPlanner(
            RRTConfig(
                step_size_m=float(config.get("step_size_m", 0.35)),
                max_iter=int(config.get("max_iter", 200)),
                goal_ahead_m=float(config.get("goal_ahead_m", 2.0)),
            )
        )
        self.tracker_config = PurePursuitConfig(
            lookahead_m=float(config.get("lookahead_m", 0.55)),
            max_speed_mps=float(config.get("target_speed_mps", 1.2)),
            min_speed_mps=float(config.get("min_speed_mps", 0.35)),
            max_steer_rad=float(safety.get("max_steer_rad", 0.42)),
        )

    def reset(self) -> None:
        pass

    def update(self, state: VehicleState, perception: PerceptionFrame) -> ModeOutput:
        if perception.scan is None:
            return self._stop("no_scan")
        self.planner.update_scan(
            np.asarray(perception.scan.ranges, dtype=np.float32),
            perception.scan.angle_min,
            perception.scan.angle_increment,
            perception.scan.range_min,
            perception.scan.range_max,
        )
        local_path = self.planner.plan()
        if local_path is None or len(local_path) < 2:
            return self._stop("no_rrt_path")

        world_path = self._local_to_world(local_path, state)
        tracker = PurePursuitTracker(world_path, self.tracker_config)
        track = tracker.track(state.x, state.y, state.yaw)
        return ModeOutput(
            command=ControlCommand(
                speed=track.speed_mps,
                steering_angle=track.steering_rad,
                mode=self.name,
                debug={"tracking_error_m": track.tracking_error_m, "path_points": len(world_path)},
            ),
            planned_path=world_path,
            goal_xy=track.goal_xy,
        )

    def _stop(self, reason: str) -> ModeOutput:
        return ModeOutput(
            command=ControlCommand(0.0, 0.0, self.name, {"safe_stop": True, "reason": reason})
        )

    @staticmethod
    def _local_to_world(local_path: np.ndarray, state: VehicleState) -> np.ndarray:
        c = math.cos(state.yaw)
        s = math.sin(state.yaw)
        x = state.x + c * local_path[:, 0] - s * local_path[:, 1]
        y = state.y + s * local_path[:, 0] + c * local_path[:, 1]
        return np.column_stack([x, y])
