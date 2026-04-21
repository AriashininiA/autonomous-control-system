from __future__ import annotations

from unified_autonomy.control.mpc_tracker import KinematicMPCTracker, MPCConfig
from unified_autonomy.interfaces import ControlCommand, ModeOutput, PerceptionFrame, VehicleState


class MPCAdapter:
    """Adapter for the lab-8 MPC implementation migrated into a ROS-free tracker."""

    name = "mpc"

    def __init__(self, config: dict, safety: dict, waypoint_csv: str | None = None):
        if waypoint_csv is None:
            raise ValueError("MPC mode requires assets.waypoint_csv")
        max_speed = float(safety.get("max_speed_mps", 4.0))
        max_steer = float(safety.get("max_steer_rad", 0.42))
        self.tracker = KinematicMPCTracker(
            waypoint_csv,
            MPCConfig(
                horizon=int(config.get("horizon", 8)),
                dt_s=float(config.get("dt_s", 0.1)),
                max_speed_mps=max_speed,
                max_steer_rad=max_steer,
                min_steer_rad=-max_steer,
            ),
        )

    def reset(self) -> None:
        pass

    def update(self, state: VehicleState, perception: PerceptionFrame) -> ModeOutput:
        result = self.tracker.compute(state)
        return ModeOutput(
            command=ControlCommand(
                speed=result.speed_mps,
                steering_angle=result.steering_rad,
                mode=self.name,
                debug={"solved": result.solved},
            ),
            planned_path=result.predicted_path,
        )
