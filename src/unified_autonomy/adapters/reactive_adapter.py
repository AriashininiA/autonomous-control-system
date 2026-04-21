import numpy as np

from unified_autonomy.control.reactive_follow_gap import FollowGapConfig, FollowGapController
from unified_autonomy.interfaces import ControlCommand, ModeOutput, PerceptionFrame, VehicleState


class ReactiveFollowGapAdapter:
    name = "reactive"

    def __init__(self, config: dict, safety: dict):
        self.controller = FollowGapController(
            FollowGapConfig(
                target_speed_mps=float(config.get("target_speed_mps", 1.5)),
                range_clip_m=float(config.get("range_clip_m", 3.0)),
                fov_deg=float(config.get("fov_deg", 70.0)),
                bubble_radius_m=float(config.get("bubble_radius_m", 0.6)),
                max_steer_rad=float(safety.get("max_steer_rad", 0.42)),
            )
        )

    def reset(self) -> None:
        pass

    def update(self, state: VehicleState, perception: PerceptionFrame) -> ModeOutput:
        if perception.scan is None:
            return self._command(0.0, 0.0, no_scan=True)

        result = self.controller.compute(
            np.asarray(perception.scan.ranges, dtype=np.float32),
            perception.scan.angle_min,
            perception.scan.angle_increment,
        )
        return self._command(result.speed_mps, result.steering_rad, **result.debug)

    def _command(self, speed: float, steer: float, **debug) -> ModeOutput:
        return ModeOutput(
            command=ControlCommand(
                speed=float(speed),
                steering_angle=float(steer),
                mode=self.name,
                debug=debug,
            )
        )

