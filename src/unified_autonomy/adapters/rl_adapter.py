from __future__ import annotations

from pathlib import Path

from unified_autonomy.interfaces import ControlCommand, ModeOutput, PerceptionFrame, VehicleState


class RLAdapter:
    """Adapter placeholder for the ninth RL project while it is still working.

    Expected contract for the finished policy:
    - observation = normalized scan + vehicle speed + optional waypoint goal
    - action = [speed_mps, steering_rad] or policy-specific action mapped to Ackermann
    - no ROS publishers inside the policy code
    """

    name = "rl"

    def __init__(self, config: dict, safety: dict, policy_path: str | None = None):
        self.config = config
        self.safety = safety
        self.policy_path = Path(policy_path).expanduser() if policy_path else None
        self.policy = None

    def reset(self) -> None:
        pass

    def update(self, state: VehicleState, perception: PerceptionFrame) -> ModeOutput:
        return ModeOutput(
            command=ControlCommand(
                speed=0.0,
                steering_angle=0.0,
                mode=self.name,
                debug={"placeholder": True, "reason": "rl_project_9_in_progress"},
            )
        )
