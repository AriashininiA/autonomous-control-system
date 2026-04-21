from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 0.0


@dataclass
class PerceptionFrame:
    scan: Any | None = None
    obstacles_xy: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))


@dataclass
class ControlCommand:
    speed: float
    steering_angle: float
    mode: str
    debug: dict[str, float | str | bool] = field(default_factory=dict)


@dataclass
class ModeOutput:
    command: ControlCommand
    planned_path: Any | None = None
    goal_xy: tuple[float, float] | None = None
    obstacle_points: np.ndarray | None = None


class ControllerMode(Protocol):
    name: str

    def reset(self) -> None:
        ...

    def update(self, state: VehicleState, perception: PerceptionFrame) -> ModeOutput:
        ...


def state_from_odom(msg: Any) -> VehicleState:
    q = msg.pose.pose.orientation
    sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
    cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return VehicleState(
        x=float(msg.pose.pose.position.x),
        y=float(msg.pose.pose.position.y),
        yaw=float(np.arctan2(sin_yaw, cos_yaw)),
        speed=float(msg.twist.twist.linear.x),
    )
