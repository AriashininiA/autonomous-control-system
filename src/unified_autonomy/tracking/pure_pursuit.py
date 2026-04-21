from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PurePursuitConfig:
    lookahead_m: float = 0.55
    wheelbase_m: float = 0.33
    max_steer_rad: float = 0.42
    max_speed_mps: float = 1.8
    min_speed_mps: float = 0.45
    search_window: int = 120
    loop_close_tol_m: float = 0.4


@dataclass
class TrackingResult:
    speed_mps: float
    steering_rad: float
    goal_xy: tuple[float, float]
    closest_idx: int
    tracking_error_m: float


class PurePursuitTracker:
    """ROS-free pure-pursuit tracker migrated from lab-5."""

    def __init__(self, waypoints_xy: np.ndarray, config: PurePursuitConfig):
        if waypoints_xy.ndim != 2 or waypoints_xy.shape[1] < 2:
            raise ValueError("waypoints_xy must be an Nx2 or wider array")
        self.waypoints = np.asarray(waypoints_xy[:, :2], dtype=float)
        self.config = config
        self.closest_idx = 0
        self.closed_loop = (
            len(self.waypoints) >= 3
            and float(np.linalg.norm(self.waypoints[0] - self.waypoints[-1])) < config.loop_close_tol_m
        )

    @classmethod
    def from_csv(cls, path: str | Path, config: PurePursuitConfig) -> "PurePursuitTracker":
        data = np.loadtxt(str(path), delimiter=",", skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return cls(data[:, :2], config)

    def track(self, x: float, y: float, yaw: float) -> TrackingResult:
        goal_x, goal_y, goal_idx, tracking_error = self._lookahead_point(x, y)
        dx = goal_x - x
        dy = goal_y - y
        goal_x_car = np.cos(yaw) * dx + np.sin(yaw) * dy
        goal_y_car = -np.sin(yaw) * dx + np.cos(yaw) * dy

        lookahead = float(np.hypot(goal_x_car, goal_y_car))
        curvature = 2.0 * goal_y_car / (lookahead * lookahead) if lookahead > 1e-3 else 0.0
        steering = float(np.clip(np.arctan(self.config.wheelbase_m * curvature), -self.config.max_steer_rad, self.config.max_steer_rad))
        steer_ratio = min(abs(steering) / max(self.config.max_steer_rad, 1e-6), 1.0)
        speed = self.config.max_speed_mps - (self.config.max_speed_mps - self.config.min_speed_mps) * steer_ratio
        return TrackingResult(
            speed_mps=float(speed),
            steering_rad=steering,
            goal_xy=(float(goal_x), float(goal_y)),
            closest_idx=goal_idx,
            tracking_error_m=float(tracking_error),
        )

    def _next_index(self, i: int) -> int | None:
        if self.closed_loop:
            return (i + 1) % len(self.waypoints)
        if i + 1 < len(self.waypoints):
            return i + 1
        return None

    def _lookahead_point(self, x: float, y: float) -> tuple[float, float, int, float]:
        n = len(self.waypoints)
        win = min(self.config.search_window, n)
        if self.closed_loop:
            candidate_idxs = np.array([(self.closest_idx + k) % n for k in range(win)])
        else:
            candidate_idxs = np.arange(self.closest_idx, min(self.closest_idx + win, n))

        if candidate_idxs.size == 0:
            p = self.waypoints[-1]
            return float(p[0]), float(p[1]), n - 1, float(np.linalg.norm(p - np.array([x, y])))

        dists = np.linalg.norm(self.waypoints[candidate_idxs] - np.array([x, y]), axis=1)
        local = int(np.argmin(dists))
        self.closest_idx = int(candidate_idxs[local])
        tracking_error = float(dists[local])

        acc = 0.0
        i = self.closest_idx
        for _ in range(n * 2 + 5):
            j = self._next_index(i)
            if j is None:
                break
            p0 = self.waypoints[i]
            p1 = self.waypoints[j]
            seg = float(np.linalg.norm(p1 - p0))
            if seg < 1e-9:
                i = j
                continue
            if acc + seg >= self.config.lookahead_m:
                t = float(np.clip((self.config.lookahead_m - acc) / seg, 0.0, 1.0))
                goal = (1.0 - t) * p0 + t * p1
                return float(goal[0]), float(goal[1]), i, tracking_error
            acc += seg
            i = j

        p = self.waypoints[-1]
        return float(p[0]), float(p[1]), n - 1, tracking_error

