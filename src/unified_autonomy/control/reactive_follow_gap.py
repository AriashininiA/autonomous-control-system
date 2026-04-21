from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class FollowGapConfig:
    target_speed_mps: float = 1.5
    range_clip_m: float = 3.0
    fov_deg: float = 70.0
    bubble_radius_m: float = 0.6
    max_steer_rad: float = 0.42


@dataclass
class FollowGapResult:
    speed_mps: float
    steering_rad: float
    debug: dict


class FollowGapController:
    """ROS-free version of the lab-4 reactive follow-the-gap controller."""

    def __init__(self, config: FollowGapConfig):
        self.config = config
        self.fov_rad = math.radians(config.fov_deg)

    def compute(self, ranges: np.ndarray, angle_min: float, angle_increment: float) -> FollowGapResult:
        if ranges.size == 0:
            return self._result(0.0, 0.0, no_scan=True)

        angles = angle_min + np.arange(ranges.size) * angle_increment
        mask = np.logical_and(angles > -self.fov_rad, angles < self.fov_rad)
        ranges = np.asarray(ranges[mask], dtype=np.float32)
        angles = np.asarray(angles[mask], dtype=np.float32)

        if ranges.size == 0:
            return self._result(0.0, 0.0, no_fov_points=True)

        finite = np.isfinite(ranges)
        ranges[~finite] = self.config.range_clip_m
        ranges[np.isnan(ranges)] = 0.0
        ranges = np.clip(ranges, 0.0, self.config.range_clip_m)

        if not np.any(ranges > 1e-3):
            return self._result(0.4, 0.0, blocked=True)

        closest_idx = int(np.argmin(np.where(ranges > 1e-3, ranges, np.inf)))
        free = self._apply_safety_bubble(ranges, angles, closest_idx)
        gap_start, gap_end = self._max_gap(free > 1e-3)
        if gap_end <= gap_start:
            return self._result(0.4, 0.0, no_gap=True)

        best_idx = gap_start + int(np.argmax(free[gap_start:gap_end]))
        steering = float(np.clip(angles[best_idx], -self.config.max_steer_rad, self.config.max_steer_rad))
        steer_ratio = min(abs(steering) / max(self.config.max_steer_rad, 1e-6), 1.0)
        speed = self.config.target_speed_mps * (1.0 - 0.45 * steer_ratio)
        return self._result(speed, steering, gap_width=int(gap_end - gap_start), closest_range_m=float(ranges[closest_idx]))

    def _apply_safety_bubble(self, ranges: np.ndarray, angles: np.ndarray, closest_idx: int) -> np.ndarray:
        free = ranges.copy()
        closest_range = float(max(ranges[closest_idx], 1e-6))
        closest_angle = float(angles[closest_idx])
        half_angle = math.atan2(self.config.bubble_radius_m, closest_range)
        bubble = np.logical_and(angles > closest_angle - half_angle, angles < closest_angle + half_angle)
        free[bubble] = 0.0
        return free

    def _result(self, speed: float, steering: float, **debug) -> FollowGapResult:
        return FollowGapResult(speed_mps=float(speed), steering_rad=float(steering), debug=debug)

    @staticmethod
    def _max_gap(is_free: np.ndarray) -> tuple[int, int]:
        best_start = 0
        best_end = 0
        current_start = None
        for i, ok in enumerate(is_free):
            if ok and current_start is None:
                current_start = i
            if (not ok or i == len(is_free) - 1) and current_start is not None:
                end = i + 1 if ok and i == len(is_free) - 1 else i
                if end - current_start > best_end - best_start:
                    best_start, best_end = current_start, end
                current_start = None
        return best_start, best_end

