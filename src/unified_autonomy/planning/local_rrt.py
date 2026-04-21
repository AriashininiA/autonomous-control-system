from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class RRTConfig:
    resolution_m: float = 0.1
    width: int = 40
    height: int = 40
    step_size_m: float = 0.35
    goal_tolerance_m: float = 0.5
    max_iter: int = 200
    goal_ahead_m: float = 2.0


@dataclass
class RRTNode:
    x: float
    y: float
    parent: int | None = None


class LocalRRTPlanner:
    """Local occupancy-grid RRT migrated from lab-6 into a ROS-free planner."""

    def __init__(self, config: RRTConfig):
        self.config = config
        self.origin_x = 0.0
        self.origin_y = -config.width * config.resolution_m / 2.0
        self.grid = -np.ones((config.height, config.width), dtype=int)
        self.rng = np.random.default_rng()

    def update_scan(self, ranges: np.ndarray, angle_min: float, angle_increment: float, range_min: float, range_max: float) -> np.ndarray:
        grid = -np.ones((self.config.height, self.config.width), dtype=int)
        ix0 = int((0.0 - self.origin_x) / self.config.resolution_m)
        iy0 = int((0.0 - self.origin_y) / self.config.resolution_m)

        for i, r in enumerate(ranges):
            if not np.isfinite(r) or r < range_min or r > range_max:
                continue
            angle = angle_min + i * angle_increment
            x = float(r) * math.cos(angle)
            y = float(r) * math.sin(angle)
            ix1 = int((x - self.origin_x) / self.config.resolution_m)
            iy1 = int((y - self.origin_y) / self.config.resolution_m)
            dx = ix1 - ix0
            dy = iy1 - iy0
            steps = max(abs(dx), abs(dy))
            if steps == 0:
                continue
            for k in range(steps):
                ix = int(ix0 + k * dx / steps)
                iy = int(iy0 + k * dy / steps)
                if 0 <= ix < self.config.width and 0 <= iy < self.config.height:
                    grid[iy, ix] = 0
            if 0 <= ix1 < self.config.width and 0 <= iy1 < self.config.height:
                for ox in (-1, 0, 1):
                    for oy in (-1, 0, 1):
                        ix = ix1 + ox
                        iy = iy1 + oy
                        if 0 <= ix < self.config.width and 0 <= iy < self.config.height:
                            grid[iy, ix] = 100

        self.grid = grid
        return grid

    def plan(self) -> np.ndarray | None:
        tree = [RRTNode(0.0, 0.0, None)]
        goal = (self.config.goal_ahead_m, 0.0)
        for _ in range(self.config.max_iter):
            sample = self._sample_free()
            if sample is None:
                continue
            nearest_idx = self._nearest(tree, sample)
            new_node = self._steer(tree[nearest_idx], sample)
            if new_node is None or not self._collision_free(tree[nearest_idx], new_node):
                continue
            new_node.parent = nearest_idx
            tree.append(new_node)
            if math.hypot(new_node.x - goal[0], new_node.y - goal[1]) < self.config.goal_tolerance_m:
                return self._trace_path(tree, len(tree) - 1)
        return None

    def _sample_free(self) -> tuple[float, float] | None:
        free = np.argwhere(self.grid == 0)
        if free.size == 0:
            return None
        row, col = free[int(self.rng.integers(0, len(free)))]
        x = self.origin_x + col * self.config.resolution_m
        y = self.origin_y + row * self.config.resolution_m
        return float(x), float(y)

    @staticmethod
    def _nearest(tree: list[RRTNode], sample: tuple[float, float]) -> int:
        dists = [math.hypot(node.x - sample[0], node.y - sample[1]) for node in tree]
        return int(np.argmin(dists))

    def _steer(self, nearest: RRTNode, sample: tuple[float, float]) -> RRTNode | None:
        dx = sample[0] - nearest.x
        dy = sample[1] - nearest.y
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            return None
        scale = min(self.config.step_size_m, dist) / dist
        return RRTNode(nearest.x + scale * dx, nearest.y + scale * dy, None)

    def _collision_free(self, start: RRTNode, end: RRTNode) -> bool:
        dist = math.hypot(end.x - start.x, end.y - start.y)
        steps = max(int(dist / self.config.resolution_m), 1)
        for i in range(steps + 1):
            t = i / steps
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            gx = int((x - self.origin_x) / self.config.resolution_m)
            gy = int((y - self.origin_y) / self.config.resolution_m)
            if gx < 0 or gx >= self.config.width or gy < 0 or gy >= self.config.height:
                return False
            if self.grid[gy, gx] != 0:
                return False
        return True

    @staticmethod
    def _trace_path(tree: list[RRTNode], idx: int) -> np.ndarray:
        points = []
        while idx is not None:
            node = tree[idx]
            points.append((node.x, node.y))
            idx = node.parent
        points.reverse()
        return np.asarray(points, dtype=float)

