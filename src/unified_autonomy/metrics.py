from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from unified_autonomy.interfaces import ControlCommand, PerceptionFrame, VehicleState


@dataclass
class MetricsSnapshot:
    mode: str
    elapsed_s: float
    average_speed_mps: float
    max_speed_mps: float
    collisions: int
    success: bool
    completion_status: str


class MetricsLogger:
    def __init__(self, output_dir: Path, run_tag: str, collision_distance_m: float):
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = output_dir / f"{run_tag}_{stamp}.csv"
        self.summary_path = output_dir / f"{run_tag}_{stamp}_summary.json"
        self.collision_distance_m = collision_distance_m
        self.start_time = time.monotonic()
        self.last_collision_time = 0.0
        self.collision_count = 0
        self.speed_samples: list[float] = []
        self.max_speed = 0.0
        self.mode = "unknown"

        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._csv_file,
            fieldnames=["t_s", "mode", "x", "y", "yaw", "speed", "cmd_speed", "cmd_steer", "collisions"],
        )
        self._writer.writeheader()

    def update(self, state: VehicleState, perception: PerceptionFrame, command: ControlCommand) -> None:
        now = time.monotonic()
        elapsed = now - self.start_time
        self.mode = command.mode
        self.speed_samples.append(max(0.0, state.speed))
        self.max_speed = max(self.max_speed, state.speed)

        if perception.scan is not None:
            ranges = np.asarray(perception.scan.ranges, dtype=np.float32)
            finite_ranges = ranges[np.isfinite(ranges)]
            if finite_ranges.size and float(np.min(finite_ranges)) < self.collision_distance_m:
                if now - self.last_collision_time > 1.0:
                    self.collision_count += 1
                    self.last_collision_time = now

        self._writer.writerow(
            {
                "t_s": f"{elapsed:.3f}",
                "mode": command.mode,
                "x": f"{state.x:.4f}",
                "y": f"{state.y:.4f}",
                "yaw": f"{state.yaw:.4f}",
                "speed": f"{state.speed:.4f}",
                "cmd_speed": f"{command.speed:.4f}",
                "cmd_steer": f"{command.steering_angle:.4f}",
                "collisions": self.collision_count,
            }
        )
        self._csv_file.flush()

    def snapshot(self, status: str = "running") -> MetricsSnapshot:
        elapsed = time.monotonic() - self.start_time
        avg_speed = float(np.mean(self.speed_samples)) if self.speed_samples else 0.0
        return MetricsSnapshot(
            mode=self.mode,
            elapsed_s=elapsed,
            average_speed_mps=avg_speed,
            max_speed_mps=float(self.max_speed),
            collisions=self.collision_count,
            success=status == "success" and self.collision_count == 0,
            completion_status=status,
        )

    def close(self, status: str = "stopped") -> None:
        snap = self.snapshot(status)
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(snap.__dict__, f, indent=2)
        self._csv_file.close()

