from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


VALID_MODES = ("mpc", "rl", "rrt", "reactive")


class DashboardStateStore:
    """Small file-backed bridge between the FastAPI dashboard and ROS node."""

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return self.default_state()
        try:
            with self.path.open("r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            return self.default_state()
        return {**self.default_state(), **state}

    def write(self, state: dict[str, Any]) -> None:
        merged = {**self.default_state(), **state, "updated_at": time.time()}
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2, sort_keys=True)
        tmp.replace(self.path)

    def request_mode(self, mode: str) -> dict[str, Any]:
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode: {mode}")
        state = self.read()
        state["requested_mode"] = mode
        state["last_error"] = ""
        self.write(state)
        return self.read()

    def update_runtime(
        self,
        *,
        active_mode: str,
        requested_mode: str | None = None,
        metrics: dict[str, Any] | None = None,
        command: dict[str, Any] | None = None,
        vehicle: dict[str, Any] | None = None,
        status: str = "running",
        last_error: str = "",
    ) -> None:
        state = self.read()
        state["active_mode"] = active_mode
        state["requested_mode"] = requested_mode or state.get("requested_mode") or active_mode
        state["status"] = status
        state["last_error"] = last_error
        if metrics is not None:
            state["metrics"] = metrics
        if command is not None:
            state["command"] = command
        if vehicle is not None:
            state["vehicle"] = vehicle
        self.write(state)

    @staticmethod
    def default_state() -> dict[str, Any]:
        return {
            "requested_mode": "reactive",
            "active_mode": "reactive",
            "available_modes": list(VALID_MODES),
            "status": "idle",
            "last_error": "",
            "updated_at": time.time(),
            "metrics": {
                "elapsed_s": 0.0,
                "average_speed_mps": 0.0,
                "max_speed_mps": 0.0,
                "collisions": 0,
                "success": False,
                "completion_status": "idle",
            },
            "command": {"speed": 0.0, "steering_angle": 0.0},
            "vehicle": {"x": 0.0, "y": 0.0, "yaw": 0.0, "speed": 0.0},
        }

