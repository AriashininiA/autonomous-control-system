from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from unified_autonomy.config import load_config
from unified_autonomy.dashboard_state import DashboardStateStore, VALID_MODES


PROJECT_ROOT = Path(__file__).resolve().parents[4]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "demo.yaml"


class ModeRequest(BaseModel):
    mode: str


def create_app(config_path: str | Path = DEFAULT_CONFIG) -> FastAPI:
    config = load_config(config_path)
    dashboard_cfg = config.raw.get("dashboard", {})
    state_file = config.resolve(str(dashboard_cfg.get("state_file", "dashboard/state.json")))
    store = DashboardStateStore(state_file)

    app = FastAPI(
        title="Unified Autonomy Dashboard",
        description="Control and telemetry dashboard for MPC, RL, and RRT autonomy modes.",
        version="0.1.0",
    )
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.on_event("startup")
    def ensure_state() -> None:
        state = store.read()
        store.write(state)

    @app.get("/")
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/status")
    def status():
        return store.read()

    @app.post("/api/mode")
    def set_mode(request: ModeRequest):
        mode = request.mode.lower().strip()
        if mode not in VALID_MODES:
            raise HTTPException(status_code=400, detail=f"mode must be one of: {', '.join(VALID_MODES)}")
        return store.request_mode(mode)

    return app


app = create_app()

