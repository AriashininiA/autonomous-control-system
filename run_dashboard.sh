#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"

python3 - <<'PY'
try:
    import fastapi  # noqa: F401
    import uvicorn  # noqa: F401
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dashboard dependency: {exc.name}. Run: pip install -r requirements.txt"
    )
PY

python3 -m uvicorn unified_autonomy.dashboard.app:app --host "$HOST" --port "$PORT" --reload
