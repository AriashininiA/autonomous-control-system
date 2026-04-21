from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only on minimal Python installs
    yaml = None


@dataclass(frozen=True)
class DemoConfig:
    raw: dict[str, Any]
    path: Path

    @property
    def topics(self) -> dict[str, str]:
        return self.raw.get("topics", {})

    @property
    def runtime(self) -> dict[str, Any]:
        return self.raw.get("runtime", {})

    @property
    def modes(self) -> dict[str, Any]:
        return self.raw.get("modes", {})

    @property
    def safety(self) -> dict[str, Any]:
        return self.raw.get("safety", {})

    @property
    def assets(self) -> dict[str, str]:
        return self.raw.get("assets", {})

    @property
    def metrics(self) -> dict[str, Any]:
        return self.raw.get("metrics", {})

    @property
    def visualization(self) -> dict[str, Any]:
        return self.raw.get("visualization", {})

    def mode_config(self, mode: str) -> dict[str, Any]:
        return self.modes.get(mode, {})

    def resolve(self, maybe_relative_path: str) -> Path:
        path = Path(maybe_relative_path).expanduser()
        if path.is_absolute():
            return path
        base = self.path.parent.parent if self.path.parent.name == "configs" else self.path.parent
        return (base / path).resolve()


def load_config(path: str | Path) -> DemoConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        if yaml is not None:
            raw = yaml.safe_load(f) or {}
        else:
            raw = _load_simple_yaml(f.read())
    return DemoConfig(raw=raw, path=config_path)


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Small fallback parser for this repo's simple nested YAML config."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, sep, value = line.strip().partition(":")
        if not sep:
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            node: dict[str, Any] = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    if value in ("true", "True"):
        return True
    if value in ("false", "False"):
        return False
    if value in ("null", "None"):
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value
