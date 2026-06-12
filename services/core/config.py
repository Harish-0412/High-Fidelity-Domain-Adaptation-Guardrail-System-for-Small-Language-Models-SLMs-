from __future__ import annotations

from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none"}:
        return None
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load the flat YAML files used by Week 1 without requiring PyYAML."""
    yaml_path = Path(path)
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{yaml_path} must contain a mapping")
        return loaded
    except ModuleNotFoundError:
        data: dict[str, Any] = {}
        for line_no, raw_line in enumerate(yaml_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Unsupported YAML syntax in {yaml_path}:{line_no}: {raw_line}")
            key, value = line.split(":", 1)
            data[key.strip()] = _parse_scalar(value)
        return data


def load_base_config(root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    return load_simple_yaml(root / "configs" / "base.yaml")

