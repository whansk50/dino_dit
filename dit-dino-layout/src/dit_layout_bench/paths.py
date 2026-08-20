"""Repository path discovery without relying on the current directory."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_ROOT = PACKAGE_ROOT / "resources"
DINO_ROOT = PACKAGE_ROOT / "_vendor" / "dino"


def require_path(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path
