"""Source-tree import and subprocess environment for validation scripts."""

from __future__ import annotations

import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"

source_path = str(SOURCE_ROOT)
if source_path not in sys.path:
    sys.path.insert(0, source_path)


def project_environment() -> dict[str, str]:
    """Return an environment in which child entrypoints can import the package."""
    environment = os.environ.copy()
    inherited = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((source_path, inherited)) if inherited else source_path
    )
    return environment


def expected_distributed_tag(world_size: int) -> str:
    """Return the MLflow distributed tag expected for a validation run."""
    if world_size < 1:
        raise ValueError("world_size must be positive")
    return str(world_size > 1).lower()


def execution_mode(world_size: int) -> str:
    """Return a human-readable execution mode for validation output."""
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if world_size == 1:
        return "single GPU"
    return f"{world_size}-process DDP"
