"""Bootstrap repository scripts without requiring an editable installation."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


def run(entrypoint: str) -> None:
    source = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(source))
    getattr(import_module("dit_layout_bench.cli"), entrypoint)()
