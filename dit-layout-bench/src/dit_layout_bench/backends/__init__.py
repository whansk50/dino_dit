"""Lazy detector backend dispatch."""

from __future__ import annotations

from importlib import import_module


def get_backend(name: str):
    if name not in {"cascade_rcnn", "dino"}:
        raise ValueError(f"Unsupported detector: {name}")
    return import_module(f"dit_layout_bench.backends.{name}")

