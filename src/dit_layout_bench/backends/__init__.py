"""Lazy detector backend dispatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dit_layout_bench.config import RunConfig


Prediction = dict[str, Any]
Predictor = Callable[[Path], list[Prediction]]


class PredictorBuilder(Protocol):
    def __call__(
        self, config: RunConfig, *, score_threshold: float = 0.5
    ) -> Predictor: ...


@dataclass(frozen=True)
class Backend:
    """Operations exposed by every detector backend."""

    train: Callable[[RunConfig], None]
    build_predictor: PredictorBuilder


def get_backend(name: str) -> Backend:
    if name == "cascade_rcnn":
        from . import cascade_rcnn

        module = cascade_rcnn
    elif name == "dino":
        from . import dino

        module = dino
    else:
        raise ValueError(f"Unsupported detector: {name}")
    return Backend(train=module.train, build_predictor=module.build_predictor)
