"""Shared runtime configuration and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DETECTORS = ("cascade_rcnn", "dino")


@dataclass(frozen=True)
class RunConfig:
    detector: str
    data_root: Path
    output_dir: Path
    pretrained: Path | None
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 4
    batch_size: int = 2
    epochs: int = 12
    resume: Path | None = None
    amp: bool = False

    def validate(self, *, require_data: bool = True) -> None:
        if self.detector not in DETECTORS:
            raise ValueError(f"detector must be one of {DETECTORS}, got {self.detector!r}")
        if require_data and not self.data_root.is_dir():
            raise FileNotFoundError(f"PubLayNet root not found: {self.data_root}")
        if require_data:
            from .data import validate_publaynet

            validate_publaynet(self.data_root)
        if self.pretrained is not None and not self.pretrained.is_file():
            raise FileNotFoundError(f"DiT checkpoint not found: {self.pretrained}")
        if self.resume is not None and not self.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {self.resume}")
        if self.batch_size < 1 or self.epochs < 1 or self.num_workers < 0:
            raise ValueError("batch_size/epochs must be positive and num_workers non-negative")
