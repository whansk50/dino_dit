"""Detector-specific YAML configuration, CLI overrides and validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .paths import CONFIG_ROOT, RECENT_CHECKPOINT_NAME
from .settings_validation import validate_settings


DETECTORS = ("cascade_rcnn", "dino")
DEFAULT_DETECTOR = "dino"
DEFAULT_CONFIGS = {
    detector: CONFIG_ROOT / f"{detector}.yaml" for detector in DETECTORS
}

_ALIASES = {
    "batch_size": "training.batch_size",
    "epochs": "training.epochs",
    "lr": "training.detector_lr",
    "lr_backbone": "training.backbone_lr",
    "num_queries": "dino.num_queries",
    "num_select": "dino.num_select",
    "drop_path": "dit.drop_path",
    "use_checkpoint": "dit.use_checkpoint",
}


def per_process_batch_size(global_batch_size: int, world_size: int) -> int:
    """Derive the DDP process batch and reject an ambiguous partial batch."""
    if world_size < 1:
        raise ValueError("world_size must be positive")
    if global_batch_size < 1:
        raise ValueError("training.batch_size must be positive")
    if global_batch_size % world_size:
        raise ValueError(
            f"training.batch_size={global_batch_size} is the global batch size "
            f"and must be divisible by world_size={world_size}"
        )
    return global_batch_size // world_size


def _coerce_like(reference: Any, value: Any, path: str) -> Any:
    """Validate a value against the type established by the default config."""
    if isinstance(reference, bool):
        if not isinstance(value, bool):
            raise ValueError(f"Config value {path} must be a boolean")
        return value
    if isinstance(reference, int):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"Config value {path} must be an integer")
        return value
    if isinstance(reference, float):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Config value {path} must be numeric")
        return float(value)
    if isinstance(reference, str):
        if not isinstance(value, str):
            raise ValueError(f"Config value {path} must be a string")
        return value
    if isinstance(reference, list):
        if not isinstance(value, list):
            raise ValueError(f"Config value {path} must be an array")
        if reference:
            return [
                _coerce_like(reference[0], item, f"{path}[{index}]")
                for index, item in enumerate(value)
            ]
        return value
    raise TypeError(f"Unsupported default config type at {path}: {type(reference).__name__}")


def _merge(base: dict[str, Any], update: dict[str, Any], prefix: str = "") -> None:
    for key, value in update.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"Unknown config key: {path}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Config section {path} must be a table")
            _merge(base[key], value, path)
        else:
            base[key] = _coerce_like(base[key], value, path)


def _set_option(settings: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must use key=value syntax: {expression!r}")
    key, raw_value = expression.split("=", 1)
    key = _ALIASES.get(key, key)
    parts = key.split(".")
    current: dict[str, Any] = settings
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            raise ValueError(f"Unknown config key: {key}")
        current = current[part]
    leaf = parts[-1]
    if leaf not in current:
        raise ValueError(f"Unknown config key: {key}")
    current[leaf] = _coerce_like(current[leaf], yaml.safe_load(raw_value), key)


def _read_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"YAML config root must be a mapping: {path}")
    return document


def _declared_detector(
    document: Mapping[str, Any] | None, path: Path | None
) -> str | None:
    """Return the detector declared by an optional partial config."""
    if document is None:
        return None
    run = document.get("run")
    if run is None:
        return None
    if not isinstance(run, Mapping):
        raise ValueError(f"Config section run must be a table: {path}")
    detector = run.get("detector")
    if detector is not None and detector not in DETECTORS:
        raise ValueError(f"run.detector must be one of {DETECTORS}, got {detector!r}")
    return detector


def load_settings(
    path: Path | None = None,
    options: list[str] | None = None,
    *,
    detector: str | None = None,
) -> dict[str, Any]:
    """Load one detector's defaults, then merge a matching partial config."""
    update = _read_yaml(path) if path is not None else None
    file_detector = _declared_detector(update, path)
    selected_detector = detector or file_detector or DEFAULT_DETECTOR
    if selected_detector not in DETECTORS:
        raise ValueError(
            f"detector must be one of {DETECTORS}, got {selected_detector!r}"
        )
    if file_detector is not None and file_detector != selected_detector:
        raise ValueError(
            f"Selected detector {selected_detector} conflicts with "
            f"run.detector={file_detector} in {path}"
        )

    default_path = DEFAULT_CONFIGS[selected_detector]
    default = _read_yaml(default_path)
    settings = deepcopy(default)
    if update is not None and path.resolve() != default_path.resolve():
        _merge(settings, update)
    for expression in options or []:
        _set_option(settings, expression)
    if settings["run"]["detector"] != selected_detector:
        raise ValueError(
            "Select the detector with --detector or run.detector in the config file; "
            "do not override run.detector through --options"
        )
    return settings


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
    settings: dict[str, Any] = field(default_factory=load_settings)
    weights_dir: Path = Path("weights")

    def section(self, name: str) -> Mapping[str, Any]:
        """Return a named settings section with a stable read-only interface."""
        return self.settings[name]

    @property
    def input(self) -> Mapping[str, Any]:
        return self.section("input")

    @property
    def training(self) -> Mapping[str, Any]:
        return self.section("training")

    @property
    def dit(self) -> Mapping[str, Any]:
        return self.section("dit")

    @property
    def detector_settings(self) -> Mapping[str, Any]:
        return self.section(self.detector)

    @property
    def tracking(self) -> Mapping[str, Any]:
        return self.section("tracking")

    def score_threshold(self) -> float:
        return float(self.detector_settings["score_threshold"])

    def validate(self, *, require_data: bool = True) -> None:
        if self.detector not in DETECTORS:
            raise ValueError(f"detector must be one of {DETECTORS}, got {self.detector!r}")
        configured_detector = self.settings["run"]["detector"]
        if configured_detector != self.detector:
            raise ValueError(
                f"RunConfig detector={self.detector} does not match "
                f"settings run.detector={configured_detector}"
            )
        if require_data and not self.data_root.is_dir():
            raise FileNotFoundError(f"PubLayNet root not found: {self.data_root}")
        if require_data:
            from .data import validate_publaynet

            validate_publaynet(self.data_root)
        if (
            self.resume is None
            and self.pretrained is not None
            and not self.pretrained.is_file()
        ):
            raise FileNotFoundError(f"DiT checkpoint not found: {self.pretrained}")
        if self.resume is not None:
            expected_resume = self.weights_dir / RECENT_CHECKPOINT_NAME
            if self.resume.resolve() != expected_resume.resolve():
                raise ValueError(
                    "Resume checkpoint must be exactly "
                    f"weights_dir/recent.pth: {expected_resume}"
                )
            if not self.resume.is_file():
                raise FileNotFoundError(f"Resume checkpoint not found: {self.resume}")
        if self.batch_size < 1 or self.epochs < 1 or self.num_workers < 0:
            raise ValueError("batch_size/epochs must be positive and num_workers non-negative")
        validate_settings(self.settings)
