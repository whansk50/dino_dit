"""Shared YAML configuration, CLI overrides and runtime validation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .paths import CONFIG_ROOT


DETECTORS = ("cascade_rcnn", "dino")
DEFAULT_CONFIG = CONFIG_ROOT / "default.yaml"

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


def _parse_override(value: str) -> Any:
    return yaml.safe_load(value)


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
    current[leaf] = _coerce_like(
        current[leaf], _parse_override(raw_value), key
    )


def load_settings(
    path: Path | None = None, options: list[str] | None = None
) -> dict[str, Any]:
    """Load defaults, merge an optional partial YAML file, then CLI overrides."""
    default = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    if not isinstance(default, dict):
        raise ValueError(f"YAML config root must be a mapping: {DEFAULT_CONFIG}")
    settings = deepcopy(default)
    selected = path or DEFAULT_CONFIG
    if selected.resolve() != DEFAULT_CONFIG.resolve():
        update = yaml.safe_load(selected.read_text(encoding="utf-8"))
        if not isinstance(update, dict):
            raise ValueError(f"YAML config root must be a mapping: {selected}")
        _merge(settings, update)
    for expression in options or []:
        _set_option(settings, expression)
    return settings


def _validate_settings(settings: Mapping[str, Any]) -> None:
    input_settings = settings["input"]
    scales = input_settings["short_edge_scales"]
    if not scales or any(not isinstance(value, int) or value < 1 for value in scales):
        raise ValueError("input.short_edge_scales must contain positive integers")
    if input_settings["max_long_edge"] < max(scales):
        raise ValueError("input.max_long_edge must be >= the largest short-edge scale")
    if input_settings["random_flip"] not in {"horizontal", "none"}:
        raise ValueError("input.random_flip must be 'horizontal' or 'none'")
    if len(input_settings["mean"]) != 3 or len(input_settings["std"]) != 3:
        raise ValueError("input.mean and input.std must each contain three values")
    if any(value <= 0 for value in input_settings["std"]):
        raise ValueError("input.std values must be positive")

    training = settings["training"]
    for key in ("checkpoint_every_epochs", "evaluate_every_epochs"):
        if not isinstance(training[key], int) or training[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if training["prefetch_factor"] < 1:
        raise ValueError("training.prefetch_factor must be a positive integer")
    if training["warmup_iters"] < 0:
        raise ValueError("training.warmup_iters must be non-negative")
    for key in ("detector_lr", "backbone_lr", "weight_decay", "warmup_factor"):
        if training[key] < 0:
            raise ValueError(f"training.{key} must be non-negative")
    if not 0 < training["warmup_factor"] <= 1:
        raise ValueError("training.warmup_factor must be in (0, 1]")

    dino = settings["dino"]
    if dino["optimizer"] not in {"adam", "adamw"}:
        raise ValueError("dino.optimizer must be 'adam' or 'adamw'")
    if dino["scheduler"] not in {"step", "multistep"}:
        raise ValueError("dino.scheduler must be 'step' or 'multistep'")
    if len(dino["adam_betas"]) != 2 or any(
        not 0 <= value < 1 for value in dino["adam_betas"]
    ):
        raise ValueError("dino.adam_betas must contain two values in [0, 1)")
    if not dino["lr_drop_epochs"] or any(
        epoch < 1 for epoch in dino["lr_drop_epochs"]
    ):
        raise ValueError("dino.lr_drop_epochs must contain positive integers")
    for key in (
        "hidden_dim",
        "num_feature_levels",
        "enc_layers",
        "dec_layers",
        "nheads",
        "dim_feedforward",
        "enc_n_points",
        "dec_n_points",
        "num_queries",
        "num_select",
        "lr_drop_epoch",
    ):
        if dino[key] < 1:
            raise ValueError(f"dino.{key} must be positive")
    if dino["hidden_dim"] % dino["nheads"] or dino["hidden_dim"] % 32:
        raise ValueError(
            "dino.hidden_dim must be divisible by nheads and GroupNorm's 32 groups"
        )
    if dino["dn_number"] < 0:
        raise ValueError("dino.dn_number must be non-negative")
    if dino["num_select"] > dino["num_queries"]:
        raise ValueError("dino.num_select must not exceed dino.num_queries")
    if not 0 <= dino["dropout"] < 1:
        raise ValueError("dino.dropout must be in [0, 1)")
    if dino["clip_max_norm"] < 0:
        raise ValueError("dino.clip_max_norm must be non-negative")
    if dino["ema_epoch"] < 0:
        raise ValueError("dino.ema_epoch must be non-negative")
    if not 0 < dino["ema_decay"] < 1:
        raise ValueError("dino.ema_decay must be between 0 and 1")
    for key in ("dn_box_noise_scale", "dn_label_noise_ratio"):
        if not 0 <= dino[key] <= 1:
            raise ValueError(f"dino.{key} must be between 0 and 1")
    for key in (
        "set_cost_class",
        "set_cost_bbox",
        "set_cost_giou",
        "cls_loss_coef",
        "bbox_loss_coef",
        "giou_loss_coef",
    ):
        if dino[key] < 0:
            raise ValueError(f"dino.{key} must be non-negative")
    if not 0 <= dino["focal_alpha"] <= 1:
        raise ValueError("dino.focal_alpha must be between 0 and 1")
    if not 0 <= dino["score_threshold"] <= 1:
        raise ValueError("dino.score_threshold must be between 0 and 1")

    if settings["dit"]["pyramid_channels"] < 1:
        raise ValueError("dit.pyramid_channels must be positive")

    cascade = settings["cascade"]
    if len(cascade["anchor_sizes"]) != 4:
        raise ValueError(
            "cascade.anchor_sizes must contain one size for each of p2/p3/p4/p5"
        )
    if any(value < 1 for value in cascade["anchor_sizes"]):
        raise ValueError("cascade.anchor_sizes values must be positive")
    if not cascade["aspect_ratios"] or any(
        not value > 0 for value in cascade["aspect_ratios"]
    ):
        raise ValueError("cascade.aspect_ratios must contain positive values")
    for key in ("roi_batch_size_per_image", "rpn_batch_size_per_image"):
        if cascade[key] < 1:
            raise ValueError(f"cascade.{key} must be positive")
    if any(not 0 < step < 1 for step in cascade["lr_steps"]):
        raise ValueError("cascade.lr_steps values must be fractions between 0 and 1")
    if not 0 <= cascade["nms_threshold"] <= 1:
        raise ValueError("cascade.nms_threshold must be between 0 and 1")
    if not 0 <= cascade["score_threshold"] <= 1:
        raise ValueError("cascade.score_threshold must be between 0 and 1")

    tracking = settings["tracking"]
    if tracking["log_every_steps"] < 1:
        raise ValueError("tracking.log_every_steps must be a positive integer")
    if not tracking["experiment_name"]:
        raise ValueError("tracking.experiment_name must not be empty")


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
        section = "cascade" if self.detector == "cascade_rcnn" else self.detector
        return self.section(section)

    @property
    def tracking(self) -> Mapping[str, Any]:
        return self.section("tracking")

    def score_threshold(self) -> float:
        return float(self.detector_settings["score_threshold"])

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
        _validate_settings(self.settings)
