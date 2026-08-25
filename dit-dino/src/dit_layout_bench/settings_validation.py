"""Validation rules for detector-specific YAML settings."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _validate_dino(settings: Mapping[str, Any]) -> None:
    if settings["optimizer"] not in {"adam", "adamw"}:
        raise ValueError("dino.optimizer must be 'adam' or 'adamw'")
    if settings["scheduler"] not in {"step", "multistep"}:
        raise ValueError("dino.scheduler must be 'step' or 'multistep'")
    if len(settings["adam_betas"]) != 2 or any(
        not 0 <= value < 1 for value in settings["adam_betas"]
    ):
        raise ValueError("dino.adam_betas must contain two values in [0, 1)")
    if not settings["lr_drop_epochs"] or any(
        epoch < 1 for epoch in settings["lr_drop_epochs"]
    ):
        raise ValueError("dino.lr_drop_epochs must contain positive integers")

    positive_keys = (
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
    )
    for key in positive_keys:
        if settings[key] < 1:
            raise ValueError(f"dino.{key} must be positive")
    if settings["hidden_dim"] % settings["nheads"] or settings["hidden_dim"] % 32:
        raise ValueError(
            "dino.hidden_dim must be divisible by nheads and GroupNorm's 32 groups"
        )
    if settings["dn_number"] < 0:
        raise ValueError("dino.dn_number must be non-negative")
    if settings["num_select"] > settings["num_queries"]:
        raise ValueError("dino.num_select must not exceed dino.num_queries")
    if not 0 <= settings["dropout"] < 1:
        raise ValueError("dino.dropout must be in [0, 1)")
    if settings["clip_max_norm"] < 0:
        raise ValueError("dino.clip_max_norm must be non-negative")
    if settings["ema_epoch"] < 0:
        raise ValueError("dino.ema_epoch must be non-negative")
    if not 0 < settings["ema_decay"] < 1:
        raise ValueError("dino.ema_decay must be between 0 and 1")
    for key in ("dn_box_noise_scale", "dn_label_noise_ratio"):
        if not 0 <= settings[key] <= 1:
            raise ValueError(f"dino.{key} must be between 0 and 1")
    for key in (
        "set_cost_class",
        "set_cost_bbox",
        "set_cost_giou",
        "cls_loss_coef",
        "bbox_loss_coef",
        "giou_loss_coef",
    ):
        if settings[key] < 0:
            raise ValueError(f"dino.{key} must be non-negative")
    if not 0 <= settings["focal_alpha"] <= 1:
        raise ValueError("dino.focal_alpha must be between 0 and 1")
    if not 0 <= settings["score_threshold"] <= 1:
        raise ValueError("dino.score_threshold must be between 0 and 1")


def _validate_cascade_rcnn(settings: Mapping[str, Any]) -> None:
    if len(settings["anchor_sizes"]) != 4:
        raise ValueError(
            "cascade_rcnn.anchor_sizes must contain one size for each of p2/p3/p4/p5"
        )
    if any(value < 1 for value in settings["anchor_sizes"]):
        raise ValueError("cascade_rcnn.anchor_sizes values must be positive")
    if not settings["aspect_ratios"] or any(
        value <= 0 for value in settings["aspect_ratios"]
    ):
        raise ValueError("cascade_rcnn.aspect_ratios must contain positive values")
    for key in ("roi_batch_size_per_image", "rpn_batch_size_per_image"):
        if settings[key] < 1:
            raise ValueError(f"cascade_rcnn.{key} must be positive")
    if any(not 0 < step < 1 for step in settings["lr_steps"]):
        raise ValueError(
            "cascade_rcnn.lr_steps values must be fractions between 0 and 1"
        )
    for key in ("nms_threshold", "score_threshold"):
        if not 0 <= settings[key] <= 1:
            raise ValueError(f"cascade_rcnn.{key} must be between 0 and 1")


def _validate_common(settings: Mapping[str, Any]) -> None:
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
    if (
        not isinstance(training["evaluate_every_epochs"], int)
        or training["evaluate_every_epochs"] < 1
    ):
        raise ValueError("training.evaluate_every_epochs must be a positive integer")
    if training["warmup_iters"] < 0:
        raise ValueError("training.warmup_iters must be non-negative")
    for key in ("detector_lr", "backbone_lr", "weight_decay", "warmup_factor"):
        if training[key] < 0:
            raise ValueError(f"training.{key} must be non-negative")
    if not 0 < training["warmup_factor"] <= 1:
        raise ValueError("training.warmup_factor must be in (0, 1]")

    if settings["dit"]["pyramid_channels"] < 1:
        raise ValueError("dit.pyramid_channels must be positive")

    tracking = settings["tracking"]
    if tracking["log_every_steps"] < 1:
        raise ValueError("tracking.log_every_steps must be a positive integer")
    if not tracking["experiment_name"]:
        raise ValueError("tracking.experiment_name must not be empty")


def validate_settings(settings: Mapping[str, Any]) -> None:
    """Validate shared values, then only the selected detector's section."""
    _validate_common(settings)
    detector = settings["run"]["detector"]
    if detector == "dino":
        if settings["training"]["prefetch_factor"] < 1:
            raise ValueError("training.prefetch_factor must be a positive integer")
        _validate_dino(settings["dino"])
    elif detector == "cascade_rcnn":
        _validate_cascade_rcnn(settings["cascade_rcnn"])
    else:
        raise ValueError(f"Unsupported run.detector: {detector!r}")
