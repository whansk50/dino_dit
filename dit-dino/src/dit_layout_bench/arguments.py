"""Argument parsing and validated runtime configuration for root scripts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from dit_layout_bench.config import (
    DETECTORS,
    RunConfig,
    load_settings,
    per_process_batch_size,
)
from dit_layout_bench.paths import RECENT_CHECKPOINT_NAME
from dit_layout_bench.tracking import process_world_size


def probability(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return number


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--detector",
        choices=DETECTORS,
        help="selects detector defaults and must match config run.detector",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="detector-specific YAML; defaults to the selected detector's defaults",
    )
    parser.add_argument(
        "--options",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="YAML-style overrides, e.g. dino.num_queries=500 training.epochs=24",
    )
    parser.add_argument("--output-dir", type=Path, help="overrides paths.output_dir")
    parser.add_argument("--weights-dir", type=Path, help="overrides paths.weights_dir")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="use WEIGHTS_DIR/recent.pth exclusively",
    )
    parser.add_argument("--device", help="overrides run.device")


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, help="overrides paths.data_root")
    parser.add_argument("--pretrained", type=Path, help="self-supervised DiT checkpoint")
    parser.add_argument("--seed", type=int, help="overrides run.seed")
    parser.add_argument("--num-workers", type=int, help="overrides run.num_workers")
    parser.add_argument(
        "--batch-size", type=int, help="overrides global training.batch_size"
    )
    parser.add_argument("--epochs", type=int, help="overrides training.epochs")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--devices",
        help=(
            "comma-separated CUDA IDs visible to PyTorch; one ID selects a "
            "single GPU and multiple IDs launch DDP internally"
        ),
    )


def _add_inference_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="an image file or a directory containing images",
    )
    parser.add_argument("--score-threshold", type=probability)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        help="defaults to OUTPUT_DIR/inference",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save images with predicted boxes (default: enabled)",
    )


def parser_for(action: str) -> argparse.ArgumentParser:
    if action not in {"train", "inference"}:
        raise ValueError(f"Unsupported action: {action}")
    parser = argparse.ArgumentParser(
        description=f"{action.title()} a PubLayNet detector with a shared DiT backbone"
    )
    _add_common_arguments(parser)
    if action == "train":
        _add_training_arguments(parser)
    else:
        _add_inference_arguments(parser)
    return parser


def _apply_cli_overrides(
    settings: dict[str, Any], args: argparse.Namespace
) -> None:
    """Apply flags exposed by the selected root script after YAML overrides."""
    mappings = (
        ("run", "detector", getattr(args, "detector", None)),
        ("run", "device", getattr(args, "device", None)),
        ("run", "seed", getattr(args, "seed", None)),
        ("run", "num_workers", getattr(args, "num_workers", None)),
        ("run", "amp", getattr(args, "amp", None)),
        ("paths", "data_root", getattr(args, "data_root", None)),
        ("paths", "output_dir", getattr(args, "output_dir", None)),
        ("paths", "weights_dir", getattr(args, "weights_dir", None)),
        ("paths", "pretrained", getattr(args, "pretrained", None)),
        ("training", "batch_size", getattr(args, "batch_size", None)),
        ("training", "epochs", getattr(args, "epochs", None)),
    )
    for section, key, value in mappings:
        if value is not None:
            settings[section][key] = str(value) if isinstance(value, Path) else value


def _runtime_config(
    settings: dict[str, Any], args: argparse.Namespace
) -> RunConfig:
    run = settings["run"]
    paths = settings["paths"]
    training = settings["training"]
    weights_dir = Path(paths["weights_dir"])
    resume = weights_dir / RECENT_CHECKPOINT_NAME if args.resume else None
    return RunConfig(
        detector=run["detector"],
        data_root=Path(paths["data_root"]),
        output_dir=Path(paths["output_dir"]),
        pretrained=Path(paths["pretrained"]) if paths["pretrained"] else None,
        weights_dir=weights_dir,
        device=run["device"],
        seed=run["seed"],
        num_workers=run["num_workers"],
        batch_size=training["batch_size"],
        epochs=training["epochs"],
        resume=resume,
        amp=run["amp"],
        settings=settings,
    )


def build_config(args: argparse.Namespace, *, require_data: bool) -> RunConfig:
    settings = load_settings(args.config, args.options, detector=args.detector)
    _apply_cli_overrides(settings, args)
    paths = settings["paths"]
    if require_data and not paths["data_root"]:
        raise ValueError("Set PubLayNet path with --data-root or paths.data_root")
    if not paths["weights_dir"]:
        raise ValueError(
            "Set a checkpoint directory with --weights-dir or paths.weights_dir"
        )
    config = _runtime_config(settings, args)
    config.validate(require_data=require_data)
    return config


def validated_config(
    parser: argparse.ArgumentParser,
    args,
    *,
    require_data: bool,
    training: bool = False,
    require_pretrained: bool = False,
    require_resume: bool = False,
) -> RunConfig:
    try:
        config = build_config(args, require_data=require_data)
        if require_pretrained and config.pretrained is None and config.resume is None:
            raise ValueError(
                "New training requires --pretrained or paths.pretrained in the config"
            )
        if require_resume and config.resume is None:
            raise ValueError("This command requires --resume")
        if training:
            world_size = process_world_size()
            per_process_batch_size(config.batch_size, world_size)
        return config
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
