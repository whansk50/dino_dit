"""Unified command-line interface for both detector backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dit_layout_bench.backends import get_backend
from dit_layout_bench.config import DEFAULT_CONFIG, DETECTORS, RunConfig, load_settings


def _parser(action: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"dit-layout-{action}",
        description=f"{action.title()} a PubLayNet detector with a shared DiT backbone",
    )
    parser.add_argument("--detector", choices=DETECTORS, help="overrides run.detector")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--options",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="TOML-style overrides, e.g. dino.num_queries=500 training.epochs=24",
    )
    parser.add_argument("--data-root", type=Path, help="overrides paths.data_root")
    parser.add_argument("--output-dir", type=Path, help="overrides paths.output_dir")
    parser.add_argument("--pretrained", type=Path, help="self-supervised DiT checkpoint")
    parser.add_argument("--resume", type=Path, help="detector checkpoint")
    parser.add_argument("--device", help="overrides run.device")
    parser.add_argument("--seed", type=int, help="overrides run.seed")
    parser.add_argument("--num-workers", type=int, help="overrides run.num_workers")
    parser.add_argument("--batch-size", type=int, help="overrides training.batch_size")
    parser.add_argument("--epochs", type=int, help="overrides training.epochs")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    if action == "inference":
        parser.add_argument("--image", type=Path, required=True)
        parser.add_argument("--score-threshold", type=float)
        parser.add_argument("--json-output", type=Path)
    return parser


def _apply_cli_overrides(settings, args) -> None:
    """Apply dedicated CLI flags last and keep the effective settings coherent."""
    mappings = (
        ("run", "detector", args.detector),
        ("run", "device", args.device),
        ("run", "seed", args.seed),
        ("run", "num_workers", args.num_workers),
        ("run", "amp", args.amp),
        ("paths", "data_root", args.data_root),
        ("paths", "output_dir", args.output_dir),
        ("paths", "pretrained", args.pretrained),
        ("paths", "resume", args.resume),
        ("training", "batch_size", args.batch_size),
        ("training", "epochs", args.epochs),
    )
    for section, key, value in mappings:
        if value is not None:
            settings[section][key] = str(value) if isinstance(value, Path) else value


def _config(args, *, require_data: bool) -> RunConfig:
    settings = load_settings(args.config, args.options)
    _apply_cli_overrides(settings, args)
    run = settings["run"]
    paths = settings["paths"]
    training_settings = settings["training"]
    if require_data and not paths["data_root"]:
        raise ValueError("Set PubLayNet path with --data-root or paths.data_root in the config")
    config = RunConfig(
        detector=run["detector"],
        data_root=Path(paths["data_root"]),
        output_dir=Path(paths["output_dir"]),
        pretrained=Path(paths["pretrained"]) if paths["pretrained"] else None,
        device=run["device"],
        seed=run["seed"],
        num_workers=run["num_workers"],
        batch_size=training_settings["batch_size"],
        epochs=training_settings["epochs"],
        resume=Path(paths["resume"]) if paths["resume"] else None,
        amp=run["amp"],
        settings=settings,
    )
    config.validate(require_data=require_data)
    return config


def _validated_config(
    parser: argparse.ArgumentParser,
    args,
    *,
    require_data: bool,
    require_pretrained: bool = False,
    require_resume: bool = False,
) -> RunConfig:
    try:
        config = _config(args, require_data=require_data)
        if require_pretrained and config.pretrained is None:
            raise ValueError(
                "Training requires --pretrained or paths.pretrained in the config"
            )
        if require_resume and config.resume is None:
            raise ValueError("This command requires --resume or paths.resume in the config")
        return config
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))


def train_entrypoint(argv=None) -> None:
    parser = _parser("train")
    args = parser.parse_args(argv)
    config = _validated_config(
        parser, args, require_data=True, require_pretrained=True
    )
    get_backend(config.detector).train(config)


def evaluate_entrypoint(argv=None) -> None:
    parser = _parser("evaluate")
    args = parser.parse_args(argv)
    config = _validated_config(parser, args, require_data=True, require_resume=True)
    get_backend(config.detector).evaluate(config)


def inference_entrypoint(argv=None) -> None:
    parser = _parser("inference")
    args = parser.parse_args(argv)
    config = _validated_config(parser, args, require_data=False, require_resume=True)
    if not args.image.is_file():
        parser.error(f"Input image not found: {args.image}")
    predictions = get_backend(config.detector).predict(
        config,
        args.image,
        score_threshold=(
            args.score_threshold
            if args.score_threshold is not None
            else config.score_threshold()
        ),
    )
    payload = json.dumps(predictions, indent=2, ensure_ascii=False)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
