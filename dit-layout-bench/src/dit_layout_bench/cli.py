"""Unified command-line interface for both detector backends."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dit_layout_bench.backends import get_backend
from dit_layout_bench.config import DETECTORS, RunConfig


def _parser(action: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"dit-layout-{action}",
        description=f"{action.title()} a PubLayNet detector with a shared DiT backbone",
    )
    parser.add_argument("--detector", required=True, choices=DETECTORS)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--pretrained", type=Path, help="self-supervised DiT checkpoint")
    parser.add_argument("--resume", type=Path, help="detector checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--amp", action="store_true")
    if action == "inference":
        parser.add_argument("--image", type=Path, required=True)
        parser.add_argument("--score-threshold", type=float, default=0.5)
        parser.add_argument("--json-output", type=Path)
    return parser


def _config(args, *, training: bool) -> RunConfig:
    config = RunConfig(
        detector=args.detector,
        data_root=args.data_root,
        output_dir=args.output_dir,
        pretrained=args.pretrained,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        resume=args.resume,
        amp=args.amp,
    )
    config.validate(require_data=training)
    if training and config.pretrained is None:
        raise ValueError("Training requires --pretrained with the self-supervised DiT checkpoint")
    return config


def train_entrypoint(argv=None) -> None:
    args = _parser("train").parse_args(argv)
    get_backend(args.detector).train(_config(args, training=True))


def evaluate_entrypoint(argv=None) -> None:
    args = _parser("evaluate").parse_args(argv)
    config = _config(args, training=True)
    if config.resume is None:
        raise ValueError("Evaluation requires --resume")
    get_backend(args.detector).evaluate(config)


def inference_entrypoint(argv=None) -> None:
    args = _parser("inference").parse_args(argv)
    config = _config(args, training=False)
    if config.resume is None:
        raise ValueError("Inference requires --resume")
    if not args.image.is_file():
        raise FileNotFoundError(f"Input image not found: {args.image}")
    predictions = get_backend(args.detector).predict(
        config, args.image, score_threshold=args.score_threshold
    )
    payload = json.dumps(predictions, indent=2, ensure_ascii=False)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)

