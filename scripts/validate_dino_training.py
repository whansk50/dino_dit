"""Validate production DINO runtime features on a temporary PubLayNet subset."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import yaml

if __package__:
    from .validation_runtime import (
        PROJECT_ROOT,
        execution_mode,
        expected_distributed_tag,
        project_environment,
    )
else:
    from validation_runtime import (
        PROJECT_ROOT,
        execution_mode,
        expected_distributed_tag,
        project_environment,
    )

from dit_layout_bench.checkpoint import safe_torch_load
from dit_layout_bench.config import load_settings
from dit_layout_bench.launcher import parse_cuda_devices, validate_cuda_devices
from dit_layout_bench.paths import RECENT_CHECKPOINT_NAME

if __package__:
    from .publaynet_subset import create_publaynet_subset, prepare_empty_work_dir
else:
    from publaynet_subset import create_publaynet_subset, prepare_empty_work_dir


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument(
        "--devices",
        required=True,
        help="one or more comma-separated CUDA device IDs (for example: 2 or 0,1)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "dino_train.yaml",
        help="base DINO config; runtime validation overrides are written separately",
    )
    parser.add_argument("--train-images", type=int, default=8)
    parser.add_argument("--val-images", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full-detector",
        action="store_true",
        help=(
            "retain all detector layers instead of the default fast "
            "one-layer detector"
        ),
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="artifact directory; defaults to a temporary directory removed on success",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="retain an automatically-created work directory after success",
    )
    return parser


def _validation_settings(
    args: argparse.Namespace,
    *,
    work_dir: Path,
    world_size: int,
    epochs: int,
) -> dict[str, Any]:
    settings = load_settings(args.config, detector="dino")
    settings["run"].update(
        device="cuda", seed=args.seed, num_workers=0, amp=True
    )
    settings["paths"].update(
        data_root=str(work_dir / "publaynet"),
        output_dir=str(work_dir / "output"),
        weights_dir=str(work_dir / "weights"),
        pretrained=str(args.pretrained.resolve()),
    )
    settings["training"].update(
        batch_size=world_size,
        epochs=epochs,
        warmup_iters=0,
        evaluate_every_epochs=1,
        persistent_workers=False,
    )
    settings["input"].update(
        short_edge_scales=[args.image_size], max_long_edge=args.image_size
    )
    settings["dit"].update(drop_path=0.0, use_checkpoint=True)
    settings["dino"].update(
        fused_optimizer=True,
        ddp_gradient_as_bucket_view=True,
        ddp_static_graph=True,
    )
    if not args.full_detector:
        settings["dino"].update(
            enc_layers=1,
            dec_layers=1,
            dim_feedforward=256,
            num_queries=10,
            num_select=10,
            dn_number=2,
        )
    settings["tracking"].update(
        enabled=True,
        tracking_uri=f"sqlite:///{work_dir / 'mlflow.db'}",
        experiment_name="dino-runtime-validation",
        run_name="",
        log_every_steps=1,
    )
    return settings


def _write_config(path: Path, settings: dict[str, Any]) -> None:
    path.write_text(
        yaml.safe_dump(settings, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _run_training(config: Path, devices: tuple[int, ...], *, resume: bool) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "train.py"),
        "--devices",
        ",".join(map(str, devices)),
        "--config",
        str(config),
    ]
    if resume:
        command.append("--resume")
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=project_environment(),
        check=True,
    )


def _verify_checkpoint(path: Path, *, expected_epoch: int) -> None:
    checkpoint = safe_torch_load(path)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Checkpoint is not a mapping: {path}")
    if int(checkpoint.get("epoch", -1)) != expected_epoch:
        raise RuntimeError(
            f"Expected checkpoint epoch {expected_epoch}, got {checkpoint.get('epoch')}"
        )
    required = {"model", "optimizer", "lr_scheduler", "scaler"}
    missing = required.difference(checkpoint)
    if missing:
        raise RuntimeError(f"Checkpoint is missing state: {sorted(missing)}")


def _verify_mlflow(work_dir: Path, *, world_size: int) -> None:
    from mlflow.tracking import MlflowClient

    tracking_uri = f"sqlite:///{work_dir / 'mlflow.db'}"
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("dino-runtime-validation")
    if experiment is None:
        raise RuntimeError("MLflow validation experiment was not created")
    runs = client.search_runs([experiment.experiment_id])
    if len(runs) != 1:
        raise RuntimeError(
            f"Expected one resumed rank-zero MLflow run, found {len(runs)}"
        )
    expected_tag = expected_distributed_tag(world_size)
    for run in runs:
        if run.info.status != "FINISHED":
            raise RuntimeError(f"MLflow run did not finish: {run.info.run_id}")
        actual_tag = run.data.tags.get("distributed")
        if actual_tag != expected_tag:
            raise RuntimeError(
                "MLflow distributed tag is incorrect: "
                f"expected={expected_tag}, actual={actual_tag}, "
                f"run={run.info.run_id}"
            )
        if run.data.params.get("runtime.world_size") != str(world_size):
            raise RuntimeError(f"MLflow world size is incorrect: {run.info.run_id}")
        if run.data.tags.get("resumed") != "true":
            raise RuntimeError(f"MLflow run was not resumed: {run.info.run_id}")
        required_metrics = {
            "train/loss",
            "train/loss_bbox",
            "train/loss_bbox_scaled",
            "train_epoch/loss",
            "eval/bbox_mAP",
        }
        missing = required_metrics.difference(run.data.metrics)
        if missing:
            raise RuntimeError(f"MLflow metrics are missing: {sorted(missing)}")
        for key in ("train/loss", "eval/bbox_mAP"):
            history = client.get_metric_history(run.info.run_id, key)
            steps = [metric.step for metric in history]
            if len(steps) < 2 or len(steps) != len(set(steps)):
                raise RuntimeError(
                    f"MLflow {key} history is incomplete or duplicated: "
                    f"{run.info.run_id}"
                )
        artifacts = client.list_artifacts(
            run.info.run_id, "resume-configs/resume-1"
        )
        if {Path(item.path).name for item in artifacts} != {
            "effective-config.yaml",
            "runtime.yaml",
        }:
            raise RuntimeError("MLflow resume config artifacts are incomplete")


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        devices = parse_cuda_devices(args.devices)
        validate_cuda_devices(devices)
        if args.image_size < 32 or args.image_size % 32:
            raise ValueError("--image-size must be a multiple of 32 and at least 32")
        if args.train_images < len(devices) * 2:
            raise ValueError("--train-images must provide at least two steps per rank")
        if args.val_images < 1:
            raise ValueError("--val-images must be positive")
        if not args.config.is_file():
            raise FileNotFoundError(f"Base config not found: {args.config}")
        if not args.pretrained.is_file():
            raise FileNotFoundError(f"DiT checkpoint not found: {args.pretrained}")
        work_dir, temporary = prepare_empty_work_dir(
            args.work_dir, prefix="dit-dino-runtime-validation-"
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    succeeded = False
    try:
        subset_root = work_dir / "publaynet"
        create_publaynet_subset(
            args.data_root,
            subset_root,
            split="train",
            image_count=args.train_images,
            seed=args.seed,
        )
        create_publaynet_subset(
            args.data_root,
            subset_root,
            split="val",
            image_count=args.val_images,
            seed=args.seed + 1,
        )

        fresh_config = work_dir / "fresh.yaml"
        _write_config(
            fresh_config,
            _validation_settings(
                args, work_dir=work_dir, world_size=len(devices), epochs=1
            ),
        )
        _run_training(fresh_config, devices, resume=False)
        checkpoint = work_dir / "weights" / RECENT_CHECKPOINT_NAME
        _verify_checkpoint(checkpoint, expected_epoch=0)

        resume_config = work_dir / "resume.yaml"
        _write_config(
            resume_config,
            _validation_settings(
                args, work_dir=work_dir, world_size=len(devices), epochs=2
            ),
        )
        _run_training(resume_config, devices, resume=True)
        _verify_checkpoint(checkpoint, expected_epoch=1)
        _verify_mlflow(work_dir, world_size=len(devices))
        succeeded = True
        features = ["AMP", "activation checkpointing"]
        if len(devices) > 1:
            features.append("static-graph DDP")
        features.extend(("fused optimizer", "resume", "rank-zero MLflow"))
        print(
            f"DINO runtime validation OK ({execution_mode(len(devices))}): "
            f"{', '.join(features)}"
        )
    finally:
        if temporary and succeeded and not args.keep_work_dir:
            shutil.rmtree(work_dir)
        else:
            print(f"Validation artifacts: {work_dir}")


if __name__ == "__main__":
    main()
