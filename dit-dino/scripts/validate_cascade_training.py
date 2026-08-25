"""Validate Cascade R-CNN training on a temporary PubLayNet subset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping

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

EXPERIMENT_NAME = "cascade-runtime-validation"


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
        default=PROJECT_ROOT / "configs" / "cascade_rcnn_train.yaml",
        help="base Cascade config; runtime validation overrides are written separately",
    )
    parser.add_argument("--train-images", type=int, default=8)
    parser.add_argument("--val-images", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
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
    settings = load_settings(args.config, detector="cascade_rcnn")
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
    )
    settings["input"].update(
        short_edge_scales=[args.image_size], max_long_edge=args.image_size
    )
    settings["dit"].update(drop_path=0.0, use_checkpoint=True)
    settings["cascade_rcnn"].update(
        roi_batch_size_per_image=32,
        rpn_batch_size_per_image=32,
    )
    settings["tracking"].update(
        enabled=True,
        tracking_uri=f"sqlite:///{work_dir / 'mlflow.db'}",
        experiment_name=EXPERIMENT_NAME,
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


def _nested_mapping(value: object, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Checkpoint {key} state is not a mapping")
    return value


def _verify_checkpoint(path: Path, *, expected_iteration: int) -> None:
    checkpoint = _nested_mapping(safe_torch_load(path), "root")
    _nested_mapping(checkpoint.get("model"), "model")
    trainer = _nested_mapping(checkpoint.get("trainer"), "trainer")
    if int(trainer.get("iteration", -1)) != expected_iteration:
        raise RuntimeError(
            f"Expected checkpoint iteration {expected_iteration}, "
            f"got {trainer.get('iteration')}"
        )
    inner_trainer = _nested_mapping(trainer.get("_trainer"), "trainer._trainer")
    _nested_mapping(inner_trainer.get("optimizer"), "optimizer")
    _nested_mapping(inner_trainer.get("grad_scaler"), "AMP grad_scaler")
    _nested_mapping(trainer.get("hooks"), "scheduler hooks")


def _verify_mlflow(work_dir: Path, *, world_size: int) -> None:
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=f"sqlite:///{work_dir / 'mlflow.db'}")
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise RuntimeError("MLflow validation experiment was not created")
    runs = client.search_runs([experiment.experiment_id])
    if len(runs) != 2:
        raise RuntimeError(
            f"Expected exactly two rank-zero MLflow runs, found {len(runs)}"
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
        try:
            import detectron2  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "Cascade validation requires a PyTorch-compatible "
                "Detectron2 installation"
            ) from error
        work_dir, temporary = prepare_empty_work_dir(
            args.work_dir, prefix="dit-cascade-runtime-validation-"
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

        iterations_per_epoch = math.ceil(args.train_images / len(devices))
        fresh_config = work_dir / "fresh.yaml"
        _write_config(
            fresh_config,
            _validation_settings(
                args, work_dir=work_dir, world_size=len(devices), epochs=1
            ),
        )
        _run_training(fresh_config, devices, resume=False)
        checkpoint = work_dir / "weights" / RECENT_CHECKPOINT_NAME
        _verify_checkpoint(
            checkpoint, expected_iteration=iterations_per_epoch - 1
        )

        resume_config = work_dir / "resume.yaml"
        _write_config(
            resume_config,
            _validation_settings(
                args, work_dir=work_dir, world_size=len(devices), epochs=2
            ),
        )
        _run_training(resume_config, devices, resume=True)
        _verify_checkpoint(
            checkpoint, expected_iteration=iterations_per_epoch * 2 - 1
        )
        _verify_mlflow(work_dir, world_size=len(devices))
        succeeded = True
        print(
            "Cascade runtime validation OK "
            f"({execution_mode(len(devices))}): AMP, activation checkpointing, "
            "optimizer/scheduler state, resume, and rank-zero MLflow"
        )
    finally:
        if temporary and succeeded and not args.keep_work_dir:
            shutil.rmtree(work_dir)
        else:
            print(f"Validation artifacts: {work_dir}")


if __name__ == "__main__":
    main()
