"""Train a PubLayNet layout detector from a YAML configuration."""

from __future__ import annotations

from dataclasses import replace
import sys

from dit_layout_bench.backends import get_backend
from dit_layout_bench.arguments import parser_for, validated_config
from dit_layout_bench.config import RunConfig, per_process_batch_size
from dit_layout_bench.launcher import (
    LocalLaunch,
    parse_cuda_devices,
    validate_cuda_devices,
)
from dit_layout_bench.runtime import distributed_session
from dit_layout_bench.tracking import MLflowTracker


def _train(config: RunConfig, *, rank: int, world_size: int, init_method=None) -> None:
    with distributed_session(
        config.device,
        rank=rank,
        world_size=world_size,
        init_method=init_method,
    ):
        with MLflowTracker(config, "train"):
            get_backend(config.detector).train(config)


def _distributed_worker(
    rank: int, launch: LocalLaunch, config: RunConfig
) -> None:
    worker_config = replace(config, device=f"cuda:{launch.devices[rank]}")
    _train(
        worker_config,
        rank=rank,
        world_size=launch.world_size,
        init_method=launch.init_method,
    )


def _launch_distributed(devices: tuple[int, ...], config: RunConfig) -> None:
    LocalLaunch.create(devices).spawn(_distributed_worker, config)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = parser_for("train")
    args = parser.parse_args(argv)
    devices = None
    if args.devices is not None:
        try:
            devices = parse_cuda_devices(args.devices)
        except ValueError as error:
            parser.error(str(error))
    distributed = devices is not None and len(devices) > 1
    if devices is not None:
        try:
            validate_cuda_devices(devices)
        except RuntimeError as error:
            parser.error(str(error))
    if devices is not None and len(devices) == 1:
        args.device = f"cuda:{devices[0]}"
    config = validated_config(
        parser,
        args,
        require_data=True,
        training=True,
        require_pretrained=True,
    )
    if distributed:
        assert devices is not None
        if config.device != "cuda":
            parser.error("multi-GPU training requires run.device=cuda")
        try:
            per_process_batch_size(config.batch_size, len(devices))
        except ValueError as error:
            parser.error(str(error))
        _launch_distributed(devices, config)
        return
    _train(config, rank=0, world_size=1)


if __name__ == "__main__":
    main()
