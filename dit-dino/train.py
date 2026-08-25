"""Train a PubLayNet layout detector from a YAML configuration."""

from __future__ import annotations

import sys

from dit_layout_bench.backends import get_backend
from dit_layout_bench.arguments import parser_for, validated_config
from dit_layout_bench.config import per_process_batch_size
from dit_layout_bench.launcher import (
    LocalLaunch,
    parse_cuda_devices,
    validate_cuda_devices,
)
from dit_layout_bench.tracking import MLflowTracker, process_world_size


def _distributed_worker(argv: tuple[str, ...]) -> None:
    main(list(argv), _internal_worker=True)


def _launch_distributed(devices: tuple[int, ...], argv: list[str]) -> None:
    LocalLaunch.create(devices).spawn(_distributed_worker, tuple(argv))


def main(argv=None, *, _internal_worker: bool = False) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = parser_for("train")
    args = parser.parse_args(argv)
    devices = None
    if args.devices is not None and not _internal_worker:
        if process_world_size() > 1:
            parser.error("--devices cannot be combined with an external DDP launcher")
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
        require_data=not distributed,
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
        _launch_distributed(devices, argv)
        return
    with MLflowTracker(config, "train"):
        get_backend(config.detector).train(config)


if __name__ == "__main__":
    main()
