"""Train a PubLayNet layout detector from a YAML configuration."""

from __future__ import annotations

import os
import socket
import sys

from dit_layout_bench.backends import get_backend
from dit_layout_bench.arguments import parser_for, validated_config
from dit_layout_bench.config import per_process_batch_size
from dit_layout_bench.tracking import MLflowTracker, process_world_size


def _parse_devices(value: str) -> tuple[int, ...]:
    try:
        devices = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--devices must contain comma-separated integers") from error
    if not devices or any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative CUDA device IDs")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicate CUDA device IDs")
    return devices


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _validate_cuda_devices(devices: tuple[int, ...]) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("--devices requires CUDA")
    available = torch.cuda.device_count()
    if any(device >= available for device in devices):
        raise RuntimeError(
            f"--devices {','.join(map(str, devices))} exceeds the "
            f"{available} CUDA devices visible to PyTorch"
        )


def _distributed_worker(
    process_index: int,
    world_size: int,
    devices: tuple[int, ...],
    argv: tuple[str, ...],
    master_addr: str,
    master_port: int,
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "RANK": str(process_index),
            "WORLD_SIZE": str(world_size),
            "LOCAL_RANK": str(devices[process_index]),
        }
    )
    main(list(argv), _internal_worker=True)


def _launch_distributed(devices: tuple[int, ...], argv: list[str]) -> None:
    import torch.multiprocessing as multiprocessing

    multiprocessing.spawn(
        _distributed_worker,
        args=(
            len(devices),
            devices,
            tuple(argv),
            "127.0.0.1",
            _free_local_port(),
        ),
        nprocs=len(devices),
        join=True,
    )


def main(argv=None, *, _internal_worker: bool = False) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = parser_for("train")
    args = parser.parse_args(argv)
    devices = None
    if args.devices is not None and not _internal_worker:
        if process_world_size() > 1:
            parser.error("--devices cannot be combined with an external DDP launcher")
        try:
            devices = _parse_devices(args.devices)
        except ValueError as error:
            parser.error(str(error))
    distributed = devices is not None and len(devices) > 1
    if devices is not None:
        try:
            _validate_cuda_devices(devices)
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
        if config.detector != "dino":
            parser.error("multi-GPU training is currently supported only for DINO")
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
