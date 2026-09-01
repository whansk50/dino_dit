"""Runtime device and process-group lifecycle helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator


def activate_device(device: str):
    """Select an explicit CUDA device and return its normalized torch device."""
    import torch

    selected = torch.device(device)
    if selected.type != "cuda":
        return selected
    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
    index = selected.index
    if index is None:
        index = torch.cuda.current_device()
    if not 0 <= index < torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA device {index} exceeds the {torch.cuda.device_count()} "
            "devices visible to PyTorch"
        )
    torch.cuda.set_device(index)
    return torch.device("cuda", index)


@contextmanager
def distributed_session(
    device: str,
    *,
    rank: int = 0,
    world_size: int = 1,
    init_method: str | None = None,
) -> Iterator[object]:
    """Select a device and own a PyTorch process group with explicit topology."""
    import torch
    import torch.distributed as dist

    if world_size < 1:
        raise ValueError("world_size must be positive")
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world_size {world_size}")

    requested = torch.device(device)
    if world_size > 1:
        if requested.type != "cuda":
            raise RuntimeError("Multi-process training requires a CUDA device")
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        if init_method is None:
            raise ValueError("Distributed training requires an explicit init_method")
        if dist.is_initialized():
            raise RuntimeError("A torch.distributed process group is already initialized")

    selected = activate_device(device)
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    try:
        yield selected
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
