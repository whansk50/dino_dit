"""Runtime device and process-group lifecycle helpers."""

from __future__ import annotations

from contextlib import contextmanager
import os
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
def managed_process_group() -> Iterator[None]:
    """Release a process group initialized inside a library-style runner."""
    try:
        yield
    finally:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _environment_integer(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be an integer, got {value!r}") from error
    if parsed < 0:
        raise RuntimeError(f"{name} must be non-negative, got {parsed}")
    return parsed


def local_process_world_size() -> int:
    """Return the launcher-provided process count on this machine."""
    return _environment_integer(
        "LOCAL_WORLD_SIZE", _environment_integer("WORLD_SIZE", 1)
    )


@contextmanager
def distributed_session(device: str):
    """Select this process's device and own an env-configured NCCL group."""
    import torch
    import torch.distributed as dist

    world_size = _environment_integer("WORLD_SIZE", 1)
    rank = _environment_integer("RANK", 0)
    local_rank = _environment_integer("LOCAL_RANK", 0)
    local_world_size = local_process_world_size()
    if world_size < 1:
        raise RuntimeError("WORLD_SIZE must be positive")
    if rank >= world_size:
        raise RuntimeError(f"RANK {rank} is outside WORLD_SIZE {world_size}")
    if local_world_size < 1 or world_size % local_world_size:
        raise RuntimeError(
            f"LOCAL_WORLD_SIZE {local_world_size} must divide WORLD_SIZE {world_size}"
        )
    if local_rank >= local_world_size:
        raise RuntimeError(
            f"LOCAL_RANK {local_rank} is outside LOCAL_WORLD_SIZE {local_world_size}"
        )

    requested = torch.device(device)
    if world_size > 1:
        if requested.type != "cuda":
            raise RuntimeError("Multi-process training requires a CUDA device")
        selected = activate_device(f"cuda:{local_rank}")
    else:
        selected = activate_device(device)

    created_group = False
    if world_size > 1:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable")
        if dist.is_initialized():
            if dist.get_world_size() != world_size or dist.get_rank() != rank:
                raise RuntimeError(
                    "Existing process group disagrees with DDP environment"
                )
        else:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=world_size,
                rank=rank,
            )
            created_group = True
    try:
        yield selected
    finally:
        if created_group and dist.is_initialized():
            dist.destroy_process_group()
