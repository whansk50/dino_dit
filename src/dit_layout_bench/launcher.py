"""Local multi-GPU process launcher used by the training entrypoint."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import socket
from typing import Any


def parse_cuda_devices(value: str) -> tuple[int, ...]:
    """Parse an ordered, duplicate-free list of visible CUDA device IDs."""
    try:
        devices = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("--devices must contain comma-separated integers") from error
    if not devices or any(device < 0 for device in devices):
        raise ValueError("--devices must contain non-negative CUDA device IDs")
    if len(set(devices)) != len(devices):
        raise ValueError("--devices must not contain duplicate CUDA device IDs")
    return devices


def validate_cuda_devices(devices: tuple[int, ...]) -> None:
    """Fail before spawning when a requested CUDA device is unavailable."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("--devices requires CUDA")
    available = torch.cuda.device_count()
    if any(device >= available for device in devices):
        raise RuntimeError(
            f"--devices {','.join(map(str, devices))} exceeds the "
            f"{available} CUDA devices visible to PyTorch"
        )


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@dataclass(frozen=True)
class LocalLaunch:
    """Immutable PyTorch rank-to-device topology for one local DDP job."""

    devices: tuple[int, ...]
    master_addr: str = "127.0.0.1"
    master_port: int = 0

    @classmethod
    def create(cls, devices: tuple[int, ...]) -> "LocalLaunch":
        if len(devices) < 2:
            raise ValueError("A distributed launch requires at least two devices")
        return cls(
            devices=devices,
            master_port=_free_local_port(),
        )

    @property
    def world_size(self) -> int:
        return len(self.devices)

    @property
    def init_method(self) -> str:
        return f"tcp://{self.master_addr}:{self.master_port}"

    def spawn(self, worker: Callable[..., None], *args: Any) -> None:
        """Run a picklable worker once per selected device and wait for all."""
        import torch.multiprocessing as multiprocessing

        multiprocessing.spawn(
            _run_process,
            args=(self, worker, args),
            nprocs=self.world_size,
            join=True,
        )


def _run_process(
    process_index: int,
    launch: LocalLaunch,
    worker: Callable[..., None],
    args: tuple[Any, ...],
) -> None:
    if not 0 <= process_index < launch.world_size:
        raise ValueError(f"Invalid local process index: {process_index}")
    worker(process_index, launch, *args)
