"""Local multi-GPU process launcher used by the training entrypoint."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import os
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


@contextmanager
def _temporary_environment(values: dict[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass(frozen=True)
class LocalLaunch:
    """Immutable rank-to-device topology for one local DDP job."""

    devices: tuple[int, ...]
    master_addr: str = "127.0.0.1"
    master_port: int = 0
    cuda_visible_devices: str | None = None

    @classmethod
    def create(cls, devices: tuple[int, ...]) -> "LocalLaunch":
        if len(devices) < 2:
            raise ValueError("A distributed launch requires at least two devices")
        return cls(
            devices=devices,
            master_port=_free_local_port(),
            cuda_visible_devices=_selected_cuda_visibility(devices),
        )

    @property
    def world_size(self) -> int:
        return len(self.devices)

    def environment(self, process_index: int) -> dict[str, str]:
        if not 0 <= process_index < self.world_size:
            raise ValueError(f"Invalid local process index: {process_index}")
        return {
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
            "RANK": str(process_index),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_WORLD_SIZE": str(self.world_size),
            # Restrict every child to the same selected devices. This keeps the
            # standard LOCAL_RANK contract used by PyTorch and Detectron2 even
            # when the caller selects non-contiguous physical GPUs.
            "LOCAL_RANK": str(process_index),
            "CUDA_VISIBLE_DEVICES": self.cuda_visible_devices
            or ",".join(map(str, self.devices)),
        }

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
    with _temporary_environment(launch.environment(process_index)):
        worker(*args)


def _selected_cuda_visibility(devices: tuple[int, ...]) -> str:
    """Map PyTorch-visible device indices back to the parent's visibility list."""
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is None:
        return ",".join(map(str, devices))
    visible = [item.strip() for item in inherited.split(",")]
    if any(device >= len(visible) for device in devices):
        raise RuntimeError("Selected CUDA device is absent from CUDA_VISIBLE_DEVICES")
    return ",".join(visible[device] for device in devices)
