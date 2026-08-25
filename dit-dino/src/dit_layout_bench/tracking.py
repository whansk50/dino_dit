"""MLflow tracking shared by both detector backends."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import math
import os
from typing import Any, Mapping

from dit_layout_bench.config import RunConfig


_ACTIVE: ContextVar["MLflowTracker | None"] = ContextVar("mlflow_tracker", default=None)


def process_rank() -> int:
    """Return the launcher-provided global rank before torch.distributed starts."""
    for name in ("RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value not in (None, ""):
            try:
                return int(value)
            except ValueError as error:
                raise RuntimeError(f"{name} must be an integer, got {value!r}") from error
    return 0


def process_world_size() -> int:
    """Return the launcher-provided world size, defaulting to one process."""
    for name in ("WORLD_SIZE", "SLURM_NTASKS", "SLURM_NPROCS"):
        value = os.environ.get(name)
        if value not in (None, ""):
            try:
                world_size = int(value)
            except ValueError as error:
                raise RuntimeError(f"{name} must be an integer, got {value!r}") from error
            if world_size < 1:
                raise RuntimeError(f"{name} must be positive, got {world_size}")
            return world_size
    return 1


def _flatten(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            flattened.update(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            flattened[name] = ",".join(map(str, value))
        else:
            flattened[name] = value
    return flattened


@dataclass
class MLflowTracker:
    config: RunConfig
    action: str
    _mlflow: Any = None
    _token: Any = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.tracking["enabled"])

    def __enter__(self) -> "MLflowTracker":
        self._token = _ACTIVE.set(self)
        # The tracker is created before DINO initializes torch.distributed, so
        # use launcher/Slurm environment variables to keep MLflow rank-zero-only.
        if not self.enabled or process_rank() != 0:
            return self
        try:
            import mlflow
        except ImportError as error:
            raise RuntimeError(
                "MLflow tracking is enabled; install dependencies from requirements.txt"
            ) from error
        self._mlflow = mlflow
        tracking_uri = self.config.tracking["tracking_uri"]
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config.tracking["experiment_name"])
        run_name = self.config.tracking["run_name"] or None
        mlflow.start_run(
            run_name=run_name,
            tags={
                "detector": self.config.detector,
                "action": self.action,
                "distributed": str(process_world_size() > 1).lower(),
            },
        )
        mlflow.log_params(_flatten(self.config.settings))
        world_size = process_world_size()
        runtime_params = {"runtime.world_size": world_size}
        if self.action == "train":
            from dit_layout_bench.config import per_process_batch_size

            runtime_params.update(
                {
                    "runtime.batch_size_per_gpu": per_process_batch_size(
                        self.config.batch_size, world_size
                    ),
                    "runtime.global_batch_size": self.config.batch_size,
                }
            )
        mlflow.log_params(runtime_params)
        mlflow.log_dict(self.config.settings, "effective-config.yaml")
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            if self._mlflow is not None:
                status = "FAILED" if exc_type is not None else "FINISHED"
                self._mlflow.end_run(status=status)
        finally:
            if self._token is not None:
                _ACTIVE.reset(self._token)

    def log_metrics(self, metrics: Mapping[str, Any], step: int | None = None) -> None:
        if self._mlflow is None:
            return
        clean = {}
        for name, value in metrics.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                clean[name] = number
        if clean:
            self._mlflow.log_metrics(clean, step=step)


def active_tracker() -> MLflowTracker | None:
    return _ACTIVE.get()
