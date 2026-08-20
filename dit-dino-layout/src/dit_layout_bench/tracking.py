"""MLflow tracking shared by both detector backends."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Any, Mapping


_ACTIVE: ContextVar["MLflowTracker | None"] = ContextVar("mlflow_tracker", default=None)


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
    config: Any
    action: str
    _mlflow: Any = None
    _token: Any = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.tracking["enabled"])

    def __enter__(self) -> "MLflowTracker":
        self._token = _ACTIVE.set(self)
        if not self.enabled:
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
            tags={"detector": self.config.detector, "action": self.action},
        )
        mlflow.log_params(_flatten(self.config.settings))
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
