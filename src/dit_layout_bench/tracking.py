"""MLflow tracking shared by both detector backends."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import math
from typing import Any, Mapping

from dit_layout_bench.config import RunConfig
from dit_layout_bench.paths import MLFLOW_RUN_ID_NAME


_ACTIVE: ContextVar["MLflowTracker | None"] = ContextVar("mlflow_tracker", default=None)


def process_rank() -> int:
    """Return the active PyTorch process-group rank."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def process_world_size() -> int:
    """Return the active PyTorch process-group size."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
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
        experiment = mlflow.set_experiment(
            self.config.tracking["experiment_name"]
        )
        run_name = self.config.tracking["run_name"] or None
        run_id_path = None
        resume_run_id = None
        resume_source_id = None
        resume_source_status = None
        resume_index = 0
        weights_dir = getattr(self.config, "weights_dir", None)
        is_resume = self.action == "train" and bool(
            getattr(self.config, "resume", None)
        )
        if self.action == "train" and weights_dir is not None:
            run_id_path = weights_dir / MLFLOW_RUN_ID_NAME
            if is_resume and run_id_path.is_file():
                resume_source_id = (
                    run_id_path.read_text(encoding="utf-8").strip() or None
                )
        if resume_source_id:
            try:
                source_run = mlflow.get_run(resume_source_id)
            except mlflow.exceptions.MlflowException:
                # The checkpoint may have been copied without its original
                # tracking database. Start a linked local attempt instead of
                # failing with an opaque missing-run error.
                resume_source_status = "MISSING"
            else:
                resume_source_status = source_run.info.status
                compatible = (
                    source_run.info.experiment_id == experiment.experiment_id
                    and source_run.data.tags.get("detector") == self.config.detector
                    and source_run.data.tags.get("action") == self.action
                )
                if resume_source_status == "FINISHED" and compatible:
                    resume_run_id = resume_source_id
                    resume_index = int(
                        source_run.data.tags.get("resume_count", "0")
                    ) + 1
                elif not compatible:
                    resume_source_status = f"{resume_source_status}_INCOMPATIBLE"
        tags = {
            "detector": self.config.detector,
            "action": self.action,
            "distributed": str(process_world_size() > 1).lower(),
        }
        if resume_run_id:
            mlflow.start_run(run_id=resume_run_id)
            mlflow.set_tags(
                tags
                | {
                    "resumed": "true",
                    "resume_count": str(resume_index),
                }
            )
        else:
            if is_resume:
                tags["resumed"] = "true"
                tags["resume_count"] = "1"
            if resume_source_id:
                tags["resume_of"] = resume_source_id
                tags["resume_source_status"] = resume_source_status or "UNKNOWN"
            run = mlflow.start_run(run_name=run_name, tags=tags)
            if run_id_path is not None:
                run_id_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_id_path = run_id_path.with_suffix(
                    f"{run_id_path.suffix}.tmp"
                )
                temporary_id_path.write_text(run.info.run_id, encoding="utf-8")
                temporary_id_path.replace(run_id_path)
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
        if not resume_run_id:
            mlflow.log_params(runtime_params)
            mlflow.log_dict(self.config.settings, "effective-config.yaml")
        else:
            artifact_root = f"resume-configs/resume-{resume_index}"
            mlflow.log_dict(
                self.config.settings,
                f"{artifact_root}/effective-config.yaml",
            )
            mlflow.log_dict(runtime_params, f"{artifact_root}/runtime.yaml")
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
