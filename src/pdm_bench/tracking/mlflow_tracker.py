from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


def _coerce_param_value(value: object) -> str | float | int | bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float | str):
        return value
    return str(value)


class MlflowTracker:
    """Minimal MLflow-backend tracker implementation."""

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        experiment_name: str = "pdm-bench",
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        mlflow = importlib.import_module("mlflow")
        mlflow.set_tracking_uri(tracking_uri or _default_tracking_uri())
        mlflow.set_experiment(experiment_name)
        mlflow.start_run(run_name=run_name, tags=tags)
        self._mlflow = mlflow
        self._active = True

    def log_params(self, params: dict[str, object]) -> None:
        if not params:
            return
        payload = {str(k): _coerce_param_value(v) for k, v in params.items()}
        self._mlflow.log_params(payload)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not metrics:
            return
        payload = {str(k): float(v) for k, v in metrics.items()}
        self._mlflow.log_metrics(payload, step=step)

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def close(self) -> None:
        if self._active:
            self._mlflow.end_run()
            self._active = False


def _default_tracking_uri() -> str:
    """Resolve a publication-safe MLflow URI with env-var override support."""
    env_uri = (
        os.getenv("PDM_BENCH_MLFLOW_URI")
        or os.getenv("PDM_TOOLS_MLFLOW_URI")
        or os.getenv("MLFLOW_TRACKING_URI")
    )
    if env_uri:
        return env_uri

    repo_root = Path(__file__).resolve().parents[4]
    default_db = repo_root / "artifacts" / "mlflow" / "mlflow.db"
    return f"sqlite:///{default_db.as_posix()}"
