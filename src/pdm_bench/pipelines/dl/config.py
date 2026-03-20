from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from pdm_bench.pipelines.common.config import (
    ArtifactsSpec,
    DatasetSpec,
    RunSpec,
    TrackingSpec,
    WindowingSpec,
    as_bool,
    warn_unknown,
)
from pdm_bench.training.dl.config import OptimizerCfg, SchedulerCfg, TrainCfg
from pdm_bench.training.dl.utils import cfg_to_jsonable


def _warn_unknown(section: str, data: dict[str, Any], allowed: set[str]) -> None:
    """Log a warning when config sections contain unknown keys."""
    warn_unknown(section, data, allowed, logger_name="pdm_bench.pipelines.dl")


def _as_bool(value: object, *, default: bool = False) -> bool:
    """Parse a bool-like value from config inputs."""
    return as_bool(value, default=default)


@dataclass(frozen=True)
class ViewsSpec:
    """Torch view configuration."""

    flatten: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ViewsSpec:
        """Create a ViewsSpec from a dict."""
        allowed = {"flatten"}
        _warn_unknown("views", data, allowed)
        return cls(flatten=_as_bool(data.get("flatten", False)))


@dataclass(frozen=True)
class DLPipelineConfig:
    """Top-level DL pipeline configuration."""

    run: RunSpec
    dataset: DatasetSpec
    windowing: WindowingSpec
    views: ViewsSpec
    models: list[str]
    train: TrainCfg
    artifacts: ArtifactsSpec
    tracking: TrackingSpec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DLPipelineConfig:
        """Create a DLPipelineConfig from a dict."""
        allowed = {
            "run",
            "dataset",
            "windowing",
            "views",
            "models",
            "train",
            "artifacts",
            "tracking",
        }
        _warn_unknown("root", data, allowed)

        run = RunSpec.from_dict(data.get("run", {}))
        dataset = DatasetSpec.from_dict(data.get("dataset", {}))
        windowing = WindowingSpec.from_dict(data.get("windowing", {}))
        views = ViewsSpec.from_dict(data.get("views", {}))
        artifacts = ArtifactsSpec.from_dict(data.get("artifacts", {}))
        tracking = TrackingSpec.from_dict(data.get("tracking", {}))

        models = data.get("models", [])
        if not isinstance(models, list) or not models:
            raise ValueError("models must be a non-empty list.")

        train = _parse_train_cfg(data.get("train", {}))

        return cls(
            run=run,
            dataset=dataset,
            windowing=windowing,
            views=views,
            models=[str(m) for m in models],
            train=train,
            artifacts=artifacts,
            tracking=tracking,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a JSON-friendly dict."""
        return {
            "run": asdict(self.run),
            "dataset": asdict(self.dataset),
            "windowing": asdict(self.windowing),
            "views": asdict(self.views),
            "models": list(self.models),
            "train": cfg_to_jsonable(self.train),
            "artifacts": asdict(self.artifacts),
            "tracking": asdict(self.tracking),
        }


def _parse_train_cfg(data: dict[str, Any]) -> TrainCfg:
    """Build a TrainCfg from a config dict."""
    allowed = {
        "epochs",
        "optimizer",
        "scheduler",
        "label_smoothing",
        "batch_size",
        "num_workers",
        "device",
        "amp",
        "log_every",
        "log_train_metrics",
        "class_weights",
        "random_state",
    }
    _warn_unknown("train", data, allowed)

    optimizer_data = data.get("optimizer", {})
    scheduler_data = data.get("scheduler", {})

    optimizer = OptimizerCfg(
        name=str(optimizer_data.get("name", "adamw")),
        lr=float(optimizer_data.get("lr", 1e-3)),
        weight_decay=float(optimizer_data.get("weight_decay", 1e-4)),
    )

    scheduler = SchedulerCfg(
        name=str(scheduler_data.get("name", "exponential")),
        gamma=float(scheduler_data.get("gamma", 0.98)),
        factor=float(scheduler_data.get("factor", 0.5)),
        patience=int(scheduler_data.get("patience", 2)),
    )

    class_weights = data.get("class_weights")
    if class_weights is not None and not isinstance(class_weights, torch.Tensor):
        class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_kwargs = {
        "epochs": int(data.get("epochs", 10)),
        "label_smoothing": float(data.get("label_smoothing", 0.05)),
        "batch_size": int(data.get("batch_size", 64)),
        "num_workers": int(data.get("num_workers", 2)),
        "device": str(data.get("device", TrainCfg().device)),
        "amp": _as_bool(data.get("amp", True), default=True),
        "log_every": int(data.get("log_every", 0)),
        "log_train_metrics": _as_bool(
            data.get("log_train_metrics", False),
            default=False,
        ),
        "class_weights": class_weights,
        "random_state": int(data.get("random_state", 42)),
    }

    return TrainCfg(
        optimizer=optimizer,
        scheduler=scheduler,
        **train_kwargs,
    )
