from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PredictionArtifact:
    """Canonical prediction payload for one evaluated split."""

    task_type: str  # classification, regression, etc.
    pipeline: str  # dl, ml, da
    model_name: str
    run_id: str
    split: str
    y_true: list[int]
    y_pred: list[int]
    y_score: list[list[float]] | None = None
    labels: list[int] | None = None
    # Optional per-sample identity columns: "recording_id", "window_id", "view_id"
    sample_ids: dict[str, list[Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_str(self.task_type, "task_type")
        _require_nonempty_str(self.pipeline, "pipeline")
        _require_nonempty_str(self.model_name, "model_name")
        _require_nonempty_str(self.run_id, "run_id")
        _require_nonempty_str(self.split, "split")

        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

        n_samples = len(self.y_true)
        if self.y_score is not None and len(self.y_score) != n_samples:
            raise ValueError("y_score must have the same length as y_true/y_pred.")

        if self.sample_ids is not None:
            for key, values in self.sample_ids.items():
                _require_nonempty_str(key, "sample_ids key")
                if len(values) != n_samples:
                    raise ValueError(
                        f"sample_ids['{key}'] must match y_true/y_pred length."
                    )


@dataclass(slots=True)
class EvaluationSummary:
    """Compact evaluation report derived from predictions."""

    task_type: str
    split: str
    metrics: dict[str, float]
    per_class_metrics: dict[str, dict[str, float]] | None = None
    confusion_matrix: list[list[int]] | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str(self.task_type, "task_type")
        _require_nonempty_str(self.split, "split")
        if not self.metrics:
            raise ValueError("metrics must not be empty.")

        for name, value in self.metrics.items():
            _require_nonempty_str(name, "metrics key")
            _require_finite_number(value, f"metrics['{name}']")

        if self.per_class_metrics is not None:
            for class_name, class_metrics in self.per_class_metrics.items():
                _require_nonempty_str(class_name, "per_class_metrics key")
                for metric_name, metric_value in class_metrics.items():
                    _require_nonempty_str(metric_name, "per_class_metrics metric key")
                    _require_finite_number(
                        metric_value,
                        f"per_class_metrics['{class_name}']['{metric_name}']",
                    )

        if self.confusion_matrix is not None:
            if not self.confusion_matrix:
                raise ValueError("confusion_matrix must not be empty when provided.")
            width = len(self.confusion_matrix[0])
            if width == 0:
                raise ValueError("confusion_matrix rows must not be empty.")
            if any(len(row) != width for row in self.confusion_matrix):
                raise ValueError("confusion_matrix rows must all have equal length.")
            if len(self.confusion_matrix) != width:
                raise ValueError("confusion_matrix must be square.")


@dataclass(slots=True)
class TrainingTelemetry:
    """Train-time diagnostic history kept separate from official evaluation."""

    pipeline: str
    run_id: str
    history: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_nonempty_str(self.pipeline, "pipeline")
        _require_nonempty_str(self.run_id, "run_id")
        for idx, row in enumerate(self.history):
            if not isinstance(row, dict):
                raise ValueError(f"history[{idx}] must be a dictionary.")


@dataclass(slots=True)
class EvaluationResult:
    """Evaluator return payload."""

    predictions: PredictionArtifact
    summary: EvaluationSummary
    telemetry: TrainingTelemetry | None = None

    def __post_init__(self) -> None:
        if self.predictions.task_type != self.summary.task_type:
            raise ValueError("predictions.task_type must match summary.task_type.")
        if self.predictions.split != self.summary.split:
            raise ValueError("predictions.split must match summary.split.")
        if (
            self.telemetry is not None
            and self.telemetry.run_id != self.predictions.run_id
        ):
            raise ValueError("telemetry.run_id must match predictions.run_id.")


def _require_nonempty_str(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string.")


def _require_finite_number(value: Any, field_name: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a real number.")
    if not math.isfinite(float(value)):
        raise ValueError(f"{field_name} must be finite.")
