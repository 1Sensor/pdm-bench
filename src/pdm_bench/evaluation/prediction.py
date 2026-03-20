from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .schemas import PredictionArtifact


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


def build_prediction_artifact(
    *,
    task_type: str,
    pipeline: str,
    model_name: str,
    run_id: str,
    split: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_score: Sequence[Sequence[float]] | None = None,
    labels: Sequence[int] | None = None,
    sample_ids: Mapping[str, Sequence[Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PredictionArtifact:
    """Create a PredictionArtifact from common sequence-like inputs."""
    sample_ids_dict: dict[str, list[Any]] | None = None
    if sample_ids is not None:
        sample_ids_dict = {str(key): list(values) for key, values in sample_ids.items()}

    return PredictionArtifact(
        task_type=task_type,
        pipeline=pipeline,
        model_name=model_name,
        run_id=run_id,
        split=split,
        y_true=list(y_true),
        y_pred=list(y_pred),
        y_score=[list(row) for row in y_score] if y_score is not None else None,
        labels=list(labels) if labels is not None else None,
        sample_ids=sample_ids_dict,
        metadata=dict(metadata) if metadata is not None else {},
    )
