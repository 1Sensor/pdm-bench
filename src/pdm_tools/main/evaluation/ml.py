from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from sklearn.base import BaseEstimator

from .artifacts import save_evaluation_result
from .classification import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
)
from .prediction import build_prediction_artifact
from .schemas import EvaluationResult, EvaluationSummary


if TYPE_CHECKING:
    from pathlib import Path


def _extract_ml_metadata(model: BaseEstimator) -> dict[str, Any]:
    """Extract lightweight model metadata for evaluation artifacts."""
    metadata: dict[str, Any] = {}

    final_estimator = getattr(model, "_final_estimator", None)
    if final_estimator is not None:
        metadata["model_params"] = str(final_estimator)

    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, Mapping) and "fe" in named_steps:
        fe_step = named_steps["fe"]
        fe_method = getattr(fe_step, "method", None)
        if fe_method is not None:
            fe_info: dict[str, Any] = {"fe_method": str(fe_method)}
            if fe_method == "poly" and hasattr(fe_step, "degree"):
                fe_info["fe_degree"] = int(fe_step.degree)
            metadata["feature_engineering"] = fe_info

    return metadata


def evaluate_ml_classification_models(
    trained_models: Mapping[str, Any],
    *,
    run_id: str,
    split: str = "test",
    pipeline: str = "ml",
    artifacts_dir: Path | str | None = None,
) -> dict[str, EvaluationResult]:
    """Evaluate trained ML models and return typed in-memory results."""
    if "test_data" not in trained_models:
        raise ValueError("trained_models must include a 'test_data' entry.")

    X_test, y_test = trained_models["test_data"]
    evaluated: dict[str, EvaluationResult] = {}

    for model_name, model in trained_models.items():
        if model_name == "test_data":
            continue
        if not isinstance(model, BaseEstimator):
            continue

        y_pred = model.predict(X_test)
        y_score = (
            model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        )
        labels = sorted(set(y_test).union(set(y_pred)))

        predictions = build_prediction_artifact(
            task_type="classification",
            pipeline=pipeline,
            model_name=model_name,
            run_id=run_id,
            split=split,
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            labels=labels,
            metadata=_extract_ml_metadata(model),
        )

        summary = EvaluationSummary(
            task_type="classification",
            split=split,
            metrics=compute_classification_metrics(
                predictions.y_true,
                predictions.y_pred,
            ),
            per_class_metrics=compute_per_class_metrics(
                predictions.y_true,
                predictions.y_pred,
                labels=predictions.labels,
            ),
            confusion_matrix=compute_confusion_matrix(
                predictions.y_true,
                predictions.y_pred,
                labels=predictions.labels,
            ),
        )

        evaluated[model_name] = EvaluationResult(
            predictions=predictions,
            summary=summary,
        )
        if artifacts_dir is not None:
            save_evaluation_result(
                evaluated[model_name],
                artifacts_dir,
                stem=f"{model_name}_{split}",
            )

    return evaluated
