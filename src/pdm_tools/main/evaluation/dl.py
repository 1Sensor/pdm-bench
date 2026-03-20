from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader

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


def evaluate_dl_classification_models(
    trained_models: Mapping[str, Any],
    eval_views: Sequence[torch.utils.data.Dataset] | None,
    *,
    run_id: str,
    split: str = "test",
    pipeline: str = "dl",
    artifacts_dir: Path | str | None = None,
    device: str = "cpu",
    batch_size: int = 256,
    num_workers: int = 0,
    include_probabilities: bool = False,
) -> dict[str, EvaluationResult]:
    """Evaluate trained DL/DA classification models on one evaluation split.

    This function runs inference for each torch model found in ``trained_models``,
    computes canonical classification summaries, and returns typed
    ``EvaluationResult`` objects. Optionally, it also persists canonical artifact
    files via :func:`save_evaluation_result`.

    Args:
        trained_models: Mapping of model names to training outputs. Each model entry
            is expected to be a mapping that contains a ``"model"`` key with a
            ``torch.nn.Module`` value. Non-conforming entries are skipped.
        eval_views: Evaluation torch views for a single split (for example
            ``test_views`` or ``target_test_views``). When ``None`` or empty, no
            evaluation is performed and an empty dict is returned.
        run_id: Stable run identifier stored in produced artifacts.
        split: Evaluated split name (for example ``"test"`` or ``"target_test"``).
        pipeline: Pipeline identifier stored in artifacts (for example ``"dl"`` or
            ``"da"``).
        artifacts_dir: Optional output directory for persisted canonical artifacts.
            When provided, writes ``*_predictions.json`` and ``*_summary.json`` per
            model.
        device: Torch device used for inference.
        batch_size: Evaluation batch size used by the internal DataLoader.
        num_workers: Number of DataLoader workers.
        include_probabilities: Whether to persist class probabilities (softmax over
            logits) in prediction artifacts as ``y_score``.

    Returns:
        A dictionary keyed by model name with :class:`EvaluationResult` values.
        Returns an empty dictionary when no valid models or no evaluation data is
        available.
    """
    if not eval_views:
        return {}

    test_concat = ConcatDataset(list(eval_views))
    if len(test_concat) == 0:
        return {}

    target_device = torch.device(device)
    use_pin_memory = target_device.type == "cuda"
    loader = DataLoader(
        test_concat,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(num_workers > 0),
    )

    view_ids = _build_view_ids(eval_views)

    evaluated: dict[str, EvaluationResult] = {}
    for model_name, payload in trained_models.items():
        model = payload.get("model") if isinstance(payload, Mapping) else None
        if not isinstance(model, torch.nn.Module):
            continue

        y_true, y_pred, y_score = _collect_predictions(
            model=model,
            loader=loader,
            device=target_device,
            include_probabilities=include_probabilities,
        )
        labels = sorted(set(y_true).union(set(y_pred)))
        sample_ids = {"view_id": view_ids} if view_ids is not None else None

        predictions = build_prediction_artifact(
            task_type="classification",
            pipeline=pipeline,
            model_name=model_name,
            run_id=run_id,
            split=split,
            y_true=y_true,
            y_pred=y_pred,
            y_score=y_score,
            labels=labels,
            sample_ids=sample_ids,
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
        result = EvaluationResult(predictions=predictions, summary=summary)
        evaluated[model_name] = result

        if artifacts_dir is not None:
            save_evaluation_result(
                result,
                artifacts_dir,
                stem=f"{model_name}_{split}",
            )

    return evaluated


def _collect_predictions(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    include_probabilities: bool,
) -> tuple[list[int], list[int], list[list[float]] | None]:
    model = model.to(device)
    model.eval()

    y_true_batches: list[np.ndarray] = []
    y_pred_batches: list[np.ndarray] = []
    y_score_batches: list[np.ndarray] = []

    with torch.inference_mode():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            y_pred_batches.append(logits.argmax(1).cpu().numpy())
            y_true_batches.append(yb.cpu().numpy())

            if include_probabilities:
                y_score_batches.append(torch.softmax(logits, dim=1).cpu().numpy())

    y_true = np.concatenate(y_true_batches).tolist() if y_true_batches else []
    y_pred = np.concatenate(y_pred_batches).tolist() if y_pred_batches else []
    y_score = (
        np.concatenate(y_score_batches).tolist()
        if include_probabilities and y_score_batches
        else None
    )
    return y_true, y_pred, y_score


def _build_view_ids(
    eval_views: Sequence[torch.utils.data.Dataset] | None,
) -> list[int] | None:
    if eval_views is None or len(eval_views) <= 1:
        return None
    counts = [len(view) for view in eval_views]
    if not counts:
        return None
    indices = np.arange(len(counts), dtype=np.int32)
    return np.repeat(indices, counts).tolist()
