from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


if TYPE_CHECKING:
    from collections.abc import Sequence


def _validate_targets(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> tuple[list[int], list[int]]:
    true = list(y_true)
    pred = list(y_pred)

    if not true:
        raise ValueError("y_true must not be empty.")
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length.")

    return true, pred


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    zero_division: int = 0,
) -> dict[str, float]:
    """Compute headline and supporting classification metrics."""
    true, pred = _validate_targets(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(
            f1_score(true, pred, average="macro", zero_division=zero_division)
        ),
        "weighted_f1": float(
            f1_score(true, pred, average="weighted", zero_division=zero_division)
        ),
        "macro_precision": float(
            precision_score(true, pred, average="macro", zero_division=zero_division)
        ),
        "macro_recall": float(
            recall_score(true, pred, average="macro", zero_division=zero_division)
        ),
    }


def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
) -> list[list[int]]:
    """Compute confusion matrix with deterministic label ordering."""
    true, pred = _validate_targets(y_true, y_pred)
    label_order = (
        list(labels) if labels is not None else sorted(set(true).union(set(pred)))
    )
    matrix = confusion_matrix(true, pred, labels=label_order)
    return matrix.tolist()


def compute_per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: Sequence[int] | None = None,
    zero_division: int = 0,
) -> dict[str, dict[str, float | int]]:
    """Compute precision/recall/f1/support per class label."""
    true, pred = _validate_targets(y_true, y_pred)
    label_order = (
        list(labels) if labels is not None else sorted(set(true).union(set(pred)))
    )
    precision, recall, f1, support = precision_recall_fscore_support(
        true,
        pred,
        labels=label_order,
        zero_division=zero_division,
    )

    per_class: dict[str, dict[str, float | int]] = {}
    for idx, label in enumerate(label_order):
        per_class[str(label)] = {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
    return per_class
