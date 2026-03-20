import pytest

from pdm_bench.evaluation.classification import (
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
)


def test_compute_classification_metrics_binary_case():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = compute_classification_metrics(y_true, y_pred)

    assert set(metrics) == {
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
    }
    assert metrics["accuracy"] == pytest.approx(0.75, abs=1e-9)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75, abs=1e-9)
    assert metrics["macro_f1"] == pytest.approx(0.7333333333, rel=1e-8)
    assert metrics["weighted_f1"] == pytest.approx(0.7333333333, rel=1e-8)
    assert metrics["macro_precision"] == pytest.approx(0.8333333333, rel=1e-8)
    assert metrics["macro_recall"] == pytest.approx(0.75, abs=1e-9)


def test_compute_confusion_matrix_uses_given_label_order():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    matrix = compute_confusion_matrix(y_true, y_pred, labels=[1, 0])

    assert matrix == [[2, 0], [1, 1]]


def test_compute_per_class_metrics_contains_expected_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    per_class = compute_per_class_metrics(y_true, y_pred, labels=[0, 1])

    assert set(per_class) == {"0", "1"}
    assert per_class["0"]["precision"] == pytest.approx(1.0, abs=1e-9)
    assert per_class["0"]["recall"] == pytest.approx(0.5, abs=1e-9)
    assert per_class["0"]["f1"] == pytest.approx(0.6666666667, rel=1e-8)
    assert per_class["0"]["support"] == 2

    assert per_class["1"]["precision"] == pytest.approx(2 / 3, rel=1e-8)
    assert per_class["1"]["recall"] == pytest.approx(1.0, abs=1e-9)
    assert per_class["1"]["f1"] == pytest.approx(0.8, abs=1e-9)
    assert per_class["1"]["support"] == 2


def test_compute_per_class_metrics_handles_zero_division():
    y_true = [0, 1]
    y_pred = [0, 0]

    per_class = compute_per_class_metrics(
        y_true,
        y_pred,
        labels=[0, 1],
        zero_division=0,
    )

    assert per_class["1"]["precision"] == pytest.approx(0.0, abs=1e-9)
    assert per_class["1"]["recall"] == pytest.approx(0.0, abs=1e-9)
    assert per_class["1"]["f1"] == pytest.approx(0.0, abs=1e-9)


def test_classification_functions_reject_empty_targets():
    with pytest.raises(ValueError, match="must not be empty"):
        compute_classification_metrics([], [])


def test_classification_functions_reject_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        compute_classification_metrics([0, 1], [0])
