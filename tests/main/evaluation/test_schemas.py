import pytest

from pdm_tools.main.evaluation.schemas import (
    EvaluationResult,
    EvaluationSummary,
    PredictionArtifact,
    TrainingTelemetry,
)


def _prediction_artifact(**overrides):
    base = {
        "task_type": "classification",
        "pipeline": "ml",
        "model_name": "rf",
        "run_id": "run-001",
        "split": "test",
        "y_true": [0, 1, 1],
        "y_pred": [0, 1, 0],
        "y_score": [[0.9, 0.1], [0.1, 0.9], [0.7, 0.3]],
        "labels": [0, 1],
        "sample_ids": {
            "recording_id": ["r1", "r1", "r2"],
            "window_index": [0, 1, 0],
        },
    }
    base.update(overrides)
    return PredictionArtifact(**base)


def _summary(**overrides):
    base = {
        "task_type": "classification",
        "split": "test",
        "metrics": {
            "accuracy": 2 / 3,
            "macro_f1": 0.6667,
        },
        "per_class_metrics": {
            "0": {"precision": 0.5, "recall": 1.0},
            "1": {"precision": 1.0, "recall": 0.5},
        },
        "confusion_matrix": [[1, 0], [1, 1]],
    }
    base.update(overrides)
    return EvaluationSummary(**base)


def _telemetry(**overrides):
    base = {
        "pipeline": "da",
        "run_id": "run-001",
        "history": [
            {"epoch": 1, "source_train_loss": 0.9},
            {"epoch": 2, "source_train_loss": 0.7},
        ],
    }
    base.update(overrides)
    return TrainingTelemetry(**base)


def test_prediction_artifact_accepts_valid_payload():
    artifact = _prediction_artifact()
    assert artifact.task_type == "classification"
    assert artifact.pipeline == "ml"
    assert artifact.split == "test"


def test_prediction_artifact_rejects_mismatched_target_lengths():
    with pytest.raises(ValueError, match="same length"):
        _prediction_artifact(y_pred=[0, 1])


def test_prediction_artifact_rejects_mismatched_score_lengths():
    with pytest.raises(ValueError, match="y_score"):
        _prediction_artifact(y_score=[[0.9, 0.1]])


def test_prediction_artifact_rejects_mismatched_sample_ids_lengths():
    with pytest.raises(ValueError, match="sample_ids"):
        _prediction_artifact(sample_ids={"recording_id": ["r1", "r2"]})


def test_evaluation_summary_rejects_empty_metrics():
    with pytest.raises(ValueError, match="metrics must not be empty"):
        _summary(metrics={})


def test_evaluation_summary_rejects_non_square_confusion_matrix():
    with pytest.raises(ValueError, match="must be square"):
        _summary(confusion_matrix=[[1, 0, 0], [0, 1, 0]])


def test_training_telemetry_rejects_non_dict_history_rows():
    with pytest.raises(ValueError, match="history\\[0\\]"):
        _telemetry(history=["not-a-dict"])


def test_evaluation_result_accepts_consistent_payloads():
    result = EvaluationResult(
        predictions=_prediction_artifact(),
        summary=_summary(),
        telemetry=_telemetry(),
    )
    assert result.summary.task_type == "classification"


def test_evaluation_result_rejects_task_type_mismatch():
    with pytest.raises(ValueError, match="task_type"):
        EvaluationResult(
            predictions=_prediction_artifact(task_type="classification"),
            summary=_summary(task_type="regression"),
        )


def test_evaluation_result_rejects_split_mismatch():
    with pytest.raises(ValueError, match="split"):
        EvaluationResult(
            predictions=_prediction_artifact(split="test"),
            summary=_summary(split="val"),
        )


def test_evaluation_result_rejects_run_id_mismatch_when_telemetry_present():
    with pytest.raises(ValueError, match="telemetry.run_id"):
        EvaluationResult(
            predictions=_prediction_artifact(run_id="run-001"),
            summary=_summary(),
            telemetry=_telemetry(run_id="run-002"),
        )
