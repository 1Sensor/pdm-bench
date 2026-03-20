import json

from pdm_bench.evaluation.artifacts import (
    save_evaluation_result,
    save_evaluation_summary,
    save_prediction_artifact,
    save_training_telemetry,
)
from pdm_bench.evaluation.schemas import (
    EvaluationResult,
    EvaluationSummary,
    PredictionArtifact,
    TrainingTelemetry,
)


def _prediction() -> PredictionArtifact:
    return PredictionArtifact(
        task_type="classification",
        pipeline="ml",
        model_name="rf",
        run_id="run-001",
        split="test",
        y_true=[0, 1],
        y_pred=[0, 1],
        y_score=[[0.8, 0.2], [0.1, 0.9]],
        labels=[0, 1],
        sample_ids={"recording_id": ["r1", "r2"]},
    )


def _summary() -> EvaluationSummary:
    return EvaluationSummary(
        task_type="classification",
        split="test",
        metrics={"accuracy": 1.0, "macro_f1": 1.0},
        confusion_matrix=[[1, 0], [0, 1]],
    )


def _telemetry() -> TrainingTelemetry:
    return TrainingTelemetry(
        pipeline="da",
        run_id="run-001",
        history=[{"epoch": 1, "source_train_loss": 0.9}],
    )


def test_save_single_artifacts(tmp_path):
    pred_path = save_prediction_artifact(_prediction(), tmp_path / "pred.json")
    summary_path = save_evaluation_summary(_summary(), tmp_path / "summary.json")
    telemetry_path = save_training_telemetry(_telemetry(), tmp_path / "telemetry.json")

    pred_payload = json.loads(pred_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    telemetry_payload = json.loads(telemetry_path.read_text(encoding="utf-8"))

    assert pred_payload["run_id"] == "run-001"
    assert summary_payload["metrics"]["accuracy"] == 1.0
    assert telemetry_payload["pipeline"] == "da"


def test_save_evaluation_result_writes_canonical_files(tmp_path):
    result = EvaluationResult(
        predictions=_prediction(),
        summary=_summary(),
        telemetry=_telemetry(),
    )

    written = save_evaluation_result(result, tmp_path, stem="final_eval")

    assert set(written) == {"predictions", "summary", "telemetry"}
    assert written["predictions"].name == "final_eval_predictions.json"
    assert written["summary"].name == "final_eval_summary.json"
    assert written["telemetry"].name == "final_eval_telemetry.json"


def test_save_evaluation_result_skips_telemetry_when_absent(tmp_path):
    result = EvaluationResult(
        predictions=_prediction(),
        summary=_summary(),
        telemetry=None,
    )

    written = save_evaluation_result(result, tmp_path)

    assert set(written) == {"predictions", "summary"}
    assert written["predictions"].exists()
    assert written["summary"].exists()
