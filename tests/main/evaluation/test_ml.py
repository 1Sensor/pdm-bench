from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.base import BaseEstimator

from pdm_tools.main.evaluation.ml import evaluate_ml_classification_models


class DummyEstimator(BaseEstimator):
    def __init__(self, preds, probs=None):
        self._preds = np.asarray(preds, dtype=int)
        self._probs = None if probs is None else np.asarray(probs, dtype=float)
        self._final_estimator = "DummyEstimator(param=1)"

    def predict(self, X):
        return self._preds

    def predict_proba(self, X):
        if self._probs is None:
            raise RuntimeError("predict_proba not configured for this dummy.")
        return self._probs


class DummyFEStep:
    def __init__(self, method, degree=None):
        self.method = method
        self.degree = degree


def test_evaluate_ml_models_returns_typed_results_for_estimators():
    X_test = np.array([[0.0], [1.0], [2.0]])
    y_test = np.array([0, 1, 1])
    model = DummyEstimator(
        preds=[0, 1, 0],
        probs=[[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]],
    )
    model.named_steps = {"fe": DummyFEStep(method="poly", degree=2)}

    trained_models = {
        "rf": model,
        "test_data": (X_test, y_test),
    }

    out = evaluate_ml_classification_models(
        trained_models,
        run_id="run-ml-001",
        split="test",
        pipeline="ml",
    )

    assert set(out.keys()) == {"rf"}
    result = out["rf"]

    assert result.predictions.model_name == "rf"
    assert result.predictions.run_id == "run-ml-001"
    assert result.predictions.split == "test"
    assert result.predictions.pipeline == "ml"
    assert result.predictions.y_true == [0, 1, 1]
    assert result.predictions.y_pred == [0, 1, 0]
    assert result.predictions.y_score == [[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]]
    assert result.predictions.metadata["model_params"] == "DummyEstimator(param=1)"
    assert result.predictions.metadata["feature_engineering"] == {
        "fe_method": "poly",
        "fe_degree": 2,
    }

    assert result.summary.task_type == "classification"
    assert result.summary.split == "test"
    assert "accuracy" in result.summary.metrics
    assert "macro_f1" in result.summary.metrics
    assert result.summary.confusion_matrix is not None
    assert result.summary.per_class_metrics is not None


def test_evaluate_ml_models_skips_non_estimator_entries():
    trained_models = {
        "not_a_model": object(),
        "test_data": (np.array([[0.0], [1.0]]), np.array([0, 1])),
    }

    out = evaluate_ml_classification_models(trained_models, run_id="run-ml-002")

    assert out == {}


def test_evaluate_ml_models_handles_models_without_probabilities():
    class DummyNoProbaEstimator(BaseEstimator):
        def __init__(self, preds):
            self._preds = np.asarray(preds, dtype=int)
            self._final_estimator = "DummyNoProbaEstimator()"

        def predict(self, X):
            return self._preds

    X_test = np.array([[0.0], [1.0], [2.0]])
    y_test = np.array([0, 1, 1])
    model = DummyNoProbaEstimator(preds=[0, 1, 0])

    trained_models = {
        "svc_no_prob": model,
        "test_data": (X_test, y_test),
    }

    out = evaluate_ml_classification_models(trained_models, run_id="run-ml-003")

    result = out["svc_no_prob"]
    assert result.predictions.y_score is None


def test_evaluate_ml_models_requires_test_data_entry():
    with pytest.raises(ValueError, match="test_data"):
        evaluate_ml_classification_models({"rf": DummyEstimator(preds=[0])}, run_id="r")


def test_evaluate_ml_models_persists_canonical_artifacts_when_enabled(tmp_path):
    X_test = np.array([[0.0], [1.0], [2.0]])
    y_test = np.array([0, 1, 1])
    model = DummyEstimator(
        preds=[0, 1, 0],
        probs=[[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]],
    )

    trained_models = {
        "rf": model,
        "test_data": (X_test, y_test),
    }

    out = evaluate_ml_classification_models(
        trained_models,
        run_id="run-ml-004",
        split="test",
        pipeline="ml",
        artifacts_dir=tmp_path,
    )

    assert "rf" in out
    pred_path = tmp_path / "rf_test_predictions.json"
    summary_path = tmp_path / "rf_test_summary.json"
    telemetry_path = tmp_path / "rf_test_telemetry.json"

    assert pred_path.exists()
    assert summary_path.exists()
    assert not telemetry_path.exists()

    pred_payload = json.loads(pred_path.read_text(encoding="utf-8"))
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))

    assert pred_payload["model_name"] == "rf"
    assert pred_payload["run_id"] == "run-ml-004"
    assert summary_payload["split"] == "test"
    assert "accuracy" in summary_payload["metrics"]
