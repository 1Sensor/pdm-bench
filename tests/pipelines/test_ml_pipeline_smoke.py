from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pdm_bench.pipelines.ml import pipeline as ml_pipeline
from pdm_bench.signals.recordings import Recording


if TYPE_CHECKING:
    from pathlib import Path


def _make_recording(rid: str, label: str, rpm: float, n_samples: int = 8) -> Recording:
    data = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)[None, :]
    return Recording(
        rid=rid,
        data=data,
        fs=1.0,
        label=label,
        source="test",
        unit="u",
        channels=["ch0"],
        rpm=rpm,
    )


@pytest.mark.filterwarnings(
    "ignore:The following available features were not used.*:UserWarning"
)
@pytest.mark.parametrize(
    ("mode", "expected_n_features"),
    [
        ("time", 1),
        ("freq", 1),
        ("time_freq", 2),
    ],
    ids=["time", "freq", "time_freq"],
)
def test_ml_pipeline_smoke(
    monkeypatch, tmp_path: Path, mode: str, expected_n_features: int
):
    root = tmp_path / "root"
    root.mkdir()

    def loader(_path: str):
        return [
            _make_recording("r1", "a", 1000),
            _make_recording("r2", "b", 2000),
        ]

    calls = {
        "train": False,
        "eval": False,
        "tracker_params": 0,
        "tracker_metrics": 0,
        "tracker_artifacts": 0,
        "tracker_closed": 0,
    }

    class FakeTracker:
        def log_params(self, _params):
            calls["tracker_params"] += 1

        def log_metrics(self, _metrics, step=None):
            _ = step
            calls["tracker_metrics"] += 1

        def log_artifact(self, _path, artifact_path=None):
            _ = artifact_path
            calls["tracker_artifacts"] += 1

        def close(self):
            calls["tracker_closed"] += 1

    def fake_train_ml_models(**kwargs):
        calls["train"] = True
        x_train, y_train = kwargs["train_dataset"]
        x_test, y_test = kwargs["test_dataset"]
        assert x_train.shape[0] > 0 and x_train.shape[1] > 0
        assert x_test.shape[0] > 0 and x_test.shape[1] > 0
        assert x_train.shape[1] == expected_n_features
        assert x_test.shape[1] == expected_n_features
        assert y_train.shape[0] == x_train.shape[0]
        assert y_test.shape[0] == x_test.shape[0]
        return {"KNN": object(), "test_data": (x_test, y_test)}

    def fake_evaluate_ml_classification_models(models: dict, **_kwargs):
        calls["eval"] = True
        assert "KNN" in models
        return {}

    monkeypatch.setattr(ml_pipeline, "LOADER_REGISTRY", {"fake": loader})
    monkeypatch.setattr(ml_pipeline, "train_ml_models", fake_train_ml_models)
    monkeypatch.setattr(ml_pipeline, "create_tracker", lambda **_kwargs: FakeTracker())
    monkeypatch.setattr(
        ml_pipeline,
        "evaluate_ml_classification_models",
        fake_evaluate_ml_classification_models,
    )

    selected_time_features = ["rms"] if mode in {"time", "time_freq"} else []
    selected_freq_features = ["mf"] if mode in {"freq", "time_freq"} else []

    output_dir = tmp_path / "out"
    cfg = {
        "run": {
            "name": "smoke-ml",
            "output_dir": str(output_dir),
            "log_to_file": False,
        },
        "dataset": {
            "loader": "fake",
            "root": str(root),
            "train_query": "rpm == 1000",
            "test_query": "rpm == 2000",
        },
        "windowing": {"size": 4, "train_overlap": 0.5, "test_overlap": 0.5},
        "features": {
            "mode": mode,
            "time_features": selected_time_features,
            "freq_features": selected_freq_features,
        },
        "train": {"classifier_names": ["KNN"], "use_bayesian_search": False},
    }

    out = ml_pipeline.run_ml_pipeline_from_dict(
        cfg, config_dir=tmp_path, config_path=tmp_path / "ml_config.yaml"
    )

    assert "KNN" in out
    assert "test_data" in out
    assert calls["train"] is True
    assert calls["eval"] is True
    assert calls["tracker_params"] > 0
    assert calls["tracker_metrics"] > 0
    assert calls["tracker_artifacts"] > 0
    assert calls["tracker_closed"] == 1

    run_dirs = [
        p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("run_")
    ]
    assert run_dirs, "Expected a run directory to be created."
    run_dir = run_dirs[0]

    assert (run_dir / "pipeline_config.json").exists()
    assert (run_dir / "run_meta.json").exists()
    feature_path = run_dir / "feature_names.json"
    assert feature_path.exists()

    feature_payload = json.loads(feature_path.read_text("utf-8"))
    assert feature_payload["mode"] == mode
    assert feature_payload["n_features"] == expected_n_features
    assert feature_payload["selected_time_features"] == selected_time_features
    assert feature_payload["selected_freq_features"] == selected_freq_features
    assert len(feature_payload["feature_names"]) == expected_n_features


def test_ml_pipeline_time_freq_rejects_misaligned_labels(monkeypatch, tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()

    def loader(_path: str):
        return [
            _make_recording("r1", "a", 1000),
            _make_recording("r2", "a", 2000),
        ]

    def fake_extract_train_test_branch(*, branch: str, **_kwargs):
        feature_names = [f"{branch}_f0"]
        x_train = np.ones((2, 1), dtype=np.float32)
        x_test = np.ones((2, 1), dtype=np.float32)
        if branch == "time":
            y_train = np.array([0, 0], dtype=np.int8)
            y_test = np.array([0, 0], dtype=np.int8)
        else:
            y_train = np.array([1, 1], dtype=np.int8)
            y_test = np.array([1, 1], dtype=np.int8)
        return feature_names, x_train, y_train, x_test, y_test

    monkeypatch.setattr(ml_pipeline, "LOADER_REGISTRY", {"fake": loader})
    monkeypatch.setattr(
        ml_pipeline,
        "_extract_train_test_branch",
        fake_extract_train_test_branch,
    )

    cfg = {
        "run": {
            "name": "smoke-ml-freq",
            "output_dir": str(tmp_path),
            "log_to_file": False,
        },
        "dataset": {
            "loader": "fake",
            "root": str(root),
            "train_query": "rpm == 1000",
            "test_query": "rpm == 2000",
        },
        "windowing": {"size": 4, "train_overlap": 0.5, "test_overlap": 0.5},
        "features": {
            "mode": "time_freq",
            "time_features": ["rms"],
            "freq_features": ["mf"],
        },
        "train": {"classifier_names": ["KNN"]},
    }

    with pytest.raises(
        ValueError, match="Label alignment check failed for train split"
    ):
        ml_pipeline.run_ml_pipeline_from_dict(
            cfg, config_dir=tmp_path, config_path=tmp_path / "ml_config.yaml"
        )
