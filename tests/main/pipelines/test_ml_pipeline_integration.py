from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pdm_tools.main.pipelines.ml import pipeline as ml_pipeline
from pdm_tools.main.signals.recordings import Recording


if TYPE_CHECKING:
    from pathlib import Path


def _make_recording(rid: str, label: str, rpm: float, value: float) -> Recording:
    data = np.full((1, 8), value, dtype=np.float32)
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
def test_ml_pipeline_integration_cpu(monkeypatch, tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()

    def loader(_path: str):
        return [
            _make_recording("tr_a", "a", 1000, 0.1),
            _make_recording("tr_b", "b", 1000, 1.0),
            _make_recording("te_a", "a", 2000, 0.2),
            _make_recording("te_b", "b", 2000, 0.9),
        ]

    monkeypatch.setattr(ml_pipeline, "LOADER_REGISTRY", {"fake": loader})

    output_dir = tmp_path / "out"
    cfg = {
        "run": {
            "name": "integration-ml",
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
        "features": {"mode": "time", "time_features": ["rms"]},
        "train": {
            "classifier_names": ["KNN"],
            "use_bayesian_search": False,
            "n_jobs": 1,
        },
    }

    out = ml_pipeline.run_ml_pipeline_from_dict(
        cfg, config_dir=tmp_path, config_path=tmp_path / "ml_config.yaml"
    )

    assert "KNN" in out
    assert "test_data" in out

    run_dirs = [
        p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("run_")
    ]
    assert run_dirs, "Expected a run directory to be created."
    run_dir = run_dirs[0]

    models_dir = run_dir / "models"
    assert models_dir.exists()
    assert any(models_dir.glob("KNN_model_*.pkl"))

    results_dir = run_dir / "results"
    assert results_dir.exists()
    assert any(results_dir.glob("*_predictions.json"))
    assert any(results_dir.glob("*_summary.json"))

    feature_payload = json.loads((run_dir / "feature_names.json").read_text("utf-8"))
    assert feature_payload["mode"] == "time"
    assert feature_payload["n_features"] > 0
    assert "feature_names" in feature_payload
    assert feature_payload["feature_names"]

    run_meta = json.loads((run_dir / "run_meta.json").read_text("utf-8"))
    assert "data_paths" in run_meta
    assert run_meta["data_paths"]["root"] == str(root)
