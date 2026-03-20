from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pdm_tools.main.pipelines.dl import pipeline as dl_pipeline
from pdm_tools.main.signals.recordings import Recording


if TYPE_CHECKING:
    from pathlib import Path


def _make_recording(
    rid: str,
    label: str,
    n_samples: int = 8,
    rpm: float = 1000,
) -> Recording:
    data = np.zeros((1, n_samples), dtype=np.float32)
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


def test_dl_pipeline_smoke(monkeypatch, tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()

    def loader(_path: str):
        return [
            _make_recording("r1", "a", rpm=1000),
            _make_recording("r2", "b", rpm=2000),
        ]

    calls = {
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

    def fake_evaluate_dl_classification_models(*_args, **kwargs):
        calls["eval"] = True
        assert kwargs["split"] == "test"
        assert kwargs["pipeline"] == "dl"
        assert kwargs["eval_views"] is not None
        return {}

    monkeypatch.setattr(dl_pipeline, "LOADER_REGISTRY", {"fake": loader})
    monkeypatch.setattr(dl_pipeline, "train_dl_models", lambda **_: {})
    monkeypatch.setattr(dl_pipeline, "create_tracker", lambda **_kwargs: FakeTracker())
    monkeypatch.setattr(
        dl_pipeline,
        "evaluate_dl_classification_models",
        fake_evaluate_dl_classification_models,
    )

    cfg = {
        "run": {"name": "smoke", "output_dir": str(tmp_path), "log_to_file": False},
        "dataset": {
            "loader": "fake",
            "root": str(root),
            "train_query": "rpm == 1000",
            "test_query": "rpm == 2000",
        },
        "windowing": {"size": 4, "train_overlap": 0.5},
        "views": {"flatten": False},
        "models": ["cnn1d"],
        "train": {"epochs": 1, "batch_size": 1, "num_workers": 0, "device": "cpu"},
    }

    out = dl_pipeline.run_dl_pipeline_from_dict(
        cfg, config_dir=tmp_path, config_path=tmp_path / "config.yaml"
    )

    assert out == {}
    assert calls["eval"] is True
    assert calls["tracker_params"] > 0
    assert calls["tracker_metrics"] > 0
    assert calls["tracker_artifacts"] > 0
    assert calls["tracker_closed"] == 1
