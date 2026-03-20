import json
from pathlib import Path

import numpy as np

from pdm_tools.main.pipelines.dl import pipeline as dl_pipeline
from pdm_tools.main.signals.recordings import Recording


def _make_recording(rid: str, label: str, rpm: float, n_samples: int = 8) -> Recording:
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


def test_dl_pipeline_integration_cpu(monkeypatch, tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()

    def loader(_path: str):
        return [
            _make_recording("r1", "a", 1000),
            _make_recording("r2", "b", 2000),
            _make_recording("r3", "a", 3000),
            _make_recording("r4", "b", 4000),
        ]

    monkeypatch.setattr(dl_pipeline, "LOADER_REGISTRY", {"fake": loader})

    output_dir = tmp_path / "out"
    cfg = {
        "run": {
            "name": "integration",
            "output_dir": str(output_dir),
            "log_to_file": False,
        },
        "dataset": {
            "loader": "fake",
            "root": str(root),
            "train_query": "rpm <= 2000",
            "test_query": "rpm >= 3000",
        },
        "windowing": {"size": 4, "train_overlap": 0.5, "test_overlap": 0.5},
        "views": {"flatten": True},
        "models": ["mlp"],
        "train": {
            "epochs": 1,
            "batch_size": 2,
            "num_workers": 0,
            "device": "cpu",
            "amp": False,
            "optimizer": {"name": "adamw", "lr": 1e-3, "weight_decay": 0.0},
            "scheduler": {"name": "constant"},
        },
        "artifacts": {"save_confusion_matrix": True, "save_predictions": True},
    }

    dl_pipeline.run_dl_pipeline_from_dict(
        cfg, config_dir=tmp_path, config_path=tmp_path / "config.yaml"
    )

    run_dirs = [
        p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("run_")
    ]
    assert run_dirs, "Expected a run directory to be created."
    run_dir = run_dirs[0]

    models_dir = run_dir / "models"
    assert models_dir.exists()
    assert any(models_dir.glob("*.pt"))

    metrics_files = list(models_dir.glob("*_metrics_*.json"))
    assert len(metrics_files) == 1
    metrics_payload = json.loads(metrics_files[0].read_text(encoding="utf-8"))
    for key in ["epoch", "train_acc", "train_loss", "val_acc", "val_loss"]:
        assert key in metrics_payload

    results_dir = run_dir / "results"
    assert results_dir.exists()
    assert any(results_dir.glob("*_predictions.json"))
    assert any(results_dir.glob("*_summary.json"))
