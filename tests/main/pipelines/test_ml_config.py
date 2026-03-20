from __future__ import annotations

import logging

import pytest

from pdm_tools.main.pipelines.ml.config import (
    MLPipelineConfig,
    MLTrainSpec,
)


def test_ml_train_spec_bool_parsing():
    spec = MLTrainSpec.from_dict(
        {
            "classifier_names": ["KNN"],
            "use_bayesian_search": "true",
        }
    )
    assert spec.use_bayesian_search is True


def test_ml_train_spec_invalid_bool():
    with pytest.raises(ValueError, match="Expected bool"):
        MLTrainSpec.from_dict(
            {
                "classifier_names": ["KNN"],
                "use_bayesian_search": "not-a-bool",
            }
        )


def test_ml_pipeline_config_minimal():
    cfg = MLPipelineConfig.from_dict(
        {
            "dataset": {"loader": "fake", "root": "/tmp"},
            "windowing": {"size": 8},
            "train": {"classifier_names": ["KNN"]},
        }
    )

    assert cfg.run.log_to_file is True
    assert cfg.features.mode == "time_freq"
    assert cfg.features.time_features == []
    assert cfg.features.freq_features == []
    assert cfg.train.use_bayesian_search is False
    assert cfg.train.bayes_n_iter == 10
    assert cfg.train.bayes_cv == 3
    assert cfg.train.bayes_n_points == 1
    assert cfg.train.n_jobs == -1
    assert cfg.train.search_spaces is None
    assert cfg.artifacts.save_confusion_matrix is True
    assert cfg.tracking.enabled is False


def test_ml_pipeline_config_requires_classifier_names():
    with pytest.raises(
        ValueError, match="train.classifier_names must be a non-empty list"
    ):
        MLPipelineConfig.from_dict(
            {
                "dataset": {"loader": "fake", "root": "/tmp"},
                "windowing": {"size": 8},
                "train": {},
            }
        )


def test_ml_pipeline_config_requires_windowing_size():
    with pytest.raises(ValueError, match="windowing.size is required"):
        MLPipelineConfig.from_dict(
            {
                "dataset": {"loader": "fake", "root": "/tmp"},
                "windowing": {},
                "train": {"classifier_names": ["KNN"]},
            }
        )


def test_ml_pipeline_config_requires_dataset_loader():
    with pytest.raises(ValueError, match="dataset.loader is required"):
        MLPipelineConfig.from_dict(
            {
                "dataset": {},
                "windowing": {"size": 8},
                "train": {"classifier_names": ["KNN"]},
            }
        )


def test_ml_pipeline_config_rejects_invalid_feature_mode():
    with pytest.raises(ValueError, match="Invalid features.mode"):
        MLPipelineConfig.from_dict(
            {
                "dataset": {"loader": "fake", "root": "/tmp"},
                "windowing": {"size": 8},
                "features": {"mode": "invalid_mode"},
                "train": {"classifier_names": ["KNN"]},
            }
        )


def test_ml_pipeline_config_warns_on_unknown_keys(caplog):
    caplog.set_level(logging.WARNING)

    MLPipelineConfig.from_dict(
        {
            "dataset": {"loader": "fake", "root": "/tmp"},
            "windowing": {"size": 8},
            "features": {"mode": "time", "unknown_feature_field": 1},
            "train": {"classifier_names": ["KNN"]},
            "unknown_root_field": 1,
        }
    )

    assert "Unknown keys in 'root': ['unknown_root_field']" in caplog.text
    assert "Unknown keys in 'features': ['unknown_feature_field']" in caplog.text


def test_ml_pipeline_config_round_trip():
    data = {
        "run": {"name": "ml-test", "output_dir": "artifacts/ml", "log_to_file": False},
        "dataset": {"loader": "fake", "root": "/tmp"},
        "windowing": {"size": 256, "train_overlap": 0.5, "test_overlap": 0.25},
        "features": {
            "mode": "time_freq",
            "time_features": [
                {"name": "rms"},
                {"name": "kurt", "params": {"ddof": 1}},
            ],
            "freq_features": ["mf"],
        },
        "train": {
            "classifier_names": ["KNN", "RF"],
            "use_bayesian_search": True,
            "bayes_n_iter": 25,
            "bayes_cv": 4,
            "bayes_n_points": 2,
            "n_jobs": 2,
            "search_spaces": {"KNN": {"estimator__n_neighbors": [1, 3, 5]}},
        },
        "artifacts": {
            "save_predictions": True,
            "save_probs": False,
            "save_confusion_matrix": True,
        },
        "tracking": {
            "enabled": True,
        },
    }

    cfg = MLPipelineConfig.from_dict(data)
    out = cfg.to_dict()

    assert out["run"]["name"] == "ml-test"
    assert out["dataset"]["loader"] == "fake"
    assert out["windowing"]["size"] == 256
    assert out["features"]["mode"] == "time_freq"
    assert out["features"]["time_features"] == [
        {"name": "rms", "params": {}},
        {"name": "kurt", "params": {"ddof": 1}},
    ]
    assert out["features"]["freq_features"] == [{"name": "mf", "params": {}}]
    assert out["train"]["classifier_names"] == ["KNN", "RF"]
    assert out["train"]["use_bayesian_search"] is True
    assert out["train"]["bayes_n_iter"] == 25
    assert out["train"]["bayes_cv"] == 4
    assert out["train"]["bayes_n_points"] == 2
    assert out["train"]["n_jobs"] == 2
    assert out["train"]["search_spaces"] == {
        "KNN": {"estimator__n_neighbors": [1, 3, 5]}
    }
    assert out["tracking"] == {"enabled": True, "experiment_name": "pdm-bench"}


def test_ml_pipeline_config_rejects_invalid_tracking_enabled_value():
    with pytest.raises(ValueError, match="Expected bool"):
        MLPipelineConfig.from_dict(
            {
                "dataset": {"loader": "fake", "root": "/tmp"},
                "windowing": {"size": 8},
                "train": {"classifier_names": ["KNN"]},
                "tracking": {"enabled": "not-a-bool"},
            }
        )
