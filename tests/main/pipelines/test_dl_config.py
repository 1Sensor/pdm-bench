import pytest

from pdm_tools.main.pipelines.dl.config import (
    ArtifactsSpec,
    DLPipelineConfig,
    RunSpec,
    ViewsSpec,
)


def test_run_spec_bool_parsing():
    spec = RunSpec.from_dict({"log_to_file": "false"})
    assert spec.log_to_file is False


def test_views_spec_bool_parsing():
    spec = ViewsSpec.from_dict({"flatten": "true"})
    assert spec.flatten is True


def test_artifacts_spec_bool_parsing():
    spec = ArtifactsSpec.from_dict({"save_predictions": "true", "save_probs": "false"})
    assert spec.save_predictions is True
    assert spec.save_probs is False


def test_run_spec_invalid_bool():
    with pytest.raises(ValueError, match="Expected bool"):
        RunSpec.from_dict({"log_to_file": "not-a-bool"})


def test_pipeline_config_minimal():
    cfg = DLPipelineConfig.from_dict(
        {
            "dataset": {"loader": "fake", "root": "/tmp"},
            "windowing": {"size": 8},
            "models": ["cnn1d"],
            "train": {},
        }
    )
    assert cfg.models == ["cnn1d"]
    assert cfg.windowing.size == 8
    assert cfg.tracking.enabled is False


def test_pipeline_config_requires_models():
    with pytest.raises(ValueError, match="models must be a non-empty list"):
        DLPipelineConfig.from_dict(
            {
                "dataset": {"loader": "fake", "root": "/tmp"},
                "windowing": {"size": 8},
                "models": [],
                "train": {},
            }
        )


def test_pipeline_config_requires_windowing_size():
    with pytest.raises(ValueError, match="windowing.size is required"):
        DLPipelineConfig.from_dict(
            {
                "dataset": {"loader": "fake", "root": "/tmp"},
                "windowing": {},
                "models": ["cnn1d"],
                "train": {},
            }
        )


def test_pipeline_config_requires_dataset_loader():
    with pytest.raises(ValueError, match="dataset.loader is required"):
        DLPipelineConfig.from_dict(
            {
                "dataset": {},
                "windowing": {"size": 8},
                "models": ["cnn1d"],
                "train": {},
            }
        )


def test_pipeline_config_accepts_tracking_enabled():
    cfg = DLPipelineConfig.from_dict(
        {
            "dataset": {"loader": "fake", "root": "/tmp"},
            "windowing": {"size": 8},
            "models": ["cnn1d"],
            "train": {},
            "tracking": {
                "enabled": True,
            },
        }
    )
    assert cfg.tracking.enabled is True


def test_pipeline_config_rejects_invalid_tracking_enabled_value():
    with pytest.raises(ValueError, match="Expected bool"):
        DLPipelineConfig.from_dict(
            {
                "dataset": {"loader": "fake", "root": "/tmp"},
                "windowing": {"size": 8},
                "models": ["cnn1d"],
                "train": {},
                "tracking": {"enabled": "not-a-bool"},
            }
        )
