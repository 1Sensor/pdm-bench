from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from pdm_bench.evaluation.ml import evaluate_ml_classification_models
from pdm_bench.loaders.cwru import load_cwru_dataset
from pdm_bench.loaders.pu import load_pu_dataset
from pdm_bench.pipelines.common.data_utils import (
    ensure_nonempty,
    load_datasets,
    warn_label_coverage,
)
from pdm_bench.pipelines.common.io_utils import (
    configure_logging,
    load_env_file,
    prepare_run_dir,
    redirect_output_to_logger,
)
from pdm_bench.pipelines.ml.config import MLPipelineConfig
from pdm_bench.signals.features_config import ExtractionConfig, FeatureRequest
from pdm_bench.tracking import create_tracker
from pdm_bench.training.dl.utils import save_json
from pdm_bench.training.ml_classifiers import train_ml_models


LOADER_REGISTRY = {
    "cwru": load_cwru_dataset,
    "pu": load_pu_dataset,
}


def run_ml_pipeline_from_dict(
    data: dict[str, Any],
    *,
    config_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    """Run the ML pipeline using a resolved config dictionary.

    Args:
        data: Configuration mapping (already resolved, e.g. by Hydra/OmegaConf).
        config_dir: Base directory used to resolve relative dataset/config paths.
        config_path: Optional path to the source config file (stored in run metadata).

    Returns:
        Dictionary returned by model training, including trained estimators
        keyed by classifier name and the shared "test_data" payload.
    """
    if config_dir is not None:
        load_env_file(Path(".env"))
        load_env_file(Path(config_dir) / ".env")
    cfg = MLPipelineConfig.from_dict(data)
    return _run_ml_pipeline(cfg, config_path=config_path, config_dir=config_dir)


@hydra.main(
    version_base="1.3", config_path="../../../../config", config_name="ml_config"
)
def main(cfg) -> int:
    """Hydra CLI entrypoint for the ML pipeline.

    Args:
        cfg: Hydra config object for the ML pipeline.

    Returns:
        Exit code (0 on success).
    """
    config_dir = Path(__file__).resolve().parents[4] / "config"

    original_cwd = Path(hydra.utils.get_original_cwd())
    load_env_file(original_cwd / ".env")
    load_env_file(config_dir / ".env")

    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("Hydra config root must be a mapping.")
    data.pop("hydra", None)

    run_ml_pipeline_from_dict(
        data,
        config_dir=config_dir,
        config_path=config_dir / "ml_config.yaml",
    )
    return 0


def _run_ml_pipeline(
    cfg: MLPipelineConfig,
    *,
    config_path: Path | None,
    config_dir: Path | None,
) -> dict:
    """Execute the pipeline from a parsed ML config."""
    config_dir = config_dir or Path.cwd()

    if HydraConfig.initialized():
        run_dir = Path.cwd()
        run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        run_dir, run_timestamp = prepare_run_dir(cfg.run)

    logger = configure_logging(run_dir, cfg.run.log_to_file)
    save_json(cfg.to_dict(), run_dir / "pipeline_config.json")
    tracker = create_tracker(
        enabled=cfg.tracking.enabled,
        experiment_name=cfg.tracking.experiment_name,
        run_name=run_dir.name,
        tags={"pipeline": "ml"},
    )

    run_meta = {
        "config_path": str(config_path) if config_path else None,
        "config_dir": str(config_dir),
        "run_dir": str(run_dir),
        "log_path": str(run_dir / "training_log.txt") if cfg.run.log_to_file else None,
        "timestamp": run_timestamp,
    }

    def _execute() -> dict:
        logger.info("[RUN] output_dir=%s", run_dir)
        tracker.log_params(
            {
                "pipeline": "ml",
                "run_name": cfg.run.name or run_dir.name,
                "run_timestamp": run_timestamp,
                "git_sha": _resolve_git_sha() or "unknown",
                "features_mode": cfg.features.mode,
                "n_classifiers": len(cfg.train.classifier_names),
            }
        )
        tracker.log_artifact(run_dir / "pipeline_config.json", artifact_path="config")
        train_ds, val_ds, test_ds, resolved = load_datasets(
            cfg.dataset,
            config_dir,
            LOADER_REGISTRY,
        )
        run_meta["data_paths"] = resolved

        ensure_nonempty(train_ds, "train", cfg.dataset.train_query)
        if cfg.dataset.val_query or cfg.dataset.val_path:
            ensure_nonempty(val_ds, "val", cfg.dataset.val_query)
        if cfg.dataset.test_query or cfg.dataset.test_path:
            ensure_nonempty(test_ds, "test", cfg.dataset.test_query)

        if test_ds is None:
            raise ValueError(
                "ML pipeline requires a test split. Set dataset.test_query or dataset.test_path."
            )

        warn_label_coverage(train_ds, val_ds, test_ds, logger)
        save_json(run_meta, run_dir / "run_meta.json")
        tracker.log_artifact(run_dir / "run_meta.json", artifact_path="run")

        train_windows, test_windows = _build_windows(train_ds, test_ds, cfg)
        feat_names, x_train, y_train, x_test, y_test = _build_features(
            train_ds=train_ds,
            test_ds=test_ds,
            train_windows=train_windows,
            test_windows=test_windows,
            cfg=cfg,
            logger=logger,
        )

        save_json(
            {
                "mode": cfg.features.mode,
                "feature_names": feat_names,
                "n_features": len(feat_names),
                "selected_time_features": [
                    req.name for req in cfg.features.time_features
                ],
                "selected_freq_features": [
                    req.name for req in cfg.features.freq_features
                ],
            },
            run_dir / "feature_names.json",
        )
        tracker.log_artifact(run_dir / "feature_names.json", artifact_path="features")

        trained_models = train_ml_models(
            classifier_names=cfg.train.classifier_names,
            train_dataset=(x_train, y_train),
            test_dataset=(x_test, y_test),
            use_bayesian_search=cfg.train.use_bayesian_search,
            search_spaces=cfg.train.search_spaces,
            n_jobs=cfg.train.n_jobs,
            bayes_n_iter=cfg.train.bayes_n_iter,
            bayes_cv=cfg.train.bayes_cv,
            bayes_n_points=cfg.train.bayes_n_points,
            random_state=42,
            save_path=str(run_dir / "models"),
        )
        tracker.log_metrics({"models_trained": float(len(cfg.train.classifier_names))})

        evaluated = evaluate_ml_classification_models(
            trained_models,
            run_id=run_timestamp,
            split="test",
            pipeline="ml",
            artifacts_dir=run_dir / "results",
        )
        for model_name, result in evaluated.items():
            tracker.log_metrics(
                {
                    f"{model_name}.{metric_name}": float(metric_value)
                    for metric_name, metric_value in result.summary.metrics.items()
                }
            )
        _log_artifact_tree(tracker, run_dir / "results", artifact_path="results")

        return trained_models

    try:
        with redirect_output_to_logger(logger):
            return _execute()
    finally:
        tracker.close()


def _build_feature_config(cfg: MLPipelineConfig) -> ExtractionConfig:
    """Build extraction config based on selected ML feature mode."""
    mode = cfg.features.mode
    if mode in {"time", "time_freq"} and not cfg.features.time_features:
        raise ValueError(
            "features.time_features must be non-empty when features.mode includes time."
        )
    if mode in {"freq", "time_freq"} and not cfg.features.freq_features:
        raise ValueError(
            "features.freq_features must be non-empty when features.mode includes freq."
        )

    time_features = [
        FeatureRequest(name=req.name, params=dict(req.params))
        for req in cfg.features.time_features
    ]
    freq_features = [
        FeatureRequest(name=req.name, params=dict(req.params))
        for req in cfg.features.freq_features
    ]

    return ExtractionConfig(
        time_features=time_features,
        freq_features=freq_features,
    )


def _build_windows(train_ds, test_ds, cfg: MLPipelineConfig):
    """Create train/test windowed datasets using configured overlaps."""
    test_overlap = cfg.windowing.test_overlap
    if test_overlap is None:
        test_overlap = cfg.windowing.train_overlap

    train_windows = train_ds.window_dataset(
        window_size=cfg.windowing.size,
        overlap=cfg.windowing.train_overlap,
    )
    test_windows = test_ds.window_dataset(
        window_size=cfg.windowing.size,
        overlap=test_overlap,
    )
    return train_windows, test_windows


def _build_features(
    *,
    train_ds,
    test_ds,
    train_windows,
    test_windows,
    cfg: MLPipelineConfig,
    logger,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract and validate features for train/test splits."""
    mode = cfg.features.mode
    feature_cfg = _build_feature_config(cfg)

    if mode == "time":
        feat_names, x_train, y_train, x_test, y_test = _extract_train_test_branch(
            train_ds=train_ds,
            test_ds=test_ds,
            train_windows=train_windows,
            test_windows=test_windows,
            feature_cfg=feature_cfg,
            branch="time",
            logger=logger,
        )
    elif mode == "freq":
        feat_names, x_train, y_train, x_test, y_test = _extract_train_test_branch(
            train_ds=train_ds,
            test_ds=test_ds,
            train_windows=train_windows,
            test_windows=test_windows,
            feature_cfg=feature_cfg,
            branch="freq",
            logger=logger,
        )
    elif mode == "time_freq":
        feat_names_time, x_train_time, y_train, x_test_time, y_test = (
            _extract_train_test_branch(
                train_ds=train_ds,
                test_ds=test_ds,
                train_windows=train_windows,
                test_windows=test_windows,
                feature_cfg=feature_cfg,
                branch="time",
                logger=logger,
            )
        )

        feat_names_freq, x_train_freq, y_train_freq, x_test_freq, y_test_freq = (
            _extract_train_test_branch(
                train_ds=train_ds,
                test_ds=test_ds,
                train_windows=train_windows,
                test_windows=test_windows,
                feature_cfg=feature_cfg,
                branch="freq",
                logger=logger,
            )
        )

        _ensure_aligned_labels(
            split="train",
            first_name="time",
            first_labels=y_train,
            second_name="freq",
            second_labels=y_train_freq,
        )
        _ensure_aligned_labels(
            split="test",
            first_name="time",
            first_labels=y_test,
            second_name="freq",
            second_labels=y_test_freq,
        )

        x_train = np.concatenate([x_train_time, x_train_freq], axis=1)
        x_test = np.concatenate([x_test_time, x_test_freq], axis=1)
        feat_names = feat_names_time + feat_names_freq
    else:
        raise ValueError(f"Unsupported features.mode: {mode!r}")

    if x_train.shape[0] == 0 or x_test.shape[0] == 0:
        raise ValueError(
            "Feature extraction produced empty matrices. Check queries/windowing."
        )
    if x_train.shape[1] == 0:
        raise ValueError(
            "No feature columns were produced. Configure features for selected mode."
        )

    return feat_names, x_train, y_train, x_test, y_test


def _extract_train_test_branch(
    *,
    train_ds,
    test_ds,
    train_windows,
    test_windows,
    feature_cfg: ExtractionConfig,
    branch: str,
    logger,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract a single feature branch for train/test and warn on name mismatch."""
    feat_names, x_train, y_train = _extract_branch_features(
        ds=train_ds,
        windows=train_windows,
        feature_cfg=feature_cfg,
        branch=branch,
    )
    feat_names_test, x_test, y_test = _extract_branch_features(
        ds=test_ds,
        windows=test_windows,
        feature_cfg=feature_cfg,
        branch=branch,
    )
    if feat_names_test != feat_names:
        logger.warning(
            "Train/test %s feature name lists differ. Train=%d Test=%d",
            branch,
            len(feat_names),
            len(feat_names_test),
        )
    return feat_names, x_train, y_train, x_test, y_test


def _ensure_aligned_labels(
    *,
    split: str,
    first_name: str,
    first_labels: np.ndarray,
    second_name: str,
    second_labels: np.ndarray,
) -> None:
    """Ensure two label arrays are exactly aligned before feature concatenation."""
    if first_labels.shape != second_labels.shape or not np.array_equal(
        first_labels,
        second_labels,
    ):
        raise ValueError(
            f"Label alignment check failed for {split} split: "
            f"{first_name} labels are not aligned with {second_name} labels."
        )


def _extract_branch_features(
    *,
    ds,
    windows,
    feature_cfg: ExtractionConfig,
    branch: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Extract one feature branch ('time' or 'freq') for a dataset split."""
    if branch == "time":
        return ds.time_features_dataset(
            windows,
            feature_cfg,
            dtype=np.float32,
        )
    if branch == "freq":
        return ds.frequency_features_dataset(
            windows,
            feature_cfg,
            dtype=np.float32,
        )
    raise ValueError(f"Unsupported feature branch: {branch!r}")


def _resolve_git_sha() -> str | None:
    """Return current git HEAD SHA, or None if unavailable."""
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return output.strip() or None
    except Exception:
        return None


def _log_artifact_tree(tracker, directory: Path, *, artifact_path: str) -> None:
    """Log all files from one artifact directory if it exists."""
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.is_file():
            tracker.log_artifact(path, artifact_path=artifact_path)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
