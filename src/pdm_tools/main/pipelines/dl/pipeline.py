from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from pdm_tools.main.evaluation.dl import evaluate_dl_classification_models
from pdm_tools.main.loaders.cwru import load_cwru_dataset
from pdm_tools.main.loaders.pu import load_pu_dataset
from pdm_tools.main.pipelines.common.data_utils import (
    ensure_nonempty,
    load_datasets,
    warn_label_coverage,
)
from pdm_tools.main.pipelines.common.io_utils import (
    configure_logging,
    load_env_file,
    prepare_run_dir,
    redirect_output_to_logger,
)
from pdm_tools.main.pipelines.dl.config import (
    DLPipelineConfig,
)
from pdm_tools.main.tracking import create_tracker
from pdm_tools.main.training.dl.engine import train_dl_models
from pdm_tools.main.training.dl.utils import save_json


if TYPE_CHECKING:
    from pdm_tools.main.signals.dataset import Dataset

LOADER_REGISTRY = {
    "cwru": load_cwru_dataset,
    "pu": load_pu_dataset,
}


def run_dl_pipeline_from_dict(
    data: dict[str, Any],
    *,
    config_dir: Path | None = None,
    config_path: Path | None = None,
) -> dict:
    """Run the DL pipeline using a resolved config dict.

    Args:
        data: Configuration mapping (Hydra-resolved).
        config_dir: Base directory for resolving relative paths.
        config_path: Optional path to the source config (for metadata).

    Returns:
        Dict mapping model names to outputs produced by training.
    """
    if config_dir is not None:
        load_env_file(Path(".env"))
        load_env_file(Path(config_dir) / ".env")
    cfg = DLPipelineConfig.from_dict(data)
    return _run_dl_pipeline(cfg, config_path=config_path, config_dir=config_dir)


@hydra.main(
    version_base="1.3", config_path="../../../../../config", config_name="config"
)
def main(cfg) -> int:
    """Hydra entrypoint for the DL pipeline.

    Args:
        cfg: Hydra config object for the pipeline.

    Returns:
        Exit code (0 on success).
    """
    config_dir = Path(__file__).resolve().parents[5] / "config"

    original_cwd = Path(hydra.utils.get_original_cwd())
    load_env_file(original_cwd / ".env")
    load_env_file(config_dir / ".env")

    data = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(data, dict):
        raise ValueError("Hydra config root must be a mapping.")
    data.pop("hydra", None)

    run_dl_pipeline_from_dict(
        data, config_dir=config_dir, config_path=config_dir / "config.yaml"
    )
    return 0


def _run_dl_pipeline(
    cfg: DLPipelineConfig,
    *,
    config_path: Path | None,
    config_dir: Path | None,
) -> dict:
    """Execute the pipeline from a parsed config.

    Args:
        cfg: Parsed pipeline configuration.
        config_path: Optional path to the source config (for metadata).
        config_dir: Base directory for resolving relative paths.

    Returns:
        Dict mapping model names to outputs produced by training.
    """
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
        tags={"pipeline": "dl"},
    )

    run_meta = {
        "config_path": str(config_path) if config_path else None,
        "config_dir": str(config_dir),
        "run_dir": str(run_dir),
        "log_path": str(run_dir / "training_log.txt") if cfg.run.log_to_file else None,
        "timestamp": run_timestamp,
    }

    def _execute():
        logger.info("[RUN] output_dir=%s", run_dir)
        tracker.log_params(
            {
                "pipeline": "dl",
                "run_name": cfg.run.name or run_dir.name,
                "run_timestamp": run_timestamp,
                "git_sha": _resolve_git_sha() or "unknown",
                "n_models": len(cfg.models),
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

        warn_label_coverage(train_ds, val_ds, test_ds, logger)

        save_json(run_meta, run_dir / "run_meta.json")
        tracker.log_artifact(run_dir / "run_meta.json", artifact_path="run")

        train_views, val_views, test_views = _build_views(
            train_ds,
            val_ds,
            test_ds,
            cfg,
        )

        results = train_dl_models(
            model_names=cfg.models,
            train_views=train_views,
            val_views=val_views,
            test_views=test_views,
            n_classes=train_ds.n_classes,
            cfg=cfg.train,
            save_path=str(run_dir / "models"),
        )
        eval_views = test_views
        eval_split = "test"
        eval_pipeline = "dl"

        tracker.log_metrics({"models_trained": float(len(results))})
        _log_training_telemetry_from_models_dir(tracker, run_dir / "models")
        _log_artifact_tree(tracker, run_dir / "models", artifact_path="models")

        if eval_views:
            persist_canonical = (
                cfg.artifacts.save_predictions or cfg.artifacts.save_confusion_matrix
            )
            evaluated = evaluate_dl_classification_models(
                results,
                eval_views=eval_views,
                run_id=run_timestamp,
                split=eval_split,
                pipeline=eval_pipeline,
                artifacts_dir=(run_dir / "results") if persist_canonical else None,
                device=cfg.train.device,
                batch_size=cfg.train.batch_size,
                num_workers=cfg.train.num_workers,
                include_probabilities=cfg.artifacts.save_probs,
            )
            for model_name, result in evaluated.items():
                tracker.log_metrics(
                    {
                        f"{model_name}.{metric_name}": float(metric_value)
                        for metric_name, metric_value in result.summary.metrics.items()
                    }
                )
            _log_artifact_tree(tracker, run_dir / "results", artifact_path="results")

        return results

    try:
        with redirect_output_to_logger(logger):
            return _execute()
    finally:
        tracker.close()


def _build_views(
    train_ds: Dataset,
    val_ds: Dataset | None,
    test_ds: Dataset | None,
    cfg: DLPipelineConfig,
):
    """Create torch dataset views for each split."""
    train_overlap = cfg.windowing.train_overlap
    val_overlap = cfg.windowing.val_overlap
    test_overlap = cfg.windowing.test_overlap

    if val_overlap is None:
        val_overlap = train_overlap
    if test_overlap is None:
        test_overlap = train_overlap

    train_windows = train_ds.window_dataset(cfg.windowing.size, train_overlap)
    train_views = train_ds.torch_dataset(train_windows, flatten=cfg.views.flatten)

    val_views = None
    if val_ds is not None:
        val_windows = val_ds.window_dataset(cfg.windowing.size, val_overlap)
        val_views = val_ds.torch_dataset(val_windows, flatten=cfg.views.flatten)

    test_views = None
    if test_ds is not None:
        test_windows = test_ds.window_dataset(cfg.windowing.size, test_overlap)
        test_views = test_ds.torch_dataset(test_windows, flatten=cfg.views.flatten)

    return train_views, val_views, test_views


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


def _log_training_telemetry_from_models_dir(tracker, models_dir: Path) -> None:
    """Log step metrics from trainer-exported metrics JSON files."""
    if not models_dir.exists():
        return

    for path in models_dir.glob("*_metrics_*.json"):
        model_name = path.name.split("_metrics_")[0]
        payload = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(payload.get("epoch"), list):
            epochs = payload["epoch"]
            for idx, epoch in enumerate(epochs):
                step_metrics: dict[str, float] = {}
                for key in ("train_acc", "train_loss", "val_acc", "val_loss"):
                    values = payload.get(key)
                    if isinstance(values, list) and idx < len(values):
                        value = values[idx]
                        if value is not None:
                            step_metrics[f"{model_name}.{key}"] = float(value)
                if step_metrics:
                    tracker.log_metrics(step_metrics, step=int(epoch))

        history = payload.get("history")
        if isinstance(history, list):
            for row in history:
                if not isinstance(row, dict):
                    continue
                step = row.get("epoch")
                step_metrics = {
                    f"{model_name}.{key}": float(value)
                    for key, value in row.items()
                    if key != "epoch" and value is not None
                }
                if step_metrics:
                    tracker.log_metrics(step_metrics, step=int(step) if step else None)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
