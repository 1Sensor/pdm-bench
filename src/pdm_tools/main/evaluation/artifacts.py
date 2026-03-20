from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from .schemas import (
        EvaluationResult,
        EvaluationSummary,
        PredictionArtifact,
        TrainingTelemetry,
    )


def _json_default(value: Any) -> Any:
    """Convert non-standard scalar/array values to JSON-serializable objects."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _save_json(payload: dict, path: Path | str) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return out_path


def save_prediction_artifact(artifact: PredictionArtifact, path: Path | str) -> Path:
    """Serialize prediction artifact to one JSON file."""
    return _save_json(asdict(artifact), path)


def save_evaluation_summary(summary: EvaluationSummary, path: Path | str) -> Path:
    """Serialize evaluation summary to one JSON file."""
    return _save_json(asdict(summary), path)


def save_training_telemetry(telemetry: TrainingTelemetry, path: Path | str) -> Path:
    """Serialize training telemetry to one JSON file."""
    return _save_json(asdict(telemetry), path)


def save_evaluation_result(
    result: EvaluationResult,
    out_dir: Path | str,
    *,
    stem: str = "evaluation",
) -> dict[str, Path]:
    """Serialize an evaluation result into canonical artifact files."""
    directory = Path(out_dir)
    written: dict[str, Path] = {
        "predictions": save_prediction_artifact(
            result.predictions, directory / f"{stem}_predictions.json"
        ),
        "summary": save_evaluation_summary(
            result.summary, directory / f"{stem}_summary.json"
        ),
    }
    if result.telemetry is not None:
        written["telemetry"] = save_training_telemetry(
            result.telemetry, directory / f"{stem}_telemetry.json"
        )
    return written
