from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


class NoopTracker:
    """Tracker implementation that intentionally performs no actions."""

    def log_params(self, params: dict[str, object]) -> None:
        _ = params

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        _ = (metrics, step)

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        _ = (path, artifact_path)

    def close(self) -> None:
        return None
