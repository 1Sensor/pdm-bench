from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from pathlib import Path


class Tracker(Protocol):
    """Minimal tracking contract used by pipeline orchestration code."""

    def log_params(self, params: dict[str, object]) -> None:
        """Log run-level parameters."""

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log numeric metrics, optionally bound to a step."""

    def log_artifact(self, path: Path | str, artifact_path: str | None = None) -> None:
        """Log one local artifact path."""

    def close(self) -> None:
        """Close resources / run context."""
