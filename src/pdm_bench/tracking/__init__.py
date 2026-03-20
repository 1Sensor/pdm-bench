from __future__ import annotations

from typing import TYPE_CHECKING

from .mlflow_tracker import MlflowTracker
from .noop_tracker import NoopTracker
from .tracker import Tracker


if TYPE_CHECKING:
    from collections.abc import Mapping


def create_tracker(
    *,
    enabled: bool,
    tracking_uri: str | None = None,
    experiment_name: str = "pdm-bench",
    run_name: str | None = None,
    tags: Mapping[str, str] | None = None,
) -> Tracker:
    """Create tracker backend for current run.

    When disabled, returns no-op tracker. When enabled, returns MLflow tracker
    using local defaults unless overrides are provided.
    """
    if not enabled:
        return NoopTracker()
    return MlflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        tags=dict(tags) if tags is not None else None,
    )


__all__ = ["create_tracker", "Tracker", "NoopTracker", "MlflowTracker"]  # noqa: RUF022
