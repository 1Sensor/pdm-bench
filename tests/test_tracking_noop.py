from __future__ import annotations

from pdm_bench.tracking import NoopTracker, create_tracker


def test_noop_tracker_methods_are_safe(tmp_path):
    tracker = NoopTracker()
    tracker.log_params({"run_id": "r1", "n_estimators": 100})
    tracker.log_metrics({"accuracy": 0.92}, step=1)
    tracker.log_artifact(tmp_path / "artifact.txt", artifact_path="results")
    tracker.close()


def test_create_tracker_returns_noop_when_disabled():
    tracker = create_tracker(enabled=False)
    assert isinstance(tracker, NoopTracker)
