from __future__ import annotations

from typing import TYPE_CHECKING

from pdm_bench.tracking import MlflowTracker, create_tracker


if TYPE_CHECKING:
    from pathlib import Path


class _FakeMlflow:
    def __init__(self) -> None:
        self.tracking_uris: list[str] = []
        self.experiments: list[str] = []
        self.started_runs: list[tuple[str | None, dict[str, str] | None]] = []
        self.logged_params: list[dict[str, object]] = []
        self.logged_metrics: list[tuple[dict[str, float], int | None]] = []
        self.logged_artifacts: list[tuple[str, str | None]] = []
        self.ended_runs = 0

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uris.append(uri)

    def set_experiment(self, name: str) -> None:
        self.experiments.append(name)

    def start_run(
        self, run_name: str | None = None, tags: dict[str, str] | None = None
    ) -> None:
        self.started_runs.append((run_name, tags))

    def log_params(self, params: dict[str, object]) -> None:
        self.logged_params.append(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.logged_metrics.append((metrics, step))

    def log_artifact(self, path: str, artifact_path: str | None = None) -> None:
        self.logged_artifacts.append((path, artifact_path))

    def end_run(self) -> None:
        self.ended_runs += 1


def test_mlflow_tracker_logs_lifecycle_and_payloads(monkeypatch, tmp_path: Path):
    fake = _FakeMlflow()

    import pdm_bench.tracking.mlflow_tracker as mlflow_tracker_module

    monkeypatch.setattr(
        mlflow_tracker_module.importlib,
        "import_module",
        lambda name: fake,
    )

    tracker = MlflowTracker(
        tracking_uri="file:./mlruns-test",
        experiment_name="pdm-tools-test",
        run_name="run-001",
        tags={"pipeline": "ml"},
    )

    artifact = tmp_path / "summary.json"
    artifact.write_text("{}", encoding="utf-8")

    tracker.log_params({"k": "v", "obj": object(), "n": 1})
    tracker.log_metrics({"accuracy": 0.95}, step=2)
    tracker.log_artifact(artifact, artifact_path="results")
    tracker.close()
    tracker.close()

    assert fake.tracking_uris == ["file:./mlruns-test"]
    assert fake.experiments == ["pdm-tools-test"]
    assert fake.started_runs == [("run-001", {"pipeline": "ml"})]
    assert fake.logged_params
    assert fake.logged_params[0]["k"] == "v"
    assert fake.logged_params[0]["n"] == 1
    assert isinstance(fake.logged_params[0]["obj"], str)
    assert fake.logged_metrics == [({"accuracy": 0.95}, 2)]
    assert fake.logged_artifacts == [(str(artifact), "results")]
    assert fake.ended_runs == 1


def test_create_tracker_returns_mlflow_when_enabled(monkeypatch):
    fake = _FakeMlflow()

    import pdm_bench.tracking.mlflow_tracker as mlflow_tracker_module

    monkeypatch.setattr(
        mlflow_tracker_module.importlib,
        "import_module",
        lambda name: fake,
    )

    tracker = create_tracker(
        enabled=True,
        tracking_uri="file:./mlruns-factory",
        experiment_name="pdm-tools-factory",
        run_name="factory-run",
        tags={"pipeline": "dl"},
    )

    assert isinstance(tracker, MlflowTracker)
    tracker.close()
    assert fake.tracking_uris == ["file:./mlruns-factory"]
    assert fake.experiments == ["pdm-tools-factory"]
    assert fake.started_runs == [("factory-run", {"pipeline": "dl"})]
    assert fake.ended_runs == 1
