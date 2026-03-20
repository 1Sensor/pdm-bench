from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from pdm_tools.main.pipelines.common.config import (
    ArtifactsSpec,
    DatasetSpec,
    RunSpec,
    TrackingSpec,
    WindowingSpec,
    as_bool,
    warn_unknown,
)
from pdm_tools.main.signals.features_config import FeatureRequest


if TYPE_CHECKING:
    from collections.abc import Iterable

VALID_FEATURE_MODES = {"time", "freq", "time_freq"}


def _warn_unknown(section: str, data: dict[str, Any], allowed: Iterable[str]) -> None:
    """Log a warning when config sections contain unknown keys."""
    warn_unknown(section, data, allowed, logger_name="pdm_tools.pipelines.ml")


def _as_bool(value: object, *, default: bool = False) -> bool:
    """Parse a bool-like value from config inputs."""
    return as_bool(value, default=default)


@dataclass(frozen=True)
class MLFeatureSpec:
    """Feature extraction plan for ML pipeline."""

    mode: str = "time_freq"
    time_features: list[FeatureRequest] = field(default_factory=list)
    freq_features: list[FeatureRequest] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLFeatureSpec:
        """Create a MLFeatureSpec from a dict."""
        allowed = {"mode", "time_features", "freq_features"}
        _warn_unknown("features", data, allowed)

        mode = str(data.get("mode", "time_freq")).lower()
        if mode not in VALID_FEATURE_MODES:
            expected = ", ".join(sorted(VALID_FEATURE_MODES))
            raise ValueError(
                f"Invalid features.mode: {mode!r}. Expected one of: {expected}."
            )

        time_features = _parse_feature_requests(
            "features.time_features", data.get("time_features")
        )
        freq_features = _parse_feature_requests(
            "features.freq_features", data.get("freq_features")
        )

        return cls(
            mode=mode,
            time_features=time_features,
            freq_features=freq_features,
        )


@dataclass(frozen=True)
class MLTrainSpec:
    """Training settings for train_ml_models."""

    classifier_names: list[str]
    use_bayesian_search: bool = False
    bayes_n_iter: int = 10
    bayes_cv: int = 3
    bayes_n_points: int = 1
    n_jobs: int = -1
    search_spaces: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLTrainSpec:
        """Create a MLTrainSpec from a dict."""
        allowed = {
            "classifier_names",
            "use_bayesian_search",
            "bayes_n_iter",
            "bayes_cv",
            "bayes_n_points",
            "n_jobs",
            "search_spaces",
        }
        _warn_unknown("train", data, allowed)

        classifier_names = data.get("classifier_names")
        if not isinstance(classifier_names, list) or not classifier_names:
            raise ValueError("train.classifier_names must be a non-empty list.")

        search_spaces = data.get("search_spaces")
        if search_spaces is not None and not isinstance(search_spaces, dict):
            raise ValueError("train.search_spaces must be a mapping.")

        return cls(
            classifier_names=[str(name) for name in classifier_names],
            use_bayesian_search=_as_bool(data.get("use_bayesian_search", False)),
            bayes_n_iter=int(data.get("bayes_n_iter", 10)),
            bayes_cv=int(data.get("bayes_cv", 3)),
            bayes_n_points=int(data.get("bayes_n_points", 1)),
            n_jobs=int(data.get("n_jobs", -1)),
            search_spaces=dict(search_spaces) if search_spaces is not None else None,
        )


@dataclass(frozen=True)
class MLPipelineConfig:
    """Top-level ML pipeline configuration contract."""

    run: RunSpec
    dataset: DatasetSpec
    windowing: WindowingSpec
    features: MLFeatureSpec
    train: MLTrainSpec
    artifacts: ArtifactsSpec
    tracking: TrackingSpec

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLPipelineConfig:
        """Create a MLPipelineConfig from a dict."""
        allowed = (
            "run",
            "dataset",
            "windowing",
            "features",
            "train",
            "artifacts",
            "tracking",
        )
        _warn_unknown("root", data, allowed)

        return cls(
            run=RunSpec.from_dict(data.get("run", {})),
            dataset=DatasetSpec.from_dict(data.get("dataset", {})),
            windowing=WindowingSpec.from_dict(data.get("windowing", {})),
            features=MLFeatureSpec.from_dict(data.get("features", {})),
            train=MLTrainSpec.from_dict(data.get("train", {})),
            artifacts=ArtifactsSpec.from_dict(data.get("artifacts", {})),
            tracking=TrackingSpec.from_dict(data.get("tracking", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the config to a JSON-friendly dict."""
        return {
            "run": asdict(self.run),
            "dataset": asdict(self.dataset),
            "windowing": asdict(self.windowing),
            "features": {
                "mode": self.features.mode,
                "time_features": [asdict(req) for req in self.features.time_features],
                "freq_features": [asdict(req) for req in self.features.freq_features],
            },
            "train": asdict(self.train),
            "artifacts": asdict(self.artifacts),
            "tracking": asdict(self.tracking),
        }


def _parse_feature_requests(
    section: str, data: list[dict[str, Any]] | None
) -> list[FeatureRequest]:
    """Parse list of feature requests."""
    if data is None:
        return []
    if not isinstance(data, list):
        raise ValueError(f"{section} must be a list.")

    requests: list[FeatureRequest] = []
    for index, raw in enumerate(data):
        item_section = f"{section}[{index}]"
        if isinstance(raw, str):
            requests.append(FeatureRequest(name=raw))
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"{item_section} must be an object or string.")
        requests.append(_parse_feature_request(item_section, raw))
    return requests


def _parse_feature_request(section: str, data: dict[str, Any]) -> FeatureRequest:
    """Parse one FeatureRequest from a mapping."""
    allowed = {"name", "params"}
    _warn_unknown(section, data, allowed)

    name = data.get("name")
    if not name:
        raise ValueError(f"{section}.name is required.")

    params = data.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError(f"{section}.params must be a mapping.")

    return FeatureRequest(name=str(name), params=dict(params))
