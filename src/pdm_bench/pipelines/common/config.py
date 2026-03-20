from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any


DEFAULT_LOGGER_NAME = "pdm_bench.pipelines"


def warn_unknown(
    section: str,
    data: dict[str, Any],
    allowed: set[str],
    *,
    logger_name: str = DEFAULT_LOGGER_NAME,
) -> None:
    """Log a warning when config sections contain unknown keys."""
    unknown = set(data) - set(allowed)
    if unknown:
        logging.getLogger(logger_name).warning(
            "Unknown keys in '%s': %s", section, sorted(unknown)
        )


def as_bool(value: object, *, default: bool = False) -> bool:
    """Parse a bool-like value from config inputs."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"true", "1", "yes", "y", "on"}:
            return True
        if norm in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"Expected bool, got {value!r}.")


@dataclass(frozen=True)
class RunSpec:
    """Run-level configuration for output and logging."""

    name: str = ""
    output_dir: str = "artifacts/dl"
    log_to_file: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunSpec:
        """Create a RunSpec from a dict."""
        allowed = {"name", "output_dir", "log_to_file"}
        warn_unknown("run", data, allowed)
        return cls(
            name=str(data.get("name", "")),
            output_dir=str(data.get("output_dir", "artifacts/dl")),
            log_to_file=as_bool(data.get("log_to_file", True)),
        )


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset loading configuration."""

    loader: str
    root: str | None = None
    train_path: str | None = None
    val_path: str | None = None
    test_path: str | None = None
    train_query: str | None = None
    val_query: str | None = None
    test_query: str | None = None
    label_map: dict[str, int] | None = None
    loader_kwargs: dict[str, Any] = field(default_factory=dict)
    train_loader_kwargs: dict[str, Any] | None = None
    val_loader_kwargs: dict[str, Any] | None = None
    test_loader_kwargs: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetSpec:
        """Create a DatasetSpec from a dict."""
        allowed = {
            "loader",
            "root",
            "path",
            "train_path",
            "val_path",
            "test_path",
            "train_query",
            "val_query",
            "test_query",
            "label_map",
            "loader_kwargs",
            "train_loader_kwargs",
            "val_loader_kwargs",
            "test_loader_kwargs",
        }
        warn_unknown("dataset", data, allowed)

        loader = data.get("loader")
        if not loader:
            raise ValueError("dataset.loader is required.")

        root = data.get("root")
        if root is None and data.get("path") is not None:
            root = data.get("path")

        label_map = data.get("label_map")
        if label_map is not None:
            label_map = {str(k): int(v) for k, v in label_map.items()}

        loader_kwargs = data.get("loader_kwargs") or {}
        if not isinstance(loader_kwargs, dict):
            raise ValueError("dataset.loader_kwargs must be a mapping.")

        train_loader_kwargs = data.get("train_loader_kwargs")
        if train_loader_kwargs is not None and not isinstance(
            train_loader_kwargs, dict
        ):
            raise ValueError("dataset.train_loader_kwargs must be a mapping.")

        val_loader_kwargs = data.get("val_loader_kwargs")
        if val_loader_kwargs is not None and not isinstance(val_loader_kwargs, dict):
            raise ValueError("dataset.val_loader_kwargs must be a mapping.")

        test_loader_kwargs = data.get("test_loader_kwargs")
        if test_loader_kwargs is not None and not isinstance(test_loader_kwargs, dict):
            raise ValueError("dataset.test_loader_kwargs must be a mapping.")

        return cls(
            loader=str(loader),
            root=str(root) if root is not None else None,
            train_path=str(data.get("train_path")) if data.get("train_path") else None,
            val_path=str(data.get("val_path")) if data.get("val_path") else None,
            test_path=str(data.get("test_path")) if data.get("test_path") else None,
            train_query=(
                str(data.get("train_query")) if data.get("train_query") else None
            ),
            val_query=str(data.get("val_query")) if data.get("val_query") else None,
            test_query=str(data.get("test_query")) if data.get("test_query") else None,
            label_map=label_map,
            loader_kwargs=dict(loader_kwargs),
            train_loader_kwargs=(
                dict(train_loader_kwargs) if train_loader_kwargs else None
            ),
            val_loader_kwargs=(dict(val_loader_kwargs) if val_loader_kwargs else None),
            test_loader_kwargs=(
                dict(test_loader_kwargs) if test_loader_kwargs else None
            ),
        )

    def uses_split_paths(self) -> bool:
        """Return True when any split path is configured."""
        return any([self.train_path, self.val_path, self.test_path])

    def uses_split_loader_kwargs(self) -> bool:
        """Return True when any split-specific loader kwargs are configured."""
        return any(
            [
                self.train_loader_kwargs,
                self.val_loader_kwargs,
                self.test_loader_kwargs,
            ]
        )


@dataclass(frozen=True)
class WindowingSpec:
    """Windowing configuration for dataset splits."""

    size: int
    train_overlap: float = 0.75
    val_overlap: float | None = None
    test_overlap: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WindowingSpec:
        """Create a WindowingSpec from a dict."""
        allowed = {"size", "train_overlap", "val_overlap", "test_overlap"}
        warn_unknown("windowing", data, allowed)
        if "size" not in data:
            raise ValueError("windowing.size is required.")
        return cls(
            size=int(data["size"]),
            train_overlap=float(data.get("train_overlap", 0.75)),
            val_overlap=(
                float(data["val_overlap"])
                if data.get("val_overlap") is not None
                else None
            ),
            test_overlap=(
                float(data["test_overlap"])
                if data.get("test_overlap") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ArtifactsSpec:
    """Artifact output configuration."""

    save_predictions: bool = False
    save_probs: bool = False
    save_confusion_matrix: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactsSpec:
        """Create an ArtifactsSpec from a dict."""
        allowed = {"save_predictions", "save_probs", "save_confusion_matrix"}
        warn_unknown("artifacts", data, allowed)
        return cls(
            save_predictions=as_bool(data.get("save_predictions", False)),
            save_probs=as_bool(data.get("save_probs", False)),
            save_confusion_matrix=as_bool(data.get("save_confusion_matrix", True)),
        )


@dataclass(frozen=True)
class TrackingSpec:
    """Experiment tracking configuration."""

    enabled: bool = False
    experiment_name: str = "pdm-bench"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrackingSpec:
        """Create a TrackingSpec from a dict."""
        allowed = {"enabled", "experiment_name"}
        warn_unknown("tracking", data, allowed)

        return cls(
            enabled=as_bool(data.get("enabled", False)),
            experiment_name=str(data.get("experiment_name", "pdm-bench")),
        )
