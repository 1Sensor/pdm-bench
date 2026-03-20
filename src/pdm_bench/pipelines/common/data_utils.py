from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pdm_bench.signals.dataset import Dataset


if TYPE_CHECKING:
    import logging
    from collections.abc import Callable, Iterable

    from pdm_bench.pipelines.common.config import DatasetSpec
    from pdm_bench.signals.recordings import Recording


def load_datasets(
    spec: DatasetSpec,
    config_dir: Path,
    loader_registry: dict[str, Callable[[str], list[Recording]]],
) -> tuple[Dataset, Dataset | None, Dataset | None, dict[str, str | None]]:
    """Load train/val/test datasets based on the dataset spec.

    Args:
        spec: Dataset specification.
        config_dir: Base directory for resolving relative paths.
        loader_registry: Mapping of loader names to loader callables.

    Returns:
        Tuple of (train_ds, val_ds, test_ds, resolved_paths).
    """
    loader = _get_loader(spec.loader, loader_registry)
    root = _resolve_path(spec.root, config_dir)

    resolved_paths = {
        "root": str(root) if root else None,
        "train": None,
        "val": None,
        "test": None,
    }

    if root and _has_unresolved_env(str(root)):
        raise ValueError(
            f"Unresolved env var in dataset.root: {root}. "
            "Ensure the variable is exported or use an absolute path."
        )
    # TODO: #77 investigate multiple loader calls
    if spec.uses_split_paths() or spec.uses_split_loader_kwargs():
        train_recs, train_path = _load_split(
            loader,
            spec.train_path,
            root,
            config_dir,
            "train",
            _merge_loader_kwargs(spec.loader_kwargs, spec.train_loader_kwargs),
            path_required=True,
        )
        if train_recs is None:
            raise ValueError("dataset.train_path is required when using split paths.")
        if not train_recs:
            raise ValueError("Train split is empty. Check dataset paths or queries.")

        val_recs, val_path = _load_split(
            loader,
            spec.val_path,
            root,
            config_dir,
            "val",
            _merge_loader_kwargs(spec.loader_kwargs, spec.val_loader_kwargs),
            path_required=False,
        )
        test_recs, test_path = _load_split(
            loader,
            spec.test_path,
            root,
            config_dir,
            "test",
            _merge_loader_kwargs(spec.loader_kwargs, spec.test_loader_kwargs),
            path_required=False,
        )

        resolved_paths["train"] = str(train_path) if train_path else None
        resolved_paths["val"] = str(val_path) if val_path else None
        resolved_paths["test"] = str(test_path) if test_path else None

        label_map = _build_label_map(spec.label_map, [train_recs, val_recs, test_recs])
        train_ds = Dataset.from_recordings("train", train_recs, label_to_id=label_map)
        val_ds = (
            Dataset.from_recordings("val", val_recs, label_to_id=label_map)
            if val_recs
            else None
        )
        test_ds = (
            Dataset.from_recordings("test", test_recs, label_to_id=label_map)
            if test_recs
            else None
        )

        train_ds = _apply_query(train_ds, spec.train_query, "train")
        val_ds = _apply_query(val_ds, spec.val_query, "val")
        test_ds = _apply_query(test_ds, spec.test_query, "test")
        train_ds, val_ds, test_ds = _reencode_splits(
            train_ds,
            val_ds,
            test_ds,
            label_map=spec.label_map,
        )

        return train_ds, val_ds, test_ds, resolved_paths

    if root is None:
        raise ValueError("dataset.root is required when not using split paths.")
    if not root.exists():
        raise ValueError(
            f"dataset.root does not exist: {root}. "
            "Check the path or export required env vars."
        )

    recs = loader(str(root), **_merge_loader_kwargs(spec.loader_kwargs, None))
    if not recs:
        raise ValueError("Dataset is empty. Check dataset.root or loader.")

    label_map = _build_label_map(spec.label_map, [recs])
    base_ds = Dataset.from_recordings("base", recs, label_to_id=label_map)

    train_ds = _apply_query(base_ds, spec.train_query, "train")
    val_ds = _apply_query(base_ds, spec.val_query, "val") if spec.val_query else None
    test_ds = (
        _apply_query(base_ds, spec.test_query, "test") if spec.test_query else None
    )
    train_ds, val_ds, test_ds = _reencode_splits(
        train_ds,
        val_ds,
        test_ds,
        label_map=spec.label_map,
    )

    return train_ds, val_ds, test_ds, resolved_paths


def _get_loader(
    name: str, registry: dict[str, Callable[[str], list[Recording]]]
) -> Callable[[str], list[Recording]]:
    """Resolve a dataset loader from the registry."""
    loader = registry.get(name.lower())
    if loader is None:
        raise ValueError(f"Unknown dataset loader: {name}.")
    return loader


def _load_split(
    loader: Callable[[str], list[Recording]],
    path_str: str | None,
    root: Path | None,
    config_dir: Path,
    split: str,
    loader_kwargs: dict[str, object],
    *,
    path_required: bool,
) -> tuple[list | None, Path | None]:
    """Load a dataset split from a path or return None if not set.

    Args:
        path_required: When True, a split path must be provided unless loader_kwargs
            are used to load from the dataset root.
    """
    if not path_str and not path_required:
        if not loader_kwargs:
            return None, None
        if root is None:
            raise ValueError(f"Missing {split} dataset root.")
        return loader(str(root), **loader_kwargs), root

    base_dir = root if root else config_dir
    split_path = _resolve_path(path_str, base_dir) if path_str else root
    if split_path is None:
        raise ValueError(f"Missing {split} dataset path.")
    return loader(str(split_path), **loader_kwargs), split_path


def _resolve_path(path_str: str | None, base_dir: Path | None) -> Path | None:
    """Resolve a possibly relative path against a base directory."""
    if path_str is None:
        return None
    expanded = Path(os.path.expandvars(path_str)).expanduser()
    if not expanded.is_absolute() and base_dir is not None:
        return base_dir / expanded
    return expanded


def _has_unresolved_env(value: str) -> bool:
    """Return True if the string contains unresolved env placeholders."""
    return re.search(r"\$(\w+|\{[^}]+\})", value) is not None


def _build_label_map(
    label_map: dict[str, int] | None,
    recordings: Iterable[list | None],
) -> dict[str, int]:
    """Build a label->id mapping from provided recordings."""
    labels = sorted({rec.label for recs in recordings if recs for rec in recs})
    if label_map is None:
        return {label: idx for idx, label in enumerate(labels)}

    missing = [label for label in labels if label not in label_map]
    if missing:
        raise ValueError(f"label_map missing labels: {missing}")

    return label_map


def _reencode_splits(
    train_ds: Dataset,
    val_ds: Dataset | None,
    test_ds: Dataset | None,
    *,
    label_map: dict[str, int] | None,
) -> tuple[Dataset, Dataset | None, Dataset | None]:
    """Rebuild split datasets with a compact label map when no explicit map is provided."""
    if label_map is not None:
        return train_ds, val_ds, test_ds

    compact_map = _build_label_map(
        None,
        [
            train_ds.recordings,
            val_ds.recordings if val_ds is not None else None,
            test_ds.recordings if test_ds is not None else None,
        ],
    )

    train_ds = Dataset.from_recordings(
        train_ds.name,
        train_ds.recordings,
        label_to_id=compact_map,
    )
    val_ds = (
        Dataset.from_recordings(
            val_ds.name,
            val_ds.recordings,
            label_to_id=compact_map,
        )
        if val_ds is not None
        else None
    )
    test_ds = (
        Dataset.from_recordings(
            test_ds.name,
            test_ds.recordings,
            label_to_id=compact_map,
        )
        if test_ds is not None
        else None
    )
    return train_ds, val_ds, test_ds


def _apply_query(ds: Dataset | None, query: str | None, name: str) -> Dataset | None:
    """Apply a pandas query to a dataset and return a subset."""
    if ds is None:
        return None
    if not query:
        return ds
    if ds.meta.empty:
        raise ValueError(f"{name} split is empty before query: {query}")
    try:
        ids = ds.meta.query(query).index
    except Exception as exc:
        raise ValueError(f"Invalid {name}_query: {query} ({exc})") from exc
    if len(ids) == 0:
        raise ValueError(f"{name} split is empty after query: {query}")
    return ds.subset_by_ids(ids, name)


def ensure_nonempty(ds: Dataset | None, split: str, query: str | None) -> None:
    """Raise if a dataset split is empty."""
    if ds is None:
        return
    if ds.meta.empty:
        if query:
            raise ValueError(f"{split} split is empty after query: {query}")
        raise ValueError(f"{split} split is empty. Check dataset paths or loader.")


def warn_label_coverage(
    train_ds: Dataset,
    val_ds: Dataset | None,
    test_ds: Dataset | None,
    logger: logging.Logger,
) -> None:
    """Log a warning if val/test contain labels unseen in train."""
    train_labels = (
        set(train_ds.meta["label"].unique()) if not train_ds.meta.empty else set()
    )

    for split_name, ds in [("val", val_ds), ("test", test_ds)]:
        if ds is None or ds.meta.empty:
            continue
        labels = set(ds.meta["label"].unique())
        missing = sorted(labels - train_labels)
        if missing:
            logger.warning("%s labels not in train split: %s", split_name, missing)


def _merge_loader_kwargs(
    base: dict[str, object] | None, override: dict[str, object] | None
) -> dict[str, object]:
    merged: dict[str, object] = {}
    if base:
        merged.update(base)
    if override:
        merged.update(override)
    sensors = merged.get("sensors")
    if isinstance(sensors, list):
        merged["sensors"] = tuple(sensors)
    return merged
