from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pdm_bench.pipelines.common import data_utils
from pdm_bench.pipelines.common.data_utils import (
    _load_split,
    ensure_nonempty,
    load_datasets,
    warn_label_coverage,
)
from pdm_bench.pipelines.common.config import DatasetSpec
from pdm_bench.signals.dataset import Dataset
from pdm_bench.signals.recordings import Recording


if TYPE_CHECKING:
    from pathlib import Path


def _make_recording(rid: str, label: str, rpm: float) -> Recording:
    data = np.zeros((1, 8), dtype=np.float32)
    return Recording(
        rid=rid,
        data=data,
        fs=1.0,
        label=label,
        source="test",
        unit="u",
        channels=["ch0"],
        rpm=rpm,
    )


def test_load_datasets_root_queries(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    recs = [
        _make_recording("r1", "a", 1000),
        _make_recording("r2", "b", 2000),
    ]

    def loader(_path: str):
        return recs

    spec = DatasetSpec(
        loader="fake",
        root=str(root),
        train_query="rpm == 1000",
        test_query="rpm == 2000",
    )

    train_ds, val_ds, test_ds, resolved = load_datasets(
        spec, tmp_path, {"fake": loader}
    )

    assert len(train_ds.recordings) == 1
    assert train_ds.recordings[0].label == "a"
    assert val_ds is None
    assert len(test_ds.recordings) == 1
    assert resolved["root"] == str(root)


def test_load_datasets_compacts_filtered_label_ids(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    recs = [
        _make_recording("r1", "a", 1000),
        _make_recording("r2", "b", 1000),
        _make_recording("r3", "d", 1000),
        _make_recording("r4", "c", 2000),
    ]

    def loader(_path: str):
        return recs

    spec = DatasetSpec(
        loader="fake",
        root=str(root),
        train_query="rpm == 1000",
        test_query="rpm == 1000",
    )

    train_ds, _val_ds, test_ds, _resolved = load_datasets(
        spec, tmp_path, {"fake": loader}
    )

    assert train_ds.label_to_id == {"a": 0, "b": 1, "d": 2}
    assert test_ds.label_to_id == {"a": 0, "b": 1, "d": 2}
    assert sorted(train_ds.meta["label_id"].unique().tolist()) == [0, 1, 2]


def test_load_datasets_invalid_query(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    recs = [_make_recording("r1", "a", 1000)]

    def loader(_path: str):
        return recs

    spec = DatasetSpec(
        loader="fake",
        root=str(root),
        train_query="missing == 1",
    )

    with pytest.raises(ValueError, match="Invalid train_query"):
        load_datasets(spec, tmp_path, {"fake": loader})


def test_load_datasets_unresolved_env(tmp_path: Path):
    spec = DatasetSpec(
        loader="fake",
        root="$MISSING_ENV/some",
    )

    with pytest.raises(ValueError, match="Unresolved env var"):
        load_datasets(spec, tmp_path, {"fake": lambda _path: []})


def test_load_datasets_split_paths(tmp_path: Path):
    recs_train = [_make_recording("r1", "a", 1000)]
    recs_test = [_make_recording("r2", "b", 2000)]

    def loader(path: str):
        if path.endswith("train"):
            return recs_train
        if path.endswith("test"):
            return recs_test
        return []

    spec = DatasetSpec(
        loader="fake",
        root=str(tmp_path),
        train_path="train",
        test_path="test",
    )

    train_ds, val_ds, test_ds, resolved = load_datasets(
        spec, tmp_path, {"fake": loader}
    )

    assert len(train_ds.recordings) == 1
    assert val_ds is None
    assert len(test_ds.recordings) == 1
    assert resolved["train"] == str(tmp_path / "train")
    assert resolved["test"] == str(tmp_path / "test")


def test_load_datasets_split_loader_kwargs(tmp_path: Path):
    recs_train = [_make_recording("r1", "a", 1000)]
    recs_test = [_make_recording("r2", "b", 2000)]

    def loader(path: str, **kwargs):
        assert path == str(tmp_path)
        sensors = kwargs.get("sensors")
        if sensors == ("DE",):
            return recs_train
        if sensors == ("FE",):
            return recs_test
        return []

    spec = DatasetSpec(
        loader="fake",
        root=str(tmp_path),
        train_loader_kwargs={"sensors": ("DE",)},
        test_loader_kwargs={"sensors": ("FE",)},
    )

    train_ds, val_ds, test_ds, resolved = load_datasets(
        spec, tmp_path, {"fake": loader}
    )

    assert len(train_ds.recordings) == 1
    assert val_ds is None
    assert len(test_ds.recordings) == 1
    assert resolved["train"] == str(tmp_path)
    assert resolved["test"] == str(tmp_path)


def test_load_datasets_label_map_missing(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    recs = [_make_recording("r1", "a", 1000)]

    def loader(_path: str):
        return recs

    spec = DatasetSpec(
        loader="fake",
        root=str(root),
        label_map={"b": 0},
    )

    with pytest.raises(ValueError, match="label_map missing labels"):
        load_datasets(spec, tmp_path, {"fake": loader})


def test_load_datasets_root_missing(tmp_path: Path):
    spec = DatasetSpec(
        loader="fake",
        root=str(tmp_path / "missing"),
    )

    with pytest.raises(ValueError, match="dataset.root does not exist"):
        load_datasets(spec, tmp_path, {"fake": lambda _path: []})


def test_load_datasets_unknown_loader(tmp_path: Path):
    spec = DatasetSpec(loader="missing", root=str(tmp_path))
    with pytest.raises(ValueError, match="Unknown dataset loader"):
        load_datasets(spec, tmp_path, {})


def test_load_datasets_empty_train_split(tmp_path: Path):
    def loader(_path: str):
        return []

    spec = DatasetSpec(
        loader="fake",
        root=str(tmp_path),
        train_path="train",
    )

    with pytest.raises(ValueError, match="Train split is empty"):
        load_datasets(spec, tmp_path, {"fake": loader})


def test_load_datasets_requires_root_when_no_split_paths(tmp_path: Path):
    spec = DatasetSpec(loader="fake")
    with pytest.raises(ValueError, match="dataset.root is required"):
        load_datasets(spec, tmp_path, {"fake": lambda _path: []})


def test_load_datasets_empty_dataset(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()

    def loader(_path: str):
        return []

    spec = DatasetSpec(loader="fake", root=str(root))
    with pytest.raises(ValueError, match="Dataset is empty"):
        load_datasets(spec, tmp_path, {"fake": loader})


def test_load_split_none_returns_none(tmp_path: Path):
    def loader(_path: str):
        return []

    recs, path = _load_split(
        loader,
        None,
        tmp_path,
        tmp_path,
        "val",
        {},
        path_required=False,
    )
    assert recs is None
    assert path is None


def test_load_split_missing_path_raises(monkeypatch, tmp_path: Path):
    def loader(_path: str):
        return []

    monkeypatch.setattr(data_utils, "_resolve_path", lambda *_: None)
    with pytest.raises(ValueError, match="Missing test dataset path"):
        _load_split(
            loader,
            "test",
            tmp_path,
            tmp_path,
            "test",
            {},
            path_required=True,
        )


def test_load_split_uses_loader_kwargs_when_no_path(tmp_path: Path):
    recs = [_make_recording("r1", "a", 1000)]

    def loader(path: str, **kwargs):
        assert path == str(tmp_path)
        assert kwargs == {"sensors": ("DE",)}
        return recs

    loaded, path = _load_split(
        loader,
        None,
        tmp_path,
        tmp_path,
        "test",
        {"sensors": ("DE",)},
        path_required=False,
    )

    assert loaded == recs
    assert path == tmp_path


def testensure_nonempty_raises_for_empty_dataset():
    empty = Dataset.from_recordings("empty", [])
    with pytest.raises(ValueError, match="split is empty"):
        ensure_nonempty(empty, "train", None)


def testensure_nonempty_none_is_noop():
    ensure_nonempty(None, "train", None)


def testwarn_label_coverage_logs(caplog):
    train = Dataset.from_recordings("train", [_make_recording("r1", "a", 1000)])
    val = Dataset.from_recordings("val", [_make_recording("r2", "b", 2000)])

    caplog.set_level(logging.WARNING)
    warn_label_coverage(train, val, None, logging.getLogger("test"))

    assert "labels not in train split" in caplog.text

