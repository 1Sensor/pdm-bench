# tests/signals/test_dataset.py
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pdm_bench.signals.dataset import Dataset
from pdm_bench.signals.features_config import ExtractionConfig, FeatureRequest
from pdm_bench.signals.recordings import WindowedRecording
from pdm_bench.signals.views import TorchWindowView


def test_from_recordings_and_meta_basic(make_recording):
    r1 = make_recording("a/1", 2, 8, label="L1", rpm=1500)
    r2 = make_recording("b/2", 2, 12, label="L2", rpm=3000)
    ds = Dataset.from_recordings("ds", [r1, r2])

    # core containers
    assert ds.name == "ds"
    assert len(ds.recordings) == 2
    # meta has expected columns and index of ids
    m = ds.meta
    assert set(  # noqa: C405
        ["label", "fs", "n_samples", "n_channels", "rpm", "source", "unit"]
    ).issubset(m.columns)
    assert list(m.index) == ["a/1", "b/2"]
    # total samples = 8 + 12
    assert ds.total_samples() == 20


def test_ids_and_get_lookup(make_recording):
    r1 = make_recording("x", 1, 4, label="ok")
    r2 = make_recording("y", 1, 6, label="fault")
    ds = Dataset.from_recordings("ds", [r1, r2])

    assert set(ds.ids) == {"x", "y"}
    assert ds.get("x") is r1
    assert ds.get("y") is r2
    with pytest.raises(KeyError):
        _ = ds.get("z")


def test_subset_by_ids_and_query(make_recording):
    r1 = make_recording("rid1", 2, 10, label="ball", rpm=1750)
    r2 = make_recording("rid2", 2, 10, label="inner", rpm=1750)
    r3 = make_recording("rid3", 2, 10, label="ball", rpm=1772)
    ds = Dataset.from_recordings("base", [r1, r2, r3])

    sub_ids = ds.subset_by_ids(["rid1", "rid3"], name="by_ids")
    assert sub_ids.name == "by_ids"
    assert {r.rid for r in sub_ids.recordings} == {"rid1", "rid3"}

    sub_q = ds.subset_query("label == 'ball' and rpm >= 1750")
    assert {r.rid for r in sub_q.recordings} == {"rid1", "rid3"}


def test_subset_by_ids_preserves_parent_label_mapping(make_recording):
    r1 = make_recording("rid1", 2, 10, label="ball")
    r2 = make_recording("rid2", 2, 10, label="inner")
    r3 = make_recording("rid3", 2, 10, label="outer")
    ds = Dataset.from_recordings(
        "base",
        [r1, r2, r3],
        label_to_id={"ball": 7, "inner": 3, "outer": 9},
    )

    sub = ds.subset_by_ids(["rid1", "rid2"], name="sub")

    assert sub.label_to_id == {"ball": 7, "inner": 3, "outer": 9}
    assert int(sub.meta.loc["rid1", "label_id"]) == 7
    assert int(sub.meta.loc["rid2", "label_id"]) == 3


def test_multiple_subsets_from_same_base_keep_identical_label_ids(make_recording):
    r1 = make_recording("rid1", 2, 10, label="ball")
    r2 = make_recording("rid2", 2, 10, label="inner")
    r3 = make_recording("rid3", 2, 10, label="outer")
    ds = Dataset.from_recordings("base", [r1, r2, r3])

    sub_ball = ds.subset_by_ids(["rid1"], name="ball_only")
    sub_inner = ds.subset_by_ids(["rid2"], name="inner_only")
    sub_outer = ds.subset_by_ids(["rid3"], name="outer_only")

    assert sub_ball.label_to_id == ds.label_to_id
    assert sub_inner.label_to_id == ds.label_to_id
    assert sub_outer.label_to_id == ds.label_to_id
    assert int(sub_ball.meta.loc["rid1", "label_id"]) == ds.label_to_id["ball"]
    assert int(sub_inner.meta.loc["rid2", "label_id"]) == ds.label_to_id["inner"]
    assert int(sub_outer.meta.loc["rid3", "label_id"]) == ds.label_to_id["outer"]


def test_subset_by_ids_empty_preserves_parent_label_mapping(make_recording):
    r1 = make_recording("rid1", 2, 10, label="ball")
    r2 = make_recording("rid2", 2, 10, label="inner")
    ds = Dataset.from_recordings(
        "base",
        [r1, r2],
        label_to_id={"ball": 0, "inner": 1},
    )

    sub = ds.subset_by_ids([], name="empty")

    assert sub.meta.empty
    assert sub.label_to_id == {"ball": 0, "inner": 1}


def test_from_recordings_raises_when_external_mapping_missing_labels(make_recording):
    r1 = make_recording("rid1", 2, 10, label="ball")
    r2 = make_recording("rid2", 2, 10, label="inner")

    with pytest.raises(ValueError, match="label_to_id missing labels"):
        Dataset.from_recordings("base", [r1, r2], label_to_id={"ball": 0})


def test_window_dataset_minimal_overlaps(make_recording):
    r = make_recording("rid", 2, 10)  # 10 samples, 2 channels
    ds = Dataset.from_recordings("ds", [r])

    # overlap = 0.0 -> hop = window_size
    w0 = ds.window_dataset(window_size=4, overlap=0.0)
    assert (
        isinstance(w0, list) and len(w0) == 1 and isinstance(w0[0], WindowedRecording)
    )
    wr = w0[0]
    # starts: 0, 4  (last_valid_start = 6, więc 8 już odpada)
    assert len(wr) == 2
    np.testing.assert_array_equal(wr[0], r.data[:, 0:4])
    np.testing.assert_array_equal(wr[1], r.data[:, 4:8])

    # overlap = 0.5 -> hop = window/2
    w1 = ds.window_dataset(window_size=4, overlap=0.5)
    wr1 = w1[0]
    assert len(wr1) == 4  # starts: 0,2,4,6
    np.testing.assert_array_equal(wr1[3], r.data[:, 6:10])


def test_window_dataset_short_sequence_gives_empty(make_recording):
    r_short = make_recording("ridS", 1, 3)  # shorter than window
    ds = Dataset.from_recordings("ds", [r_short])

    w = ds.window_dataset(window_size=4, overlap=0.5)
    assert len(w) == 1
    assert len(w[0]) == 0  # no valid windows


def test_window_dataset_invalid_params(make_recording):
    ds = Dataset.from_recordings("ds", [make_recording("rid", 1, 10)])
    with pytest.raises(ValueError):
        ds.window_dataset(window_size=0, overlap=0.0)
    with pytest.raises(ValueError):
        ds.window_dataset(window_size=4, overlap=1.0)


def test_subset_query_empty_meta():
    ds = Dataset.from_recordings("ds", [])
    sub = ds.subset_query("label == 'anything'")
    assert isinstance(sub, Dataset)
    assert len(sub.recordings) == 0
    assert sub.meta.empty


def test_dataset_summary(make_recording):
    r1 = make_recording("rid1", 2, 10, label="ball", rpm=1750)
    r2 = make_recording("rid2", 2, 10, label="inner", rpm=1750)
    r3 = make_recording("rid3", 2, 10, label="ball", rpm=1772)
    ds = Dataset.from_recordings("base", [r1, r2, r3])

    summary = ds.summary()
    assert isinstance(summary["counts"], pd.DataFrame)
    assert isinstance(summary["stats"], pd.DataFrame)

    counts = summary["counts"]
    assert set(counts.columns) == {"source", "label", "count"}
    assert counts["count"].sum() == 3

    stats = summary["stats"]
    assert set(stats.columns) == {"stat", "n_samples", "n_channels", "fs"}


def test_dataset_summary_empty():
    ds = Dataset.from_recordings("empty", [])
    summary = ds.summary()

    assert isinstance(summary, dict)
    assert set(summary.keys()) == {"counts", "stats"}
    assert all(isinstance(df, pd.DataFrame) for df in summary.values())


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_time_frequency_features_dataset_basic(make_recording):
    """
    Two channels, scalar indicators (rms, std, mf) → correct column names and matrix shape.
    """
    r = make_recording("ridA", 2, 8)  # 2 channels, 8 samples
    ds = Dataset.from_recordings("ds", [r])
    windows: list[WindowedRecording] = ds.window_dataset(window_size=4, overlap=0.0)
    assert len(windows) == 1 and len(windows[0]) == 2  # two windows
    feature_cfg = ExtractionConfig(
        time_features=[
            FeatureRequest("rms"),
            FeatureRequest("std"),
        ],
        freq_features=[
            FeatureRequest("mf"),
        ],
    )
    names, X_indicators, y = ds.time_features_dataset(windows, feature_cfg)
    names_freq, X_freq, y_freq = ds.frequency_features_dataset(windows, feature_cfg)
    # Expected channel-major order: ch0_* then ch1_*
    assert names == ["ch0_rms", "ch0_std", "ch1_rms", "ch1_std"]
    assert names_freq == ["ch0_mf", "ch1_mf"]

    # X: n_windows x (n_channels * n_features_per_channel) = 2 x (2 * 2) = (2, 4)
    assert isinstance(X_indicators, np.ndarray)
    assert X_indicators.shape == (2, 4)
    assert X_indicators.dtype == np.float32

    assert isinstance(X_freq, np.ndarray)
    assert X_freq.shape == (2, 2)
    assert X_freq.dtype == np.float32


def test_frequency_features_dataset_inconsistent_fs(make_recording):
    """
    Mixed sampling frequencies in one Dataset should raise ValueError
    (we assume a single fs for the entire dataset).
    """
    r1 = make_recording("ridA", 2, 8, fs=12_000)
    r2 = make_recording("ridB", 2, 8, fs=48_000)

    ds = Dataset.from_recordings("ds", [r1, r2])

    windows: list[WindowedRecording] = ds.window_dataset(window_size=4, overlap=0.0)

    feature_cfg = ExtractionConfig(
        time_features=[FeatureRequest("rms")],
        freq_features=[FeatureRequest("mf")],
    )

    with pytest.raises(
        ValueError, match="Inconsistent sampling frequency \\(fs\\) across dataset"
    ):
        ds.frequency_features_dataset(windows, feature_cfg)


# --------------------------------------------------------------------------
# Torch integration tests
# --------------------------------------------------------------------------


@pytest.fixture
def sample_dataset(make_recording):
    rec1 = make_recording("rec1", n_ch=2, n=500)
    rec2 = make_recording("rec2", n_ch=2, n=500)
    recordings = [rec1, rec2]
    by_id = {r.rid: r for r in recordings}
    meta_df = pd.DataFrame(
        {"record_id": ["rec1", "rec2"], "label_id": [0, 1]}
    ).set_index("record_id")
    return Dataset("mock_dataset", recordings, meta_df, by_id, feature_extractor=None)


@pytest.fixture
def sample_windowed_list():
    data1 = np.random.randn(2, 500).astype(np.float32)
    data2 = np.random.randn(2, 500).astype(np.float32)
    starts = np.arange(0, 400, 100)
    rec1 = WindowedRecording(
        record_id="rec1", data=data1, window_size=100, starts=starts
    )
    rec2 = WindowedRecording(
        record_id="rec2", data=data2, window_size=100, starts=starts
    )
    return [rec1, rec2]


@pytest.mark.parametrize("flatten", [True, False])
def test_torch_dataset_creates_correct_views(
    sample_dataset, sample_windowed_list, flatten
):
    views = sample_dataset.torch_dataset(sample_windowed_list, flatten=flatten)
    assert len(views) == 2
    assert all(isinstance(v, TorchWindowView) for v in views)
    assert [v._label for v in views] == [0, 1]
    assert [v._windowed_rec.record_id for v in views] == ["rec1", "rec2"]
    if hasattr(views[0], "_flatten"):
        assert all(v._flatten == flatten for v in views)


def test_torch_dataset_empty_input(sample_dataset):
    """
    Edge case: when input list is empty, should return empty list.
    """
    views = sample_dataset.torch_dataset([], flatten=True)
    assert views == []


def test_torch_dataset_no_labels(sample_dataset, sample_windowed_list):
    """
    Edge case: when _meta_df has no valid labels for inputs.
    """
    sample_dataset._meta_df = pd.DataFrame(columns=["label_id"])  # no mapping
    with pytest.warns(UserWarning, match="not found in dataset metadata"):
        views = sample_dataset.torch_dataset(sample_windowed_list, flatten=False)
    assert views == []


def test_torch_dataset_skips_unknown_record_ids_with_warning(sample_dataset):
    data = np.random.randn(2, 500).astype(np.float32)
    starts = np.arange(0, 400, 100)
    unknown_record = WindowedRecording(
        record_id="recX", data=data, window_size=100, starts=starts
    )

    with pytest.warns(
        UserWarning, match="Record id 'recX' not found in dataset metadata"
    ):
        views = sample_dataset.torch_dataset([unknown_record], flatten=False)

    assert views == []


@pytest.mark.parametrize("flatten", [True, False])
def test_torch_windowview_smoke(sample_dataset, sample_windowed_list, flatten):
    """
    Smoke test: verify that TorchWindowView behaves like a torch Dataset.
    """
    views = sample_dataset.torch_dataset(sample_windowed_list, flatten=flatten)
    view = views[0]

    # Take first (x, y)
    x, y = view[0]

    # Check types
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, int | torch.Tensor)

    # Check shapes depending on flatten flag
    c, w = view._windowed_rec.data.shape[0], view._windowed_rec.window_size
    if flatten:
        assert x.shape == (c * w,)
    else:
        assert x.shape == (c, w)

    # check tensor dtype
    assert x.dtype == torch.float32

    # check label matches what was in meta_df
    assert y == view._label
