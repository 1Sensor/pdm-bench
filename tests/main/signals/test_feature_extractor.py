# tests/main/features/test_feature_extractor.py
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import pdm_tools.main.signals.feature_extractor as fe_mod
from pdm_tools.main.signals.dataset import Dataset
from pdm_tools.main.signals.feature_extractor import FeatureExtractor
from pdm_tools.main.signals.features_config import ExtractionConfig, FeatureRequest


if TYPE_CHECKING:
    from pdm_tools.main.signals.recordings import WindowedRecording


def f_vector(x: np.ndarray, **kwargs):
    return np.array([10.0, 20.0], dtype=float)


def f_tuple(x: np.ndarray, **kwargs):
    return np.array([1.0, 3.0], dtype=float), 7.0


def f_dummy(signal, **kwargs):
    return 0.0


custom_catalog_various_outputs_types = {
    "t": {"fn": f_tuple, "params_required": [], "params_optional": {}},
    "v": {"fn": f_vector, "params_required": [], "params_optional": {}},
}

custom_catalog_dummy_params = {
    "t": {"fn": f_dummy, "params_required": ["alpha"], "params_optional": {"beta": 0}},
}


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_empty_windows_return_empty_lists(make_recording):
    """
    When all WindowedRecording objects are empty (sequence shorter than window),
    compute_time_features returns ([], []).
    """
    r_short = make_recording("ridS", 1, 3)  # shorter than window
    ds = Dataset.from_recordings("ds", [r_short])

    empty_windows = ds.window_dataset(window_size=4, overlap=0.5)
    assert len(empty_windows) == 1 and len(empty_windows[0]) == 0

    fx = FeatureExtractor(ds)
    cols, out_x, out_y = fx.compute_time_features(
        windowed_list=empty_windows,
        features_config=ExtractionConfig(time_features=[FeatureRequest("rms")]),
        rid_to_label=ds.meta["label_id"].to_dict(),
        dtype=np.float32,
    )
    assert cols == []
    assert out_x.shape == (0, 0)
    assert out_x.dtype == np.float32
    assert out_y.shape == (0,)
    assert out_y.dtype == np.int8


def test_unknown_feature_raises_keyerror(make_recording):
    """
    Unknown feature name in features_config → KeyError listing available features.
    """
    r = make_recording("ridE", 1, 8)
    ds = Dataset.from_recordings("ds", [r])
    windows = ds.window_dataset(window_size=4, overlap=0.0)

    fx = FeatureExtractor(ds)
    with pytest.raises(KeyError):
        fx.compute_time_features(
            windowed_list=windows,
            features_config=ExtractionConfig(
                time_features=[FeatureRequest("non_existing_feature")]
            ),
            rid_to_label=ds.meta["label_id"].to_dict(),
            dtype=np.float32,
        )


def test_expanded_subcols_various_outputs(make_recording, monkeypatch):
    r = make_recording("rid_min", n_ch=2, n=8)
    ds = Dataset.from_recordings("ds", [r])
    windows: list[WindowedRecording] = ds.window_dataset(window_size=4, overlap=0.0)
    assert len(windows) == 1 and len(windows[0]) == 2

    monkeypatch.setattr(
        fe_mod,
        "available_time_features_mapping",
        custom_catalog_various_outputs_types,
        raising=True,
    )

    cfg = ExtractionConfig(
        time_features=[
            FeatureRequest("t"),
            FeatureRequest("v"),
        ]
    )

    fx = FeatureExtractor(ds)
    feature_names, out_x, out_y = fx.compute_time_features(
        windowed_list=windows,
        features_config=cfg,
        rid_to_label=ds.meta["label_id"].to_dict(),
        dtype=np.float32,
    )

    # t -> ["t0_0","t0_1","t1"], v -> ["v_0","v_1"]
    per_ch = ["t0_0", "t0_1", "t1", "v_0", "v_1"]
    expected = [f"ch0_{n}" for n in per_ch] + [f"ch1_{n}" for n in per_ch]
    assert feature_names == expected
    assert out_x.shape == (2, 2 * len(per_ch))
    assert out_y.shape == (2,)


def test_validate_params_raises_on_unexpected_key(make_recording, monkeypatch):
    r = make_recording("rid_min", n_ch=2, n=8)
    ds = Dataset.from_recordings("ds", [r])
    windows: list[WindowedRecording] = ds.window_dataset(window_size=4, overlap=0.0)

    monkeypatch.setattr(
        fe_mod,
        "available_time_features_mapping",
        custom_catalog_dummy_params,
        raising=True,
    )
    cfg = ExtractionConfig(time_features=[FeatureRequest("t", {"gamma": 123})])

    fx = FeatureExtractor(ds)
    with pytest.raises(TypeError) as exc:
        fx.compute_time_features(
            windows, cfg, rid_to_label=ds.meta["label_id"].to_dict(), dtype=np.float32
        )

    expected_msg = (
        "Feature 't' got unexpected param(s): ['gamma']. "
        "Allowed: required=['alpha'], optional=['beta']"
    )
    assert str(exc.value) == expected_msg


def test_validate_params_raises_on_missing_required(make_recording, monkeypatch):
    r = make_recording("rid_min", n_ch=2, n=8)
    ds = Dataset.from_recordings("ds", [r])
    windows: list[WindowedRecording] = ds.window_dataset(window_size=4, overlap=0.0)

    monkeypatch.setattr(
        fe_mod,
        "available_time_features_mapping",
        custom_catalog_dummy_params,
        raising=True,
    )
    cfg = ExtractionConfig(time_features=[FeatureRequest("t", {"beta": 5})])

    fx = FeatureExtractor(ds)
    with pytest.raises(TypeError) as exc:
        fx.compute_time_features(
            windows, cfg, rid_to_label=ds.meta["label_id"].to_dict(), dtype=np.float32
        )

    expected_msg = "Feature 't' is missing required param(s): ['alpha']. "
    assert str(exc.value) == expected_msg


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_compute_time_features_skips_empty_recordings_when_stacking(make_recording):
    r_empty = make_recording("rid_empty", n_ch=1, n=3)  # 0 windows
    r_full = make_recording("rid_full", n_ch=1, n=8)  # 2 windows (size=4)
    ds = Dataset.from_recordings("ds", [r_empty, r_full])

    windows = ds.window_dataset(window_size=4, overlap=0.0)

    fx = FeatureExtractor(ds)
    cols, out_x, out_y = fx.compute_time_features(
        windows,
        ExtractionConfig(time_features=[FeatureRequest("rms")]),
        rid_to_label=ds.meta["label_id"].to_dict(),
        dtype=np.float32,
    )

    assert cols == ["ch0_rms"]
    assert out_x.shape == (2, 1)
    assert out_y.shape == (2,)
    # all labels should come from rid_full
    assert set(out_y.tolist()) == {ds.meta.loc["rid_full", "label_id"]}


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_compute_time_features_empty_feature_config(make_recording):
    r = make_recording("ridZ", n_ch=2, n=8)
    ds = Dataset.from_recordings("ds", [r])
    windows = ds.window_dataset(window_size=4, overlap=0.0)

    fx = FeatureExtractor(ds)
    cols, out_x, out_y = fx.compute_time_features(
        windows,
        ExtractionConfig(time_features=[]),
        rid_to_label=ds.meta["label_id"].to_dict(),
        dtype=np.float32,
    )

    assert cols == []
    assert out_x.shape == (2, 0) and out_x.dtype == np.float32
    assert out_y.shape == (2,) and out_y.dtype == np.int8
