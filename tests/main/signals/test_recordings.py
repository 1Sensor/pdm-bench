from __future__ import annotations

import numpy as np
import pytest

from pdm_tools.main.signals.recordings import Recording, WindowedRecording


def test_recording_basic():
    # channels x samples
    data = np.vstack(
        [np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32) * 10]
    )
    rec = Recording(
        rid="exp1/1730/X101/fileA",
        data=data,
        fs=12000.0,
        label="outer_race",
        source="CWRU",
        unit="g",
        channels=["DE_time", "FE_time"],
        rpm=1730.0,
        meta={"Experiment": "12DriveEndFault"},
    )

    assert rec.rid == "exp1/1730/X101/fileA"
    assert rec.data.shape == (2, 10)
    assert rec.data.dtype == np.float32
    assert rec.fs == 12000.0
    assert rec.label == "outer_race"
    assert rec.source == "CWRU"
    assert rec.unit == "g"
    assert rec.channels == ["DE_time", "FE_time"]
    assert rec.rpm == 1730.0
    assert rec.meta["Experiment"] == "12DriveEndFault"


def test_windowed_recording_len_getitem_iter_and_views():
    # Make a simple 2x10 signal
    base = np.vstack(
        [np.arange(10, dtype=np.float32), np.arange(10, dtype=np.float32) + 100.0]
    )

    # Windows of size 4 starting at indices 0, 3, 6  -> 3 windows total
    starts = np.array([0, 3, 6], dtype=np.int64)
    wr = WindowedRecording(record_id="r1", data=base, window_size=4, starts=starts)

    # __len__
    assert len(wr) == 3

    # __getitem__ returns a view with correct slice
    w1 = wr[1]
    np.testing.assert_array_equal(w1, base[:, 3:7])
    # It should be a view (no copy)
    assert np.shares_memory(w1, base)

    # Iteration yields all windows in order and as views
    windows = list(wr)
    assert len(windows) == 3
    np.testing.assert_array_equal(windows[0], base[:, 0:4])
    np.testing.assert_array_equal(windows[2], base[:, 6:10])
    assert all(np.shares_memory(w, base) for w in windows)

    # Mutate base, window must reflect the change (proves view semantics)
    base[0, 3] = -999.0
    assert w1[0, 0] == -999.0  # first element of window 1 is base[:, 3]

    # dtype/shape
    assert w1.shape == (2, 4)
    assert w1.dtype == base.dtype


def test_windowed_recording_out_of_bounds_raises():
    base = np.zeros((2, 10), dtype=np.float32)
    # This start is invalid because 8 + window_size(4) = 12 > 10
    wr = WindowedRecording(
        record_id="r2", data=base, window_size=4, starts=np.array([0, 8])
    )

    # Valid index works
    _ = wr[0]
    # Invalid index for list of starts
    with pytest.raises(IndexError):
        _ = wr[2]

    # Accessing the second start should still raise when slicing due to OOB
    # (depending on desired contract you may prefer an explicit check in __getitem__;
    #  here we assert numpy slicing shape mismatch to signal a logic error)
    with pytest.raises(IndexError):
        _ = wr[1]  # end goes past the right edge


def test_windowed_recording_slots_prevent_new_attributes():
    base = np.zeros((1, 8), dtype=np.float32)
    wr = WindowedRecording(
        record_id="r3", data=base, window_size=4, starts=np.array([0, 4])
    )
    with pytest.raises(AttributeError):
        wr.some_new_attr = 123
