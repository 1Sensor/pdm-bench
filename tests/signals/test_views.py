# tests/signals/test_views.py
import numpy as np
import pytest
import torch

from pdm_bench.signals.recordings import Recording, WindowedRecording
from pdm_bench.signals.views import FFTView, TorchWindowView


def make_windowed(
    recording: Recording, window_size: int, starts: np.ndarray
) -> WindowedRecording:
    """Helper to build a WindowedRecording."""
    return WindowedRecording(
        record_id=recording.rid,
        data=recording.data,
        window_size=window_size,
        starts=starts.astype(int),
    )


def zscore_per_channel(x: np.ndarray) -> np.ndarray:
    """Matches TorchWindowView normalization: per channel over time axis."""
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-8)


def test_len_matches_number_of_windows(make_recording):
    rec = make_recording("r1", n_ch=3, n=20)
    starts = np.array([0, 4, 10, 15], dtype=int)
    wr = make_windowed(rec, window_size=4, starts=starts)

    view = TorchWindowView(_windowed_rec=wr, _label=2, _flatten=False)
    assert len(view) == len(starts)


def test_getitem_unflattened_shape_and_content(make_recording):
    rec = make_recording("r2", n_ch=2, n=12)
    starts = np.array([3, 5], dtype=int)
    window_size = 4
    wr = make_windowed(rec, window_size=window_size, starts=starts)

    view = TorchWindowView(_windowed_rec=wr, _label=1, _flatten=False)
    x, y = view[0]
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)
    assert x.shape == (2, window_size)
    assert y.dtype == torch.long and y.item() == 1

    expected = rec.data[:, starts[0] : starts[0] + window_size]
    np.testing.assert_allclose(x.numpy(), zscore_per_channel(expected))


def test_getitem_flattened_shape(make_recording):
    rec = make_recording("r3", n_ch=4, n=30)
    window_size = 5
    starts = np.array([0, 10, 20], dtype=int)
    wr = make_windowed(rec, window_size=window_size, starts=starts)

    view = TorchWindowView(_windowed_rec=wr, _label=0, _flatten=True)
    x, y = view[1]
    assert x.shape == (4 * window_size,)
    assert y.item() == 0


def test_zero_copy_sharing_numpy_and_tensor(make_recording):
    rec = make_recording("r4", n_ch=2, n=16)
    window_size = 4
    starts = np.array([6], dtype=int)
    wr = make_windowed(rec, window_size=window_size, starts=starts)

    view = TorchWindowView(_windowed_rec=wr, _label=7, _flatten=False)
    x, _y = view[0]

    old_data = rec.data[0, starts[0]]
    x[0, 0] = 999.0
    assert rec.data[0, starts[0]] == pytest.approx(old_data)

    old_x = x[1, 1].item()
    old_val = rec.data[1, starts[0] + 1]
    rec.data[1, starts[0] + 1] = -123.0
    assert x[1, 1].item() == pytest.approx(old_x)
    rec.data[1, starts[0] + 1] = old_val


def test_torch_windowview_return_contract_unchanged(make_recording):
    rec = make_recording("r_contract", n_ch=2, n=12)
    starts = np.array([0], dtype=int)
    wr = make_windowed(rec, window_size=4, starts=starts)

    view = TorchWindowView(_windowed_rec=wr, _label=3, _flatten=False)
    sample = view[0]

    assert isinstance(sample, tuple)
    assert len(sample) == 2
    x, y = sample
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.long


def test_fftview_len_matches_windows(make_recording):
    rec = make_recording("fft1", n_ch=2, n=40)
    starts = np.array([0, 10, 20], dtype=int)
    wr = make_windowed(rec, window_size=8, starts=starts)

    view = FFTView(windows=wr, fs=np.float32(100.0))
    assert len(view) == len(starts)


def test_fftview_freqs_expected_length_and_values(make_recording):
    rec = make_recording("fft_freqs", n_ch=1, n=32)
    window_size = 8
    starts = np.array([0], dtype=int)
    wr = make_windowed(rec, window_size=window_size, starts=starts)

    fs = np.float32(200.0)
    view = FFTView(windows=wr, fs=fs, out_dtype=np.float32)
    expected = np.fft.rfftfreq(window_size, d=1.0 / fs).astype(np.float32)

    assert view.freqs.shape == (window_size // 2 + 1,)
    assert view.freqs.dtype == np.float32
    assert np.allclose(view.freqs, expected)


def test_fftview_getitem(make_recording):
    rec = make_recording("fft_basic", n_ch=3, n=64)
    window_size = 16
    starts = np.array([5, 20], dtype=int)
    wr = make_windowed(rec, window_size=window_size, starts=starts)

    view = FFTView(windows=wr, fs=np.float32(128.0), out_dtype=np.float32)
    out = view[0]
    assert out.shape == (3, window_size // 2 + 1)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
