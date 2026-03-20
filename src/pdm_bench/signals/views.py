from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset

from .recordings import WindowedRecording


@dataclass(slots=True)
class TorchWindowView(TorchDataset):
    """
    Lazy PyTorch view over one WindowedRecording.
    Every element is one window: (x, y).
      - x: tensor from numpy view (zero-copy), shape (C, W) or flattened (C*W)
      - label: int (class id) - given during construction
    Nothing is created upfront (no Xw).
    """

    _windowed_rec: WindowedRecording
    _label: int
    _flatten: bool = True
    _normalization: bool = True

    def __len__(self) -> int:
        return len(self._windowed_rec)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self._windowed_rec.starts[idx])
        end = start + self._windowed_rec.window_size

        # (C, W) — SLICE = base array view (no copy)
        x_np = self._windowed_rec.data[:, start:end]
        if self._normalization:
            x_np = (x_np - x_np.mean(axis=1, keepdims=True)) / (
                x_np.std(axis=1, keepdims=True) + 1e-8
            )

        if self._flatten:
            x_np = x_np.reshape(-1)  # (C*W,)

        # Zero-copy from numpy to torch (shared memory)
        x = torch.from_numpy(x_np)
        y = torch.tensor(self._label, dtype=torch.long)
        return x, y


@dataclass(slots=True)
class FFTView:
    """
    Lazy, memory-efficient view of FFT-transformed windows for a single WindowedRecording.
    It computes the real FFT (rfft) on demand from the original time-domain window.

    Attributes:
        windows (WindowedRecording): Underlying time-domain windows (shape: [n_channels, window_size]).
        fs (np.float32): sampling frequency
        window_fn (Callable[[int], np.ndarray] | None):
            Function that generates a window given its length (int -> np.ndarray). Default is np.hanning.
            Use None for a rectangular window (all ones). Hann typically reduces leakage (lower sidelobes)
            but widens the main lobe (lower line resolution).
        out_dtype (np.dtype): Output dtype (float32 by default to save RAM).
    """

    windows: WindowedRecording
    fs: np.float32
    window_fn: Callable[[int], np.ndarray] | None = np.hanning
    out_dtype: np.dtype = np.float32
    _analysis_window: np.ndarray = field(init=False, repr=False)
    _freqs: np.ndarray = field(init=False, repr=False)

    @property
    def record_id(self) -> str:
        return self.windows.record_id

    def __post_init__(self):
        """Runs once after init. It builds and stores the analysis window vector (shape: (window_size,))
        and casts it to out_dtype"""
        window_length = self.windows.window_size
        if self.window_fn is None:  # Rectangular window
            self._analysis_window = np.ones(window_length, dtype=self.out_dtype)
        else:
            self._analysis_window = self.window_fn(window_length).astype(
                self.out_dtype, copy=False
            )
        self._freqs = np.fft.rfftfreq(window_length, d=1.0 / self.fs).astype(
            self.out_dtype, copy=False
        )

    def __len__(self) -> int:
        """Returns the number of windows available."""
        return len(self.windows)

    def __getitem__(self, index: int) -> np.ndarray:
        """Compute the rfft for a single window by index"""
        time_window = self.windows[index]
        windowed = (time_window * self._analysis_window).astype(
            self.out_dtype, copy=False
        )
        magnitude = np.fft.rfft(windowed, axis=1)  # (n_channels, n_freqs)
        return np.abs(magnitude).astype(self.out_dtype, copy=False)

    def __iter__(self) -> np.ndarray:
        """Iterate through all windows in the recordings, yielding one rFFT result at a time."""
        yield from (self[i] for i in range(len(self)))

    @property
    def freqs(self) -> np.ndarray:
        return self._freqs
