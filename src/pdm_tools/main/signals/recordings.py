from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import numpy as np


@dataclass
class Recording:
    """
    One recording = one file loaded into RAM.
    """

    rid: str
    data: np.ndarray  # shape (channels, n_samples), float32
    fs: float
    label: str
    source: str
    unit: str
    channels: list[str] = field(default_factory=list)
    rpm: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WindowedRecording:
    """
    Lazy, memory-efficient view of fixed-size windows for a single Recording.
    Each window is returned as a **view (slice)** of the original `data` array.

    Attributes:
        record_id (str): Unique identifier of the recording.
        data (np.ndarray): 2D array with shape (n_channels, n_samples) containing the original signal.
        window_size (int): Number of samples in each window.
        starts (np.ndarray): 1D array of start indices for each window (length equals the number of windows).
    """

    record_id: str
    data: np.ndarray
    window_size: int
    starts: np.ndarray

    def __len__(self) -> int:
        """Return the total number of windows available for this recording."""
        return int(self.starts.size)

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Get a single window as a view of the data.
        Args:
            index (int): Index of the desired window (0-based).
        Returns:
            np.ndarray: View of shape (n_channels, window_size)
        """
        start = int(self.starts[index])
        end = start + self.window_size
        n = self.data.shape[1]
        if start < 0 or end > n:
            raise IndexError(
                f"Window [{start}:{end}) is out of bounds for data with n_samples={n}"
            )
        return self.data[:, start:end]

    def __iter__(self):
        """Iterate through all windows in the recordings a view on the original data."""
        window_len = self.window_size
        for start in self.starts:
            end = int(start) + window_len
            yield self.data[:, int(start) : end]
