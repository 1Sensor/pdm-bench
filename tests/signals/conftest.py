import numpy as np
import pytest

from pdm_bench.signals.recordings import Recording


@pytest.fixture
def make_recording():
    """
    Factory fixture returning a function to easily create synthetic Recording objects.
    Default values can be changed through **kwargs.
    """

    def _make(rid: str, n_ch: int, n: int, **kwargs) -> Recording:
        defaults = {
            "fs": 12_000.0,
            "label": "normal",
            "source": "test",
            "unit": "u",
            "rpm": None,
            "meta": {"m": 1},
        }
        params = {**defaults, **kwargs}

        data = np.arange(n_ch * n, dtype=np.float32).reshape(n_ch, n)
        channels = [f"ch{i}" for i in range(n_ch)]

        return Recording(
            rid=rid,
            data=data,
            channels=channels,
            **params,
        )

    return _make
