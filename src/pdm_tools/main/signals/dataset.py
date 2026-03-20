from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from pdm_tools.main.signals.feature_extractor import FeatureExtractor
from pdm_tools.main.signals.features_config import ExtractionConfig, FeatureRequest
from pdm_tools.main.signals.recordings import WindowedRecording
from pdm_tools.main.signals.views import (
    FFTView,
    TorchWindowView,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from pdm_tools.main.signals.recordings import Recording


@dataclass(slots=True)
class Dataset:
    """
    In-memory dataset = heavy payload (list[Recording]) + light metadata (pd.DataFrame).
    Provides:
      - O(1) lookup by id
      - convenient pandas slicing/filtering
      - subset creation (returns a new Dataset)
    """

    name: str
    recordings: list[Recording]
    _meta_df: pd.DataFrame
    _by_id: dict[str, Recording]
    _features: FeatureExtractor
    label_to_id: dict[str, int]

    def __init__(
        self,
        name: str,
        recordings: list[Recording],
        _meta_df: pd.DataFrame,
        _by_id: dict[str, Recording],
        feature_extractor: FeatureExtractor | None = None,
        label_to_id: dict[str, int] | None = None,
    ):
        self.name = name
        self.recordings = list(recordings)
        self._meta_df = _meta_df
        self._by_id = _by_id
        self._features = feature_extractor or FeatureExtractor(self)
        self.label_to_id = label_to_id or {}

    # ----------- constructors -----------

    @classmethod
    def from_recordings(
        cls,
        name: str,
        recs: list[Recording],
        label_to_id: dict[str, int] | None = None,
    ) -> Dataset:
        """Build Dataset from a list of Recording objects."""
        by_id = {r.rid: r for r in recs}  ## TODO:  silent duplicate overwrite?
        rows: list[dict[str, Any]] = []
        for r in recs:
            rows.append(
                {
                    "id": r.rid,
                    "label": r.label,
                    "fs": r.fs,
                    "n_samples": int(r.data.shape[1]),
                    "n_channels": int(r.data.shape[0]),
                    "rpm": r.rpm,
                    "source": r.source,
                    "unit": r.unit,
                    **(r.meta or {}),
                }
            )
        meta_df = (
            pd.DataFrame(rows).set_index("id")
            if rows
            else pd.DataFrame(
                columns=[
                    "label",
                    "fs",
                    "n_samples",
                    "n_channels",
                    "rpm",
                    "source",
                    "unit",
                ]
            ).set_index(pd.Index([], name="id"))
        )

        meta_df, label_to_id = cls._encode_labels(meta_df, label_to_id)

        return cls(
            name=name,
            recordings=recs,
            _meta_df=meta_df,
            _by_id=by_id,
            label_to_id=label_to_id,
        )

    # ----------- basic access -----------

    @property
    def meta(self) -> pd.DataFrame:
        """Lightweight metadata table (no signal arrays inside)."""
        return self._meta_df

    @property
    def ids(self) -> list[str]:
        """All recording ids in this dataset."""
        return list(self._by_id.keys())

    @property
    def n_classes(self) -> int:
        return len(self.label_to_id)

    def get(self, rid: str) -> Recording:
        """O(1) lookup of a Recording by id."""
        return self._by_id[rid]

    # ----------- slicing / subsetting -----------

    def subset_by_ids(self, ids: Iterable[str], name: str | None = None) -> Dataset:
        """Return a new Dataset that contains only recordings with given ids."""
        ids = list(ids)
        recs = [self._by_id[i] for i in ids]
        return Dataset.from_recordings(
            name or f"{self.name}__subset",
            recs,
            label_to_id=dict(self.label_to_id),
        )

    def subset_query(self, expr: str, name: str | None = None) -> Dataset:
        """
        Pandas-like filtering using a query string, e.g.:
          ds.subset_query("label == 'ball' and rpm == 1750")
        """
        if self._meta_df.empty:
            return Dataset.from_recordings(
                name or f"{self.name}__empty",
                [],
                label_to_id=self.label_to_id,
            )
        sub_idx = self._meta_df.query(expr).index
        return self.subset_by_ids(sub_idx, name)

    # ----------- convenience helpers -----------

    def total_samples(self) -> int:
        """Sum of n_samples across all recordings (for quick sanity checks)."""
        return int(self._meta_df["n_samples"].sum()) if not self._meta_df.empty else 0

    def summary(self) -> dict[str, pd.DataFrame]:
        """Tiny summary: counts by label/source and basic stats."""
        if self._meta_df.empty:
            return {
                "counts": pd.DataFrame(columns=["source", "label", "count"]),
                "stats": pd.DataFrame(
                    columns=["stat", "n_samples", "n_channels", "fs"]
                ),
            }

        counts = (
            self._meta_df.groupby(["source", "label"])
            .size()
            .rename("count")
            .reset_index()
        )
        stats = (
            self._meta_df[["n_samples", "n_channels", "fs"]]
            .describe()
            .round(1)
            .reset_index()
            .rename(columns={"index": "stat"})
        )
        return {"counts": counts, "stats": stats}

    @staticmethod
    def _encode_labels(
        meta_df: pd.DataFrame,
        label_to_id: dict[str, int] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, int]]:
        """Adds label_id with numerical labels encoding."""
        if meta_df.empty:
            return meta_df, (label_to_id or {})

        if "label" not in meta_df.columns:
            warnings.warn(
                "Skipping label encoding - meta_df is missing 'label' column.",
                stacklevel=2,
            )
            return meta_df, (label_to_id or {})

        if label_to_id is None:
            label_to_id = {
                label: i for i, label in enumerate(sorted(meta_df["label"].unique()))
            }
        else:
            missing = sorted(set(meta_df["label"].unique()) - set(label_to_id))
            if missing:
                raise ValueError(f"label_to_id missing labels: {missing}")
        meta_df = meta_df.copy()
        meta_df["label_id"] = meta_df["label"].map(label_to_id)
        return meta_df, label_to_id

    # ----------- datasets -----------

    def window_dataset(
        self,
        window_size: int,
        overlap: float,
    ) -> list[WindowedRecording]:
        """
        Split every recording in the dataset into fixed-size, overlapping windows.
        Args:
            window_size (int): The number of samples in each window. Must be greater than 0.
            overlap (float): Overlap ratio between consecutive windows in the range [0.0, 1.0).

        Returns:
            list[WindowedRecording]: A list of `WindowedRecording` objects, one per `Recording` in the dataset.
                Each object provides:
                    - Random access by index (`wr[i]`)
                    - Iteration over windows
                    - Window views of shape `(n_channels, window_size)`.
        """
        if not (0.0 <= float(overlap) < 1.0):
            raise ValueError("overlap must be in the range [0.0, 1.0).")

        if window_size <= 0:
            raise ValueError("window_size must be greater than 0.")

        # Step size (hop) = distance between consecutive window starts
        step_size = max(1, round(window_size * (1.0 - float(overlap))))
        step_size = min(step_size, window_size)

        windowed_list: list[WindowedRecording] = []

        for recording in self.recordings:
            num_samples = int(recording.data.shape[1])

            if num_samples < window_size:
                start_indices = np.array([], dtype=np.uint32)
            else:
                last_valid_start = num_samples - window_size
                start_indices = np.arange(
                    0, last_valid_start + 1, step_size, dtype=np.uint32
                )

            windowed_list.append(
                WindowedRecording(
                    record_id=recording.rid,
                    data=recording.data,  # no copy, just a view
                    window_size=window_size,
                    starts=start_indices,
                )
            )

        return windowed_list

    def time_features_dataset(
        self,
        windowed_list: list[WindowedRecording],
        features_config: ExtractionConfig,
        dtype: np.dtype = np.float32,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Wrapper for time-domain feature extraction, allowing direct computation from a Dataset instance.
        Args:
            windowed_list (list[WindowedRecording]): list of windowed recordings to process.
            features_config (ExtractionConfig): mapping {feature_name: kwargs} defining which indicators to compute.
            dtype (np.dtype): data type of the resulting feature matrices (default: np.float32).

        Returns:
            tuple[list[str], np.ndarray, np.ndarray]: the same output as FeatureExtractor.compute_time_features():
                - list of feature column names,
                - np.ndarray - feature matrix,
                - np.ndarray - array of labels.
        """
        return self._features.compute_time_features(
            windowed_list=windowed_list,
            features_config=features_config,
            dtype=dtype,
            rid_to_label=self._meta_df["label_id"].to_dict(),
        )

    def frequency_features_dataset(
        self,
        windowed_list: list[WindowedRecording],
        features_config: ExtractionConfig,
        dtype: np.dtype = np.float32,
        window_fn: Callable[[int], np.ndarray] | None = np.hanning,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Wrapper for frequency-domain feature extraction. Builds FFTView for each WindowedRecording
        using the per-recording sampling frequency, then computes frequency features.
        Args:
            windowed_list (list[WindowedRecording]): list of WindowedRecording objects to process.
            features_config (ExtractionConfig): defining which frequency indicators to compute and their parameters.
            dtype (np.dtype): data type of the resulting feature matrices (default: np.float32).
            window_fn (Callable[[int], np.ndarray] | None) - Window function used before FFT (default: np.hanning).
                    Use None for a rectangular window.

        Returns:
            tuple[list[str], np.ndarray, np.ndarray]: same output as FeatureExtractor.compute_frequency_features():
                - list of feature column names,
                - np.ndarray feature matrix X,
                - np.ndarray label vector y.
        """
        fs_values = self._meta_df["fs"].unique()
        if len(fs_values) != 1:
            raise ValueError("Inconsistent sampling frequency (fs) across dataset")

        fft_views: list[FFTView] = []
        for window_rec in windowed_list:
            fft_views.append(FFTView(window_rec, fs_values[0], window_fn, dtype))
        return self._features.compute_frequency_features(
            fft_view_list=fft_views,
            features_config=features_config,
            dtype=dtype,
            rid_to_label=self._meta_df["label_id"].to_dict(),
        )

    def torch_dataset(
        self,
        windowed_list: list[WindowedRecording],
        flatten: bool = False,
        normalization: bool = False,
    ) -> list[TorchWindowView]:
        """
        Create lazy TorchView for every WindowedRecording in the collection.
        Nothing is materialized - every element is a view.
        Args:
            windowed_list (list[WindowedRecording]): output `window_dataset(...)`,
            flatten (bool): True -> (C*W) pod MLP; False -> (C, W) for Conv1d (default: False),
            normalization (bool): Z-score normalization (default: False).
        Returns:
            List[TorchView]: list of lazy TorchViews over WindowedRecordings.
        """
        views: list[TorchWindowView] = []
        map_rid_label = self._meta_df["label_id"].to_dict()
        for window_rec in windowed_list:
            if window_rec.record_id not in map_rid_label:
                warnings.warn(
                    f"Record id '{window_rec.record_id}' not found in dataset metadata - skipping.",
                    stacklevel=2,
                )
                continue
            tv = TorchWindowView(
                window_rec, map_rid_label[window_rec.record_id], flatten, normalization
            )
            views.append(tv)
        return views
