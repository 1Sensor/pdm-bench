# ruff: noqa: ERA001

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from pdm_bench.signals.features_config import (
    ExtractionConfig,
    FeatureRequest,
    available_frequency_features_mapping,
    available_time_features_mapping,
)
from pdm_bench.signals.views import FFTView


if TYPE_CHECKING:
    from pdm_bench.signals.dataset import Dataset
    from pdm_bench.signals.recordings import WindowedRecording


class FeatureExtractor:
    """
    Lightweight feature extractor that depends only on the Dataset interface (fs / ids / get).
    Computes numerical features for windowed signals in either time or frequency domain.
    It doesn't store raw signals — it only knows how to extract and process them from a Dataset.
    """

    def __init__(self, dataset: Dataset):
        """Store reference to the Dataset that provides recordings and sampling info."""
        self.ds = dataset

    def compute_frequency_features(
        self,
        fft_view_list: list[FFTView],
        features_config: ExtractionConfig,
        rid_to_label: dict[str, int],
        dtype: np.dtype = np.float32,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Compute frequency-domain features for a list of FFTView objects and stack them into unified (X, y) arrays.
        This method applies each configured frequency indicator to every channel of every FFT window.
        Args:
            fft_view_list (list[FFTView]): List of FFTView objects providing per-window spectra.
            features_config (ExtractionConfig): configuration of features and their parameters {feature_name: kwargs}.
            rid_to_label (dict[str, int): Mapping from recording id to integer class label.
            dtype (np.dtype): Output dtype for feature matrix (default: np.float32).

        Returns:
            tuple[list[str], np.ndarray, np.ndarray]:
                - feature_names: Ordered list of feature column names (including channel prefixes).
                - X: Feature matrix of shape (n_windows, n_channels * n_features_per_channel).
                - y: Label vector of shape (n_windows,).
        """

        feature_names, full_indicators = self._compute_features_generic(
            windowed_list=fft_view_list,
            features_config=features_config.freq_features,
            mapping=available_frequency_features_mapping,
            dtype=dtype,
            rid_to_label=rid_to_label,
        )
        return self._stack_feature_matrices(feature_names, full_indicators, dtype)

    def compute_time_features(
        self,
        windowed_list: list[WindowedRecording],
        features_config: ExtractionConfig,
        rid_to_label: dict[str, int],
        dtype=np.float32,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Compute time-domain features for a list of WindowedRecording objects and stack them into unified (X, y) arrays.
        This method applies each configured indicator function to every channel of every window.
        Args:
            windowed_list (list[WindowedRecording]): list of recordings with time windows (n_channels, n_samples).
            features_config (ExtractionConfig): configuration of features and their arguments {feature_name: kwargs}.
            rid_to_label (dict[str, int]): Mapping from recording id to integer class label.
            dtype (np.dtype): data type of the resulting feature arrays (default: np.float32).

        Returns:
            tuple[list[str], np.ndarray, np.ndarray]:
                - feature_names: ordered list of column names, e.g.['ch0_rms', 'ch0_kurt', 'ch1_rms', ...].
                - X: feature matrix of shape (n_windows, n_channels * n_features_per_channel).
                - y: label vector of shape (n_windows,).
        """
        feature_names, full_indicators = self._compute_features_generic(
            windowed_list=windowed_list,
            features_config=features_config.time_features,
            mapping=available_time_features_mapping,
            dtype=dtype,
            rid_to_label=rid_to_label,
        )
        return self._stack_feature_matrices(feature_names, full_indicators, dtype)

    def _compute_features_generic(
        self,
        windowed_list: list[WindowedRecording] | list[FFTView],
        features_config: list[FeatureRequest],
        mapping: dict[str, dict],
        rid_to_label: dict[str, int],
        dtype: np.dtype = np.float32,
    ) -> tuple[list[str], list[dict]]:
        """
        Generic feature computation logic shared by time and frequency domains.
        Validates requested features, determines output structure, and computes feature matrices.
        Args:
            windowed_list (list[WindowedRecording] | list[FFTView]): list of recordings with signal window objects.
            features_config (list[FeatureRequest]): requested features with parameters.
            mapping (dict[str, callable]): mapping of feature names to their actual implementation functions.
            rid_to_label (dict[str, int]): Mapping from recording id to integer class label.
            dtype (np.dtype): data type of the resulting feature arrays (default: np.float32).

        Returns:
            list[str]: column names corresponding to computed features for each channel.
            list[dict]: list of dictionaries with metadata and computed feature matrices.
                One entry per recording, each containing:
                    - 'record_id': recording identifier (str)
                    - 'data': feature matrix (np.ndarray) with shape (n_windows, n_channels * n_features_per_channel)
        """
        features_dict = self._features_dict(features_config, mapping)
        fn_map = {
            feature_name: feature_spec["fn"]
            for feature_name, feature_spec in mapping.items()
            if "fn" in feature_spec
        }

        first_nonempty_window = next(
            (window for window in windowed_list if len(window) > 0), None
        )
        if first_nonempty_window is None:
            warnings.warn(
                "No non-empty windows found in the provided windowed_list. "
                "Returning empty results.",
                UserWarning,
                stacklevel=2,
            )
            return [], []

        num_channels = first_nonempty_window[0].shape[0]

        sample_signal = first_nonempty_window[0][
            0
        ]  # (L,) in time domain or (n_freqs,) in freq domain
        sample_freqs = (
            first_nonempty_window.freqs
            if isinstance(first_nonempty_window, FFTView)
            else None
        )

        expanded_subcols = self._expand_feature_column_names(
            sample_signal, features_dict, fn_map, sample_freqs
        )
        feature_name_columns = self._build_feature_name_columns(
            num_channels, list(features_dict.keys()), expanded_subcols
        )

        full_indicators: list[dict] = []
        for window_rec in windowed_list:
            rid = window_rec.record_id
            static_freqs = window_rec.freqs if isinstance(window_rec, FFTView) else None
            print("Processing recording:", rid)

            recording_feature_matrix = self._compute_recording_feature_matrix(
                window_rec,
                num_channels,
                features_dict,
                fn_map,
                dtype,
                static_freqs,
                expanded_subcols,
            )

            full_indicators.append(
                {
                    "record_id": rid,
                    "label": rid_to_label[rid],
                    "data": recording_feature_matrix,
                }
            )

        return feature_name_columns, full_indicators

    @staticmethod
    def _stack_feature_matrices(
        feature_names: list[str],
        full_indicators: list[dict],
        dtype: np.dtype,
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        """
        Stack per-recording feature matrices into a single dataset.
        Takes a list of dictionaries produced by _compute_features_generic (each containing a feature matrix
        and a label for one recording) and concatenates all windows into one global X and y.
        """
        if not full_indicators:
            return (
                feature_names,
                np.empty((0, 0), dtype=dtype),
                np.empty((0,), dtype=np.int8),
            )

        total_rows = 0
        n_features = None
        for indicator_dict in full_indicators:
            indicator_data = np.asarray(indicator_dict["data"])
            if n_features is None:
                n_features = indicator_data.shape[1]
            total_rows += indicator_data.shape[0]

        x = np.empty((total_rows, n_features), dtype=dtype)
        y = np.empty((total_rows,), dtype=np.int8)

        pos = 0
        for indicator_dict in full_indicators:
            indicator_data = np.asarray(indicator_dict["data"], dtype=dtype)
            rows = indicator_data.shape[0]
            if rows == 0:
                continue
            x[pos : pos + rows] = indicator_data
            y[pos : pos + rows] = indicator_dict["label"]
            pos += rows

        return feature_names, x, y

    def _features_dict(
        self, requests: list[FeatureRequest], available_mapping: dict[str, dict]
    ) -> dict[str, dict]:
        """
        Build the {feature_name: kwargs} mapping expected by your FeatureExtractor,
        validating strictly against the available_*_features_mapping catalogs.
        """
        out: dict[str, dict] = {}
        requested_names = []

        for req in requests:
            if req.name not in available_mapping:
                raise KeyError(
                    f"Unknown feature '{req.name}'. Available: {sorted(available_mapping.keys())}"
                )
            entry = available_mapping[req.name]
            out[req.name] = self._validate_params_against_catalog(
                req.name, dict(req.params), entry
            )
            requested_names.append(req.name)

        unused = [name for name in available_mapping if name not in requested_names]
        if unused:
            warnings.warn(
                f"The following available features were not used: {unused}",
                UserWarning,
                stacklevel=2,
            )
        return out

    @staticmethod
    def _validate_params_against_catalog(
        feature_name: str, user_params: dict, catalog_entry: dict
    ) -> dict:
        """
        Validate user-provided feature parameters against the feature catalog entry.
        Checks:
            - no unexpected parameters were provided,
            - all required parameters are present.
        """
        params_req = set(catalog_entry["params_required"])
        params_opt = set(catalog_entry["params_optional"].keys())
        params_all = params_req | params_opt

        unexpected_param = [param for param in user_params if param not in params_all]
        if unexpected_param:
            raise TypeError(
                f"Feature '{feature_name}' got unexpected param(s): {unexpected_param}. "
                f"Allowed: required={sorted(params_req)}, optional={sorted(params_opt)}"
            )

        missing_param = [param for param in params_req if param not in user_params]
        if missing_param:
            raise TypeError(
                f"Feature '{feature_name}' is missing required param(s): {missing_param}. "
            )
        return dict(user_params)

    @staticmethod
    def _build_feature_name_columns(
        num_channels: int,
        feature_names: list[str],
        expanded_subcols: dict[str, list[str]],
    ) -> list[str]:
        """
        Generate ordered column names for all features and channels.
        Args:
            num_channels (int): number of channels in each recording.
            feature_names (list[str]): list of feature names to include.
            expanded_subcols (dict[str, list[str]]): mapping from feature name to subcolumn names (for tuple/array outputs)
                                                  from _expand_feature_column_names.

        Returns:
            list[str]: list of feature column names ['ch0_rms', 'ch0_kurt', 'ch1_rms', ...].
        """
        cols: list[str] = []
        for channel in range(num_channels):
            for feature_name in feature_names:
                for sub_feature in expanded_subcols.get(feature_name, [feature_name]):
                    cols.append(f"ch{channel}_{sub_feature}")
        return cols

    @staticmethod
    def _expand_feature_column_names(
        signal: np.ndarray,
        features_config: dict[str, dict],
        mapping: dict[str, callable],
        freqs: np.ndarray | None,
    ) -> dict[str, list[str]]:
        """
        Determine how each feature expands into one or more output columns.
        Some indicators return tuples or arrays instead of scalars (e.g. peaks) — this method inspects
        one example signal to detect the correct number and naming of resulting subcolumns.
        Rules:
            - tuple: element i of a tuple → 'name{i}' if scalar, or 'name{i}_j' for the j-th entry if vector
              Example: 'peaks' → (array([0.1, 0.3]), 5) → ['peaks0_0', 'peaks0_1', 'peaks1']
            - 1D ndarray: 'name_0' ... 'name_{N-1}'
              Example: 'fft_energy' → array([0.2, 0.4, 0.8]) → ['fft_energy_0', 'fft_energy_1', 'fft_energy_2']
            - scalar: 'name'
              Example: 'rms' → 0.45 → ['rms']
        """
        expanded: dict[str, list[str]] = {}
        for feature_name, kwargs in features_config.items():
            feature_function = mapping[feature_name]
            out = (
                feature_function(signal, freqs, **kwargs)
                if freqs is not None
                else feature_function(signal, **kwargs)
            )

            if isinstance(out, tuple):
                columns: list[str] = []
                for idx, vector in enumerate(out):
                    arr = np.asarray(vector)
                    if arr.ndim == 0 or arr.size == 1:
                        columns.append(f"{feature_name}{idx}")
                    else:
                        columns.extend(
                            [
                                f"{feature_name}{idx}_{element}"
                                for element in range(arr.size)
                            ]
                        )
                expanded[feature_name] = columns
            else:
                arr = np.asarray(out)
                if arr.ndim == 0 or arr.size == 1:
                    expanded[feature_name] = [feature_name]
                else:
                    expanded[feature_name] = [
                        f"{feature_name}_{element}" for element in range(arr.size)
                    ]
        return expanded

    def _compute_recording_feature_matrix(
        self,
        window_rec: WindowedRecording | FFTView,
        num_channels: int,
        features_config: dict[str, dict],
        mapping: dict[str, callable],
        dtype: np.dtype = np.float32,
        freqs: np.ndarray | None = None,
        expanded_subcols: dict[str, list[str]] | None = None,
    ) -> np.ndarray:
        """
        Compute the complete feature matrix for one recording (all windows and channels).
        Iterates over every window and channel and concatenates the results into a single array.
        Args:
            window_rec (WindowedRecording | FFTView): recording or view containing windowed signal data.
            num_channels (int): number of channels in the recording.
            features_config (dict[str, dict]): configuration of features to compute and their arguments.
            mapping (dict[str, callable]): dictionary linking feature names to indicator functions.
            dtype (np.dtype): data type of the resulting matrix (default: np.float32).
            freqs (np.ndarray | None): frequency axis, only for frequency-domain features.
            expanded_subcols (dict[str, list[str]] | None): precomputed mapping of feature names to subcolumns.

        Returns:
            np.ndarray: Array of shape (n_windows, n_channels * n_features_per_channel) with computed feature values.
        """
        num_features_per_ch = (
            sum(
                len(expanded_subcols.get(feature_name, [feature_name]))
                for feature_name in features_config
            )
            if expanded_subcols is not None
            else len(features_config)
        )
        num_cols = num_channels * num_features_per_ch
        recording_indicators = np.full((len(window_rec), num_cols), np.nan, dtype=dtype)

        for window_idx, window in enumerate(window_rec):
            col = 0
            for channel in range(num_channels):
                signal = np.ascontiguousarray(window[channel])
                feature_vector = self._compute_feature_vector_for_signal(
                    signal, features_config, mapping, freqs=freqs
                )
                recording_indicators[window_idx, col : col + num_features_per_ch] = (
                    np.asarray(feature_vector, dtype=dtype)
                )
                col += num_features_per_ch
        return recording_indicators

    @staticmethod
    def _compute_feature_vector_for_signal(
        signal: np.ndarray,
        features_config: dict[str, dict],
        mapping: dict[str, callable],
        freqs: np.ndarray | None = None,
    ) -> list[np.float32]:
        """
        Compute all configured feature values for a single-channel signal.
        Flattens the results of all indicator functions into a one-dimensional list.
        Args:
            signal (np.ndarray): single-channel time or frequency signal.
            features_config (dict[str, dict]): configuration of features to compute and their arguments.
            mapping (dict[str, callable): mapping of feature names to their implementations.
            freqs (np.ndarray | None): frequency axis, only for frequency-domain indicators.

        Returns:
            list[np.float32]: flattened list of computed feature values for the signal.
        """
        feature_vals: list[np.float32] = []
        for feature_name, kwargs in features_config.items():
            feature_function = mapping[feature_name]
            out = (
                feature_function(signal, freqs, **kwargs)
                if freqs is not None
                else feature_function(signal, **kwargs)
            )

            outputs = (
                out if isinstance(out, tuple) else (out,)
            )  # if more than one vector of indicators

            for value in outputs:
                arr = np.asarray(value)
                if arr.ndim == 0:  # scalar value
                    feature_vals.append(np.float32(arr))
                else:  # one vector value
                    feature_vals.extend(np.float32(arr.ravel()))
        return feature_vals
