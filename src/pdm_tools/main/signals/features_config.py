from dataclasses import dataclass, field

import pdm_tools.main.signals.indicators as indicators


""" Map short feature codes to indicator functions (time and frequency domain)."""
available_time_features_mapping = {
    "pp": {"fn": indicators.peak_to_peak, "params_required": [], "params_optional": {}},
    "zp": {"fn": indicators.zero_to_peak, "params_required": [], "params_optional": {}},
    "rms": {"fn": indicators.rms, "params_required": [], "params_optional": {}},
    "cf": {"fn": indicators.crest_factor, "params_required": [], "params_optional": {}},
    "std": {"fn": indicators.std, "params_required": [], "params_optional": {}},
    "kurt": {"fn": indicators.kurt, "params_required": [], "params_optional": {}},
    "sf": {"fn": indicators.shape_factor, "params_required": [], "params_optional": {}},
    "eo": {
        "fn": indicators.energy_operator,
        "params_required": [],
        "params_optional": {},
    },
    "FM4": {
        "fn": indicators.fourth_order_fom,
        "params_required": [],
        "params_optional": {},
    },
    "FM6": {
        "fn": indicators.sixth_order_fom,
        "params_required": [],
        "params_optional": {},
    },
    "FM8": {
        "fn": indicators.eight_order_fom,
        "params_required": [],
        "params_optional": {},
    },
    "clf": {
        "fn": indicators.clearance_factor,
        "params_required": [],
        "params_optional": {},
    },
    "ii": {
        "fn": indicators.impulse_indicator,
        "params_required": [],
        "params_optional": {},
    },
    "FN4": {
        "fn": indicators.fourth_order_np,
        "params_required": [],
        "params_optional": {},
    },
}

available_frequency_features_mapping = {
    "mf": {"fn": indicators.mean_freq, "params_required": [], "params_optional": {}},
    "fc": {"fn": indicators.freq_center, "params_required": [], "params_optional": {}},
    "peaks": {
        "fn": indicators.find_peaks,
        "params_required": [],
        "params_optional": {
            "peaks_num": 3,
            "height": None,
            "distance": None,
        },
    },
    "rms_f": {
        "fn": indicators.rms_frequency,
        "params_required": [],
        "params_optional": {},
    },
    "std_f": {
        "fn": indicators.std_frequency,
        "params_required": [],
        "params_optional": {},
    },
}


@dataclass(frozen=True)
class FeatureRequest:
    """
    Configuration for a single feature chosen by the user.
    Example:
        FeatureConfig(name="peaks", params={"k": 3})
    """

    name: str
    params: dict = field(default_factory=dict)


@dataclass(slots=True)
class ExtractionConfig:
    """
    High-level extraction plan for a single run of feature computation.
    The user declares what features they want from time and/or frequency domains.
    """

    dtype: str = "float32"
    time_features: list = field(default_factory=list)
    freq_features: list = field(default_factory=list)
