from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from pdm_bench.signals.recordings import Recording


AVAILABLE_SENSORS = ("DE", "FE", "BA")
DEFAULT_SENSORS = ("DE",)


def _is_valid_key(key: str, sensors: tuple[str, ...]) -> bool:
    """Return True if the key is valid for processing, else False."""
    if key.startswith("__"):
        return False
    return any(sensor in key for sensor in sensors)


def _infer_fs(file_path: Path) -> float | None:
    """Heuristic: infer sampling rate from directory names."""
    parts = set(file_path.parts)
    if "48DriveEndFault" in parts:
        return 48_000.0
    if (
        "12DriveEndFault" in parts
        or "12FanEndFault" in parts
        or "NormalBaseline" in parts
    ):
        return 12_000.0
    return None  # unknown → can be filled later if needed


def _normalize_label(stem: str) -> str:
    """Map CWRU filename to a normalized class label."""
    if "Ball" in stem:
        return "ball"
    if "InnerRace" in stem:
        return "inner_race"
    if "OuterRace" in stem:
        return "outer_race"
    if "Normal" in stem:
        return "normal"
    return "unknown"


def _parse_fault_family(stem: str) -> dict[str, str | float | None]:
    """Extract lightweight fault-family metadata from a CWRU filename stem."""
    if stem == "Normal":
        return {
            "recording_family": "Normal",
            "fault_severity_in": None,
            "fault_location_variant": None,
        }

    parts = stem.split("-", maxsplit=1)
    if len(parts) != 2:
        return {
            "recording_family": stem,
            "fault_severity_in": None,
            "fault_location_variant": None,
        }

    severity_str, family = parts
    try:
        severity = float(severity_str)
    except ValueError:
        severity = None

    location_variant: str | None = None
    if family.startswith("OuterRace"):
        location_variant = family.removeprefix("OuterRace") or None

    return {
        "recording_family": stem,
        "fault_severity_in": severity,
        "fault_location_variant": location_variant,
    }


def _load_cwru_file(
    file_path: Path,
    base_dir: Path,
    sensors: tuple[str, ...],
) -> list[Recording]:
    """
    Load a single CWRU .mat file and extract its data. File can contain multiple recordings.

    Args:
        file_path (Path): Path to the .mat file.
        base_dir (Path): Base directory path.
        sensors (tuple[str, ...]): Sensors to load from the file.

    Returns:
        list[Recording]: Returns list of recordings loaded from a file.
    """
    mat_data = loadmat(file_path)

    filtered_data = {}
    source_column = []
    last_source = None

    # Process valid keys only
    for key, value in (
        (k, v) for k, v in mat_data.items() if _is_valid_key(k, sensors)
    ):
        source, sensor = key.split("_", 1)
        signal = value.flatten()

        # Concatenate or initialize the sensor data
        if sensor in filtered_data:
            filtered_data[sensor] = np.concatenate((filtered_data[sensor], signal))
        else:
            filtered_data[sensor] = signal

        # Only extend 'Source' if we have a different source
        if last_source is None or source != last_source:
            source_column.extend([source] * len(signal))
            last_source = source

    filtered_data["Source"] = np.array(source_column)

    df = pd.DataFrame(filtered_data)

    # Create a DataFrame from the processed data
    # Extract 'Experiment' and 'Load' from the file path
    experiment, rpm_str = file_path.parent.relative_to(base_dir).parts
    fs = _infer_fs(file_path)
    label = _normalize_label(file_path.stem)
    family_meta = _parse_fault_family(file_path.stem)

    recordings: list[Recording] = []

    for source in np.unique(source_column):
        sub = df[df["Source"] == source].drop(columns=["Source"])
        # (channels, n_samples) float32
        data = sub.to_numpy(dtype=np.float32).T
        channels = sub.columns.to_list()

        rid = f"{experiment}/{rpm_str}/{source}/{file_path.stem}"

        recordings.append(
            Recording(
                rid=rid,
                data=data,  # shape (n_channels, n_samples)
                fs=fs,
                label=label,
                source="CWRU",
                unit="g",
                channels=channels,
                rpm=float(rpm_str) if rpm_str.isdigit() else None,
                meta={
                    "Experiment": experiment,
                    **family_meta,
                    "file_path": str(file_path),
                },
            )
        )

    return recordings


def load_cwru_dataset(
    base_dir: str,
    sensors: tuple[str, ...] = DEFAULT_SENSORS,
) -> list[Recording]:
    """
    Load all .mat files from the CWRU dataset into a list of recordings.

    Args:
        base_dir (str): Directory containing CWRU .mat files.
        sensors (tuple[str, ...]): Sensors to load from each file.
            Available sensors: AVAILABLE_SENSORS. Defaults to DE only.

    Returns:
        list[Recording]: Combined Recordings from all files.
    """
    invalid = set(sensors) - set(AVAILABLE_SENSORS)
    if invalid:
        invalid_list = ", ".join(sorted(invalid))
        raise ValueError(
            f"Unknown sensors requested: {invalid_list}. "
            f"Available sensors: {', '.join(AVAILABLE_SENSORS)}."
        )

    base_path = Path(base_dir)
    all_data = []

    for mat_file in base_path.rglob("*.mat"):
        print(f"Processing: {mat_file}")
        try:
            recordings_list = _load_cwru_file(mat_file, base_path, sensors)
            all_data += recordings_list
        except Exception as e:
            print(f"Failed to process {mat_file}: {e}")

    return all_data
