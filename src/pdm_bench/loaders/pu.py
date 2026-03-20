import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from pdm_bench.signals.recordings import Recording


"""
Sampling frequencies, consistent among measurements:
- motor current + vibrations: 64 kHz
- mechanic parameters (load force, load torque, speed): 4 kHz
- temperature: 1 Hz
"""
RASTER_FS_HZ = {
    "HostService": 64_000,
    "Mech_4kHz": 4_000,
    "Temp_1Hz": 1,
}

"""Operating params"""
SPEED_CODE_TO_RPM = {"N09": 900, "N15": 1500}
TORQUE_CODE_TO_NM = {"M07": 0.7, "M01": 0.1}
FORCE_CODE_TO_N = {"F10": 1000, "F04": 400}

_FILENAME_REGEX = re.compile(
    r"^(?P<speed>N\d{2})_(?P<torque>M\d{2})_(?P<force>F\d{2})_(?P<bearing>K[ABI]?\d{2,3})_(?P<meas>\d+)$"
)

ARTIFICIAL_BEARING_CODES = {
    "KA01",
    "KA03",
    "KA05",
    "KA06",
    "KA07",
    "KA08",
    "KA09",
    "KI01",
    "KI03",
    "KI05",
    "KI07",
    "KI08",
}

LIFETIME_BEARING_CODES = {
    "KA04",
    "KA15",
    "KA16",
    "KA22",
    "KA30",
    "KB23",
    "KB24",
    "KB27",
    "KI04",
    "KI14",
    "KI16",
    "KI17",
    "KI18",
    "KI21",
}


def _mat_to_1d_float(sig_data: Any) -> np.ndarray:
    """Reshape data, as it comes in form of (1, N) and cast to float32."""
    arr = np.asarray(sig_data)
    return arr.reshape(-1).astype(np.float32)


def _infer_label_from_bearing_code(bearing_code: str) -> str:
    """
    Possible labels:
    - K00x: healthy,
    - KAxx: outer ring damage (artificial or real),
    - KIxx: inner ring damage (artificial or real),
    - KBxx: multiple damages, inner and outer ring.
    """
    if bearing_code.startswith("K0"):
        return "healthy"
    if bearing_code.startswith("KA"):
        return "outer_ring"
    if bearing_code.startswith("KI"):
        return "inner_ring"
    if bearing_code.startswith("KB"):
        return "multiple"
    raise ValueError(f"Unknown bearing code: {bearing_code}")


def _parse_pu_filename(stem: str) -> dict[str, Any]:
    """
    Extract info from file name (speed, torque, force, bearing, measurement)
    """
    matches = _FILENAME_REGEX.match(stem)
    if not matches:
        raise ValueError(f"Filename does not match expected PU pattern: {stem}")

    speed_code = matches.group("speed")
    torque_code = matches.group("torque")
    force_code = matches.group("force")
    bearing = matches.group("bearing")
    meas = int(matches.group("meas"))

    setting_id = f"{speed_code}_{torque_code}_{force_code}"

    return {
        "setting_id": setting_id,
        "operating_condition": setting_id,
        "speed_rpm": SPEED_CODE_TO_RPM.get(speed_code),
        "torque_nm": TORQUE_CODE_TO_NM.get(torque_code),
        "radial_force_n": FORCE_CODE_TO_N.get(force_code),
        "bearing_code": bearing,
        "measurement_no": meas,
    }


def _infer_damage_provenance(bearing_code: str) -> str:
    """Map PU bearing codes to healthy / artificial / lifetime provenance groups."""
    if bearing_code.startswith("K0"):
        return "healthy"
    if bearing_code in ARTIFICIAL_BEARING_CODES:
        return "artificial"
    if bearing_code in LIFETIME_BEARING_CODES:
        return "lifetime"
    return "unknown"


def _extract_signals_by_raster(
    mat_record: np.void,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """
    Extract signals from a PU .mat record grouped by raster.
    For each raster (HostService, Mech_4kHz, Temp_1Hz), X contains the time vector and
    Y contains the corresponding signal channels.

    Args:
        mat_record (np.void): Single record loaded from a PU .mat file.

    Returns:
        tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
            - signals_by_raster (dict): Raster -> channel name -> 1D signal array.
            - time_by_raster (dict): Raster -> 1D time vector.
    """
    signals_by_raster = {}
    time_by_raster = {}

    x = np.asarray(mat_record["X"])
    for time_vector in range(x.shape[1]):
        sig = x[0, time_vector]
        raster = sig["Raster"].flat[0].item()
        t = _mat_to_1d_float(sig["Data"])
        time_by_raster[raster] = t

    y = np.asarray(mat_record["Y"])
    for signal in range(y.shape[1]):
        sig = y[0, signal]
        name = sig["Name"].flat[0].item()
        raster = sig["Raster"].flat[0].item()
        data = _mat_to_1d_float(sig["Data"])
        if raster not in signals_by_raster:
            signals_by_raster[raster] = {}
        signals_by_raster[raster][name] = data

    return signals_by_raster, time_by_raster


def _load_pu_file(file_path: Path) -> list[Recording]:
    """
    Load a single PU .mat file and extract its data. Each file will contain 3 recordings,
    depending on the raster, which determines sampling frequency of signals.

    Args:
        file_path (Path): Path to the .mat file.

    Returns:
        list[Recording]: Returns list of recordings loaded from a file.
    """
    mat_data = loadmat(file_path)

    # in file there is one valid data key, the same as file name(stem of file_path)
    stem = file_path.stem
    if stem in mat_data:
        root = mat_data[stem]
    else:
        raise ValueError("No valid data keys in .mat")

    # root is a ndarray (1,1) with dtype of fields Info/X/Y/Description
    recording = root[0, 0]
    info = _parse_pu_filename(stem)
    label = _infer_label_from_bearing_code(info["bearing_code"])
    damage_provenance = _infer_damage_provenance(info["bearing_code"])

    signals_by_raster, time_by_raster = _extract_signals_by_raster(recording)

    group_id = f"{info['setting_id']}/{info['bearing_code']}/{info['measurement_no']}"

    recordings: list[Recording] = []

    for raster, channel_map in signals_by_raster.items():
        fs = RASTER_FS_HZ.get(raster)
        if fs is None:
            raise ValueError(f"Unknown raster '{raster}' in PU file: {file_path}")
        channels = list(channel_map.keys())
        data = np.vstack([channel_map[channel] for channel in channels]).astype(
            np.float32, copy=False
        )

        raster_rid = f"{group_id}::{raster}"

        t = time_by_raster.get(raster)

        recordings.append(
            Recording(
                rid=raster_rid,
                data=data,
                fs=fs,
                label=label,
                source="PU",
                unit="mixed",
                channels=channels,
                rpm=info["speed_rpm"],
                meta={
                    "group_id": group_id,
                    "raster": raster,
                    "operating_condition": info["operating_condition"],
                    "damage_provenance": damage_provenance,
                    "torque_nm": info["torque_nm"],
                    "radial_force_n": info["radial_force_n"],
                    "bearing_code": info["bearing_code"],
                    "measurement_no": info["measurement_no"],
                    "file_path": str(file_path),
                    "t0_s": float(t[0]),
                    "dt_s": 1.0 / float(fs),
                },
            )
        )

    return recordings


def load_pu_dataset(base_dir: str) -> list[Recording]:
    """
    Load all .mat files from the PU dataset into a list of recordings.

    Args:
        base_dir (str): Directory containing PU bearing coded folders with .mat files.

    Returns:
        list[Recording]: Combined Recordings from all files.
    """
    base_path = Path(base_dir)
    all_data = []

    mat_files = list(base_path.rglob("*.mat"))
    if not mat_files:
        raise FileNotFoundError(f"No .mat files found under: {base_path}")

    for mat_file in mat_files:
        print(f"Processing: {mat_file}")
        try:
            recordings_list = _load_pu_file(mat_file)
            all_data += recordings_list
        except Exception as e:
            print(f"Failed to process {mat_file}: {e}")

    return all_data
