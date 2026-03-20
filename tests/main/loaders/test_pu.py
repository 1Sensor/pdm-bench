from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from pdm_tools.main.loaders.pu import _load_pu_file, load_pu_dataset


@pytest.fixture
def base_dir_fixture(tmp_path: Path) -> Generator[Path, None, None]:
    """
    A pytest fixture to create and return a base testing directory.
    This fixture is automatically cleaned up by pytest after tests run.
    """
    base_dir = tmp_path / "PU"
    base_dir.mkdir()
    yield base_dir


def _write_pu_mat(
    mat_path,
    *,
    times_by_raster: dict[str, np.ndarray] | None = None,
    signals: list[tuple[str, str, np.ndarray]] | None = None,
):
    """
    Create a minimal PU-like .mat structure that load_pu_file can read.
    - times_by_raster lets you define 1...N rasters (X field).
    - signals lets you define 1...N signals (Y field), each assigned to a raster.

    If not provided, defaults to:
      one raster HostService with time [0,1,2]
      one signal Vib on HostService with data [10,11,12]
    """
    stem = mat_path.stem
    if times_by_raster is None:
        times_by_raster = {"HostService": np.asarray([0, 1, 2], dtype=np.float32)}

    if signals is None:
        signals = [("Vib", "HostService", np.asarray([10, 11, 12], dtype=np.float32))]

    # --- X: time vectors per raster ---
    x_dtype = np.dtype([("Raster", "O"), ("Data", "O")])
    x = np.empty((1, len(times_by_raster)), dtype=x_dtype)

    for i, (raster, t) in enumerate(times_by_raster.items()):
        t = np.asarray(t, dtype=np.float32).reshape(1, -1)  # stored as (1, N)
        x[0, i]["Raster"] = np.array([raster], dtype=object)
        x[0, i]["Data"] = t

    # --- Y: signals ---
    y_dtype = np.dtype([("Name", "O"), ("Raster", "O"), ("Data", "O")])
    y = np.empty((1, len(signals)), dtype=y_dtype)

    for i, (name, raster, data) in enumerate(signals):
        data = np.asarray(data, dtype=np.float32).reshape(1, -1)  # stored as (1, N)
        y[0, i]["Name"] = np.array([name], dtype=object)
        y[0, i]["Raster"] = np.array([raster], dtype=object)
        y[0, i]["Data"] = data

    # --- root ---
    root_dtype = np.dtype([("Info", "O"), ("X", "O"), ("Y", "O"), ("Description", "O")])
    root = np.empty((1, 1), dtype=root_dtype)
    root[0, 0]["Info"] = np.array(["dummy"], dtype=object)
    root[0, 0]["X"] = x
    root[0, 0]["Y"] = y
    root[0, 0]["Description"] = np.array(["dummy"], dtype=object)

    savemat(mat_path, {stem: root})


def test_load_pu_file_multiple_rasters(base_dir_fixture: Path):
    mat_path = base_dir_fixture / "N15_M07_F10_KA04_1.mat"
    _write_pu_mat(
        mat_path,
        times_by_raster={
            "HostService": np.array([0, 1], dtype=np.float32),
            "Temp_1Hz": np.array([0, 1], dtype=np.float32),
        },
        signals=[
            ("V1", "HostService", np.array([1, 2], dtype=np.float32)),
            ("T1", "Temp_1Hz", np.array([3, 4], dtype=np.float32)),
        ],
    )

    recs = _load_pu_file(mat_path)
    assert len(recs) == 2
    assert {r.meta["raster"] for r in recs} == {"HostService", "Temp_1Hz"}


def test_load_pu_file_minimal(base_dir_fixture: Path):
    mat_path = base_dir_fixture / "N15_M07_F10_KI04_1.mat"
    _write_pu_mat(mat_path)

    recordings = _load_pu_file(mat_path)
    assert len(recordings) == 1
    rec = recordings[0]
    assert rec.source == "PU"
    assert rec.fs == 64_000
    assert rec.rpm == 1500
    assert rec.channels == ["Vib"]
    assert rec.data.shape == (1, 3)
    assert "HostService" in rec.rid
    assert rec.meta["operating_condition"] == "N15_M07_F10"
    assert rec.meta["damage_provenance"] == "lifetime"


@pytest.mark.parametrize(
    "fname,expected_label",
    [
        ("N15_M07_F10_K001_1.mat", "healthy"),  # K0...
        ("N15_M07_F10_KA04_1.mat", "outer_ring"),  # KA...
        ("N15_M07_F10_KI04_1.mat", "inner_ring"),  # KI...
        ("N15_M07_F10_KB04_1.mat", "multiple"),  # KB...
    ],
)
def test_load_pu_file_label_inference_via_public_api(
    base_dir_fixture, fname, expected_label
):
    mat_path = base_dir_fixture / fname
    _write_pu_mat(mat_path)

    recs = _load_pu_file(mat_path)
    assert len(recs) == 1
    assert recs[0].label == expected_label


def test_load_pu_file_unknown_bearing_code(base_dir_fixture):
    fname = "N15_M07_F10_K104_1.mat"  # invalid / unknown
    mat_path = base_dir_fixture / fname
    _write_pu_mat(mat_path)

    with pytest.raises(ValueError, match="Unknown bearing code"):
        _load_pu_file(mat_path)


def test_load_pu_file_unknown_raster_raises(base_dir_fixture: Path):
    mat_path = base_dir_fixture / "N15_M07_F10_KA04_1.mat"
    _write_pu_mat(
        mat_path,
        times_by_raster={"WeirdRaster": np.array([0, 1, 2], dtype=np.float32)},
        signals=[("V1", "WeirdRaster", np.array([1, 2, 3], dtype=np.float32))],
    )
    with pytest.raises(ValueError, match="Unknown raster"):
        _load_pu_file(mat_path)


def test_load_pu_file_unmatch_regex(base_dir_fixture: Path):
    mat_path = base_dir_fixture / "bad_name.mat"
    _write_pu_mat(mat_path)

    with pytest.raises(ValueError, match="Filename does not match"):
        _load_pu_file(mat_path)


def test_load_pu_file_two_signals_same_raster(base_dir_fixture: Path):
    mat_path = base_dir_fixture / "N15_M07_F10_KA04_1.mat"

    _write_pu_mat(
        mat_path,
        signals=[
            ("V1", "HostService", np.array([10, 11, 12], dtype=np.float32)),
            ("V2", "HostService", np.array([20, 21, 22], dtype=np.float32)),
        ],
    )

    recs = _load_pu_file(mat_path)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.meta["raster"] == "HostService"
    assert set(rec.channels) == {"V1", "V2"}
    assert rec.data.shape == (2, 3)


@pytest.mark.parametrize(
    ("fname", "expected_provenance"),
    [
        ("N15_M07_F10_K001_1.mat", "healthy"),
        ("N15_M07_F10_KA03_1.mat", "artificial"),
        ("N15_M07_F10_KA04_1.mat", "lifetime"),
    ],
)
def test_load_pu_file_extracts_damage_provenance(
    base_dir_fixture: Path,
    fname: str,
    expected_provenance: str,
):
    mat_path = base_dir_fixture / fname
    _write_pu_mat(mat_path)

    recs = _load_pu_file(mat_path)
    assert len(recs) == 1
    assert recs[0].meta["damage_provenance"] == expected_provenance


def test_load_pu_dataset_multiple_files(base_dir_fixture: Path):
    """
    Read 3 files, but one of them is corrupted.
    Ensure that corrupted file does not kill the loader.
    """
    f1 = base_dir_fixture / "N15_M07_F10_KA04_1.mat"
    f2 = base_dir_fixture / "N09_M01_F04_K001_2.mat"
    corrupted = base_dir_fixture / "corrupted.mat"
    _write_pu_mat(f1)
    _write_pu_mat(f2)
    corrupted.write_text("not a mat file")
    recordings = load_pu_dataset(str(base_dir_fixture))
    assert len(recordings) == 2
    assert {r.rpm for r in recordings} == {1500, 900}


def test_load_pu_file_no_valid_key(base_dir_fixture: Path):
    mat_path = base_dir_fixture / "N15_M07_F10_KA04_2.mat"
    savemat(mat_path, {"wrong_key": np.array([1, 2, 3])})

    with pytest.raises(ValueError, match="No valid data keys"):
        _load_pu_file(mat_path)


def test_load_pu_dataset_no_mat_files(base_dir_fixture: Path):
    with pytest.raises(FileNotFoundError, match="No \\.mat files found"):
        load_pu_dataset(str(base_dir_fixture))
