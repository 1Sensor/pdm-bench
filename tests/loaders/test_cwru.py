from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from pdm_bench.loaders.cwru import (
    DEFAULT_SENSORS,
    _load_cwru_file,
    load_cwru_dataset,
)


@pytest.fixture
def base_dir_fixture(tmp_path: Path) -> Generator[Path, None, None]:
    """
    A pytest fixture to create and return a base testing directory.
    This fixture is automatically cleaned up by pytest after tests run.
    """
    base_dir = tmp_path / "CWRU"
    base_dir.mkdir()
    yield base_dir


@pytest.mark.parametrize(
    "file_name,expected_label",
    [
        ("0.21-InnerRace.mat", "inner_race"),  # typical naming
        ("Normal.mat", "normal"),  # no hyphen -> fallback
    ],
)
def test_load_cwru_file_minimal_mat(
    base_dir_fixture: Path,
    file_name: str,
    expected_label: str,
):
    """
    Test that load_cwru_file can handle small .mat files with expected or unexpected naming.
    The test is parametrized with two variations of the file name.
    """
    # 1. Create subdirectories: base_dir/Experiment1/Load1
    experiment_dir = base_dir_fixture / "Experiment1"
    load_dir = experiment_dir / "Load1"
    load_dir.mkdir(parents=True, exist_ok=True)

    # 2. Create minimal .mat data with just a couple of relevant keys
    mat_content = {
        "X121_DE_time": np.array([1, 2, 3]),  # Should produce one column "time"
        "X121_FE_time": np.array([4, 5, 6]),  # Should also produce "time" data
    }
    mat_path = load_dir / file_name

    # 3. Write this minimal data to disk
    savemat(mat_path, mat_content)

    recordings = _load_cwru_file(mat_path, base_dir_fixture, sensors=("DE", "FE"))

    # 5. Validate the results
    assert len(recordings) == 1

    rec = recordings[0]
    assert set(rec.channels) == {"DE_time", "FE_time"}
    assert rec.data.shape == (2, 3)  # (channels, samples)
    assert rec.label == expected_label
    assert rec.rid.startswith("Experiment1/Load1/X121/")

    # Metadata from path
    assert rec.meta["Experiment"] == "Experiment1"
    assert rec.meta["recording_family"] == file_name.removesuffix(".mat")
    assert str(mat_path) == rec.meta["file_path"]


def test_load_cwru_file_extracts_fault_family_metadata(base_dir_fixture: Path):
    exp_dir = base_dir_fixture / "12DriveEndFault" / "1750"
    exp_dir.mkdir(parents=True, exist_ok=True)

    mat_path = exp_dir / "0.021-OuterRace12.mat"
    savemat(mat_path, {"X101_DE_time": np.array([1, 2, 3])})

    recordings = _load_cwru_file(mat_path, base_dir_fixture, sensors=("DE",))

    assert len(recordings) == 1
    rec = recordings[0]
    assert rec.meta["recording_family"] == "0.021-OuterRace12"
    assert rec.meta["fault_severity_in"] == pytest.approx(0.021)
    assert rec.meta["fault_location_variant"] == "12"


def test_load_cwru_file_no_valid_keys(base_dir_fixture: Path):
    """
    Test that load_cwru_file correctly handles a .mat file with no valid keys
    (keys do not contain 'DE', 'FE', or 'BA').
    """
    exp_dir = base_dir_fixture / "ExpA" / "LoadA"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create a .mat file with only irrelevant_data
    mat_content = {
        "irrelevant_data": np.array([10, 20, 30]),
    }
    mat_path = exp_dir / "NoValidKeys.mat"
    savemat(mat_path, mat_content)

    recordings = _load_cwru_file(mat_path, base_dir_fixture, DEFAULT_SENSORS)

    # No valid sensors → no recordings
    assert recordings == []


def test_load_cwru_dataset_multiple_files(base_dir_fixture: Path):
    """
    Test that load_cwru_dataset aggregates multiple files and skips corrupted ones.
    """
    # Create Exp1/Load1 Exp2/Load2 and Exp3/Load3
    exp1_load1 = base_dir_fixture / "Exp1" / "Load1"
    exp2_load2 = base_dir_fixture / "Exp2" / "Load2"
    exp3_load3 = base_dir_fixture / "Exp3" / "Load3"
    exp1_load1.mkdir(parents=True, exist_ok=True)
    exp2_load2.mkdir(parents=True, exist_ok=True)
    exp3_load3.mkdir(parents=True, exist_ok=True)

    # Valid file 1
    valid_file_1 = exp1_load1 / "0.07-OuterRace.mat"
    savemat(valid_file_1, {"X101_DE_time": np.array([1, 2])})

    # Valid file 2
    valid_file_2 = exp2_load2 / "0.017-InnerRace.mat"
    savemat(valid_file_2, {"X101_FE_time": np.array([3, 4, 5])})

    # Valid file 3
    valid_file_3 = exp3_load3 / "0.027-Ball.mat"
    savemat(valid_file_3, {"X101_FE_time": np.array([6, 7, 8])})

    # Corrupted file (empty, not a valid .mat)
    corrupted_file = exp2_load2 / "corrupted.mat"
    corrupted_file.touch()

    # Load all .mat files in the base directory
    recordings = load_cwru_dataset(str(base_dir_fixture), sensors=("DE", "FE"))

    # Three valid files → three recordings
    assert len(recordings) == 3
    assert {rec.meta["Experiment"] for rec in recordings} == {"Exp1", "Exp2", "Exp3"}
    samples = sorted(rec.data.shape[1] for rec in recordings)
    assert samples == [2, 3, 3]


@pytest.mark.parametrize(
    "keys_in_file,expected_data_count",
    [
        ({"ABCDE_time": np.array([10, 20])}, 2),  # 'DE' substring triggers parse
        ({"XBA_freq": np.array([30, 40, 50])}, 3),  # 'BA' substring triggers parse
        ({"123FE_data": np.array([60])}, 1),  # 'FE' substring triggers parse
    ],
)
def test_load_cwru_file_edge_case_keys(
    base_dir_fixture: Path, keys_in_file, expected_data_count
):
    """
    Test that partial matches of 'DE', 'FE', or 'BA' in the key name
    are correctly recognized (e.g., 'ABCDE_time', '123FE_data').
    """
    exp_dir = base_dir_fixture / "ExpZ" / "LoadZ"
    exp_dir.mkdir(parents=True, exist_ok=True)

    mat_path = exp_dir / "edge_case.mat"
    savemat(mat_path, keys_in_file)

    recordings = _load_cwru_file(mat_path, base_dir_fixture, sensors=("DE", "FE", "BA"))

    # One valid key → one recording
    assert len(recordings) == 1
    rec = recordings[0]
    # Sample count preserved
    assert rec.data.shape[1] == expected_data_count
    # Metadata still correct
    assert rec.meta["Experiment"] == "ExpZ"


@pytest.mark.parametrize(
    "folder_name,expected_fs",
    [
        ("48DriveEndFault", 48_000.0),
        ("12DriveEndFault", 12_000.0),
        ("12FanEndFault", 12_000.0),
        ("NormalBaseline", 12_000.0),
    ],
)
def test_infer_fs_from_path(base_dir_fixture, folder_name, expected_fs):
    exp_dir = base_dir_fixture / folder_name / "1750"
    exp_dir.mkdir(parents=True, exist_ok=True)

    mat_path = exp_dir / "Normal.mat"
    savemat(mat_path, {"X101_DE_time": np.array([1, 2, 3])})

    recs = _load_cwru_file(mat_path, base_dir_fixture, sensors=("DE",))
    assert len(recs) == 1
    assert recs[0].fs == expected_fs


def test_load_cwru_dataset_invalid_sensor_raises(base_dir_fixture):
    with pytest.raises(ValueError, match="Unknown sensors requested"):
        load_cwru_dataset(str(base_dir_fixture), sensors=("DE", "XYZ"))
