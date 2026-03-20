import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal
from pandas.testing import assert_frame_equal

from pdm_bench.signals.indicators import scale


@pytest.fixture
def input_array():
    return np.ones(5)


@pytest.fixture
def input_series(input_array):
    return pd.Series(input_array)


@pytest.fixture
def input_df(input_series):
    return pd.DataFrame({"sig1": input_series, "sig2": input_series})


def test_default_params_scaling(input_array):
    scaled_data = scale(data=input_array)

    exp_value = 98.03921568
    excepted_data = np.full(5, exp_value)
    assert_almost_equal(scaled_data, excepted_data)


def test_custom_params_scalling(input_array):
    scaled_data = scale(data=input_array, amplification=10.0, sensitivity=0.1)

    exp_value = 1.0
    excepted_data = np.full(5, exp_value)
    assert_almost_equal(scaled_data, excepted_data)


def test_series_scalling(input_series):
    scaled_data = scale(data=input_series, amplification=10.0, sensitivity=0.1)

    exp_value = 1.0
    excepted_data = np.full(5, exp_value)
    assert_almost_equal(scaled_data, excepted_data)


def test_df_scalling(input_df):
    scaled_df = input_df.apply(lambda col: scale(col))

    exp_value = 98.03921568
    expected_array = np.full(5, exp_value)
    expected_df = pd.DataFrame({"sig1": expected_array, "sig2": expected_array})
    assert_frame_equal(scaled_df, expected_df)


def test_negative_sensitivity(input_array):
    sens = -1.0

    with pytest.raises(
        ValueError, match=f"Sensitivity must be a positive value. Equals to {sens}"
    ):
        scale(data=input_array, sensitivity=sens)


def test_negative_amplification(input_array):
    amp = -1.0

    with pytest.raises(
        ValueError, match=f"Amplification must be a positive value. Equals to {amp}"
    ):
        scale(data=input_array, amplification=amp)


def test_zero_sensitivity(input_array):
    sens = 0.0

    with pytest.raises(
        ValueError, match=f"Sensitivity must be a positive value. Equals to {sens}"
    ):
        scale(data=input_array, sensitivity=sens)


def test_zero_amplification(input_array):
    amp = 0.0

    with pytest.raises(
        ValueError, match=f"Amplification must be a positive value. Equals to {amp}"
    ):
        scale(data=input_array, amplification=amp)
