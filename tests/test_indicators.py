import numpy as np  # noqa: I001
import pandas as pd
import pytest
from pandas.core.series import Series

from pdm_bench.signals.indicators import (
    crest_factor,
    kurt,
    peak_to_peak,
    rms,
    shape_factor,
    std,
    freq_center,
    mean_freq,
    zero_to_peak,
    vrms,
    find_peaks,
    energy_operator,
    zero_order_fom,
    fourth_order_fom,
    sixth_order_fom,
    eight_order_fom,
    clearance_factor,
    impulse_indicator,
    rms_frequency,
    std_frequency,
    fourth_order_np,
    shannon_entropy,
    check_frequency_input,
)


@pytest.fixture
def input_zeros_array():
    return np.zeros(5)


@pytest.fixture
def input_ones_array():
    return np.ones(5)


@pytest.fixture
def input_arange_array():
    return np.arange(1, 5)


@pytest.fixture
def input_arange_array_diff():
    return np.array([1, 2, 5, 8])


@pytest.fixture
def input_series(input_arange_array):
    return pd.Series(input_arange_array, name="col")


@pytest.fixture
def input_series_diff(input_arange_array_diff):
    return pd.Series(input_arange_array_diff, name="col")


@pytest.fixture
def input_df(input_series: Series):
    return pd.DataFrame({"sig1": input_series, "sig2": input_series})


@pytest.fixture
def input_df_diff(input_series_diff: Series):
    return pd.DataFrame({"sig1": input_series_diff, "sig2": input_series_diff})


@pytest.fixture
def test_threshold():
    return 0.00001


@pytest.fixture
def test_threshold_big():
    return 0.001


# peak to peak
def test_zero_peak_to_peak(input_zeros_array):
    assert peak_to_peak(input_zeros_array) == 0.0


def test_ones_peak_to_peak(input_ones_array):
    assert peak_to_peak(input_ones_array) == 0.0


def test_nonzero_peak_to_peak(input_arange_array):
    assert peak_to_peak(input_arange_array) == 3.0


def test_peak_to_peak_with_series(input_series):
    assert peak_to_peak(input_series) == 3.0


def test_peak_to_peak_with_df(input_df):
    exp_col1, exp_col2 = input_df.apply(lambda col: peak_to_peak(col))
    assert exp_col1 == 3.0
    assert exp_col2 == 3.0


# zero to peak
def test_zero_zero_to_peak(input_zeros_array):
    assert zero_to_peak(input_zeros_array) == 0.0


def test_zero_to_peak_with_ones(input_ones_array):
    assert zero_to_peak(data=input_ones_array) == 1.0


def test_nonzero_zero_to_peak(input_arange_array):
    assert zero_to_peak(input_arange_array) == 4.0


def test_zero_to_peak_with_series(input_series):
    assert zero_to_peak(input_series) == 4.0


def test_zero_to_peak_with_df(input_df):
    exp_col1, exp_col2 = input_df.apply(lambda col: zero_to_peak(col))
    assert exp_col1 == 4.0
    assert exp_col2 == 4.0


# rms
def test_empty_rms():
    with pytest.raises(ValueError, match="Empty input data."):
        rms(np.array([]))


def test_zero_rms(input_zeros_array):
    assert rms(input_zeros_array) == 0.0


def test_ones_rms(input_ones_array):
    assert rms(input_ones_array) == 1.0


def test_nonzero_rms(input_arange_array, test_threshold):
    assert rms(input_arange_array) == pytest.approx(2.73861, test_threshold)


def test_rms_with_series(input_series, test_threshold):
    assert rms(input_series) == pytest.approx(2.73861, test_threshold)


def test_rms_with_df(input_df, test_threshold):
    exp_col1, exp_col2 = input_df.apply(lambda col: rms(col))
    assert exp_col1 == pytest.approx(2.73861, test_threshold)
    assert exp_col2 == pytest.approx(2.73861, test_threshold)


# crest factor
def test_zero_crest_factor(input_zeros_array):
    with pytest.raises(ValueError, match="RMS value equals zero."):
        crest_factor(input_zeros_array)


def test_ones_crest_factors(input_ones_array):
    assert crest_factor(input_ones_array) == 1.0


def test_nonzero_crest_factor(input_arange_array, test_threshold):
    assert crest_factor(input_arange_array) == pytest.approx(1.46059, test_threshold)


def test_crest_factor_with_series(input_series, test_threshold):
    assert crest_factor(input_series) == pytest.approx(1.46059, test_threshold)


def test_crest_factor_with_df(input_df, test_threshold):
    exp_col1, exp_col2 = input_df.apply(lambda col: crest_factor(col))
    assert exp_col1 == pytest.approx(1.46059, test_threshold)
    assert exp_col2 == pytest.approx(1.46059, test_threshold)


# std
def test_zero_std(input_zeros_array):
    assert std(input_zeros_array) == 0.0


def test_ones_std(input_ones_array):
    assert std(input_ones_array) == 0.0


def test_nonzero_std(input_arange_array, test_threshold):
    assert std(input_arange_array) == pytest.approx(1.11803, test_threshold)


def test_std_with_series(input_series, test_threshold):
    assert std(input_series) == pytest.approx(1.11803, test_threshold)


def test_std_with_df(input_df, test_threshold):
    exp_col1, exp_col2 = input_df.apply(lambda col: std(col))
    assert exp_col1 == pytest.approx(1.11803, test_threshold)
    assert exp_col2 == pytest.approx(1.11803, test_threshold)


# kurtosis
@pytest.fixture
def kurt_test_th():
    return 0.1


def test_normal_kurt(kurt_test_th):
    mu, sigma = 0, 1  # mean and standard deviation
    # generate normally distributed input vector
    input_array = np.random.default_rng(seed=42).normal(mu, sigma, 1000)

    # for normal distribution kurtosis should be close to 3
    assert kurt(input_array) == pytest.approx(3.0, kurt_test_th)


def test_laplace_kurt(kurt_test_th):
    loc, scale = 0.0, 1.0
    input_array = np.random.default_rng(seed=42).laplace(loc, scale, 10000)

    # for laplace distribution kurtosis should be close to 6
    assert kurt(input_array) == pytest.approx(6, kurt_test_th)


def test_uniform_kurt(kurt_test_th):
    low, high = 0, 1
    input_array = np.random.default_rng(seed=42).uniform(low, high, 1000)

    # for uniform distribution kurtosis should be close to 1.8
    assert kurt(input_array) == pytest.approx(1.8, kurt_test_th)


def test_normal_kurt_with_series(kurt_test_th):
    mu, sigma = 0, 1  # mean and standard deviation
    # generate normally distributed input vector
    input_array = np.random.default_rng(seed=42).normal(mu, sigma, 1000)
    input_series = pd.Series(input_array)

    # for normal distribution kurtosis should be close to 3
    assert kurt(input_series) == pytest.approx(3.0, kurt_test_th)


def test_normal_kurt_with_df(kurt_test_th):
    mu, sigma = 0, 1  # mean and standard deviation
    # generate normally distributed input vector
    input_array = np.random.default_rng(seed=42).normal(mu, sigma, 1000)
    input_series = pd.Series(input_array)
    input_df = pd.DataFrame({"sig1": input_series, "sig2": input_series})

    exp_col1, exp_col2 = input_df.apply(lambda col: kurt(col))
    assert exp_col1 == pytest.approx(3.0, kurt_test_th)
    assert exp_col2 == pytest.approx(3.0, kurt_test_th)


# shape factor
def test_empty_shape_factor():
    with pytest.raises(ValueError, match="Empty input data."):
        shape_factor(np.array([]))


def test_ones_shape_factor(input_ones_array):
    assert shape_factor(input_ones_array) == 1.0


def test_nonzero_shape_factor(input_arange_array, test_threshold):
    assert shape_factor(input_arange_array) == pytest.approx(1.09544, test_threshold)


def test_shape_factor_with_series(input_series, test_threshold):
    assert shape_factor(input_series) == pytest.approx(1.09544, test_threshold)


def test_shape_factor_with_df(input_df, test_threshold):
    exp_col1, exp_col2 = input_df.apply(lambda col: shape_factor(col))
    assert exp_col1 == pytest.approx(1.09544, test_threshold)
    assert exp_col2 == pytest.approx(1.09544, test_threshold)


# check_frequency_input
def test_check_frequency_input_valid_array(input_arange_array):
    result = check_frequency_input(input_arange_array, input_arange_array)
    assert result is None


def test_check_frequency_input_empty():
    with pytest.raises(
        ValueError,
        match="Frequency and magnitude must be one-dimensional and of non-zero, equal length.",
    ):
        check_frequency_input(np.array([]), np.array([]))


def test_check_frequency_input_mismatched_lengths(input_arange_array):
    with pytest.raises(
        ValueError,
        match="Frequency and magnitude must be one-dimensional and of non-zero, equal length.",
    ):
        check_frequency_input(input_arange_array, input_arange_array[:-1])


def test_check_frequency_input_non1d_magnitude(input_arange_array):
    with pytest.raises(
        ValueError,
        match="Frequency and magnitude must be one-dimensional and of non-zero, equal length.",
    ):
        check_frequency_input(input_arange_array.reshape(1, -1), input_arange_array)


def test_check_frequency_input_non1d_frequency(input_arange_array):
    with pytest.raises(
        ValueError,
        match="Frequency and magnitude must be one-dimensional and of non-zero, equal length.",
    ):
        check_frequency_input(input_arange_array, input_arange_array.reshape(1, -1))


def test_check_frequency_input_valid_series(input_series):
    result = check_frequency_input(input_series, input_series)
    assert result is None


# vrms
def test_vrms_empty():
    with pytest.raises(ValueError, match="Empty input data."):
        vrms(np.array([]), fs=1000, cutoff_start=0, cutoff_stop=100)


def test_vrms_basic(input_ones_array, test_threshold):
    acc = np.divide(input_ones_array, 0.0102 * 1.0)
    expected = (1 / (2 * np.pi * acc)) * np.cos(2 * np.pi * acc)
    result = vrms(input_ones_array, 1000.0, 0, 1000)
    assert result == pytest.approx(expected, test_threshold)


# find peaks
def test_empty_find_peaks():
    with pytest.raises(ValueError, match="Empty input data."):
        find_peaks(np.array([]), np.array([]))


def test_known_signal_find_peaks():
    mag = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0])
    freq = np.linspace(0, 100, mag.size)
    max_peaks, max_freq = find_peaks(mag, freq, peaks_num=1)
    assert max_peaks == np.array([4.0])
    assert max_freq == np.array(freq[4])


def test_find_peaks_fewer_than_requested():
    # Only one peak exists but we request three.
    freq = np.linspace(0, 1000, 1000)
    mag = np.full(1000, 0.1)
    mag[300] = 4.0
    max_peaks, max_freq = find_peaks(mag, freq, peaks_num=3)
    print(max_peaks, max_freq)
    np.testing.assert_allclose(max_peaks, [4.0, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(max_freq, [freq[300], np.nan, np.nan], equal_nan=True)


def test_find_peaks_custom_height():
    mag = np.array([0, 1, 2, 3, 2, 1, 0])
    freq = np.linspace(0, 10, len(mag))
    # Set height threshold above actual peaks → expect no peaks
    max_peaks, max_freq = find_peaks(mag, freq, height=5)
    assert np.isnan(max_peaks).all()
    assert np.isnan(max_freq).all()


def test_find_peaks_custom_distance():
    mag = np.array([0, 5, 0, 5, 0, 5])
    freq = np.linspace(0, 10, len(mag))
    max_peaks, max_freq = find_peaks(mag, freq, distance=3, peaks_num=3)
    np.testing.assert_allclose(max_peaks, [5.0, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(max_freq, [freq[3], np.nan, np.nan], equal_nan=True)


# energy operator
def test_energy_operator_empty():
    with pytest.raises(ValueError, match="Signal must contain at least 3 samples"):
        energy_operator(np.array([]))


def test_ones_energy_operator_constant(input_ones_array):
    # For a constant signal, the difference is always zero, so its kurtosis is undefined (we expect nan).
    assert np.isnan(energy_operator(input_ones_array))


def test_nonconstant_energy_operator_diff(input_arange_array_diff, test_threshold):
    assert energy_operator(input_arange_array_diff) == pytest.approx(
        1.0, test_threshold
    )


# zero_order_fom
def test_zero_order_fom_empty():
    with pytest.raises(ValueError, match="Empty input data."):
        zero_order_fom(
            np.array([]),
            np.array([]),
            np.array([]),
            fundamental_freq=100.0,
            n_harmonics=2,
        )


def test_zero_order_fom_basic(input_arange_array, test_threshold):
    freq = np.linspace(0, 1000, 1000)
    fft_amp = np.zeros(1000)
    fft_amp[np.argmin(np.abs(freq - 100))] = 5.0
    fft_amp[np.argmin(np.abs(freq - 200))] = 3.0
    assert zero_order_fom(
        input_arange_array, freq, fft_amp, fundamental_freq=100.0, n_harmonics=2
    ) == pytest.approx(0.375, rel=test_threshold)


def test_zero_order_fom_zero_harmonics(input_arange_array, test_threshold):
    with pytest.raises(ValueError, match="Denominator - sum of harmonics equals zero."):
        zero_order_fom(
            input_arange_array,
            np.linspace(0, 1000, 1000),
            np.zeros(1000),
            fundamental_freq=100.0,
            n_harmonics=2,
        )


# fourth_order_fom
def test_fourth_order_fom_empty():
    with pytest.raises(ValueError, match="Empty input data."):
        fourth_order_fom(np.array([]))


def test_ones_fourth_order_fom(input_ones_array):
    # For a constant signal (e.g., [1, 1, 1, 1, 1]), the diff is [0, 0, 0, 0], so the kurtosis is undefined.
    assert np.isnan(fourth_order_fom(input_ones_array))


def test_nonzero_fourth_order_fom(input_arange_array_diff, test_threshold):
    assert fourth_order_fom(input_arange_array_diff) == pytest.approx(
        1.5, rel=test_threshold
    )


def test_fourth_order_fom_with_df(input_df_diff, test_threshold):
    exp_col1, exp_col2 = input_df_diff.apply(lambda col: fourth_order_fom(col))
    assert exp_col1 == pytest.approx(1.5, test_threshold)
    assert exp_col2 == pytest.approx(1.5, test_threshold)


# sixth_order_fom
def test_sixth_order_fom_empty():
    with pytest.raises(ValueError, match="Empty input data."):
        sixth_order_fom(np.array([]))


def test_ones_sixth_order_fom(input_ones_array):
    # For a constant signal (e.g., [1, 1, 1, 1, 1]), the diff is [0, 0, 0, 0], so the kurtosis is undefined.
    with pytest.raises(ValueError, match="Denominator - variance equals zero."):
        sixth_order_fom(input_ones_array)


def test_nonzero_sixth_order_fom(input_arange_array_diff, test_threshold):
    assert sixth_order_fom(input_arange_array_diff) == pytest.approx(
        2.75, rel=test_threshold
    )


def test_sixth_order_fom_with_df(input_df_diff, test_threshold):
    exp_col1, exp_col2 = input_df_diff.apply(lambda col: sixth_order_fom(col))
    assert exp_col1 == pytest.approx(2.75, test_threshold)
    assert exp_col2 == pytest.approx(2.75, test_threshold)


# eight_order_fom
def test_eight_order_fom_empty():
    with pytest.raises(ValueError, match="Empty input data."):
        eight_order_fom(np.array([]))


def test_ones_eight_order_fom(input_ones_array):
    # For a constant signal (e.g., [1, 1, 1, 1, 1]), the diff is [0, 0, 0, 0], so the kurtosis is undefined.
    with pytest.raises(ValueError, match="Denominator - variance equals zero."):
        eight_order_fom(input_ones_array)


def test_nonzero_eight_order_fom(input_arange_array_diff, test_threshold):
    assert eight_order_fom(input_arange_array_diff) == pytest.approx(
        43 / 24, rel=test_threshold
    )


def test_eight_order_fom_with_df(input_df_diff, test_threshold):
    exp_col1, exp_col2 = input_df_diff.apply(lambda col: eight_order_fom(col))
    assert exp_col1 == pytest.approx(43 / 24, test_threshold)
    assert exp_col2 == pytest.approx(43 / 24, test_threshold)


# clearance_factor
def test_clearance_factor_empty():
    with pytest.raises(ValueError, match="Empty input data."):
        clearance_factor(np.array([]))


def test_zeros_clearance_factor(input_zeros_array):
    with pytest.raises(ValueError, match="Denominator value equals zero."):
        clearance_factor(input_zeros_array)


def test_ones_clearance_factor(input_ones_array):
    # For an array of ones, np.sqrt(1) = 1, mean = 1, max = 1, so clearance factor = 1/1^2 = 1.
    assert clearance_factor(input_ones_array) == pytest.approx(1.0)


def test_nonzero_clearance_factor(input_arange_array, test_threshold):
    expected = 4 / (((1 + np.sqrt(2) + np.sqrt(3) + 2) / 4) ** 2)
    assert clearance_factor(input_arange_array) == pytest.approx(
        expected, rel=test_threshold
    )


def test_clearance_factor_with_df(input_df, test_threshold=1e-5):
    exp_col1, exp_col2 = input_df.apply(lambda col: clearance_factor(col))
    expected = 4 / (((1 + np.sqrt(2) + np.sqrt(3) + 2) / 4) ** 2)
    assert exp_col1 == pytest.approx(expected, rel=test_threshold)
    assert exp_col2 == pytest.approx(expected, rel=test_threshold)


# impulse_indicator
def test_impulse_indicator_empty():
    with pytest.raises(ValueError, match="Empty input data."):
        impulse_indicator(np.array([]))


def test_zeros_impulse_indicator(input_zeros_array):
    with pytest.raises(ValueError, match="Mean value equals zero."):
        impulse_indicator(input_zeros_array)


def test_nonzero_impulse_indicator(input_arange_array, test_threshold):
    assert impulse_indicator(input_arange_array) == pytest.approx(
        1.6, rel=test_threshold
    )


def test_impulse_indicator_with_df(input_df, test_threshold):
    exp_col1, exp_col2 = input_df.apply(lambda col: impulse_indicator(col))
    assert exp_col1 == pytest.approx(1.6, test_threshold)
    assert exp_col2 == pytest.approx(1.6, test_threshold)


# mean_freq
def test_mean_freq_zero_sum(input_zeros_array):
    freq = np.linspace(0, 100, len(input_zeros_array))
    with pytest.raises(
        ValueError, match="Denominator - sum of magnitudes equals zero."
    ):
        mean_freq(input_zeros_array, freq)


def test_mean_freq_basic(input_arange_array, test_threshold):
    freq = np.linspace(10, 40, len(input_arange_array))
    assert mean_freq(input_arange_array, freq) == pytest.approx(30, rel=test_threshold)


def test_mean_freq_with_df(input_df, test_threshold):
    freq = np.linspace(10, 40, len(input_df))
    exp_col1, exp_col2 = input_df.apply(lambda col: mean_freq(col, freq))
    assert exp_col1 == pytest.approx(30, test_threshold)
    assert exp_col2 == pytest.approx(30, test_threshold)


# freq_center
def test_freq_center_zero_sum(input_zeros_array):
    freq = np.linspace(0, 100, len(input_zeros_array))
    with pytest.raises(
        ValueError, match="Denominator - sum of magnitudes and frequencies equals zero."
    ):
        freq_center(input_zeros_array, freq)


def test_freq_center_basic(input_arange_array, test_threshold):
    freq = np.linspace(10, 40, len(input_arange_array))
    assert freq_center(input_arange_array, freq) == pytest.approx(
        10000 / 300, rel=test_threshold
    )


def test_freq_center_with_df(input_df, test_threshold):
    freq = np.linspace(10, 40, len(input_df))
    exp_col1, exp_col2 = input_df.apply(lambda col: freq_center(col, freq))
    assert exp_col1 == pytest.approx(10000 / 300, test_threshold)
    assert exp_col2 == pytest.approx(10000 / 300, test_threshold)


# rms_frequency
def test_rms_frequency_zero_sum(input_zeros_array):
    freq = np.linspace(0, 100, len(input_zeros_array))
    with pytest.raises(ValueError, match="Sum of magnitudes equals zero."):
        rms_frequency(input_zeros_array, freq)


def test_rms_frequency_basic(input_arange_array, test_threshold):
    freq = np.linspace(10, 40, len(input_arange_array))
    assert rms_frequency(input_arange_array, freq) == pytest.approx(
        np.sqrt(10000 / 10), rel=test_threshold
    )


def test_rms_frequency_with_df(input_df, test_threshold):
    freq = np.linspace(10, 40, len(input_df))
    exp_col1, exp_col2 = input_df.apply(lambda col: rms_frequency(col, freq))
    assert exp_col1 == pytest.approx(np.sqrt(10000 / 10), test_threshold)
    assert exp_col2 == pytest.approx(np.sqrt(10000 / 10), test_threshold)


# std_frequency
def test_std_frequency_zero_sum():
    mag = np.array([-2, -1, 0, 1, 2])
    freq = np.linspace(0, 100, len(mag))
    with pytest.raises(ValueError, match="Sum of magnitudes equals zero."):
        std_frequency(mag, freq)


def test_std_frequency_basic(input_arange_array, test_threshold):
    freq = np.linspace(10, 40, len(input_arange_array))
    assert std_frequency(input_arange_array, freq) == pytest.approx(
        10.540925, rel=test_threshold
    )


def test_std_frequency_with_df(input_df, test_threshold):
    freq = np.linspace(10, 40, len(input_df))
    exp_col1, exp_col2 = input_df.apply(lambda col: std_frequency(col, freq))
    assert exp_col1 == pytest.approx(10.540925, test_threshold)
    assert exp_col2 == pytest.approx(10.540925, test_threshold)


# fourth_order_np
def test_fourth_order_np_empty():
    with pytest.raises(ValueError, match="Empty input data."):
        fourth_order_np(np.array([]))


def test_zeros_fourth_order_np(input_zeros_array):
    with pytest.raises(ValueError, match="Denominator - power std equals zero."):
        fourth_order_np(input_zeros_array)


def test_fourth_order_np_basic(input_arange_array, test_threshold_big):
    assert fourth_order_np(input_arange_array) == pytest.approx(
        -1.28, rel=test_threshold_big
    )


def test_fourth_order_np_with_df(input_df, test_threshold_big):
    exp_col1, exp_col2 = input_df.apply(lambda col: fourth_order_np(col))
    assert exp_col1 == pytest.approx(-1.28, test_threshold_big)
    assert exp_col2 == pytest.approx(-1.28, test_threshold_big)


# shannon_entropy
def test_shannon_entropy_empty():
    with pytest.raises(ValueError, match="Empty input data."):
        shannon_entropy(np.array([]))


def test_shannon_entropy_zero_total_energy(input_zeros_array):
    with pytest.raises(ValueError, match="Total energy equals zero."):
        shannon_entropy(input_zeros_array)


def test_shannon_entropy_ones(input_ones_array, test_threshold):
    assert shannon_entropy(input_ones_array) == pytest.approx(
        np.log(len(input_ones_array)), test_threshold
    )
