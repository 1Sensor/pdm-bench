"""
Acceleration based condition inidicators.

Most of the indicators implemenations are based on the ones found in below paper.

Vikas Sharma, Anand Parey,
A Review of Gear Fault Diagnosis Using Various Condition Indicators,
Procedia Engineering,
Volume 144,
2016,
Pages 253-263,
ISSN 1877-7058,
https://doi.org/10.1016/j.proeng.2016.05.131.
(https://www.sciencedirect.com/science/article/pii/S1877705816303484)
"""

import numpy as np
from numpy import ndarray
from pandas import Series
from scipy import signal
from scipy.stats import kurtosis

from pdm_tools.main.scaler import scale


def check_frequency_input(magnitude: ndarray | Series, frequency: ndarray | Series):
    """
    Check that the provided magnitude and frequency arrays are both non-empty, one-dimensional, and of equal length.

    Args:
        magnitude (ndarray | Series): Magnitude values.
        frequency (ndarray | Series): Frequency values corresponding to the magnitudes.

    Returns:
        0
    """
    if (
        frequency.ndim != 1
        or magnitude.ndim != 1
        or frequency.shape[0] != magnitude.shape[0]
        or frequency.shape[0] == 0
    ):
        raise ValueError(
            "Frequency and magnitude must be one-dimensional and of non-zero, equal length."
        )

    return None


def peak_to_peak(data: ndarray | Series) -> ndarray:
    """Peak to peak.

    Equals to difference between maximum and minimum value of the signal.

    Args:
        data (ndarray | Series): input acceleration signal

    Returns:
        ndarray: calculated peak to peak value
    """
    return np.ptp(data)


def zero_to_peak(data: ndarray | Series) -> ndarray:
    """Zero to peak.

    Equals to maximum value of the signal.

    Args:
        data (ndarray | Series): input acceleration signal

    Returns:
        ndarray: calculated zero to peak value
    """
    return np.max(data)


def rms(data: ndarray | Series) -> ndarray:
    """Calculates root mean square value of acceleration signal.

    Reflects the vibration amplitude and energy of signal in the time domain.

    Formula:
        RMS = sqrt(sum(x^2)/N), where: N - length, x - data

    Args:
        data (ndarray | Series): input acceleration signal

    Raises:
        ValueError: RMS value equals zero.

    Returns:
        ndarray: calculated rms value
    """
    if len(data) == 0:
        raise ValueError("Empty input data.")
    return np.sqrt(np.divide(np.sum(np.square(data)), len(data)))


def crest_factor(data: ndarray | Series) -> ndarray:
    """Crest factor.

    Peak value divided by the RMS. Faults often first manifest themselves in changes in the peakiness of a signal
    before they manifest in the energy represented by the signal root mean squared.
    The crest factor can provide an early warning for faults when they first develop.

    Args:
        data (ndarray | Series): input acceleration signal

    Raises:
        ValueError: RMS value equals zero.

    Returns:
        ndarray: calculated crest factor
    """
    if rms(data) == 0:
        raise ValueError("RMS value equals zero.")
    return np.max(data) / rms(data)


def std(data: ndarray | Series) -> ndarray:
    """Standard deviation.

    Args:
        data (ndarray | Series): input acceleration signal

    Returns:
        ndarray: calculated standard deviation
    """
    return np.std(data)


def kurt(data: ndarray | Series) -> ndarray:
    """Kurtosis.

    Kurtosis is the fourth order normalized moment of a given signal and provides a measure of the
    peakedness of the signal. Developing faults can increase the number of outliers,
    and therefore increase the value of the kurtosis metric.

    Args:
        data (ndarray | Series): input acceleration signal

    Returns:
        ndarray: calculated kurtosis
    """
    return kurtosis(data, fisher=False)


def shape_factor(data: ndarray | Series) -> ndarray:
    """Shape factor

    Used to represent the time series distribution of the signal in the time domain.

    Args:
        data (ndarray | Series): input acceleration signal

    Returns:
        ndarray: calculated shape factor
    """
    return rms(data) / np.divide(np.sum(np.abs(data)), len(data))


def mean_freq(magnitude: ndarray | Series, frequency: ndarray | Series) -> ndarray:
    """Mean frequency

    It is a frequency domain parameter, extracted from the frequency spectrum of the gear vibration signal.
    MF indicates the vibration energy in the frequency domain.

    Args:
        magnitude (ndarray | Series): amplitude of the spectrum at each frequency bin
        frequency (ndarray | Series): corresponding frequency values

    Returns:
        ndarray: calculated mean frequency
    """
    check_frequency_input(magnitude, frequency)

    if np.sum(magnitude) == 0:
        raise ValueError("Denominator - sum of magnitudes equals zero.")
    return np.sum(magnitude * frequency) / np.sum(magnitude)


def freq_center(magnitude: ndarray | Series, frequency: ndarray | Series) -> ndarray:
    """Frequency center

    FC shows the position changes of the main frequencies.

    Args:
        magnitude (ndarray | Series): amplitude of the spectrum at each frequency bin
        frequency (ndarray | Series): corresponding frequency values

    Returns:
        ndarray: calculated frequency center
    """
    check_frequency_input(magnitude, frequency)

    if np.sum(magnitude * frequency) == 0:
        raise ValueError("Denominator - sum of magnitudes and frequencies equals zero.")

    return np.sum(frequency**2 * magnitude) / np.sum(frequency * magnitude)


def vrms(
    data: ndarray | Series,
    fs: float,
    cutoff_start: float,
    cutoff_stop: float,
    sensitivity: float = 0.0102,
    amplification: float = 1.0,
) -> ndarray:
    """Velocity RMS.

    Args:
        data (ndarray | Series): input acceleration signal, in [V]
        fs (float): input signal's sampling frequency
        cutoff_start (float): bandpass cutoff start frequency
        cutoff_stop (float): bandpass cutoff stop frequency
        sensitivity (float|None): sensitivity of used sensor, in V/(m/s^2), Defaults to 0.0102.
        amplification (float|None): amplification set during measurement. Defaults to 1.0.

    Returns:
        ndarray: calculated VRMS
    """
    if len(data) == 0:
        raise ValueError("Empty input data.")

    acceleration = scale(
        data, sensitivity, amplification
    )  # Scale acceleeration to m/s^2
    # Scale extracted Acceleration components to Velocity (omega arithmetics)
    velocity = (1 / (2 * np.pi * acceleration)) * np.cos(2 * np.pi * acceleration)

    df = fs / len(data)  # Spectral resolution
    idx_start = max(int(cutoff_start // df), 0)
    idx_stop = min(int(cutoff_stop // df), len(velocity))
    velocity = velocity[idx_start::idx_stop]  # Selected bandpass components

    # RMS of filtered velocity signal, from Parseval's Theorem
    return np.sqrt(np.mean(np.square(velocity)))


def find_peaks(
    magnitude: ndarray | Series,
    frequency: ndarray | Series,
    peaks_num: int = 3,
    height: float | None = None,
    distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximal peaks in signal spectrum and corresponding frequencies.

    Args:
        magnitude (ndarray | Series): Amplitude of the spectrum at each frequency bin.
        frequency (ndarray | Series): Corresponding frequency values.
        height (float), optional: Minimal height of peaks. If not provided, defaults to the mean of `magnitude`.
        distance (float), optional: Minimal distance between peaks. If not provided, defaults to 3 frequency bins.
        peaks_num (int), optional: Number of peaks to search.

    Returns:
        max_peaks (ndarray): Array of the maximal peak heights.
        max_freq (ndarray): Array of the corresponding frequency values for these peaks."""

    result_peaks = np.full(peaks_num, np.nan)
    result_freq = np.full(peaks_num, np.nan)

    if magnitude.size == 0 or frequency.size == 0:
        raise ValueError("Empty input data.")

    if height is None:
        height = np.mean(magnitude)
    if distance is None and frequency.size > 3:
        distance = frequency[3] - frequency[0]

    peaks, properties = signal.find_peaks(magnitude, height=height, distance=distance)

    if peaks.shape[0] > 0:
        peaks_found = peaks.shape[0]
        peaks_use = min(peaks_num, peaks_found)
        magnitude_value_indices = np.argsort(properties["peak_heights"])
        top_indices = magnitude_value_indices[-peaks_use:]
        selected_peaks_indices = peaks[top_indices]

        max_peaks = properties["peak_heights"][top_indices]
        max_freq = frequency[selected_peaks_indices]

        # Sort the result in descending order of peak height.
        order = np.argsort(max_peaks)[::-1]
        max_peaks = max_peaks[order]
        max_freq = max_freq[order]

        result_peaks[:peaks_use] = max_peaks
        result_freq[:peaks_use] = max_freq

    return result_peaks, result_freq


def energy_operator(data: ndarray | Series) -> ndarray:
    """
    Compute the Energy Operator (EOP) using the Teager-Kaiser Energy Operator (TKEO).

    Captures both amplitude and frequency modulations of a signal, making it useful for highlighting transient events.
    EOP is developed by first calculating TKEO operator x(i)^2-x(i-1)x(i+1) for every point of the signal.
    The energy operator is then computed by taking the kurtosis of the resulting signal.

    Args:
        data (ndarray | Series): Input vibration signal.

    Returns:
        ndarray: Calculated EOP value.
    """
    if data.shape[0] < 3:
        raise ValueError("Signal must contain at least 3 samples to compute TKEO.")
    tkeo_operator = (
        data[1:-1] ** 2 - data[:-2] * data[2:]
    )  # Avoiding first and last points
    return kurt(tkeo_operator)


def zero_order_fom(
    data: ndarray | Series,
    frequency: ndarray | Series,
    magnitude: ndarray | Series,
    fundamental_freq: float,
    n_harmonics: int,
) -> ndarray:
    """
    Zero Order Figure of Merit (FMO)

    It is calculated as the ratio between the maximum peak-to-peak amplitude of the time-domain signal and the sum of
    the amplitudes of the fundamental frequency and its harmonics in the frequency domain.

    Args:
        data (ndarray | Series): Time-domain vibration signal.
        frequency (ndarray | Series): Frequency axis corresponding to fft_amplitude.
        magnitude (ndarray | Series): Amplitude spectrum of the vibration signal.
        fundamental_freq (float): significant frequency whose harmonics are considered. Might be shaft speed,
                                  BPFO, BPFI and others (in [Hz]).
        n_harmonics (int): Number of harmonics to sum in the frequency domain.

    Returns:
        ndarray: The Zero Order Figure of Merit (FMO) value.
    """
    if len(data) == 0:
        raise ValueError("Empty input data.")
    check_frequency_input(magnitude, frequency)

    harmonic_amplitudes = []
    for harmonic_index in range(1, n_harmonics + 1):
        target_freq = harmonic_index * fundamental_freq
        # Find the closest bin in 'frequency' to this target frequency
        idx = np.argmin(np.abs(frequency - target_freq))
        harmonic_amplitudes.append(magnitude[idx])

    sum_harmonics = np.sum(harmonic_amplitudes)

    if sum_harmonics == 0:
        raise ValueError("Denominator - sum of harmonics equals zero.")

    return peak_to_peak(data) / sum_harmonics


def fourth_order_fom(data: ndarray | Series) -> ndarray:
    """
    Fourth Order Figure of Merit (FM4)

    Because FM4 is computed on the difference signal, it is sensitive to sudden changes and impulsive events.
    It is being calculated by first constructing the difference signal, and then normalized kurtosis.

    Args:
        data (ndarray | Series): Time-domain vibration signal.

    Returns:
        ndarray: The Fourth Order Figure of Merit (FMA).
    """
    if len(data) == 0:
        raise ValueError("Empty input data.")
    return kurt(np.diff(data))


def sixth_order_fom(data: ndarray | Series) -> ndarray:
    """
    M6A: Sixth-Order Figure of Merit

    Surface damage indicator for machinery components. M6A is more sensitive to peaks in the difference signal,
    compared to FM4, because of using sixth moment.

    Args:
        data (ndarray | Series): Time-domain vibration signal.

    Returns:
        ndarray: The computed M6A value.
    """
    if len(data) == 0:
        raise ValueError("Empty input data.")

    d = np.diff(data)
    d_minus_mean = d - np.mean(d)
    sixth_statistical_moment = np.sum(d_minus_mean**6)
    second_statistical_moment = np.sum(d_minus_mean**2)

    if second_statistical_moment == 0:
        raise ValueError("Denominator - variance equals zero.")

    return ((len(d)) ** 2 * sixth_statistical_moment) / (second_statistical_moment**3)


def eight_order_fom(data: ndarray | Series) -> ndarray:
    """
    M8A: Eight-Order Figure of Merit

    It applies the eighth moment normalized by the variance to the fourth power.
    More sensitive than M6A to peaks in the difference signal.

    Args:
        data (ndarray | Series): Time-domain vibration signal.

    Returns:
        ndarray: The computed M8A value.
    """
    if len(data) == 0:
        raise ValueError("Empty input data.")

    d = np.diff(data)
    d_minus_mean = d - np.mean(d)
    eight_statistical_moment = np.sum(d_minus_mean**8)
    second_statistical_moment = np.sum(d_minus_mean**2)

    if second_statistical_moment == 0:
        raise ValueError("Denominator - variance equals zero.")

    return ((len(d)) ** 2 * eight_statistical_moment) / (second_statistical_moment**4)


def clearance_factor(data: ndarray | Series) -> ndarray:
    """
    Calculate the Clearance Factor.

    Commonly used for detecting impulsive changes in rotating machinery.

    Args:
        data (ndarray | Series): Time-domain vibration signal.

    Returns:
        ndarray: The Clearance Factor value.
    """
    if len(data) == 0:
        raise ValueError("Empty input data.")
    denominator = np.mean(np.sqrt(np.abs(data)))
    if denominator == 0:
        raise ValueError("Denominator value equals zero.")

    return np.max(np.abs(data)) / (denominator**2)


def impulse_indicator(data: ndarray | Series) -> ndarray:
    """
    Calculate the Impulse Indicator.

    Often used in rotating machinery diagnostics to measure how spiky a signal is. It measures how many times larger the
    single biggest peak is compared to the average of all absolute values.
    A high impulse indicator suggests transient or impulsive events. Compared to Clearance Factor it uses
    average absolute value which is a simpler measure of “typical” amplitude than RMS.

    Args:
        data (ndarray | Series): Time-domain vibration signal.

    Returns:
        ndarray: The Impulse Indicator value.
    """
    if len(data) == 0:
        raise ValueError("Empty input data.")

    mean_abs_value = np.mean(np.abs(data))
    if mean_abs_value == 0:
        raise ValueError("Mean value equals zero.")

    return np.max(np.abs(data)) / mean_abs_value


def rms_frequency(magnitude: ndarray | Series, frequency: ndarray | Series) -> ndarray:
    """
    Root Mean Square Frequency (RMSF)

    Measures a "weighted" RMS of the frequency axis, indicating how the energy is distributed in the spectrum.

    Args:
        magnitude (ndarray | Series): Amplitude or power at each frequency bin.
        frequency (ndarray | Series): Corresponding frequency values (Hz).

    Returns:
        ndarray: The RMS Frequency
    """
    check_frequency_input(magnitude, frequency)
    sum_magnitudes = np.sum(magnitude)
    if sum_magnitudes == 0:
        raise ValueError("Sum of magnitudes equals zero.")

    return np.sqrt(np.sum(frequency**2 * magnitude) / sum_magnitudes)


def std_frequency(magnitude: ndarray | Series, frequency: ndarray | Series) -> ndarray:
    """
    Standard Deviation Frequency (STDF)

    Measures the spread of the frequency distribution around the Frequency Center.

    Args:
        magnitude (ndarray | Series): Amplitude or power at each frequency bin.
        frequency (ndarray | Series): Corresponding frequency values (Hz).

    Returns:
        ndarray: The Standard Deviation Frequency in Hz.
    """
    fc = freq_center(magnitude, frequency)
    sum_magnitudes = np.sum(magnitude)
    if sum_magnitudes == 0:
        raise ValueError("Sum of magnitudes equals zero.")

    return np.sqrt(np.sum((frequency - fc) ** 2 * magnitude) / sum_magnitudes)


def shannon_entropy(energies: ndarray | Series) -> ndarray:
    """
    Compute the Shannon Entropy from energy of signal.

    Used to detect how “dispersed” or “impulsive” the wavelet-packet energy distribution becomes under fault conditions.
    Higher Entropy - the energy is more evenly spread among sub-bands.
    Lower Entropy - one or a few sub-bands dominate the energy, which can indicate faults.

    Args:
        energies (ndarray | Series): sub-band energies.

    Returns:
        ndarray: The Shannon entropy value.
    """
    if len(energies) == 0:
        raise ValueError("Empty input data.")

    total_energy = np.sum(energies)

    if total_energy == 0:
        raise ValueError("Total energy equals zero.")

    fraction_of_total_energy = energies / total_energy
    log_terms = np.where(
        fraction_of_total_energy > 0, np.log(fraction_of_total_energy), 0
    )
    return -np.sum(fraction_of_total_energy * log_terms)


def fourth_order_np(data: ndarray | Series) -> ndarray:
    """
    Compute NP4 (Fourth Order Normalized Power)

    A higher NP4 indicates a more impulsive or heavy-tailed power distribution, more extreme spikes in power,
    which can reveal localized faults in gears or bearings.
    For small NP4 power is more Gaussian-like and has fewer impulsive events.

    Args:
        data (ndarray | Series): time-domain vibration data.

    Returns:
        ndarray: The NP4 value.
    """
    if len(data) == 0:
        raise ValueError("Empty input data.")

    power = np.square(data)  # P(t) = x(t)^2
    p_std = std(power)  # population standard deviation

    if p_std == 0:
        raise ValueError("Denominator - power std equals zero.")

    return np.array(np.mean(((power - np.mean(power)) / p_std) ** 4) - 3.0)
