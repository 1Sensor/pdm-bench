"""Scaler - gathers function for values scaling and conversion."""

import numpy as np
import pandas as pd


def scale(
    data: np.ndarray | pd.Series,
    amplification: float = 1.0,
    sensitivity: float = 0.0102,
) -> np.ndarray:
    """Functions allow to scale measured to values with given measurement parameters.

    Args:
        data (np.ndarray | pd.Series): measurement data from accelerometer in [V].
        amplification (float, optional): amplification set during measurement. Defaults to 1.0.
        sensitivity (float, optional): Sensitivity of a sensor used during measurement in [V/(m/s^2)]
            Defaults to 0.0102, which is 1/9.81 and is common for conversion to SI values.

    Returns:
        np.ndarray: scaled data array
    """
    _params_check(amplification=amplification, sensitivity=sensitivity)
    if isinstance(data, pd.Series):
        data = data.to_numpy()
    return np.divide(data, sensitivity * amplification)


def _params_check(amplification: float, sensitivity: float):
    """Checks if parameters are valid, if not throws an error.

    Args:
        amplification (float): Given amplification value. Must be positive.
        sensitivity (float): Given

    Raises:
        ValueError: Amplification must be a positive value
        ValueError: Sensitivity must be a positive value
    """
    if amplification <= 0:
        raise ValueError(
            f"Amplification must be a positive value. Equals to {amplification}"
        )

    if sensitivity <= 0:
        raise ValueError(
            f"Sensitivity must be a positive value. Equals to {sensitivity}"
        )
