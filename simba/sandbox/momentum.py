import numpy as np

from simba.utils.checks import check_float, check_valid_array
from simba.utils.enums import Formats


def momentum_magnitude(x: np.ndarray, mass: float, sample_rate: float) -> float:
    """
    Compute the magnitude of momentum given 2D positional data and mass.

    :param np.ndarray x: 2D array of shape (n_samples, 2) representing positions.
    :param float mass: Mass of the object.
    :param float sample_rate: Sampling rate in FPS.
    :returns: Magnitude of the momentum.
    :rtype: float
    """

    check_valid_array(data=x, source=f'{momentum_magnitude.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=[2,], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{momentum_magnitude.__name__} mass', value=mass, min_value=10e-6)
    check_float(name=f'{momentum_magnitude.__name__} sample_rate', value=sample_rate, min_value=10e-6)
    dx, dy = np.diff(x[:, 0].flatten()), np.diff(x[:, 1].flatten())
    speed = np.mean(np.sqrt(dx ** 2 + dy ** 2) / (1 / sample_rate))
    return mass * speed


def sliding_momentum_magnitude(x: np.ndarray, mass: np.ndarray, sample_rate: float, time_window: float) -> np.ndarray:
    """
    Compute the sliding window momentum magnitude for 2D positional data.

    :param np.ndarray x: 2D array of shape (n_samples, 2) representing positions.
    :param np.ndarray mass: Array of mass values for each frame.
    :param float sample_rate: Sampling rate in FPS.
    :param float time_window: Time window in seconds for sliding momentum calculation.
    :returns: Momentum magnitudes computed for each frame, with results from frames that cannot form a complete window filled with -1.0.
    :rtype: np.ndarray

    """
    time_window_frms = np.ceil(sample_rate * time_window)
    results = np.full(shape=(x.shape[0]), fill_value=-1.0, dtype=np.float32)
    delta_t = 1 / sample_rate
    for r in range(time_window_frms, x.shape[0] + 1):
        l = r - time_window_frms
        keypoint_sample, mass_sample = x[l:r], mass[l:r]
        mass_sample_mean = np.mean(mass_sample)
        dx, dy = np.diff(keypoint_sample[:, 0].flatten()), np.diff(keypoint_sample[:, 1].flatten())
        speed = np.mean(np.sqrt(dx ** 2 + dy ** 2) / delta_t)
        results[r - 1] = mass_sample_mean * speed
    return results
