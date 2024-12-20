__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

from typing import Optional, Tuple

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cupy as cp
import numpy as np


def sliding_rayleigh_z(x: np.ndarray,
                       time_window: float,
                       sample_rate: float,
                       batch_size: Optional[int] = int(5e+7)) -> Tuple[np.ndarray, np.ndarray]:

    """
    Computes the Rayleigh Z-statistic over a sliding window for a given time series of angles

    This function calculates the Rayleigh Z-statistic, which tests the null hypothesis that the population of angles
    is uniformly distributed around the circle. The calculation is performed over a sliding window across the input
    time series, and results are computed in batches for memory efficiency.

    Data is processed using GPU acceleration via CuPy, which allows for faster computation compared to a CPU-based approach.

    .. note::
        Adapted from ``pingouin.circular.circ_rayleigh`` and ``pycircstat.tests.rayleigh``.


    **Rayleigh Z-statistic:**

    The Rayleigh Z-statistic is given by:

    .. math::

       R = \frac{1}{n} \sqrt{\left(\sum_{i=1}^{n} \cos(\theta_i)\right)^2 + \left(\sum_{i=1}^{n} \sin(\theta_i)\right)^2}

    where:
    - :math:`\theta_i` are the angles in the window.
    - :math:`n` is the number of angles in the window.

    :param np.ndarray x: Input array of angles in degrees. Should be a 1D numpy array.
    :param float time_window: The size of the sliding window in time units (e.g., seconds).
    :param float sample_rate: The sampling rate of the input time series in samples per time unit (e.g., Hz, fps).
    :param Optional[int] batch_size: The number of samples to process in each batch. Default is 5e7 (50m). Reducing this value may save memory at the cost of longer computation time.
    :return:
       A tuple containing two numpy arrays:
       - **z_results**: Rayleigh Z-statistics for each position in the input array where the window was fully applied.
       - **p_results**: Corresponding p-values for the Rayleigh Z-statistics.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    n = x.shape[0]
    x = cp.asarray(x, dtype=cp.float16)
    z_results = cp.zeros_like(x, dtype=cp.float16)
    p_results = cp.zeros_like(x, dtype=cp.float16)
    x = np.deg2rad(x).astype(cp.float16)
    window_size = int(np.ceil(time_window * sample_rate))
    for cnt, left in enumerate(range(0, n, batch_size)):
        right = int(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size + 1
        x_batch = x[left:right]
        x_batch = cp.lib.stride_tricks.sliding_window_view(x_batch, window_size).astype(cp.float16)
        cos_sums = cp.nansum(cp.cos(x_batch), axis=1) ** 2
        sin_sums = cp.nansum(cp.sin(x_batch), axis=1) ** 2
        R = cp.sqrt(cos_sums + sin_sums) / window_size
        Z = window_size * (R**2)
        P = cp.exp(np.sqrt(1 + 4 * window_size + 4 * (window_size ** 2 - R ** 2)) - (1 + 2 * window_size))
        z_results[left + window_size - 1:right] = Z
        p_results[left + window_size - 1:right] = P

    return z_results.get(), p_results.get()