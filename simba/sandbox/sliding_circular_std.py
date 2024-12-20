__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

from typing import Optional

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cupy as cp
import numpy as np


def sliding_circular_std(x: np.ndarray,
                         time_window: float,
                         sample_rate: float,
                         batch_size: Optional[int] = int(5e+7)) -> np.ndarray:
    """
    Calculate the sliding circular standard deviation of a time series data on GPU.

    This function computes the circular standard deviation over a sliding window for a given time series array.
    The time series data is assumed to be in degrees, and the function converts it to radians for computation.
    The sliding window approach is used to handle large datasets efficiently, processing the data in batches.

    The circular standard deviation (σ) is computed using the formula:

    .. math::

       \sigma = \sqrt{-2 \cdot \log \left|\text{mean}\left(\exp(i \cdot x_{\text{batch}})\right)\right|}

    where :math:`x_{\text{batch}}` is the data within the current sliding window, and :math:`\text{mean}` and
    :math:`\log` are computed in the circular (complex plane) domain.

    :param np.ndarray x: The input time series data in degrees. Should be a 1D numpy array.
    :param float time_window: The size of the sliding window in seconds.
    :param float sample_rate: The sample rate of the time series data (i.e., hz, fps).
    :param Optional[int] batch_size: The size of each batch for processing the data. Default is 5e+7 (50m).

    :return: A numpy array containing the sliding circular standard deviation values.
    :rtype: np.ndarray
    """


    n = x.shape[0]
    x = cp.asarray(x, dtype=cp.float16)
    results = cp.zeros_like(x, dtype=cp.float16)
    x = np.deg2rad(x).astype(cp.float16)
    window_size = int(np.ceil(time_window * sample_rate))
    for cnt, left in enumerate(range(0, n, batch_size)):
        right = int(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size + 1
        x_batch = x[left:right]
        x_batch = cp.lib.stride_tricks.sliding_window_view(x_batch, window_size).astype(cp.float16)
        m = cp.log(cp.abs(cp.mean(cp.exp(1j * x_batch), axis=1)))
        stdev = cp.rad2deg(cp.sqrt(-2 * m))
        results[left + window_size - 1:right] = stdev

    return results.get()
