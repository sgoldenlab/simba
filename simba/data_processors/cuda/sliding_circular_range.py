__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

from typing import Optional

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cupy as cp
import numpy as np


def sliding_circular_range(x: np.ndarray,
                          time_window: float,
                          sample_rate: float,
                          batch_size: Optional[int] = int(5e+7)) -> np.ndarray:
    """
    Computes the sliding circular range of a time series data array using GPU.

    This function calculates the circular range of a time series data array using a sliding window approach.
    The input data is assumed to be in degrees, and the function handles the circular nature of the data
    by considering the circular distance between angles.

    .. math::

       R = \\min \\left( \\text{max}(\\Delta \\theta) - \\text{min}(\\Delta \\theta), \\, 360 - \\text{max}(\\Delta \\theta) + \\text{min}(\\Delta \\theta) \\right)

    where:

    - :math:`\\Delta \\theta` is the difference between angles within the window,
    - :math:`360` accounts for the circular nature of the data (i.e., wrap-around at 360 degrees).

    :param np.ndarray x: The input time series data in degrees. Should be a 1D numpy array.
    :param float time_window: The size of the sliding window in seconds.
    :param float sample_rate: The sample rate of the time series data (i.e., hz, fps).
    :param Optional[int] batch_size: The size of each batch for processing the data. Default is 5e+7 (50m).
    :return: A numpy array containing the sliding circular range values.
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(0, 361, (19, )).astype(np.int32)
    >>> p = sliding_circular_range(x, 1, 10)
    """

    n = x.shape[0]
    x = cp.asarray(x, dtype=cp.float16)
    results = cp.zeros_like(x, dtype=cp.int16)
    x = cp.deg2rad(x).astype(cp.float16)
    window_size = int(cp.ceil(time_window * sample_rate))
    for cnt, left in enumerate(range(0, n, batch_size)):
        right = int(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size + 1
        x_batch = x[left:right]
        x_batch = cp.lib.stride_tricks.sliding_window_view(x_batch, window_size).astype(cp.float16)
        x_batch = cp.sort(x_batch)
        results[left + window_size - 1:right] = cp.abs(cp.rint(cp.rad2deg(cp.amin(cp.vstack([x_batch[:, -1] - x_batch[:, 0], 2 * cp.pi - cp.max(cp.diff(x_batch), axis=1)]).T, axis=1))))
    return results.get()