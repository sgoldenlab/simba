__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

from typing import Optional

import cupy
import numpy as np


def sliding_circular_mean(x: np.ndarray,
                          time_window: float,
                          sample_rate: int,
                          batch_size: Optional[int] = 3e+7) -> np.ndarray:

    """
    Calculate the sliding circular mean over a time window for a series of angles.

    This function computes the circular mean of angles in the input array `x` over a specified sliding window.
    The circular mean is a measure of the average direction for angles, which is especially useful for angular data
    where traditional averaging would not be meaningful due to the circular nature of angles (e.g., 359° and 1° should average to 0°).

    The calculation is performed using a sliding window approach, where the circular mean is computed for each window
    of angles. The function leverages GPU acceleration via CuPy for efficiency when processing large datasets.

    The circular mean :math:`\\mu` for a set of angles is calculated using the following formula:

    .. math::

        \\mu = \\text{atan2}\\left(\\frac{1}{N} \\sum_{i=1}^{N} \\sin(\\theta_i), \\frac{1}{N} \\sum_{i=1}^{N} \\cos(\\theta_i)\\right)

    - :math:`\\theta_i` are the angles in radians within the sliding window
    - :math:`N` is the number of samples in the window


    :param np.ndarray x: Input array containing angle values in degrees. The array should be 1-dimensional.
    :param float time_window: Time duration for the sliding window, in seconds. This determines the number of samples in each window  based on the `sample_rate`.
    :param int sample_rate: The number of samples per second (i.e., FPS). This is used to calculate the window size in terms of array indices.
    :param Optional[int] batch_size: The maximum number of elements to process in each batch. This is used to handle large arrays by processing them in chunks to avoid memory overflow. Defaults to 3e+7 (30 million elements).
    :return np.ndarray: A 1D numpy array of the same length as `x`, containing the circular mean for each sliding window.  Values before the window is fully populated will be set to -1.

    :example:
    >>> x = np.random.randint(0, 361, (i, )).astype(np.int32)
    >>> results = sliding_circular_mean(x, 1, 10)
    """


    window_size = np.ceil(time_window * sample_rate).astype(np.int64)
    n = x.shape[0]
    results = cupy.full(x.shape[0], -1, dtype=np.int32)
    for cnt, left in enumerate(range(0, int(n), int(batch_size))):
        right = np.int32(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size+1
        x_batch = cupy.asarray(x[left:right])
        x_batch = cupy.lib.stride_tricks.sliding_window_view(x_batch, window_size)
        x_batch = np.deg2rad(x_batch)
        cos, sin = cupy.cos(x_batch).astype(np.float32), cupy.sin(x_batch).astype(np.float32)
        r = cupy.rad2deg(cupy.arctan2(cupy.mean(sin, axis=1), cupy.mean(cos, axis=1)))
        r = cupy.where(r < 0, r + 360, r)
        results[left + window_size - 1:right] = r
    return results.get()
