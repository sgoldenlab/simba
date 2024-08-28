__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

from typing import Optional

import cupy
import numpy as np


def sliding_resultant_vector_length(x: np.ndarray,
                                    time_window: float,
                                    sample_rate: int,
                                    batch_size: Optional[int] = 3e+7) -> np.ndarray:

    """
    Calculate the sliding resultant vector length over a time window for a series of angles.

    This function computes the resultant vector length (R) for each window of angles in the input array `x`.
    The resultant vector length is a measure of the concentration of angles, and it ranges from 0 to 1, where 1
    indicates all angles point in the same direction, and 0 indicates uniform distribution of angles.

    For a given sliding window of angles, the resultant vector length :math:`R` is calculated using the following formula:

    .. math::

        R = \\frac{1}{N} \\sqrt{\\left(\\sum_{i=1}^{N} \\cos(\\theta_i)\\right)^2 + \\left(\\sum_{i=1}^{N} \\sin(\\theta_i)\\right)^2}

    where:

    - :math:`\\theta_i` are the angles in radians within the sliding window
    - :math:`N` is the number of samples in the window

    The computation is performed in a sliding window manner over the entire array, utilizing GPU acceleration
    with CuPy for efficiency, especially on large datasets.


    :param np.ndarray x: Input array containing angle values in degrees. The array should be 1-dimensional.
    :param float time_window: Time duration for the sliding window, in seconds. This determines the number of samples in each window  based on the `sample_rate`.
    :param int sample_rate: The number of samples per second (i.e., FPS). This is used to calculate the window size in terms of array indices.
    :param Optional[int] batch_size: The maximum number of elements to process in each batch. This is used to handle large arrays by processing them in chunks to avoid memory overflow. Defaults to 3e+7 (30 million elements).
    :return np.ndarray: A 1D numpy array of the same length as `x`, containing the resultant vector length for each sliding window. Values before the window is fully populated will be set to -1.


    :example:
    >>> x = np.random.randint(0, 361, (5000, )).astype(np.int32)
    >>> results = sliding_resultant_vector_length(x, 1, 10)
    """

    window_size = np.ceil(time_window * sample_rate).astype(np.int64)
    n = x.shape[0]
    results = cupy.full(x.shape[0], -1, dtype=np.float32)
    for cnt, left in enumerate(range(0, int(n), int(batch_size))):
        right = np.int32(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size+1
        x_batch = cupy.asarray(x[left:right])
        x_batch = cupy.lib.stride_tricks.sliding_window_view(x_batch, window_size)
        x_batch = np.deg2rad(x_batch)
        cos, sin = cupy.cos(x_batch).astype(np.float32), cupy.sin(x_batch).astype(np.float32)
        cos_sum, sin_sum = cupy.sum(cos, axis=1), cupy.sum(sin, axis=1)
        r = np.sqrt(cos_sum ** 2 + sin_sum ** 2) / window_size
        results[left+window_size-1:right] = r
    return results.get()