__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

from typing import Optional

import cupy as cp
import numpy as np


def sliding_spearmans_rank(x: np.ndarray,
                           y: np.ndarray,
                           time_window: float,
                           sample_rate: int,
                           batch_size: Optional[int] = int(1.6e+7)) -> np.ndarray:
    """
    Computes the Spearman's rank correlation coefficient between two 1D arrays `x` and `y`
    over sliding windows of size `time_window * sample_rate`. The computation is performed
    in batches to optimize memory usage, leveraging GPU acceleration with CuPy.

    .. math::
       \rho = 1 - \frac{6 \sum d_i^2}{n_w(n_w^2 - 1)}

    .. math::
        The function uses CuPy to perform GPU-accelerated calculations. Ensure that your environment
        supports GPU computation with CuPy installed.

    Where:
    - \( \rho \) is the Spearman's rank correlation coefficient.
    - \( d_i \) is the difference between the ranks of corresponding elements in the sliding window.
    - \( n_w \) is the size of the sliding window.

    :param np.ndarray x: The first 1D array containing the values for Feature 1.
    :param np.ndarray y: The second 1D array containing the values for Feature 2.
    :param float time_window: The size of the sliding window in seconds.
    :param int sample_rate: The sampling rate (samples per second) of the data.
    :param Optional[int] batch_size: The size of each batch to process at a time for memory efficiency. Defaults to 1.6e7.
    :return: A 1D numpy array containing the Spearman's rank correlation coefficient for each sliding window.
    :rtype: np.ndarray

    :example:
    >>> x = np.array([9, 10, 13, 22, 15, 18, 15, 19, 32, 11])
    >>> y = np.array([11, 12, 15, 19, 21, 26, 19, 20, 22, 19])
    >>> sliding_spearmans_rank(x, y, time_window=0.5, sample_rate=2)
    """


    window_size = int(np.ceil(time_window * sample_rate))
    n = x.shape[0]
    results = cp.full(n, -1, dtype=cp.float32)

    for cnt, left in enumerate(range(0, n, batch_size)):
        right = int(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size + 1
        x_batch = cp.asarray(x[left:right])
        y_batch = cp.asarray(y[left:right])
        x_batch = cp.lib.stride_tricks.sliding_window_view(x_batch, window_size)
        y_batch = cp.lib.stride_tricks.sliding_window_view(y_batch, window_size)
        rank_x = cp.argsort(cp.argsort(x_batch, axis=1), axis=1)
        rank_y = cp.argsort(cp.argsort(y_batch, axis=1), axis=1)
        d_squared = cp.sum((rank_x - rank_y) ** 2, axis=1)
        n_w = window_size
        s = 1 - (6 * d_squared) / (n_w * (n_w ** 2 - 1))

        results[left + window_size - 1:right] = s

    return cp.asnumpy(results)