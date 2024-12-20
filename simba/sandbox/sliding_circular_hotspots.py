__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

from typing import Optional

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import cupy as cp
import numpy as np


def sliding_circular_hotspots(x: np.ndarray,
                              time_window: float,
                              sample_rate: float,
                              bins: np.ndarray,
                              batch_size: Optional[int] = int(3.5e+7)) -> np.ndarray:
    """
    Calculate the proportion of data points falling within specified circular bins over a sliding time window using GPU

    This function processes time series data representing angles (in degrees) and calculates the proportion of data
    points within specified angular bins over a sliding window. The calculations are performed in batches to
    accommodate large datasets efficiently.

    :param np.ndarray x: The input time series data in degrees. Should be a 1D numpy array.
    :param float time_window: The size of the sliding window in seconds.
    :param float sample_rate: The sample rate of the time series data (i.e., hz, fps).
    :param ndarray bins: 2D array of shape representing circular bins defining [start_degree, end_degree] inclusive.
    :param Optional[int] batch_size: The size of each batch for processing the data. Default is 5e+7 (50m).
    :return: A 2D numpy array where each row corresponds to a time point in `data`, and each column represents a circular bin. The values in the array represent the proportion of data points within each bin at each time point. The first column represents the first bin.
    :rtype: np.ndarray
    """

    n = x.shape[0]
    x = cp.asarray(x, dtype=cp.float16)
    results = cp.full((x.shape[0], bins.shape[0]), dtype=cp.float16, fill_value=-1)
    window_size = int(cp.ceil(time_window * sample_rate))
    for cnt, left in enumerate(range(0, n, batch_size)):
        right = int(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size + 1
        x_batch = x[left:right]
        x_batch = cp.lib.stride_tricks.sliding_window_view(x_batch, window_size).astype(cp.float16)
        batch_results = cp.full((x_batch.shape[0], bins.shape[0]), dtype=cp.float16, fill_value=-1)
        for bin_cnt in range(bins.shape[0]):
            if bins[bin_cnt][0] > bins[bin_cnt][1]:
                mask = ((x_batch >= bins[bin_cnt][0]) & (x_batch <= 360)) | ((x_batch >= 0) & (x_batch <= bins[bin_cnt][1]))
            else:
                mask = (x_batch >= bins[bin_cnt][0]) & (x_batch <= bins[bin_cnt][1])
            count_per_row = cp.array(mask.sum(axis=1) / window_size).reshape(-1, )
            batch_results[:, bin_cnt] = count_per_row
        results[left + window_size - 1:right, ] = batch_results
    return results.get()