import time

import numpy as np
from numba import njit

@njit('(float32[:], float64,)')
def mad_median_rule(data: np.ndarray, k: int) -> np.ndarray:
    """
    Detect outliers using the MAD-Median Rule. Returns 1d array of size data.shape[0] with 1 representing outlier and 0 representing inlier.

    :example:
    >>> data = np.random.randint(0, 600, (9000000,)).astype(np.float32)
    >>> mad_median_rule(data=data, k=1.0)
    """

    median = np.median(data)
    mad = np.median(np.abs(data - median))
    threshold = k * mad
    outliers = np.abs(data - median) > threshold
    return outliers * 1

@njit('(float32[:], float64, float64[:], float64)')
def sliding_mad_median_rule(data: np.ndarray, k: int, time_windows: np.ndarray, fps: float) -> np.ndarray:
    """
    Count the number of outliers in a sliding time-window using the MAD-Median Rule.

    :param np.ndarray data: 1D numerical array representing feature.
    :param int k: The outlier threshold defined as k * median absolute deviation in each time window.
    :param np.ndarray time_windows: 1D array of time window sizes in seconds.
    :param float fps: The frequency of the signal.
    :return np.ndarray: Array of size (data.shape[0], time_windows.shape[0]) with counts if outliers detected.

    :example:
    >>> data = np.random.randint(0, 50, (50000,)).astype(np.float32)
    >>> sliding_mad_median_rule(data=data, k=2, time_windows=np.array([20.0]), fps=1.0)
    """
    results = np.full((data.shape[0], time_windows.shape[0]), -1)
    for time_window in time_windows:
        w = int(fps * time_window)
        for i in range(w, data.shape[0]+1, 1):
            w_data = data[i-w:i]
            median = np.median(w_data)
            mad = np.median(np.abs(w_data - median))
            threshold = k * mad
            outliers = np.abs(w_data - median) > threshold
            results[i-1] = np.sum(outliers * 1)
    return results





# data = np.random.randint(0, 50, (50000,)).astype(np.float32)
# start = time.time()
# sliding_mad_median_rule(data=data, k=2, time_windows=np.array([20.0]), fps=1.0)
# print(time.time() - start)
#
# mad_median_rule(data=data, k=1.0)
# print(time.time() - start)
