__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 1024

@cuda.jit
def _cuda_sliding_sum(x: np.ndarray, d: np.ndarray, results: np.ndarray):
    r = cuda.grid(1)
    l = np.int32(r - (d[0]-1))
    if (r > results.shape[0]) or (l < 0):
        results[r] = -1
    else:
        x_i = x[l:r]
        local_sum = 0
        for k in range(x_i.shape[0]):
            local_sum += x_i[k]
        results[r-1] = local_sum

def sliding_sum(x: np.ndarray, time_window: float, sample_rate: int) -> np.ndarray:
    """
    Computes the sum of values within a sliding window over a 1D numpy array `x` using CUDA for acceleration.

    :param np.ndarray x: The input 1D numpy array of floats. The array over which the sliding window sum is computed.
    :param float time_window:The size of the sliding window in seconds. This window slides over the array `x` to compute the sum.
    :param int sample_rate: The number of samples per second in the array `x`. This is used to convert the time-based window size into the number of samples.
    :return np.ndarray: A numpy array containing the sum of values within each position of the sliding window.

    :example:
    >>> x = np.random.randint(1, 11, (100, )).astype(np.float32)
    >>> time_window = 1
    >>> sample_rate = 10
    >>> r_x = sliding_sum(x=x, time_window=time_window, sample_rate=10)
    """
    x = np.ascontiguousarray(x).astype(np.float32)
    window_size = np.array([np.ceil(time_window * sample_rate)])
    x_dev = cuda.to_device(x)
    delta_dev = cuda.to_device(window_size)
    results = cuda.device_array(x.shape, dtype=np.float32)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _cuda_sliding_sum[bpg, THREADS_PER_BLOCK](x_dev, delta_dev, results)
    results = results.copy_to_host()
    return results


