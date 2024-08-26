__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 1024

@cuda.jit
def _cuda_sliding_min(x: np.ndarray, d: np.ndarray, results: np.ndarray):
    def _cuda_min(a, b):
        return a if a < b else b
    r = cuda.grid(1)
    l = np.int32(r - (d[0]-1))
    if (r > results.shape[0]) or (l < 0):
        results[r] = -1
    else:
        x_i = x[l:r-1]
        local_min = x_i[0]
        for k in range(x_i.shape[0]):
            local_min = _cuda_min(local_min, x_i[k])
        results[r] = local_min

def sliding_min(x: np.ndarray, time_window: float, sample_rate: int) -> np.ndarray:
    """
    Computes the minimum value within a sliding window over a 1D numpy array `x` using CUDA for acceleration.

    .. image:: _static/img/sliding_min_cuda.png
       :width: 500
       :align: center

    :param np.ndarray x: Input 1D numpy array of floats. The array over which the sliding window minimum is computed.
    :param float time_window: The size of the sliding window in seconds.
    :param intsample_rate: The sampling rate of the data, which determines the number of samples per second.
    :return: A numpy array containing the minimum value for each position of the sliding window.

    :example:
    >>> x = np.arange(0, 10000000)
    >>> time_window = 1
    >>> sample_rate = 10
    >>> sliding_min(x=x, time_window=time_window, sample_rate=sample_rate)
    """

    x = np.ascontiguousarray(x).astype(np.float32)
    window_size = np.array([np.ceil(time_window * sample_rate)])
    x_dev = cuda.to_device(x)
    delta_dev = cuda.to_device(window_size)
    results = cuda.device_array(x.shape, dtype=np.float32)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _cuda_sliding_min[bpg, THREADS_PER_BLOCK](x_dev, delta_dev, results)
    results = results.copy_to_host()
    return results
