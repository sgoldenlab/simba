__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"


import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 256

@cuda.jit
def _get_values_in_range_kernel(values, ranges, results):
    i = cuda.grid(1)
    if i >= values.shape[0]:
        return
    v = values[i]
    for j in range(ranges.shape[0]):
        l, u = ranges[j][0], ranges[j][1]
        cnt = 0
        for k in v:
            if k <= u and k >= l:
                cnt += 1
        results[i, j] = cnt


def count_values_in_ranges(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Counts the number of values in each feature within specified ranges for each row in a 2D array using CUDA.

    .. image:: _static/img/get_euclidean_distance_cuda.png
       :width: 500
       :align: center

    :param np.ndarray x: 2d array with feature values.
    :param np.ndarray r: 2d array with lower and upper boundaries.
    :return np.ndarray: 2d array of size len(x) x len(r) with the counts of values in each feature range (inclusive).

    :example:
    >>> x = np.random.randint(1, 11, (10, 10)).astype(np.int8)
    >>> r = np.array([[1, 6], [6, 11]])
    >>> r_x = count_values_in_ranges(x=x, r=r)
    """

    x = np.ascontiguousarray(x).astype(np.float32)
    r = np.ascontiguousarray(r).astype(np.float32)
    n, m = x.shape[0], r.shape[0]
    values_dev = cuda.to_device(x)
    ranges_dev = cuda.to_device(r)
    results = cuda.device_array((n, m), dtype=np.int32)
    bpg = (n + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _get_values_in_range_kernel[bpg, THREADS_PER_BLOCK](values_dev, ranges_dev, results)
    results = results.copy_to_host()
    cuda.current_context().memory_manager.deallocations.clear()
    return results