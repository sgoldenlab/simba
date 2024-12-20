__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import math

import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 1024
@cuda.jit
def _cuda_is_inside_circle(x, y, r, results):
    i = cuda.grid(1)
    if i > results.shape[0]:
        return
    else:
        p = (math.sqrt((x[i][0] - y[0][0]) ** 2 + (x[i][1] - y[0][1]) ** 2))
        if p <= r[0]:
            results[i] = 1
def is_inside_circle(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    """
    Determines whether points in array `x` are inside the rectangle defined by the top left and bottom right vertices in array `y`.

    :param np.ndarray x: 2d numeric np.ndarray size (N, 2).
    :param np.ndarray y: 2d numeric np.ndarray size (2, 2) (top left[x, y], bottom right[x, y])
    :return np.ndarray: 2d numeric boolean (N, 1) with 1s representing the point being inside the rectangle and 0 if the point is outside the rectangle.
    """

    x = np.ascontiguousarray(x).astype(np.int32)
    y = np.ascontiguousarray(y).astype(np.int32)
    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    r = np.array([r]).astype(np.float32)
    r_dev = cuda.to_device(r)
    results = cuda.device_array((x.shape[0]), dtype=np.int8)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    del x, y
    _cuda_is_inside_circle[bpg, THREADS_PER_BLOCK](x_dev, y_dev, r_dev, results)
    results = results.copy_to_host()
    return results
