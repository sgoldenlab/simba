__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import math

import numpy as np
from numba import cuda, int32

THREADS_PER_BLOCK = 1024

@cuda.jit()
def _cuda_direction_from_two_bps(x, y, results):
    i = cuda.grid(1)
    if i > x.shape[0]:
        return
    else:
        a = math.atan2(x[i][0] - y[i][0], y[i][1] - x[i][1]) * (180 / math.pi)
        a = int32(a + 360 if a < 0 else a)
        results[i] = a


def direction_from_two_bps(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the directionality in degrees from two body-parts. E.g., ``nape`` and ``nose``,
    or ``swim_bladder`` and ``tail`` with GPU acceleration.

    .. image:: _static/img/direction_from_two_bps_cuda.png
       :width: 1200
       :align: center


    :parameter np.ndarray x: Size len(frames) x 2 representing x and y coordinates for first body-part.
    :parameter np.ndarray y: Size len(frames) x 2 representing x and y coordinates for second body-part.
    :return np.ndarray: Frame-wise directionality in degrees.

    """
    x = np.ascontiguousarray(x).astype(np.int32)
    y = np.ascontiguousarray(y).astype(np.int32)
    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    results = cuda.device_array((x.shape[0]), dtype=np.int32)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _cuda_direction_from_two_bps[bpg, THREADS_PER_BLOCK](x_dev, y_dev, results)
    results = results.copy_to_host()
    return results
