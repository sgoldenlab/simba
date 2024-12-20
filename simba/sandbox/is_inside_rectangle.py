__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 1024
@cuda.jit
def _cuda_is_inside_rectangle(x, y, r):
    i = cuda.grid(1)
    if i > r.shape[0]:
        return
    else:
        if (x[i][0] >= y[0][0]) and (x[i][0] <= y[1][0]):
            if (x[i][1] >= y[0][1]) and (x[i][1] <= y[1][1]):
                r[i] = 1

def is_inside_rectangle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Determines whether points in array `x` are inside the rectangle defined by the top left and bottom right vertices in array `y`.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/is_inside_rectangle.csv
       :widths: 10, 45, 45
       :align: center
       :class: simba-table
       :header-rows: 1

    :param np.ndarray x: 2d numeric np.ndarray size (N, 2).
    :param np.ndarray y: 2d numeric np.ndarray size (2, 2) (top left[x, y], bottom right[x, y])
    :return np.ndarray: 2d numeric boolean (N, 1) with 1s representing the point being inside the rectangle and 0 if the point is outside the rectangle.

    """


    x = np.ascontiguousarray(x).astype(np.int32)
    y = np.ascontiguousarray(y).astype(np.int32)
    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    results = cuda.device_array((x.shape[0]), dtype=np.int8)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _cuda_is_inside_rectangle[bpg, THREADS_PER_BLOCK](x_dev, y_dev, results)
    results = results.copy_to_host()
    return results