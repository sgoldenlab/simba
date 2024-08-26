__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import numpy as np
from numba import cuda

THREADS_PER_BLOCK = 1024

@cuda.jit
def _cuda_is_inside_polygon(x, p, r):
    i = cuda.grid(1)
    if i > r.shape[0]:
        return
    else:
        x, y, n = x[i][0], x[i][1], len(p)
        p2x, p2y, xints, inside = 0.0, 0.0, 0.0, False
        p1x, p1y = p[0]
        for j in range(n + 1):
            p2x, p2y = p[j % n]
            if (
                    (y > min(p1y, p2y))
                    and (y <= max(p1y, p2y))
                    and (x <= max(p1x, p2x))
            ):
                if p1y != p2y:
                    xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xints:
                    inside = not inside
            p1x, p1y = p2x, p2y
        if inside:
            r[i] = 1


def is_inside_polygon(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Determines whether points in array `x` are inside the polygon defined by the vertices in array `y`.

    This function uses GPU acceleration to perform the point-in-polygon test. The points in `x` are tested against
    the polygon defined by the vertices in `y`. The result is an array where each element indicates whether
    the corresponding point is inside the polygon.

    .. image:: _static/img/is_inside_polygon_cuda.webp
       :width: 500
       :align: center

    :param np.ndarray x: An array of shape (N, 2) where each row represents a point in 2D space. The points are checked against the polygon.
    :param np.ndarray y: An array of shape (M, 2) where each row represents a vertex of the polygon in 2D space.
    :return np.ndarray: An array of shape (N,) where each element is 1 if the corresponding point in `x` is inside the polygon defined by `y`, and 0 otherwise.

    :example:
    >>> x = np.random.randint(0, 200, (i, 2)).astype(np.int8)
    >>> y = np.random.randint(0, 200, (4, 2)).astype(np.int8)
    >>> results = is_inside_polygon(x=x, y=y)
    >>> print(results)
    >>> [1 0 1 0 1 1 0 0 1 0]
    """
    x = np.ascontiguousarray(x).astype(np.int32)
    y = np.ascontiguousarray(y).astype(np.int32)
    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    results = cuda.device_array((x.shape[0]), dtype=np.int8)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _cuda_is_inside_polygon[bpg, THREADS_PER_BLOCK](x_dev, y_dev, results)
    results = results.copy_to_host()
    return results