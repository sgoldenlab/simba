from typing import Optional

import cupy as cp
import numpy as np

from simba.utils.checks import check_int, check_valid_array
from simba.utils.enums import Formats


def get_euclidean_distance_cupy(x: np.ndarray,
                                y: np.ndarray,
                                batch_size: Optional[int] = int(3.5e10+7)) -> np.ndarray:
    """
    Computes the Euclidean distance between corresponding pairs of points in two 2D arrays
    using CuPy for GPU acceleration. The computation is performed in batches to handle large
    datasets efficiently.

    :param np.ndarray x: A 2D NumPy array with shape (n, 2), where each row represents a point in a 2D space.
    :param np.ndarray y: A 2D NumPy array with shape (n, 2), where each row represents a point in a 2D space. The shape of `y` must match the shape of `x`.
    :param Optional[int] batch_size: The number of points to process in a single batch. This parameter controls memory usage and can be adjusted based on available GPU memory. The default value is large (`3.5e10 + 7`) to maximize GPU utilization, but it can be lowered if memory issues arise.
    :return: A 1D NumPy array of shape (n,) containing the Euclidean distances between corresponding points in `x` and `y`.
    :return: A 1D NumPy array of shape (n,) containing the Euclidean distances between corresponding points in `x` and `y`.
    :rtype: np.ndarray

    :example:
    >>> x = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([[7, 8], [9, 10], [11, 12]])
    >>> distances = get_euclidean_distance_cupy(x, y)
    """
    check_valid_array(data=x, source=check_valid_array.__name__, accepted_ndims=[2,], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, source=check_valid_array.__name__, accepted_ndims=[2, ], accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_shapes=(x.shape,))
    check_int(name='batch_size', value=batch_size, min_value=1)
    results = cp.full((x.shape[0]), fill_value=cp.nan, dtype=cp.float32)
    for l in range(0, x.shape[0], batch_size):
        r = l + batch_size
        batch_x, batch_y = cp.array(x[l:r]), cp.array(y[l:r])
        results[l:r] = (cp.sqrt((batch_x[:, 0] - batch_y[:, 0]) ** 2 + (batch_x[:, 1] - batch_y[:, 1]) ** 2))
    return results.get()