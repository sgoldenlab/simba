import time

import numpy as np
from simba.utils.checks import check_valid_array


def manhattan_distance_cdist(data: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Manhattan distance matrix between points in a 2D array.

    Can be preferred over Euclidean distance in scenarios where the movement is restricted
    to grid-based paths and/or the data is high dimensional.

    .. math::
       D_{\text{Manhattan}} = |x_2 - x_1| + |y_2 - y_1|

    :param data: 2D array where each row represents a featurized observation (e.g., frame)
    :return np.ndarray: Pairwise Manhattan distance matrix where element (i, j) represents the distance between points i and j.

    :example:
    >>> data = np.random.randint(0, 50, (10000, 2))
    >>> manhattan_distance_cdist(data=data)
    """
    check_valid_array(data=data, source=f'{manhattan_distance_cdist} data', accepted_ndims=(2,), accepted_dtypes=(np.float32, np.float64, np.int64, np.int32, int, float, np.float16, np.int8, np.int16))
    differences = np.abs(data[:, np.newaxis, :] - data)
    results = np.sum(differences, axis=-1)
    return results

data = np.random.randint(0, 50, (10000, 2))
y = manhattan_distance_cdist(data=data)
start = time.time()
x = manhattan_distance_cdist(data=data)
print(time.time() - start)
start = time.time()
y = manhattan_distance_cdist_2(data=data)
print(time.time() - start)
