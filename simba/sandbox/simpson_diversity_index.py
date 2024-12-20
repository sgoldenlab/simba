from typing import Optional
import numpy as np
from numba import jit

@jit(nopython=True)
def simpson_index(x: np.ndarray) -> float:
    """
    Calculate Simpson's diversity index for a given array of values.

    Simpson's diversity index is a measure of diversity that takes into account the number of different categories
    present in the input data as well as the relative abundance of each category.

    :param np.ndarray x: 1-dimensional numpy array containing the values representing categories for which Simpson's index is calculated.
    :return float: Simpson's diversity index value for the input array `x`
    """

    unique_v = np.unique(x)
    n_unique = np.unique(x).shape[0]
    results = np.full((n_unique, 3), np.nan)
    for i in range(unique_v.shape[0]):
        v = unique_v[i]
        cnt = np.argwhere(x == v).flatten().shape[0]
        squared = cnt * (cnt-1)
        results[i, :] = np.array([v, cnt, squared])
    return (np.sum(results[:, 2])) / (x.shape[0] * (x.shape[0] -1))




x = np.random.randint(0, 5, (10000000,))
simpson_index(x=x)
