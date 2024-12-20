import pandas as pd

from simba.utils.checks import check_valid_array
import numpy as np
from numba import jit

def relative_risk(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the relative risk between two binary arrays.

    Relative risk (RR) is the ratio of the probability of an event occurring in one group/feature/cluster/variable (x)
    to the probability of the event occurring in another group/feature/cluster/variable (y).

    :param np.ndarray x: The first 1D binary array.
    :param np.ndarray y: The second 1D binary array.
    :return float: The relative risk between arrays x and y.

    :example:
    >>> relative_risk(x=np.array([0, 1, 1]), y=np.array([0, 1, 0]))
    >>> 2.0
    """
    check_valid_array(data=x, source=f'{relative_risk.__name__} x', accepted_ndims=(1,), accepted_values=[0, 1])
    check_valid_array(data=y, source=f'{relative_risk.__name__} y', accepted_ndims=(1,), accepted_values=[0, 1])
    if np.sum(y) == 0:
        return -1.0
    elif np.sum(x) == 0:
        return 0.0
    else:
        return (np.sum(x) / x.shape[0]) / (np.sum(y) / y.shape[0])


@jit(nopython=True)
def sliding_relative_risk(x: np.ndarray, y: np.ndarray, window_sizes: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Calculate sliding relative risk values between two binary arrays using different window sizes.


    :param np.ndarray x: The first 1D binary array.
    :param np.ndarray y: The second 1D binary array.
    :param np.ndarray window_sizes:
    :param int sample_rate:
    :return np.ndarray: Array of size  x.shape[0] x window_sizes.shape[0] with sliding eta squared values.
    """
    results = np.full((x.shape[0], window_sizes.shape[0]), -1.0)
    for i in range(window_sizes.shape[0]):
        window_size = int(window_sizes[i] * sample_rate)
        for l, r in zip(range(0, x.shape[0] + 1), range(window_size, x.shape[0] + 1)):
            sample_x, sample_y = x[l:r], y[l:r]
            print(sample_x, sample_y)
            if np.sum(sample_y) == 0:
                results[r - 1, i] = -1.0
            elif np.sum(sample_x) == 0:
                results[r - 1, i] = 0.0
            else:
                results[r - 1, i] = (np.sum(sample_x) / sample_x.shape[0]) / (np.sum(sample_y) / sample_y.shape[0])
    return results





x = np.array([0, 1, 1, 0])
y = np.array([0, 1, 0, 0])
sliding_relative_risk(x=x, y=y, window_sizes=np.array([1.0]), sample_rate=2)