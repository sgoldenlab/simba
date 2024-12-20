import time

import numpy as np
import pandas as pd
from numba import jit

@jit(nopython=True)
def sliding_eta_squared(x: np.ndarray, y: np.ndarray, window_sizes: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Calculate sliding window eta-squared, a measure of effect size for between-subjects designs,
    over multiple window sizes.

    :param np.ndarray x: The array containing the dependent variable data.
    :param np.ndarray y: The array containing the grouping variable (categorical) data.
    :param np.ndarray window_sizes: 1D array of window sizes in seconds.
    :param int sample_rate: The sampling rate of the data in frames per second.
    :return np.ndarray:

    :example:
    >>> x = np.random.randint(0, 10, (10000,))
    >>> y = np.random.randint(0, 2, (10000,))
    >>> p = sliding_eta_squared(x=x, y=y, window_sizes=np.array([1.0, 2.0]), sample_rate=10)

    """
    results = np.full((x.shape[0], window_sizes.shape[0]), -1.0)
    for i in range(window_sizes.shape[0]):
        window_size = int(window_sizes[i] * sample_rate)
        for l, r in zip(range(0, x.shape[0] + 1), range(window_size, x.shape[0] + 1)):
            sample_x = x[l:r]
            sample_y = y[l:r]
            sum_square_within, sum_square_between = 0, 0
            for lbl in np.unique(sample_y):
                g = sample_x[np.argwhere(sample_y == lbl).flatten()]
                sum_square_within += np.sum((g - np.mean(g)) ** 2)
                sum_square_between += len(g) * (np.mean(g) - np.mean(sample_x)) ** 2
            if sum_square_between + sum_square_within == 0:
                results[r - 1, i] = 0.0
            else:
                results[r - 1, i] = (sum_square_between / (sum_square_between + sum_square_within)) ** .5
    return results


x = np.random.randint(0, 10, (10000,))
y = np.random.randint(0, 2, (10000,))
p = sliding_eta_squared(x=x, y=y, window_sizes=np.array([1.0, 2.0]), sample_rate=10)


#print(p, o)