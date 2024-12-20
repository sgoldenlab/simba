import time

import numpy as np
from numba import prange, jit, njit
import pandas as pd
from pingouin.reliability import cronbach_alpha


@njit('(float32[:,:], )',)
def cov_matrix(data: np.ndarray):
    """
    Jitted helper to compute the covariance matrix of the input data. Helper for computing cronbach alpha,
    multivariate analysis, and distance computations.

    :param np.ndarray data: 2-dimensional numpy array representing the input data with shape (n, m), where n is the number of observations and m is the number of variables.
    :return: Covariance matrix of the input data with shape (m, m). The (i, j)-th element of the matrix represents the covariance between the i-th and j-th variables in the data.

    :example:
    >>> data = np.random.randint(0,2, (200, 40)).astype(np.float32)
    >>> covariance_matrix = cov_matrix(data=data)
    """
    n, m = data.shape
    cov = np.full((m, m), 0.0)
    for i in prange(m):
        mean_i = np.sum(data[:, i]) / n
        for j in range(m):
            mean_j = np.sum(data[:, j]) / n
            cov[i, j] = np.sum((data[:, i] - mean_i) * (data[:, j] - mean_j)) / (n - 1)
    return cov

@njit('(float32[:,:], )')
def cronbach_a(data: np.ndarray):
    """
    Cronbach's alpha is a way of assessing reliability by comparing the amount of shared variance, or covariance,
    among the items making up an instrument to the amount of overall variance.
    Cronbach’s alpha can be used to assess internal consistency reliability when the variables
    (e.g., survey items, measure items) analyzed are continuous (interval or ratio measurement scale);

    :example:
    >>> data = np.random.randint(0,2, (200, 40)).astype(np.float32)
    >>> x = cronbach_a(data=data)
    """

    cov = cov_matrix(data=data)
    return (data.shape[1] / (data.shape[1] - 1)) * (1 - np.trace(cov) / np.sum(cov))






data = np.random.randint(0,2, (200, 40)).astype(np.float32)
covariance_matrix = cov_matrix(data=data)
# start = time.time()
# d = cov_matrix(data=data)
# print(time.time() - start)
#
# start = time.time()
# x = cronbach_a(data=data)
# print(time.time() - start)
# start = time.time()
# y = cronbach_alpha(data=pd.DataFrame(data))
# print(time.time() - start)

#data.coV