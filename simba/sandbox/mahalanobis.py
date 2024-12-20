import time

import numpy as np
from simba.mixins.statistics_mixin import Statistics
from numba import jit, njit, prange
from scipy.spatial.distance import cdist

@jit('(float32[:,:],)')
def mahalanobis_distance_cdist(data: np.ndarray) -> np.ndarray:
    """
    Compute the Mahalanobis distance between every pair of observations in a 2D array using Numba.

    The Mahalanobis distance is a measure of the distance between a point and a distribution. It accounts for correlations between variables and the scales of the variables, making it suitable for datasets where features are not independent and have different variances.

    However, Mahalanobis distance may not be suitable in certain scenarios, such as:
    - When the dataset is small and the covariance matrix is not accurately estimated.
    - When the dataset contains outliers that significantly affect the estimation of the covariance matrix.
    - When the assumptions of multivariate normality are violated.

    :param np.ndarray data: 2D array with feature observations. Frames on axis 0 and feature values on axis 1
    :return np.ndarray: Pairwise Mahalanobis distance matrix where element (i, j) represents the Mahalanobis distance between  observations i and j.

    :example:
    >>> data = np.random.randint(0, 50, (1000, 200)).astype(np.float32)
    >>> x = mahalanobis_distance_cdist(data=data)
    """

    covariance_matrix = np.cov(data, rowvar=False)
    inv_covariance_matrix = np.linalg.inv(covariance_matrix).astype(np.float32)
    n = data.shape[0]
    distances = np.zeros((n, n))
    for i in prange(n):
        for j in range(n):
            diff = data[i] - data[j]
            diff = diff.astype(np.float32)
            distances[i, j] = np.sqrt(np.dot(np.dot(diff, inv_covariance_matrix), diff.T))
    return distances










    pass




data = np.random.randint(0, 50, (1000, 200)).astype(np.float32)
#data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float64)
start = time.time()
x = mahalanobis_distance(data=data)
print(time.time() - start)

start = time.time()
covariance_matrix = np.cov(data, rowvar=False)
inv_covariance_matrix = np.linalg.inv(covariance_matrix)
distances = cdist(data, data, 'mahalanobis', VI=inv_covariance_matrix)
print(time.time() - start)

