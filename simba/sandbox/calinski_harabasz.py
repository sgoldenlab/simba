import numpy as np
from numba import jit, prange, njit
import time

from sklearn.metrics.cluster import calinski_harabasz_score

@njit("(float32[:,:], int64[:])", cache=True)
def calinski_harabasz(x: np.ndarray,
                      y: np.ndarray) -> float:
    """
    Compute the Calinski-Harabasz score to evaluate clustering quality.

    The Calinski-Harabasz score is a measure of cluster separation and compactness.
    It is calculated as the ratio of the between-cluster dispersion to the
    within-cluster dispersion. A higher score indicates better clustering.

    :param x: 2D array representing the data points. Shape (n_samples, n_features/n_dimension).
    :param y: 2D array representing cluster labels for each data point. Shape (n_samples,).
    :return float: Calinski-Harabasz score.

    :example:
    >>> x = np.random.random((100, 2)).astype(np.float32)
    >>> y = np.random.randint(0, 100, (100,)).astype(np.int64)
    >>> calinski_harabasz(x=x, y=y)
    """
    n_labels = np.unique(y).shape[0]
    labels = np.unique(y)
    extra_dispersion, intra_dispersion = 0.0, 0.0
    global_mean = np.full((x.shape[1]), np.nan)
    for i in range(x.shape[1]):
        global_mean[i] = np.mean(x[:, i].flatten())
    for k in range(n_labels):
        cluster_k = x[np.argwhere(y == labels[k]).flatten(), :]
        mean_k = np.full((x.shape[1]), np.nan)
        for i in prange(cluster_k.shape[1]): mean_k[i] = np.mean(cluster_k[:, i].flatten())
        extra_dispersion += len(cluster_k) * np.sum((mean_k - global_mean) ** 2)
        intra_dispersion += np.sum((cluster_k - mean_k) ** 2)
    return extra_dispersion * (x.shape[0] - n_labels) / (intra_dispersion * (n_labels - 1.0))


x = np.random.random((100, 2)).astype(np.float32)
y = np.random.randint(-1, 2, (100,)).astype(np.int64)
calinski_harabasz(x=x, y=y)