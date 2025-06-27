import time

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np

from simba.utils.warnings import GPUToolsWarning

try:
    import cupy as cp
    from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
    from cupyx.scipy.spatial.distance import cdist
except:
    GPUToolsWarning(msg='GPU tools not detected, reverting to CPU')
    import numpy as cp
    from scipy.spatial.distance import cdist
    from sklearn.metrics import silhouette_score as cython_silhouette_score

from simba.utils.checks import check_str, check_valid_array
from simba.utils.enums import Formats


def euclid_cdist_gpu(X, Y=None):
    Y = X if Y is None else Y
    X_norm = cp.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_norm = cp.sum(Y ** 2, axis=1).reshape(1, -1)
    return cp.sqrt(X_norm - 2 * X @ Y.T + Y_norm)


def silhouette_score_gpu(x: np.ndarray,
                         y: np.ndarray,
                         metric: Literal["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "sqeuclidean"] =  'euclidean') -> float:
    """
    Compute the Silhouette Score for clustering assignments on GPU using a specified distance metric.

    :param np.ndarray x: Feature matrix of shape (n_samples, n_features) containing numeric data.
    :param np.ndarray y: Cluster labels array of shape (n_samples,) with numeric labels.
    :param Literal["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "sqeuclidean"] metric:  Distance metric to use (default='euclidean'). Must be one of: "cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", or "sqeuclidean".
    :return: Mean silhouette score as a float.
    :rtype: float

    :example:
    >>> x, y = make_blobs(n_samples=50000, n_features=20, centers=5, cluster_std=10, center_box=(-1, 1))
    >>> score_gpu = silhouette_score_gpu(x=x, y=y)
    """
    VALID_METRICS = ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "sqeuclidean"]
    check_str(name=f'{silhouette_score_gpu.__name__} metric', value=metric, options=VALID_METRICS)
    check_valid_array(data=x, source=f'{silhouette_score_gpu.__name__} x', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, source=f'{silhouette_score_gpu.__name__} y', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0]])
    x = cython_silhouette_score(X=x, labels=y, metric=metric)
    return x

    # x, y = cp.ascontiguousarray(cp.asarray(x)), cp.ascontiguousarray(cp.asarray(y))
    # dists = euclid_cdist_gpu(X=x)
    #
    # cluster_ids = np.unique(y)
    # print(cluster_ids)




def silhouette_score(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the silhouette score for the given dataset and labels.

    :param np.ndarray x: The dataset as a 2D NumPy array of shape (n_samples, n_features).
    :param np.ndarray y: Cluster labels for each data point as a 1D NumPy array of shape (n_samples,).
    :returns: The average silhouette score for the dataset.
    :rtype: float

    :example:
    >>> x, y = make_blobs(n_samples=10000, n_features=400, centers=5, cluster_std=10, center_box=(-1, 1))
    >>> score = silhouette_score(x=x, y=y)

    >>> from sklearn.metrics import silhouette_score as sklearn_silhouette # SKLEARN ALTERNATIVE
    >>> score_sklearn = sklearn_silhouette(x, y)

    """
    dists = cdist(x, x)
    results = np.full(x.shape[0], fill_value=-1.0, dtype=np.float32)
    cluster_ids = np.unique(y)
    cluster_indices = {cluster_id: np.argwhere(y == cluster_id).flatten() for cluster_id in cluster_ids}

    for i in range(x.shape[0]):
        intra_idx = cluster_indices[y[i]]
        if len(intra_idx) <= 1:
            a_i = 0.0
        else:
            intra_distances = dists[i, intra_idx]
            a_i = np.sum(intra_distances) / (intra_distances.shape[0] - 1)
        b_i = np.inf
        for cluster_id in cluster_ids:
            if cluster_id != y[i]:
                inter_idx = cluster_indices[cluster_id]
                inter_distances = dists[i, inter_idx]
                b_i = min(b_i, np.mean(inter_distances))
        results[i] = (b_i - a_i) / max(a_i, b_i)

    return np.mean(results)

from sklearn.datasets import make_blobs

start = time.time()

x, y = make_blobs(n_samples=50000, n_features=20, centers=5, cluster_std=10, center_box=(-1, 1))
score_gpu = silhouette_score_gpu(x=x, y=y)
print(time.time() - start)

from sklearn.metrics import \
    silhouette_score as sklearn_silhouette  # SKLEARN ALTERNATIVE

start = time.time()
score_sklearn = sklearn_silhouette(x, y)
print(time.time() - start)
print(score_gpu, score_sklearn)