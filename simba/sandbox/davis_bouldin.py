import time

import numpy as np
from sklearn.metrics.cluster import davies_bouldin_score

from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import check_valid_array

def davis_bouldin(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Davis-Bouldin index for evaluating clustering performance.

    Davis-Bouldin index measures the clustering quality based on the within-cluster
    similarity and between-cluster dissimilarity. Lower values indicate better clustering.

    .. note::
       Modified from `scikit-learn <https://github.com/scikit-learn/scikit-learn/blob/f07e0138bfee41cd2c0a5d0251dc3fe03e6e1084/sklearn/metrics/cluster/_unsupervised.py#L390>`_

    :param np.ndarray x: 2D array representing the data points. Shape (n_samples, n_features/n_dimension).
    :param np.ndarray y: 2D array representing cluster labels for each data point. Shape (n_samples,).
    :return float: Davis-Bouldin score.

    :example:
    >>> x = np.random.randint(0, 100, (100, 2))
    >>> y = np.random.randint(0, 3, (100,))
    >>> Statistics.davis_bouldin(x=x, y=y)
    """

    check_valid_array(data=x, source=Statistics.davis_bouldin.__name__, accepted_ndims=(2,), accepted_dtypes=(int, float))
    check_valid_array(data=y, source=Statistics.davis_bouldin.__name__, accepted_ndims=(1,), accepted_shapes=[(x.shape[0],)], accepted_dtypes=(int, float))
    n_labels = np.unique(y).shape[0]
    intra_dists = np.full((n_labels), 0.0)
    centroids = np.full((n_labels, x.shape[1]), 0.0)
    for k in range(n_labels):
        cluster_k = x[np.argwhere(y == k)].reshape(-1, 2)
        cluster_mean = np.full((x.shape[1]), np.nan)
        for i in range(cluster_mean.shape[0]):
            cluster_mean[i] = np.mean(cluster_k[:, i].flatten())
        centroids[k] = cluster_mean
        intra_dists[k] = np.average(FeatureExtractionMixin.framewise_euclidean_distance(location_1=cluster_k,
                                                                                        location_2=np.full(cluster_k.shape, cluster_mean),
                                                                                        px_per_mm=1))
    centroid_distances = FeatureExtractionMixin.cdist(array_1=centroids.astype(np.float32), array_2=centroids.astype(np.float32))
    if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
        return 0.0
    centroid_distances[centroid_distances == 0] = np.inf
    combined_intra_dists = intra_dists[:, None] + intra_dists
    return np.mean(np.max(combined_intra_dists / centroid_distances, axis=1))






x = np.random.random((1000000, 2))
y = np.random.randint(0, 25, (1000000,))
start = time.time()
z = davis_bouldin(x=x, y=y)
print(time.time() - start)

start = time.time()
p = davis_bouldin(x, y)
print(time.time() - start)
print(z, p)