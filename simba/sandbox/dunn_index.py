import numpy as np
from itertools import permutations
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import check_valid_array

def dunn_index(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Dunn index to evaluate the quality of clustered labels.

    This function calculates the Dunn index, which is a measure of clustering quality.
    The index considers the ratio of the minimum inter-cluster distance to the maximum
    intra-cluster distance. A higher Dunn index indicates better clustering.

    .. note::
       Modified from `jqmviegas <https://github.com/jqmviegas/jqm_cvi/>`_
       Wiki `https://en.wikipedia.org/wiki/Dunn_index <https://en.wikipedia.org/wiki/Dunn_index>`_
       Uses Euclidean distances.

    :param np.ndarray x: 2D array representing the data points. Shape (n_samples, n_features).
    :param np.ndarray y: 2D array representing cluster labels for each data point. Shape (n_samples,).
    :return float: The Dunn index value

    :example:
    >>> x = np.random.randint(0, 100, (100, 2))
    >>> y = np.random.randint(0, 3, (100,))
    >>> dunn_index(x=x, y=y)
    """

    check_valid_array(data=x, source=dunn_index.__name__, accepted_ndims=(2,), accepted_dtypes=(int, float))
    check_valid_array(data=y, source=dunn_index.__name__, accepted_ndims=(1,), accepted_shapes=[(x.shape[0],)], accepted_dtypes=(int, float))
    distances = FeatureExtractionMixin.cdist(array_1=x.astype(np.float32), array_2=x.astype(np.float32))
    ks = np.sort(np.unique(y)).astype(np.int64)
    deltas = np.full((ks.shape[0], ks.shape[0]), np.inf)
    big_deltas = np.zeros([ks.shape[0], 1])
    for (i, l) in list(permutations(ks, 2)):
        values = distances[np.where((y == i))][:, np.where((y == l))]
        deltas[i, l] = np.min(values[np.nonzero(values)])
    for k in ks:
        values = distances[np.where((y == ks[k]))][:, np.where((y == ks[k]))]
        big_deltas[k] = np.max(values)

    return np.min(deltas) / np.max(big_deltas)





