import numpy as np
from sklearn.metrics import adjusted_rand_score
from simba.utils.checks import check_valid_array

def adjusted_rand_index(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index (ARI)

    A value close to 0.0 represent random labeling and exactly 1.0 when the clusterings are identical.

    :param np.ndarray x: 1D array representing cluster labels for model one. Shape (n_samples,).
    :param np.ndarray y: 1D array representing cluster labels for model two. Shape (x.shape[0],).
    :return float: Adjusted Rand Index value, ranges from -1 to 1. A value close to 1 indicates a perfect match between the two clusterings, while a value close to 0 indicates random labeling.

    :example:
    >>> x = np.random.randint(0, 2, (10000,))
    >>> y = np.random.randint(0, 2, (10000,))
    >>> adjusted_rand_index(x=x, y=y)
    """
    check_valid_array(data=x, source=adjusted_rand_index.__name__, accepted_ndims=(1,))
    check_valid_array(data=x, source=adjusted_rand_index.__name__, accepted_shapes=[(x.shape)])
    return  adjusted_rand_score(x, y)




#
#
