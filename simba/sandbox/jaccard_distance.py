import numpy as np
from simba.utils.checks import check_valid_array

def jaccard_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Jaccard distance between two 1D NumPy arrays.

    The Jaccard distance is a measure of dissimilarity between two sets. It is defined as the size of the
    intersection of the sets divided by the size of the union of the sets.

    :param np.ndarray x: The first 1D NumPy array.
    :param np.ndarray y: The second 1D NumPy array.
    :return float: The Jaccard distance between arrays x and y.

    :example:
    >>> x = np.random.randint(0, 5, (100))
    >>> y = np.random.randint(0, 7, (100))
    >>> jaccard_distance(x=x, y=y)
    >>> 0.71428573
    """
    check_valid_array(data=x, source=f'{jaccard_distance.__name__} x', accepted_ndims=(1,))
    check_valid_array(data=y, source=f'{jaccard_distance.__name__} y', accepted_ndims=(1,), accepted_dtypes=[x.dtype.type])
    u_x, u_y = np.unique(x), np.unique(y)
    return np.float32(1 -(len(np.intersect1d(u_x, u_y)) / len(np.unique(np.hstack((u_x, u_y))))))



    #union = np.unique(np.hstack((x, y)))


x = np.random.randint(0, 5, (100))
y = np.random.randint(0, 7, (100))
jaccard_distance(x=x, y=y)