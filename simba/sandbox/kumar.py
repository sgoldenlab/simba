from numba import njit, jit
import numpy as np
from simba.utils.enums import Formats
from simba.utils.checks import check_valid_array

def kumar_hassebrook_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """
    Kumar-Hassebrook similarity is a measure used to quantify the similarity between two vectors.

    .. note::
        Kumar-Hassebrook similarity score of 1 indicates identical vectors and 0 indicating no similarity

    :param np.ndarray x: 1D array representing the first feature values.
    :param np.ndarray y: 1D array representing the second feature values.
    :return: Kumar-Hassebrook similarity between vectors x and y.

    :example:
    >>> x, y = np.random.randint(0, 500, (1000,)), np.random.randint(0, 500, (1000,))
    >>> kumar_hassebrook_similarity(x=x, y=y)
    """
    check_valid_array(data=x, source=f'{kumar_hassebrook_similarity.__name__} x', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, source=f'{kumar_hassebrook_similarity.__name__} y', accepted_ndims=(1,), accepted_shapes=(x.shape,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    dot_product = np.dot(x, y)
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    return dot_product / (norm_x**2 + norm_y**2 - dot_product)


def wave_hedges_distance(x: np.ndarray, y: np.ndarray) -> float:
    """

    Computes the Wave-Hedges distance between two 1-dimensional arrays `x` and `y`. The Wave-Hedges distance is a measure of dissimilarity between arrays.

    .. note::
        Wave-Hedges distance score of 0 indicate identical arrays. There is no upper bound.


    :example:
    >>> x = np.random.randint(0, 500, (1000,))
    >>> y = np.random.randint(0, 500, (1000,))
    >>> wave_hedges_distance(x=x, y=y)
    """

    check_valid_array(data=x, source=f'{kumar_hassebrook_similarity.__name__} x', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, source=f'{kumar_hassebrook_similarity.__name__} y', accepted_ndims=(1,), accepted_shapes=(x.shape,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    x_y = abs(x - y)
    xy_max = np.maximum(x, y)
    return np.sum(np.where(((x_y != 0) & (xy_max != 0)), x_y / xy_max, 0))


def gower_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Gower-like distance vector between corresponding rows of two numerical matrices.
    Gower distance is a measure of dissimilarity between two vectors (or rows in this case).

    .. note::
       This function assumes x and y have the same shape and only considers numerical attributes.
        Each observation in x is compared to the corresponding observation in y based on normalized
        absolute differences across numerical columns.

    :param np.ndarray x: First numerical matrix with shape (m, n).
    :param np.ndarray y: Second numerical matrix with shape (m, n).
    :return np.ndarray: Gower-like distance vector with shape (m,).

    :example:
    >>> x, y = np.random.randint(0, 500, (1000, 6000)), np.random.randint(0, 500, (1000, 6000))
    >>> gower_distance(x=x, y=y)

    """
    check_valid_array(data=x, source=f'{gower_distance.__name__} x', accepted_ndims=(1, 2), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, source=f'{gower_distance.__name__} y', accepted_ndims=(x.ndim,), accepted_shapes=(x.shape,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    field_ranges = np.max(x, axis=0) - np.min(x, axis=0)
    results = np.full((x.shape[0]), np.nan)
    for i in range(x.shape[0]):
        u, v = x[i], y[i]
        dist = 0.0
        for j in range(u.shape[0]):
            if field_ranges[j] != 0:
                dist += np.abs(u[j] - v[j]) / field_ranges[j]
        results[i] = dist / u.shape[0]
    return results


def normalized_google_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Normalized Google Distance (NGD) between two vectors or matrices.

    .. note::
       This function assumes x and y have the same shape. It computes NGD based on the sum of elements and the minimum values between corresponding elements of x and y.

    :param np.ndarray x: First numerical matrix with shape (m, n).
    :param np.ndarray y: Second array or matrix with shape (m, n).
    :return float:  Normalized Google Distance between x and y.

    :example:
    >>> x, y = np.random.randint(0, 500, (1000,200)), np.random.randint(0, 500, (1000,200))
    >>> normalized_google_distance(x=y, y=x)
    """
    check_valid_array(data=x, source=f'{normalized_google_distance.__name__} x', accepted_ndims=(1, 2), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, source=f'{normalized_google_distance.__name__} y', accepted_ndims=(x.ndim,), accepted_shapes=(x.shape,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)

    sum_x, sum_y = np.sum(x), np.sum(y)
    sum_min = np.sum(np.minimum(x, y))
    D = (sum_x + sum_y) - np.min([sum_x, sum_y])
    N = np.max([sum_x, sum_y]) - sum_min
    if D == 0:
        return -1.0
    else:
        return N / D













#kumar_hassebrook_similarity(x=x, y=y)

#




# def gower_distance(x: np.ndarray, y: np.ndarray) -> float:
#     check_
#
#
#
#
#     np.sum(np.abs(x - y)) / x.size
#
