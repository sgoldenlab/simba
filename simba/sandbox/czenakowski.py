import numpy as np
from simba.utils.checks import check_valid_array
from simba.utils.enums import Formats
from numba import jit, njit
import time
from typing import Optional

@jit(nopython=True)
def czebyshev_distance(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
    """
    Calculate the Czebyshev distance between two N-dimensional samples.

    The Czebyshev distance is defined as the maximum absolute difference
    between the corresponding elements of the two arrays.

    .. math::
        D_\infty(p, q) = \max_i \left| p_i - q_i \right|

    :param np.ndarray sample_1: The first sample, an N-dimensional NumPy array.
    :param np.ndarray sample_2: The second sample, an N-dimensional NumPy array.
    :return float: The Czebyshev distance between the two samples.

    :example:
    >>> sample_1 = np.random.randint(0, 10, (10000,100))
    >>> sample_2 = np.random.randint(0, 10, (10000,100))
    >>> czebyshev_distance(sample_1=sample_1, sample_2=sample_2)
    """

    c = 0.0
    for idx in np.ndindex(sample_1.shape):
        c = max((c, np.abs(sample_1[idx] - sample_2[idx])))
    return c


@njit(["(float32[:, :], float64[:], int64)",])
def sliding_czebyshev_distance(x: np.ndarray, window_sizes: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Calculate the sliding Chebyshev distance for a given signal with different window sizes.

    This function computes the sliding Chebyshev distance for a signal `x` using
    different window sizes specified by `window_sizes`. The Chebyshev distance measures
    the maximum absolute difference between the corresponding elements of two signals.

    .. note::
       Normalize array x before passing it to ensure accurate results.

    .. math::
       D_\infty(p, q) = \max_i \left| p_i - q_i \right|

    :param np.ndarray x: Input signal, a 2D array with shape (n_samples, n_features).
    :param np.ndarray window_sizes: Array containing window sizes for sliding computation.
    :param float sample_rate: Sampling rate of the signal.
    :return np.ndarray: 2D array of Chebyshev distances for each window size and position.
    """

    result = np.full((x.shape[0], window_sizes.shape[0]), 0.0)
    for i in range(window_sizes.shape[0]):
        window_size = int(window_sizes[i] * sample_rate)
        for l, r in zip(range(0, x.shape[0] + 1), range(window_size, x.shape[0] + 1)):
            sample, c = x[l:r, :], 0.0
            for j in range(sample.shape[1]):
                c = max(c, (np.abs(np.min(sample[:, j]) - np.max(sample[:, j]))))
            result[r-1, i] = c
    return result



@njit(["(int64[:], int64[:], float64[:])", "(int64[:], int64[:], types.misc.Omitted(None))",
       "(int64[:, :], int64[:, :], float64[:])", "(int64[:, :], int64[:, :], types.misc.Omitted(None))"])
def sokal_michener(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """
    Jitted compute of the Sokal-Michener dissimilarity between two binary vectors or matrices.

    Higher values indicate more dissimilar vectors or matrices, while lower values indicate more similar vectors or matrices.

    The Sokal-Michener dissimilarity is a measure of dissimilarity between two sets
    based on the presence or absence of attributes, commonly used in ecological and
    biological studies. This implementation supports weighted dissimilarity.

    :param np.ndarray x: First binary vector or matrix.
    :param np.ndarray y: Second binary vector or matrix.
    :param Optional[np.ndarray] w: Optional weight vector. If None, all weights are considered as 1.
    :return float: Sokal-Michener dissimilarity between `x` and `y`.

    :example:
    >>> x = np.random.randint(0, 2, (200,))
    >>> y = np.random.randint(0, 2, (200,))
    >>> sokal_michener = sokal_michener(x=x, y=y)
    """
    if w is None:
        w = np.ones(x.shape[0]).astype(np.float64)
    unequal_cnt = 0.0
    for i in np.ndindex(x.shape):
        x_i, y_i = x[i], y[i]
        if x_i != y_i:
            unequal_cnt += 1 * w[i[0]]
    return (2.0 * unequal_cnt) / (x.size + unequal_cnt)

#
# x = np.random.randint(0, 2, (200,))
# y = np.random.randint(0, 2, (200,))
# sokal_michener(x=x, y=y)



#sliding_czebyshev_distance(x=sample_1, window_sizes=np.array([1.0, 2.0]), sample_rate=10.0)
#print(time.time() - start)

# check_valid_array(data=sample_1, source=f'{czebyshev_distance.__name__} sample_1', accepted_dtypes=Formats.NUMERIC_DTYPES.value)
# check_valid_array(data=sample_2, source=f'{czebyshev_distance.__name__} sample_2', accepted_ndims=(sample_1.ndim,), accepted_shapes=(sample_1.shape,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)