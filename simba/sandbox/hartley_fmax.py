import numpy as np
from simba.utils.checks import check_valid_array

def hartley_fmax(x: np.ndarray, y: np.ndarray):
    """
    Compute Hartley's Fmax statistic to test for equality of variances between two features or groups.

    Values close to one represents closer to equal variance.

    :param np.ndarray x: 1D array representing numeric data of the first group/feature.
    :param np.ndarray x: 1D array representing numeric data of the second group/feature.

    :example:
    >>> x = np.random.random((100,))
    >>> y = np.random.random((100,))
    >>> hartley_fmax(x=x, y=y)
    """
    check_valid_array(data=x, source=hartley_fmax.__name__, accepted_ndims=(1,), accepted_dtypes=(np.float32, np.float64, np.int64, np.float32))
    check_valid_array(data=y, source=hartley_fmax.__name__, accepted_ndims=(1,), accepted_dtypes=(np.float32, np.float64, np.int64, np.float32))
    max_var = np.max((np.var(x), np.var(y)))
    min_var = np.min((np.var(x), np.var(y)))
    return max_var / min_var