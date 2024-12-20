import numpy as np
from simba.utils.checks import check_valid_array
from typing import Optional

def grubbs_test(x: np.ndarray, left_tail: Optional[bool] = False) -> float:
    """
    Perform Grubbs' test to detect outliers if the minimum or maximum value in a feature series is an outlier.

    :param np.ndarray x: 1D array representing numeric data.
    :param Optional[bool] left_tail: If True, the test calculates the Grubbs' test statistic for the left tail (minimum value). If False (default), it calculates the statistic for the right tail (maximum value).
    :return float: The computed Grubbs' test statistic.

    :example:
    >>> x = np.random.random((100,))
    >>> grubbs_test(x=x)
    """
    check_valid_array(data=x, source=grubbs_test.__name__, accepted_ndims=(1,), accepted_dtypes=(np.float32, np.float64, np.int64, np.float32))
    x = np.sort(x)
    if left_tail:
        return (np.mean(x) - np.min(x)) / np.std(x)
    else:
        return (np.max(x) - np.mean(x)) / np.std(x)






