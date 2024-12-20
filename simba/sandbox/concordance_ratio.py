import time

import numpy as np
from numba import jit, njit


@njit('(int64[:, :]), bool_')
def concordance_ratio(x: np.ndarray, invert: bool) -> float:
    """
    Calculate the concordance ratio of a 2D numpy array.

    :param np.ndarray x: A 2D numpy array with ordinals represented as integers.
    :param bool invert: If True, the concordance ratio is inverted, and disconcordance ratio is returned
    :return float: The concordance ratio, representing the count of rows with only one unique value divided by the total number of rows in the array.

    :example:
    >>> x = np.random.randint(0, 2, (5000, 4))
    >>> results = concordance_ratio(x=x, invert=False)
    """
    conc_count = 0
    for i in prange(x.shape[0]):
        unique_cnt = np.unique((x[i])).shape[0]
        if unique_cnt == 1:
            conc_count += 1
    if invert:
        conc_count = x.shape[0] - conc_count
    return conc_count / x.shape[0]

# x = np.random.randint(0, 2, (5000, 4))
# start = time.time()
# results = concordance_ratio(x=x, invert=False)
# print(time.time() - start)
# print(results)





