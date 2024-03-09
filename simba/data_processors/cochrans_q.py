from typing import Optional
import numpy as np
from simba.utils.checks import check_valid_array

def cochrans_q(data: np.ndarray) -> float:
    """
    Compute Cochrans Q for 2-dimensional boolean array.

    Can be used to evaluate if the performance of multiple (>=2) classifiers on the same data is the same or significantly different.

    .. note::
       If two classifiers, consider ``simba.mixins.statistics.Statistics.mcnemar``.

    :param np.ndarray data: Two dimensional array of boolean values where axis 1 represents classifiers or features and rows represent frames.
    :return float: Cochran's Q statistic

    :example:
    >>> data = np.random.randint(0, 2, (100000, 4))
    >>> cochrans_q(data=data)
    """
    check_valid_array(data=data, source=cochrans_q.__name__, accepted_ndims=(2,))
    additional = list(set(list(np.sort(np.unique(data)))) - {0, 1})
    if len(additional) > 0:
        raise InvalidInputError(msg=f'Cochrans Q requires binary input data but found {additional}', source=Statistics.mcnemar.__name__)
    col_sums = np.sum(data, axis=0)
    row_sum_sum = np.sum(np.sum(data, axis=1))
    row_sum_square_sum = np.sum(np.square(np.sum(data, axis=1)))
    k = data.shape[1]
    g2 = np.sum(sum(np.square(col_sums)))
    nominator = (k-1) * ((k*g2) - np.square(np.sum(col_sums)))
    denominator = (k * row_sum_sum) - row_sum_square_sum

    if nominator == 0 or denominator == 0:
        return -1.0
    else:
        return nominator / denominator


