from typing import Optional, Tuple
import numpy as np
from scipy.stats.distributions import chi2
from simba.utils.checks import check_valid_array
from simba.utils.errors import CountError, InvalidInputError


def mcnemar(x: np.ndarray, y: np.ndarray, ground_truth: np.ndarray, continuity_corrected: Optional[bool] = True) -> Tuple[float, float]:
    """
    McNemar's Test to compare the difference in predictive accuracy of two models.

    E.g., can be used to compute if the accuracy of two classifiers are significantly different when transforming the same data.

    .. note::
       `mlextend <https://github.com/rasbt/mlxtend/blob/master/mlxtend/evaluate/mcnemar.py>`__.


    :param np.ndarray x: 1-dimensional Boolean array with predictions of the first model.
    :param np.ndarray x: 1-dimensional Boolean array with predictions of the second model.
    :param np.ndarray x: 1-dimensional Boolean array with ground truth labels.
    :param Optional[bool] continuity_corrected : Whether to apply continuity correction. Default is True.

    :example:
    >>> x = np.random.randint(0, 2, (100000, ))
    >>> y = np.random.randint(0, 2, (100000, ))
    >>> ground_truth = np.random.randint(0, 2, (100000, ))
    >>> mcnemar(x=x, y=y, ground_truth=ground_truth)
    """

    check_valid_array(data=x, source=mcnemar.__name__, accepted_ndims=(1,), accepted_dtypes=(np.int64, np.int32, np.int8))
    check_valid_array(data=y, source=mcnemar.__name__, accepted_ndims=(1,), accepted_dtypes=(np.int64, np.int32, np.int8))
    check_valid_array(data=ground_truth, source=mcnemar.__name__, accepted_ndims=(1,), accepted_dtypes=(np.int64, np.int32, np.int8))
    if len(list({x.shape[0], y.shape[0], ground_truth.shape[0]})) != 1:
        raise CountError(msg=f'The three arrays has to be equal lengths but got: {x.shape[0], y.shape[0], ground_truth.shape[0]}', source=mcnemar.__name__)
    for i in [x, y, ground_truth]:
        additional = list(set(list(np.sort(np.unique(i)))) - {0, 1})
        if len(additional) > 0: raise InvalidInputError(msg=f'Mcnemar requires binary input data but found {additional}', source=mcnemar.__name__)
    data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), ground_truth.reshape(-1, 1)))
    b = np.where((data == (0, 1, 0)).all(axis=1))[0].shape[0] + np.where((data == (1, 0, 1)).all(axis=1))[0].shape[0]
    c = np.where((data == (1, 0, 0)).all(axis=1))[0].shape[0] + np.where((data == (0, 1, 1)).all(axis=1))[0].shape[0]
    if not continuity_corrected:
        x = (np.square(b-c)) / (b+c)
    else:
        x = (np.square(np.abs(b-c)-1)) / (b+c)
    p = chi2.sf(x, 1)
    return x, p








x = np.random.randint(0, 2, (100000, ))
y = np.random.randint(0, 2, (100000, ))
ground_truth = np.random.randint(0, 2, (100000, ))
mcnemar(x=x, y=y, ground_truth=ground_truth)