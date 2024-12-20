import numpy as np
from simba.utils.checks import check_valid_array


def menhinicks_index(x: np.array) -> float:
    """
    Calculate the Menhinick's Index for a given array of values.

    Menhinick's Index is a measure of category richness.
    It quantifies the number of categories relative to the square root of the total number of observations.

    :example:
    >>> x = np.random.randint(0, 5, (1000,))
    >>> menhinicks_index(x=x)
    """
    check_valid_array(source=f'{menhinicks_index.__name__} x', accepted_ndims=(1,), data=x, accepted_dtypes=(np.float32, np.float64, np.int32, np.int64, np.int8), min_axis_0=2)
    return np.unique(x).shape[0] / np.sqrt(x.shape[0])