import numpy as np
from simba.utils.checks import check_valid_array


def margalef_diversification_index(x: np.array) -> float:
    """
    Calculate the Margalef Diversification Index for a given array of values.

    The Margalef Diversification Index is a measure of category diversity. It quantifies the richness of a community
    relative to the number of individuals.

    :example:
    >>> x = np.random.randint(0, 100, (100,))
    >>> margalef_diversification_index(x=x)
    """
    check_valid_array(source=f'{margalef_diversification_index.__name__} x', accepted_ndims=(1,), data=x, accepted_dtypes=(np.float32, np.float64, np.int32, np.int64, np.int8), min_axis_0=2)
    n_unique = np.unique(x).shape[0]
    return (n_unique-1) / np.log(x.shape[0])



x = np.random.randint(0, 100, (100,))
margalef_diversification_index(x=x)