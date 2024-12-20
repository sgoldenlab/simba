import numpy as np
from simba.utils.checks import check_valid_array


def brillouins_index(x: np.array) -> float:
    """
    Calculate Brillouin's Diversity Index for a given array of values.

    Brillouin's Diversity Index is a measure of species diversity that accounts for both species richness
    and evenness of distribution.

    Brillouin's Diversity Index (H) is calculated using the formula:

    .. math::

       H = \\frac{1}{\\log(S)} \\sum_{i=1}^{S} \\frac{N_i(N_i - 1)}{n(n-1)}

    where:
    - \( H \) is Brillouin's Diversity Index,
    - \( S \) is the total number of unique species,
    - \( N_i \) is the count of individuals in the i-th species,
    - \( n \) is the total number of individuals.

    :param np.array x: One-dimensional numpy array containing the values for which Brillouin's Index is calculated.
    :return float: Brillouin's Diversity Index value for the input array `x`

    :example:
    >>> x = np.random.randint(0, 10, (100,))
    >>> brillouins_index(x)
    """

    check_valid_array(
        source=f"{brillouins_index.__name__} x",
        accepted_ndims=(1,),
        data=x,
        accepted_dtypes=(np.float32, np.float64, np.int32, np.int64, np.int8),
        min_axis_0=2,
    )
    n_total = x.shape[0]
    n_unique = np.unique(x, return_counts=True)[1]
    if n_unique.shape[0] == 1:
        return 1.0
    else:
        S = len(n_unique)
        h = 0
        for count in n_unique:
            h += count * (count - 1)
        h /= (n_total * (n_total - 1))
        h /= np.log(S)
        return h