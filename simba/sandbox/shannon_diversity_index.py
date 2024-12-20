import numpy as np
from numba import jit

@jit(nopython=True)
def shannon_diversity_index(x: np.ndarray) -> float:
    """
    Calculate the Shannon Diversity Index for a given array of categories. The Shannon Diversity Index is a measure of diversity in a
    categorical feature, taking into account both the number of different categories (richness)
    and their relative abundances (evenness).

    :example:
    >>> x = np.random.randint(0, 100, (100, ))
    >>> shannon_diversity_index(x=x)
    """


    unique_v = np.unique(x)
    n_unique = np.unique(x).shape[0]
    results = np.full((n_unique,), np.nan)
    for i in range(unique_v.shape[0]):
        v = unique_v[i]
        cnt = np.argwhere(x == v).flatten().shape[0]
        pi = cnt / x.shape[0]
        results[i] = pi * np.log(pi)
    return np.sum(np.abs(results))

x = np.random.randint(0, 100, (100, ))
y = np.random.randint(0, 1, (300, ))
x = np.append(x, y)

shannon_diversity_index(x=x)


