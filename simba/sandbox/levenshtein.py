import numpy as np
from numba import jit

#@jit(nopython=True)
def _levenshtein(x, y):
    D = np.zeros((len(x) + 1, len(y) + 1), dtype=int)
    D[0, 1:] = range(1, len(y) + 1)
    D[1:, 0] = range(1, len(x) + 1)

    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            delta = 2 if x[i - 1] != y[j - 1] else 0
            D[i, j] = min(D[i - 1, j - 1] + delta, D[i - 1, j] + 1, D[i, j - 1] + 1)
    return D[-1, -1], D

def levenshtein(x, y):
    """ levenshtein distance for iterable sequences
    """
    # check type
    if (np.all(map(type, x)) is str) and (np.all(map(type, y)) is str):
        _x = np.array(x, dtype=np.str)
        _y = np.array(y, dtype=np.str)
    elif (np.all(map(type, x)) is int) and (np.all(map(type, y)) is int):
        _x = np.array(x, dtype=np.int)
        _y = np.array(y, dtype=np.int)
    elif type(x) is str and type(y) is str:
        _x = np.array(list(x), dtype=np.str)
        _y = np.array(list(y), dtype=np.str)
    else:
        raise TypeError
    print(_x, _y)
    d, D = _levenshtein(_x, _y)
    return d, D

x = np.array(['kitten'])
y = np.array(['kitten'])
# x = "kitten"
# y = "sitting"
r = levenshtein(x, y)