import time

import numpy as np
from numba import jit, typed, types
from simba.utils.checks import check_valid_array
from simba.utils.data import get_mode

def berger_parker(x: np.ndarray) -> float:
    """
    Berger-Parker index for the given one-dimensional array.
    The Berger-Parker index is a measure of category dominance, calculated as the ratio of
    the frequency of the most abundant category to the total number of observations

    :example:
    x = np.random.randint(0, 25, (100,)).astype(np.float32)
    z = berger_parker(x=x)
    """
    check_valid_array(source=f'{berger_parker.__name__} x', accepted_ndims=(1,), data=x, accepted_dtypes=(np.float32, np.float64, np.int32, np.int64, np.int8))
    return get_mode(x=x) / x.shape[0]








x = np.random.randint(0, 25, (100,)).astype(np.float32)
start = time.time()
p = berger_parker(x=x)
print(time.time() - start)

# start = time.time()
# u = mode_2(x=x)
# print(time.time() - start)



