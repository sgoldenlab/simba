import numpy as np
from pingouin.correlation import bicor
from numba import jit
import time


@jit(nopython=True)
def biweight_midcorrelation(x: np.ndarray, y: np.ndarray, c: int = 9):
    x_median = np.median(x)
    y_median = np.median(y)

    x_mad = np.median(np.abs(x - x_median))
    y_mad = np.median(np.abs(y - y_median))

    if x_mad == 0 or y_mad == 0:
        return -1.0, -1.0

    u = (x - x_median) / (c * x_mad)
    v = (y - y_median) / (c * y_mad)
    w_x = (1 - u ** 2) ** 2 * ((1 - np.abs(u)) > 0)
    w_y = (1 - v ** 2) ** 2 * ((1 - np.abs(v)) > 0)

    x_norm = (x - x_median) * w_x
    y_norm = (y - y_median) * w_y
    denom = np.sqrt((x_norm ** 2).sum()) * np.sqrt((y_norm ** 2).sum())
    r = (x_norm * y_norm).sum() / denom
    #print(r)



x = np.random.randint(0, 50, (10000000,))
y = np.random.randint(0, 50, (10000000,))
start = time.time()
biweight_midcorrelation(x=x, y=y)
print(time.time() - start)
start = time.time()
bicor(x=x, y=y)
print(time.time() - start)


