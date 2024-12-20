import time

import numpy as np
from numba import njit, jit, prange

@njit('(float32[:,:], float32[:,:], int64)')
def linear_frechet_distance(x: np.ndarray, y: np.ndarray, sample: int = 100) -> float:
    """
    Compute the Linear Fréchet Distance between two trajectories.

    The Fréchet Distance measures the dissimilarity between two continuous
    curves or trajectories represented as sequences of points in a 2-dimensional
    space.

    :param ndarray data: First 2D array of size len(frames) representing body-part coordinates x and y.
    :param ndarray data: Second 2D array of size len(frames) representing body-part coordinates x and y.
    :param int sample: The downsampling factor for the trajectories (default is 100If sample > 1, the trajectories are downsampled by selecting every sample-th point.

    .. note::
       Slightly modified from `João Paulo Figueira <https://github.com/joaofig/discrete-frechet/blob/ff5629e5a43cfad44d5e962f4105dd25c90b9289/distances/discrete.py#L67>`_

    :example:
    >>> x = np.random.randint(0, 100, (10000, 2)).astype(np.float32)
    >>> y = np.random.randint(0, 100, (10000, 2)).astype(np.float32)
    >>> distance = linear_frechet_distance(x=x, y=y, sample=100)

    """
    if sample > 1: x, y = x[::sample], y[::sample]
    n_p, n_q = x.shape[0], y.shape[0]
    ca = np.full((n_p, n_q), 0.0)
    for i in prange(n_p):
        for j in range(n_q):
            d = x[i] - y[j]
            d = np.sqrt(np.dot(d, d))
            if i > 0 and j > 0:
                ca[i, j] = max(min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]), d)
            elif i > 0 and j == 0:
                ca[i, j] = max(ca[i - 1, 0], d)
            elif i == 0 and j > 0:
                ca[i, j] = max(ca[0, j - 1], d)
            else:
                ca[i, j] = d
    return ca[n_p - 1, n_q - 1]


# x = np.random.randint(0, 100, (10000, 2)).astype(np.float32)
# y = np.random.randint(0, 100, (10000, 2)).astype(np.float32)
# distance = linear_frechet_distance(x=x, y=y, sample=100)
#
# start = time.time()
# results = linear_frechet_distance(x=x, y=y, sample=100)
# print(time.time() - start)


x1 = np.full((1000, 1), 1)
y1 = np.full((1000,), 2)
for i in range(y1.shape[0]):
    y1[i] = i
line_1 = np.hstack((x1, y1.reshape(-1, 1))).astype(np.float32)

x1 = np.full((1000, 1), 5)
y1 = np.full((1000,), 2)
for i in range(y1.shape[0]):
    y1[i] = i
line_2 = np.hstack((x1, y1.reshape(-1, 1))).astype(np.float32)


linear_frechet_distance(x=line_1, y=line_2, sample=1)

