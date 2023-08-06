__author__ = "Simon Nilsson"

import numpy as np
from numba import njit, jit, prange
from numba.np.extensions import cross2d

@njit('(float32[:,:], int64[:], int64, int64)')
def process(S, P, a, b):
    signed_dist = cross2d(S[P] - S[a], S[b] - S[a])
    K = np.array([i for s, i in zip(signed_dist, P) if s > 0 and i != a and i != b], dtype=np.int64)
    if len(K) == 0:
        return [a, b]
    c = P[np.argmax(signed_dist)]
    return process(S, K, a, c)[:-1] + process(S, K, c, b)

@njit('(float32[:, :,:], types.unicode_type)', fastmath=True)
def jitted_hull(points: np.ndarray, target: str = 'perimeter') -> np.ndarray:
    """
    Compute attributes (e.g., perimeter or area) of a polygon.

    :param array points: 3d array FRAMESxBODY-PARTxCOORDINATE
    :param str target: Options [perimeter, area, hull]
    :return: 1d np.array representing perimeter length or area of polygon on each frame

    .. note::
       Modified from `Jérôme Richard <https://stackoverflow.com/questions/74812556/computing-quick-convex-hull-using-numba/74817179#74817179>`_

    .. image:: _static/img/jitted_hull.png
       :width: 400
       :align: center



    :example:
    >>> points = np.random.randint(1, 50, size=(50, 5, 2)).astype(np.float32)
    >>> results = jitted_hull(points, target='area')
    """

    def perimeter(xy):
        perimeter = np.linalg.norm(xy[0] - xy[-1])
        for i in prange(xy.shape[0]-1):
            p = np.linalg.norm(xy[i] - xy[i+1])
            perimeter+=p
        return perimeter

    def area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    results = np.full((points.shape[0]), np.nan)
    for i in range(points.shape[0]):
        S = points[i, :, :]
        a, b = np.argmin(S[:,0]), np.argmax(S[:,0])
        max_index = np.argmax(S[:,0])
        idx = process(S, np.arange(S.shape[0]), a, max_index)[:-1] + process(S, np.arange(S.shape[0]), max_index, a)[:-1]
        x, y = np.full((len(idx)), np.nan), np.full((len(idx)), np.nan)
        for j in prange(len(idx)):
            x[j], y[j] = S[idx[j], 0], S[idx[j], 1]
        x0, y0 = np.mean(x), np.mean(y)
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        angles = np.where((y - y0) > 0, np.arccos((x - x0) / r), 2 * np.pi - np.arccos((x - x0) / r))
        mask = np.argsort(angles)
        x_sorted, y_sorted = x[mask], y[mask]
        if target == 'perimeter':
            xy = np.vstack((x_sorted, y_sorted)).T
            results[i] = perimeter(xy)
        if target == 'area':
            results[i] = area(x_sorted, y_sorted)

    return results



@jit(nopython=True)
def jitted_centroid(points: np.ndarray) -> np.ndarray:
    """
    Compute the centroid of polygons.

    :param array points: 3d array FRAMESxBODY-PARTxCOORDINATE
    :param str target: Options [perimeter, area]
    :return 1d np.array

    :example:
    >>> points = np.random.randint(1, 50, size=(50, 5, 2)).astype(float)
    >>> results = jitted_centroid(points)
    """

    results = np.full((points.shape[0], 2), np.nan)
    for i in range(points.shape[0]):
        S = points[i, :, :]
        a, b = np.argmin(S[:,0]), np.argmax(S[:,0])
        max_index = np.argmax(S[:,0])
        idx = process(S, np.arange(S.shape[0]), a, max_index)[:-1] + process(S, np.arange(S.shape[0]), max_index, a)[:-1]
        perimeter_points = np.full((len(idx), 2), np.nan)
        for j in prange(len(idx)):
            perimeter_points[j] = points[i][j]
        results[i][0] = np.int(np.mean(perimeter_points[:, 0].flatten()))
        results[i][1] = np.int(np.mean(perimeter_points[:, 1].flatten()))
    return results

# points = np.random.randint(1, 5, size=(1, 10, 2)).astype(np.float32)
# points[0][1] = np.nan
# results = jitted_hull(points, target='area')
# print(results)




# points = np.random.randint(1, 10, size=(50, 5, 2)).astype(np.float32)
# results = jitted_centroid(points)
#
#
# results = np.full((points.shape[0]), np.nan)
