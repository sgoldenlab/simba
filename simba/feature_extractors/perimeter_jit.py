__author__ = "Simon Nilsson; sronilsson@gmail.com"
from typing import Optional

import numpy as np
from numba import jit, njit, prange
from numba.np.extensions import cross2d

from simba.utils.checks import check_float, check_str, check_valid_array
from simba.utils.enums import Formats


@njit("(float32[:,:], int64[:], int64, int64)")
def process(S, P, a, b):
    """
    One step of the quickhull algorithm: partition points by signed distance to line (a,b), recurse on the subset above the line.
    Uses 2D cross product for signed distance; farthest point becomes new hull vertex. Returns hull vertex indices from a to b (excluding b).
    """
    signed_dist = cross2d(S[P] - S[a], S[b] - S[a])
    K = np.array(
        [i for s, i in zip(signed_dist, P) if s > 0 and i != a and i != b],
        dtype=np.int64,
    )
    if len(K) == 0:
        return [a, b]
    c = P[np.argmax(signed_dist)]
    return process(S, K, a, c)[:-1] + process(S, K, c, b)


@njit("(float32[:, :, :], types.unicode_type)", fastmath=True)
def jitted_hull(points: np.ndarray, target: str = "perimeter") -> np.ndarray:
    """
    Convex hull perimeter or area per frame from body-part (x,y) coordinates.

    For each frame, builds the 2D convex hull (quickhull), then:
    - **perimeter**: sum of edge lengths \|v_{i+1} - v_i\| (with wrap).
    - **area**: shoelace formula, 0.5 * |Σ(x_i y_{i+1} - y_i x_{i+1})|.

    :param points: (n_frames, n_body_parts, 2) float32, [x, y] per point.
    :param target: ``'perimeter'`` or ``'area'``.
    :return: (n_frames,) float64; NaN where hull fails.

    .. seealso::
       :func:`simba.feature_extractors.perimeter_jit.get_hull_sizes` (wrapper with validation).
    """

    def perimeter(xy):
        perimeter = np.linalg.norm(xy[0] - xy[-1])
        for i in prange(xy.shape[0] - 1):
            p = np.linalg.norm(xy[i] - xy[i + 1])
            perimeter += p
        return perimeter

    def area(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    results = np.full((points.shape[0]), np.nan)
    for i in range(points.shape[0]):
        S = points[i, :, :]
        a, b = np.argmin(S[:, 0]), np.argmax(S[:, 0])
        max_index = np.argmax(S[:, 0])
        idx = (
            process(S, np.arange(S.shape[0]), a, max_index)[:-1]
            + process(S, np.arange(S.shape[0]), max_index, a)[:-1]
        )
        x, y = np.full((len(idx)), np.nan), np.full((len(idx)), np.nan)
        for j in prange(len(idx)):
            x[j], y[j] = S[idx[j], 0], S[idx[j], 1]
        x0, y0 = np.mean(x), np.mean(y)
        r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
        angles = np.where(
            (y - y0) > 0, np.arccos((x - x0) / r), 2 * np.pi - np.arccos((x - x0) / r)
        )
        mask = np.argsort(angles)
        x_sorted, y_sorted = x[mask], y[mask]
        if target == "perimeter":
            xy = np.vstack((x_sorted, y_sorted)).T
            results[i] = perimeter(xy)
        if target == "area":
            results[i] = area(x_sorted, y_sorted)

    return results


@jit(nopython=True)
def jitted_centroid(points: np.ndarray) -> np.ndarray:
    """
    Centroid of the convex hull per frame (mean of hull vertex coordinates).

    For each frame: quickhull → hull vertex indices → centroid = (mean(x), mean(y)) of those vertices.

    :param points: (n_frames, n_body_parts, 2) [x, y] coordinates.
    :return: (n_frames, 2) int32; centroid (x, y) per frame; NaN where hull fails.

    :example:
    >>> points = np.random.randint(1, 50, size=(50, 5, 2)).astype(np.float32)
    >>> centroids = jitted_centroid(points)
    """
    results = np.full(shape=(points.shape[0], 2), fill_value=np.nan, dtype=np.int32)
    for i in range(points.shape[0]):
        S = points[i, :, :]
        a, b = np.argmin(S[:, 0]), np.argmax(S[:, 0])
        max_index = np.argmax(S[:, 0])
        idx = (
            process(S, np.arange(S.shape[0]), a, max_index)[:-1]
            + process(S, np.arange(S.shape[0]), max_index, a)[:-1]
        )
        perimeter_points = np.full((len(idx), 2), np.nan)
        for j in prange(len(idx)):
            perimeter_points[j, 0] = S[idx[j], 0]
            perimeter_points[j, 1] = S[idx[j], 1]
        results[i][0] = np.int32(np.mean(perimeter_points[:, 0]))
        results[i][1] = np.int32(np.mean(perimeter_points[:, 1]))
    return results



def get_hull_sizes(points: np.ndarray,
                   target: str = "perimeter",
                   pixels_per_mm: Optional[float] = None):
    """
    Convex hull perimeter or area per frame, with validation and optional conversion to mm.

    Calls :func:`jitted_hull` after checking array shape/dtype; if ``pixels_per_mm`` is given, divides result to get mm.

    :param points: (n_frames, n_body_parts, 2) non-negative numeric [x, y] in pixels.
    :param target: ``'perimeter'`` or ``'area'``.
    :param pixels_per_mm: If set, result is divided by this (output in mm).
    :return: (n_frames,) float32; hull size per frame (pixels or mm); NaN where hull fails.

    :example:
    >>> points = np.random.randint(0, 500, size=(100, 7, 2)).astype(np.float32)
    >>> get_hull_sizes(points, target='perimeter')
    >>> get_hull_sizes(points, target='area', pixels_per_mm=2.5)
    """
    if pixels_per_mm is not None:
        check_float(name=f'{get_hull_sizes.__name__} pixels_per_mm', value=pixels_per_mm, min_value=1e-16, raise_error=True)
    check_valid_array(data=points, source=f'{get_hull_sizes.__name__} points', accepted_ndims=(3,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_value=0.0, raise_error=True)
    check_str(name=f'{get_hull_sizes.__name__} target', options=('perimeter', 'area'), value=target, allow_blank=False, raise_error=True)
    size = jitted_hull(points=points.astype(np.float32), target=target).astype(np.float32)

    return size if pixels_per_mm is None else size / pixels_per_mm

# points = np.random.randint(1, 5, size=(1, 10, 2)).astype(np.float32)
# points[0][1] = np.nan
# results = jitted_hull(points, target='area')
# print(results)


# points = np.random.randint(1, 10, size=(50, 5, 2)).astype(np.float32)
# results = jitted_centroid(points)
#
#
# results = np.full((points.shape[0]), np.nan)
