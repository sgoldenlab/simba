__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import math
from typing import Optional

import numpy as np
from numba import cuda, njit

from simba.utils.checks import check_float, check_valid_array
from simba.utils.enums import Formats

try:
    import cupy as cp

except ModuleNotFoundError:
    import numpy as cp

THREADS_PER_BLOCK = 1024

@cuda.jit
def _cuda_is_inside_rectangle(x, y, r):
    i = cuda.grid(1)
    if i > r.shape[0]:
        return
    else:
        if (x[i][0] >= y[0][0]) and (x[i][0] <= y[1][0]):
            if (x[i][1] >= y[0][1]) and (x[i][1] <= y[1][1]):
                r[i] = 1

def is_inside_rectangle(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Determines whether points in array `x` are inside the rectangle defined by the top left and bottom right vertices in array `y`.
    |:heart_eyes:|

    .. image:: _static/img/simba.data_processors.cuda.geometry.is_inside_rectangle.webp
       :width: 450
       :align: center

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/is_inside_rectangle.csv
       :widths: 10, 45, 45
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       For numba CPU function see :func:`~simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_inside_rectangle_roi`

    :param np.ndarray x: 2d numeric np.ndarray size (N, 2).
    :param np.ndarray y: 2d numeric np.ndarray size (2, 2) (top left[x, y], bottom right[x, y])
    :return: 2d numeric boolean (N, 1) with 1s representing the point being inside the rectangle and 0 if the point is outside the rectangle.
    :rtype: np.ndarray
    """

    x = np.ascontiguousarray(x).astype(np.int32)
    y = np.ascontiguousarray(y).astype(np.int32)
    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    results = cuda.device_array((x.shape[0]), dtype=np.int8)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _cuda_is_inside_rectangle[bpg, THREADS_PER_BLOCK](x_dev, y_dev, results)
    results = results.copy_to_host()
    return results

@cuda.jit
def _cuda_is_inside_circle(x, y, r, results):
    i = cuda.grid(1)
    if i > results.shape[0]:
        return
    else:
        p = (math.sqrt((x[i][0] - y[0][0]) ** 2 + (x[i][1] - y[0][1]) ** 2))
        if p <= r[0]:
            results[i] = 1
def is_inside_circle(x: np.ndarray, y: np.ndarray, r: float) -> np.ndarray:
    """
    Determines whether points in array `x` are inside the circle with center ``y`` and radius ``r``

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/is_inside_circle.csv
       :widths: 10, 90
       :align: center
       :class: simba-table
       :header-rows: 1

    :param np.ndarray x: 2d numeric np.ndarray size (N, 2).
    :param np.ndarray y: 2d numeric np.ndarray size (1, 2) representing the center of the circle.
    :param float r: The radius of the circle.
    :return: 2d numeric boolean (N, 1) with 1s representing the point being inside the circle and 0 if the point is outside the rectangle.
    :rtype: 1d np.ndarray vector.
    """

    x = np.ascontiguousarray(x).astype(np.int32)
    y = np.ascontiguousarray(y).astype(np.int32)
    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    r = np.array([r]).astype(np.float32)
    r_dev = cuda.to_device(r)
    results = cuda.device_array((x.shape[0]), dtype=np.int8)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    del x, y
    _cuda_is_inside_circle[bpg, THREADS_PER_BLOCK](x_dev, y_dev, r_dev, results)
    results = results.copy_to_host()
    return results


@cuda.jit
def _cuda_is_inside_polygon(x, p, r):
    i = cuda.grid(1)
    if i > r.shape[0]:
        return
    else:
        x, y, n = x[i][0], x[i][1], len(p)
        p2x, p2y, xints, inside = 0.0, 0.0, 0.0, False
        p1x, p1y = p[0]
        for j in range(n + 1):
            p2x, p2y = p[j % n]
            if (
                    (y > min(p1y, p2y))
                    and (y <= max(p1y, p2y))
                    and (x <= max(p1x, p2x))
            ):
                if p1y != p2y:
                    xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                if p1x == p2x or x <= xints:
                    inside = not inside
            p1x, p1y = p2x, p2y
        if inside:
            r[i] = 1


def is_inside_polygon(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Determines whether points in array `x` are inside the polygon defined by the vertices in array `y`.

    This function uses GPU acceleration to perform the point-in-polygon test. The points in `x` are tested against
    the polygon defined by the vertices in `y`. The result is an array where each element indicates whether
    the corresponding point is inside the polygon.

    .. image:: _static/img/simba.data_processors.cuda.geometry.is_inside_polygon.webp
       :width: 450
       :align: center

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/is_inside_polygon.csv
       :widths: 10, 45, 45
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       For jitted CPU function see :func:`~simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_inside_polygon_roi`

    :param np.ndarray x: An array of shape (N, 2) where each row represents a point in 2D space. The points are checked against the polygon.
    :param np.ndarray y: An array of shape (M, 2) where each row represents a vertex of the polygon in 2D space.
    :return: An array of shape (N,) where each element is 1 if the corresponding point in `x` is inside the polygon defined by `y`, and 0 otherwise.
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(0, 200, (i, 2)).astype(np.int8)
    >>> y = np.random.randint(0, 200, (4, 2)).astype(np.int8)
    >>> results = is_inside_polygon(x=x, y=y)
    >>> print(results)
    >>> [1 0 1 0 1 1 0 0 1 0]
    """

    x = np.ascontiguousarray(x).astype(np.int32)
    y = np.ascontiguousarray(y).astype(np.int32)
    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    results = cuda.device_array((x.shape[0]), dtype=np.int8)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _cuda_is_inside_polygon[bpg, THREADS_PER_BLOCK](x_dev, y_dev, results)
    results = results.copy_to_host()
    return results


@cuda.jit(device=True)
def _cross_test(x, y, x1, y1, x2, y2):
    """Cross product test for determining whether left of line."""
    cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    return cross < 0

@cuda.jit
def _convex_hull_kernel(pts: np.ndarray, results: np.ndarray) -> np.ndarray:
    """
    CUDA kernel for the Jarvis March algorithm.

    .. note::
       `Modified from Jacob Hultman <https://github.com/jhultman/rotating-calipers>`_

    :param pts: M x N x 2 array where M is the number of frames, N is the number of body-parts, and 2 representing the x and y coordinates of the body-parts.
    :param results: M x N array where M is the number of frames, and N is the indexes of the body-parts belonging to the hull. If -1, the body-part does not belong to the hull.
    """
    row = cuda.grid(1)
    if row >= pts.shape[0]:
        return

    point_on_hull = 0
    min_x = pts[row, 0, 0]
    for j in range(pts.shape[1]):
        x = pts[row, j, 0]
        if x < min_x:
            min_x = x
            point_on_hull = j
    startpoint = point_on_hull
    count = 0
    while True:
        results[row, count] = point_on_hull
        count += 1
        endpoint = 0
        for j in range(pts.shape[1]):
            if endpoint == point_on_hull:
                endpoint = j
            elif _cross_test(
                pts[row, j, 0],
                pts[row, j, 1],
                pts[row, point_on_hull, 0],
                pts[row, point_on_hull, 1],
                pts[row, endpoint, 0],
                pts[row, endpoint, 1],
            ):
                endpoint = j
        point_on_hull = endpoint
        if endpoint == startpoint:
            break
    for j in range(count, pts.shape[1], 1):
        results[row, j] = -1


#@jit(nopython=True)
@njit("(int32[:, :, :], int32[:, :])")
def _slice_hull_idx(points: np.ndarray, point_idx: np.ndarray):
    results = np.zeros_like(points)
    for i in range(point_idx.shape[0]):
        results[i] = points[i][point_idx[i]]
    return results


def get_convex_hull(pts: np.ndarray) -> np.ndarray:
    """
    Compute the convex hull for each set of 2D points in parallel using CUDA and the Jarvis March algorithm.
    This function processes a batch of 2D point sets (frames) and computes the convex hull for each set. The convex hull of a set of points is the smallest convex polygon that contains all the points.

    The function uses a variant of the Gift Wrapping algorithm (Jarvis March) to compute the convex hull. It finds the leftmost point, then iteratively determines the next point on the hull by checking the orientation of the remaining points. The results are stored in the `results` array, where each row corresponds to a frame and contains the indices of the points forming the convex hull. Points not on the hull are marked with `-1`.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/get_convex_hull.csv
       :widths: 10, 90
       :align: center
       :class: simba-table
       :header-rows: 1

    .. note::
       `Modified from Jacob Hultman <https://github.com/jhultman/rotating-calipers>`_

    .. seealso::
       :func:`~simba.feature_extractors.perimeter_jit.jitted_hull`.
       :func:`~simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.convex_hull_calculator_mp`.

    .. video:: _static/img/get_convex_hull_cuda.mp4
       :width: 800
       :autoplay:
       :loop:

    :param pts: A 3D numpy array of shape (M, N, 2) where: - M is the number of frames. - N is the number of points (body-parts) in each frame. - The last dimension (2) represents the x and y coordinates of each point.
    :return: An upated 3D numpy array of shape (M, N, 2) consisting of the points in the hull.
    :rtype: np.ndarray


    :example:
    >>> video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0514.mp4"
    >>> data_path = r"/mnt/c/troubleshooting/mitra/project_folder/csv/outlier_corrected_movement_location/501_MA142_Gi_CNO_0514 - test.csv"
    >>> df = read_df(file_path=data_path, file_type='csv')
    >>> frame_data = df.values.reshape(len(df), -1, 2)
    >>> x = get_convex_hull(frame_data)
    """

    pts = np.ascontiguousarray(pts).astype(np.int32)
    n, m, _ = pts.shape
    bpg = (n + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    pts_dev = cuda.to_device(pts)
    results = cuda.device_array((n, m), dtype=np.int32)
    _convex_hull_kernel[bpg, THREADS_PER_BLOCK](pts_dev, results)
    hull = results.copy_to_host().astype(np.int32)
    hull = _slice_hull_idx(pts, hull)
    return hull


def poly_area(data: np.ndarray,
              pixels_per_mm: Optional[float] = 1.0,
              batch_size: Optional[int] = int(0.5e+7)) -> np.ndarray:

    """
    Compute the area of a polygon using GPU acceleration.

    This function calculates the area of polygons defined by sets of points in a 3D array.
    Each 2D slice along the first dimension represents a polygon, with each row corresponding
    to a point in the polygon and each column representing the x and y coordinates.

    The computation is done in batches to handle large datasets efficiently.

    .. seealso::
       :func:`~simba.feature_extractors.perimeter_jit.jitted_hull`.


    .. image:: _static/img/simba.data_processors.cuda.geometry.poly_area_cuda.webp
       :width: 450
       :align: center


    :param data: A 3D numpy array of shape (N, M, 2), where N is the number of polygons, M is the number of points per polygon, and 2 represents the x and y coordinates.
    :param pixels_per_mm: Optional scaling factor to convert the area from pixels squared  to square millimeters. Default is 1.0.
    :param batch_size: Optional batch size for processing the data in chunks to fit in memory. Default is 0.5e+7.
    :return: A 1D numpy array of shape (N,) containing the computed area of each polygon in square millimeters.
    :rtype: np.ndarray
    """

    check_valid_array(data=data, source=f'{poly_area} data', accepted_ndims=(3,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{poly_area} pixels_per_mm', min_value=10e-16, value=pixels_per_mm)
    results = cp.full((data.shape[0]), fill_value=cp.nan, dtype=cp.int32)
    for l in range(0, data.shape[0], batch_size):
        r = l + batch_size
        x = cp.asarray(data[l:r, :, 0])
        y = cp.asarray(data[l:r, :, 1])
        x_r = cp.roll(x, shift=1, axis=1)
        y_r = cp.roll(y, shift=1, axis=1)
        dot_xy_roll_y = cp.sum(x * y_r, axis=1)
        dot_y_roll_x = cp.sum(y * x_r, axis=1)
        results[l:r] = (0.5 * cp.abs(dot_xy_roll_y - dot_y_roll_x)) / pixels_per_mm

    return results.get()

def find_midpoints(x: np.ndarray,
                   y: np.ndarray,
                   percentile: Optional[float] = 0.5,
                   batch_size: Optional[int] = int(1.5e+7)) -> np.ndarray:

    """
    Calculate the midpoints between corresponding points in arrays `x` and `y`
    based on a given percentile using GPU acceleration.

    For example, calculate the midpoint between the animal ears (to get presumed ``nape``) or lateral sides (to get presumed center of mass),
    or nose and left ear (to get left eye) etc.

    This function computes the midpoints between each pair of points (x[i], y[i])
    from the input arrays `x` and `y`. The midpoint is calculated by taking a
    weighted sum of the differences along each axis, where the weight is determined
    by the specified percentile. The computation is performed in batches to handle
    large datasets efficiently.

    .. seealso:
       For CPU function see :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.find_midpoints`.

    .. image:: _static/img/find_midpoints.webp
       :width: 600
       :align: center

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/find_midpoints.csv
       :widths: 10, 45, 45
       :align: center
       :class: simba-table
       :header-rows: 1

    :param np.ndarray x: An array of shape (n, 2) representing the x-coordinates of n points.
    :param np.ndarray y: An array of shape (n, 2) representing the y-coordinates of n points.
    :param percentile: A float value between 0 and 1 indicating the percentile to use when calculating the midpoints. The default value is 0.5, which corresponds to the middle.
    :param Optional[int] batch_size: An integer specifying the batch size for processing the input arrays. Larger batch sizes will use more memory but may be faster. The default value is 15 million (1.5e+7).
    :return: An array of shape (n, 2) containing the calculated midpoints for each pair of corresponding points in `x` and `y`.
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(0, 100, (100, 2)).astype(np.int8)
    >>> y = np.random.randint(0, 100, (100, 2)).astype(np.int8)
    >>> p = find_midpoints(x=x, y=y)
    """

    n = x.shape[0]
    x = cp.asarray(x)
    y = cp.asarray(y)
    results = cp.full((n, 2), -1, dtype=cp.int32)
    for left in range(0, n, batch_size):
        right = int(min(left + batch_size, n))
        x_batch = x[left:right]
        y_batch = y[left:right]
        axis_0_diff = cp.abs(x_batch[:, 0] - y_batch[:, 0])
        axis_1_diff = cp.abs(x_batch[:, 1] - y_batch[:, 1])
        x_dist_percentile = (axis_0_diff * percentile).astype(cp.int32)
        y_dist_percentile = (axis_1_diff * percentile).astype(cp.int32)
        new_x = cp.minimum(x_batch[:, 0], y_batch[:, 0]) + x_dist_percentile
        new_y = cp.minimum(x_batch[:, 1], y_batch[:, 1]) + y_dist_percentile
        results[left:right, 0] = new_x
        results[left:right, 1] = new_y

    return results


@cuda.jit()
def _directionality_to_static_targets_kernel(left_ear, right_ear, nose, target, results):
    i = cuda.grid(1)
    if i > left_ear.shape[0]:
        return
    else:
        LE, RE = left_ear[i], right_ear[i]
        N, Tx, Ty = nose[i], target[0], target[1]

        Px = abs(LE[0] - Tx)
        Py = abs(LE[1] - Ty)
        Qx = abs(RE[0] - Tx)
        Qy = abs(RE[1] - Ty)
        Nx = abs(N[0] - Tx)
        Ny = abs(N[1] - Ty)
        Ph = math.sqrt(Px * Px + Py * Py)
        Qh = math.sqrt(Qx * Qx + Qy * Qy)
        Nh = math.sqrt(Nx * Nx + Ny * Ny)
        if Nh < Ph and Nh < Qh and Qh < Ph:
            results[i][0] = 0
            results[i][1] = RE[0]
            results[i][2] = RE[1]
            results[i][3] = 1
        elif Nh < Ph and Nh < Qh and Ph < Qh:
            results[i][0] = 1
            results[i][1] = LE[0]
            results[i][2] = LE[1]
            results[i][3] = 1
        else:
            results[i][0] = 2
            results[i][1] = -1
            results[i][2] = -1
            results[i][3] = 0


def directionality_to_static_targets(left_ear: np.ndarray,
                                     right_ear: np.ndarray,
                                     nose: np.ndarray,
                                     target: np.ndarray) -> np.ndarray:
    """
    GPU helper to calculate if an animal is directing towards a static location (e.g., ROI centroid), given the target location and the left ear, right ear, and nose coordinates of the observer.

    .. note::
       Input left ear, right ear, and nose coordinates of the observer is returned by :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.check_directionality_viable`

    .. seealso::
        For numba based CPU method, see :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.jitted_line_crosses_to_static_targets`
        If the target is moving, consider :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.jitted_line_crosses_to_nonstatic_targets`.

    .. image:: _static/img/directing_static_targets.png
       :width: 400
       :align: center

    .. video:: _static/img/ts_example.webm
       :width: 800
       :autoplay:
       :align: center
       :loop:

    ..  youtube:: vqdYUS3bM68
       :width: 640
       :height: 480
       :align: center

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/directionality_to_static_targets.csv
       :widths: 10, 45, 45
       :align: center
       :class: simba-table
       :header-rows: 1

    :param np.ndarray left_ear: 2D array of size len(frames) x 2 with the coordinates of the observer animals left ear
    :param np.ndarray right_ear: 2D array of size len(frames) x 2 with the coordinates of the observer animals right ear
    :param np.ndarray nose: 2D array of size len(frames) x 2 with the coordinates of the observer animals nose
    :param np.ndarray target: 1D array of with x,y of target location
    :return: 2D array of size len(frames) x 4. First column represent the side of the observer that the target is in view. 0 = Left side, 1 = Right side, 2 = Not in view. Second and third column represent the x and y location of the observer animals ``eye`` (half-way between the ear and the nose). Fourth column represent if target is view (bool).
    :rtype: np.ndarray

    :example:
    >>> left_ear = np.random.randint(0, 500, (100, 2))
    >>> right_ear = np.random.randint(0, 500, (100, 2))
    >>> nose = np.random.randint(0, 500, (100, 2))
    >>> target = np.random.randint(0, 500, (2))
    >>> directionality_to_static_targets(left_ear=left_ear, right_ear=right_ear, nose=nose, target=target)

    """

    left_ear = np.ascontiguousarray(left_ear).astype(np.int32)
    right_ear = np.ascontiguousarray(right_ear).astype(np.int32)
    nose = np.ascontiguousarray(nose).astype(np.int32)
    target = np.ascontiguousarray(target).astype(np.int32)

    left_ear_dev = cuda.to_device(left_ear)
    right_ear_dev = cuda.to_device(right_ear)
    nose_dev = cuda.to_device(nose)
    target_dev = cuda.to_device(target)
    results = cuda.device_array((left_ear.shape[0], 4), dtype=np.int32)
    bpg = (left_ear.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _directionality_to_static_targets_kernel[bpg, THREADS_PER_BLOCK](left_ear_dev, right_ear_dev, nose_dev, target_dev, results)

    results = results.copy_to_host()
    return results


@cuda.jit()
def _directionality_to_nonstatic_targets_kernel(left_ear, right_ear, nose, target, results):
    i = cuda.grid(1)
    if i > left_ear.shape[0]:
        return
    else:
        LE, RE = left_ear[i], right_ear[i]
        N, T = nose[i], target[i]

        Px = abs(LE[0] - T[0])
        Py = abs(LE[1] - T[1])
        Qx = abs(RE[0] - T[0])
        Qy = abs(RE[1] - T[1])
        Nx = abs(N[0] - T[0])
        Ny = abs(N[1] - T[1])
        Ph = math.sqrt(Px * Px + Py * Py)
        Qh = math.sqrt(Qx * Qx + Qy * Qy)
        Nh = math.sqrt(Nx * Nx + Ny * Ny)
        if Nh < Ph and Nh < Qh and Qh < Ph:
            results[i][0] = 0
            results[i][1] = RE[0]
            results[i][2] = RE[1]
            results[i][3] = 1
        elif Nh < Ph and Nh < Qh and Ph < Qh:
            results[i][0] = 1
            results[i][1] = LE[0]
            results[i][2] = LE[1]
            results[i][3] = 1
        else:
            results[i][0] = 2
            results[i][1] = -1
            results[i][2] = -1
            results[i][3] = 0


def directionality_to_nonstatic_target(left_ear: np.ndarray,
                                       right_ear: np.ndarray,
                                       nose: np.ndarray,
                                       target: np.ndarray) -> np.ndarray:

    """
    GPU method to calculate if an animal is directing towards a moving point location given the target location and the left ear, right ear, and nose coordinates of the observer.


    .. image:: _static/img/directing_moving_targets.png
       :width: 400
       :align: center

    .. seealso::
       Input left ear, right ear, and nose coordinates of the observer is returned by :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.check_directionality_viable`

       For non-GPU numba based CPU method, see :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.jitted_line_crosses_to_nonstatic_targets`

       If the target is static, consider :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.jitted_line_crosses_to_static_targets`
       or :func:`simba.data_processors.cuda.geometry.directionality_to_static_targets` for GPU acceleration.

    :param np.ndarray left_ear: 2D array of size len(frames) x 2 with the coordinates of the observer animals left ear
    :param np.ndarray right_ear: 2D array of size len(frames) x 2 with the coordinates of the observer animals right ear
    :param np.ndarray nose: 2D array of size len(frames) x 2 with the coordinates of the observer animals nose
    :param np.ndarray target: 1D array of with x,y of target location
    :return: 2D array of size len(frames) x 4. First column represent the side of the observer that the target is in view. 0 = Left side, 1 = Right side, 2 = Not in view. Second and third column represent the x and y location of the observer animals ``eye`` (half-way between the ear and the nose). Fourth column represent if target is view (bool).
    :rtype: np.ndarray

    :example:
    >>> left_ear = np.random.randint(0, 500, (100, 2))
    >>> right_ear = np.random.randint(0, 500, (100, 2))
    >>> nose = np.random.randint(0, 500, (100, 2))
    >>> target = np.random.randint(0, 500, (100, 2))
    >>> directionality_to_static_targets(left_ear=left_ear, right_ear=right_ear, nose=nose, target=target)
    """

    left_ear = np.ascontiguousarray(left_ear).astype(np.int32)
    right_ear = np.ascontiguousarray(right_ear).astype(np.int32)
    nose = np.ascontiguousarray(nose).astype(np.int32)
    target = np.ascontiguousarray(target).astype(np.int32)

    left_ear_dev = cuda.to_device(left_ear)
    right_ear_dev = cuda.to_device(right_ear)
    nose_dev = cuda.to_device(nose)
    target_dev = cuda.to_device(target)
    results = cuda.device_array((left_ear.shape[0], 4), dtype=np.int32)
    bpg = (left_ear.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _directionality_to_nonstatic_targets_kernel[bpg, THREADS_PER_BLOCK](left_ear_dev, right_ear_dev, nose_dev, target_dev, results)

    results = results.copy_to_host()
    return results






