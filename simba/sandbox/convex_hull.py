__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import numpy as np
from numba import cuda, njit

THREADS_PER_BLOCK = 128

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


    .. image:: _static/img/get_convex_hull_cuda.png
       :width: 300
       :align: center

    .. note::
       `Modified from Jacob Hultman <https://github.com/jhultman/rotating-calipers>`_

    :param pts: A 3D numpy array of shape (M, N, 2) where:
                - M is the number of frames.
                - N is the number of points (body-parts) in each frame.
                - The last dimension (2) represents the x and y coordinates of each point.

    :return: An upated 3D numpy array of shape (M, N, 2) consisting of the points in the hull.


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
