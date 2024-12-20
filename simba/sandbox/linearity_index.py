import numpy as np

from simba.utils.checks import check_float, check_valid_array
from simba.utils.enums import Formats


def linearity_index(x: np.ndarray) -> float:

    """
    Calculates the straightness (linearity) index of a path.

    :param np.ndarray x: An (N, M) array representing the path, where N is the number of points and M is the number of spatial dimensions (e.g., 2 for 2D or 3 for 3D). Each row represents the coordinates of a point along the path.
    :return: The straightness index of the path, a value between 0 and 1, where 1 indicates a perfectly straight path.
    :rtype: float

    :example:
    >>> x = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    >>> linearity_index(x=x)
    >>> x = np.random.randint(0, 100, (100, 2))
    >>> linearity_index(x=x)
    """

    check_valid_array(data=x, source=f'{linearity_index.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=[2, ], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    straight_line_distance = np.linalg.norm(x[0] - x[-1])
    path_length  = np.sum(np.linalg.norm(np.diff(x, axis=0), axis=1))
    if path_length == 0:
        return 0.0
    else:
        return straight_line_distance / path_length


def sliding_linearity_index(x: np.ndarray,
                            window_size: float,
                            sample_rate: float) -> np.ndarray:

    """
    Calculates the Linearity Index (Path Straightness) over a sliding window for a path represented by an array of points.

    The Linearity Index measures how straight a path is by comparing the straight-line distance between the start and end points of each window to the total distance traveled along the path.

    :param np.ndarray x: An (N, M) array representing the path, where N is the number of points and M is the number of spatial dimensions (e.g., 2 for 2D or 3 for 3D). Each row represents the coordinates of a point along the path.
    :param float x: The size of the sliding window in seconds. This defines the time window over which the linearity index is calculated. The window size should be specified in seconds.
    :param float sample_rate: The sample rate in Hz (samples per second), which is used to convert the window size from seconds to frames.
    :return: A 1D array of length N, where each element represents the linearity index of the path within a sliding  window. The value is a ratio between the straight-line distance and the actual path length for each window. Values range from 0 to 1, with 1 indicating a perfectly straight path.
    :rtype: np.ndarray
    """

    frame_step = int(max(1.0, window_size * sample_rate))
    results = np.full(x.shape[0], fill_value=0.0, dtype=np.float32)
    for r in range(frame_step, x.shape[0]):
        l =  r - frame_step
        sample_x = x[l:r, :]
        straight_line_distance = np.linalg.norm(sample_x[0] - sample_x[-1])
        path_length = np.sum(np.linalg.norm(np.diff(sample_x, axis=0), axis=1))
        if path_length == 0:
            results[r] = 0.0
        else:
            results[r] = straight_line_distance / path_length
    return results


x = np.random.randint(0, 100, (100, 2))
sliding_linearity_index(x=x, window_size=1, sample_rate=10)