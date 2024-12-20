import numpy as np

from simba.utils.checks import check_str, check_valid_array

try:
    from typing import Literal
except:
    from typing_extensions import Literal

    from simba.utils.enums import Formats


def path_curvature(x: np.ndarray, agg_type: Literal['mean', 'median', 'max'] = 'mean') -> float:
    """
    Calculate aggregate curvature of a 2D path given an array of points.

    :param x: A 2D numpy array of shape (N, 2), where N is the number of points and each row is (x, y).
    :param Literal['mean', 'median', 'max'] agg_type: The type of summary statistic to return. Options are 'mean', 'median', or 'max'.
    :return: A single float value representing the path curvature based on the specified summary type.
    :rtype: float

    :example:
    >>> x = np.array([[0, 0], [1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]])
    >>> low = path_curvature(x)
    >>> x = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]])
    >>> high = path_curvature(x)
    """
    check_valid_array(data=x, source=f'{path_curvature.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=[2, ], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_str(name=f'{path_curvature.__name__} agg_type', value=agg_type, options=('mean', 'median', 'max'))
    dx, dy = np.diff(x[:, 0]),np.diff(x[:, 1])
    x_prime, y_prime = dx[:-1], dy[:-1]
    x_double_prime, y_double_prime = dx[1:] - dx[:-1], dy[1:] - dy[:-1]
    curvature = np.abs(x_prime * y_double_prime - y_prime * x_double_prime) / (x_prime ** 2 + y_prime ** 2) ** (3 / 2)
    if agg_type == 'mean':
        return np.float32(np.nanmean(curvature))
    elif agg_type == 'median':
        return np.float32(np.nanmedian(curvature))
    else:
        return np.float32(np.nanmax(curvature))


def sliding_path_curvature(x: np.ndarray,
                           agg_type: Literal['mean', 'median', 'max'],
                           window_size: float,
                           sample_rate: float) -> np.ndarray:
    """
    Computes the curvature of a path over sliding windows along the path points, providing a measure of the path’s bending
    or turning within each window.

    This function calculates curvature for each window segment by evaluating directional changes. It provides the option to
    aggregate curvature values within each window using the mean, median, or maximum, depending on the desired level of
    sensitivity to bends and turns. A higher curvature value indicates a sharper or more frequent directional change within
    the window, while a lower curvature suggests a straighter or smoother path.

    :param x: A 2D array of shape (N, 2) representing the path, where N is the number of points, and each point has two spatial coordinates (e.g., x and y for 2D space).
    :param Literal['mean', 'median', 'max'] agg_type: Type of aggregation for the curvature within each window.
    :param float window_size: Duration of the window in seconds, used to define the size of each segment over which curvature  is calculated.
    :param float sample_rate: The rate at which path points were sampled (in points per second), used to convert the window size from seconds to frames
    :return: An array of shape (N,) containing the computed curvature values for each window position along the path. Each element represents the aggregated curvature within a specific window, with `NaN` values for frames where the window does not fit.
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(0, 500, (91, 2))
    >>> sliding_path_curvature(x=x, agg_type='mean', window_size=1, sample_rate=30)
    """

    frame_step = int(max(1.0, window_size * sample_rate))
    results = np.full(shape=(x.shape[0]), fill_value=np.nan, dtype=np.float32)
    for r in range(frame_step, x.shape[0]+1):
        l = r - frame_step
        sample_x = x[l:r]
        dx, dy = np.diff(sample_x[:, 0]), np.diff(sample_x[:, 1])
        x_prime, y_prime = dx[:-1], dy[:-1]
        x_double_prime, y_double_prime = dx[1:] - dx[:-1], dy[1:] - dy[:-1]
        curvature = np.abs(x_prime * y_double_prime - y_prime * x_double_prime) / (x_prime ** 2 + y_prime ** 2) ** (3 / 2)
        if agg_type == 'mean':
            results[r-1] = np.float32(np.nanmean(curvature))
        elif agg_type == 'median':
            results[r-1] = np.float32(np.nanmedian(curvature))
        else:
            results[r-1] = np.float32(np.nanmax(curvature))

    return results


x = np.random.randint(0, 500, (91, 2))
sliding_path_curvature(x=x, agg_type='mean', window_size=1, sample_rate=30)




# x = np.array([[0, 0], [1, 0.1], [2, 0.2], [3, 0.3], [4, 0.4]])
# #x = np.random.randint(0, 500, (100, 2))
# low = path_curvature(x)
# x = np.array([[0, 0], [1, 1], [2, 0], [3, 1], [4, 0]])
# high = path_curvature(x)
# print(low, high)