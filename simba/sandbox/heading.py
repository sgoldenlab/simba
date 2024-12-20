import numpy as np
from numba import njit

@njit('(int32[:,:], float64, float64)')
def sliding_bearing(x: np.ndarray, lag: float, fps: float) -> np.ndarray:
    """
    Calculates the sliding bearing (direction) of movement in degrees for a sequence of 2D points representing a single body-part.

    .. note::
       To calculate frame-by-frame bearing, pass fps == 1 and lag == 1.

    .. image:: _static/img/sliding_bearing.png
       :width: 600
       :align: center

    :param np.ndarray x: An array of shape (n, 2) representing the time-series sequence of 2D points.
    :param float lag: The lag time (in seconds) used for calculating the sliding bearing. E.g., if 1, then bearing will be calculated using coordinates in the current frame vs the frame 1s previously.
    :param float fps: The sample rate (frames per second) of the sequence.
    :return np.ndarray: An array containing the sliding bearings (in degrees) for each point in the sequence.

    :example:
    >>> x = np.array([[10, 10], [20, 10]])
    >>> sliding_bearing(x=x, lag=1, fps=1)
    >>> [-1. 90.]
    """

    results = np.full((x.shape[0]), -1.0)
    lag = int(lag * fps)
    for i in range(lag, x.shape[0]):
        x1, y1 = x[i-lag, 0], x[i-lag, 1]
        x2, y2 = x[i, 1], x[i, 1]
        degree = 90 - np.degrees(np.arctan2(y1 - y2, x2 - x1))
        results[i] = degree + 360 if degree < 0 else degree
    return results