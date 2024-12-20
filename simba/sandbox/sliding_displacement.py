import numpy as np
from numba import njit

@njit('(int32[:,:], float64[:], float64, float64)')
def sliding_displacement(x: np.ndarray,
                         time_windows: np.ndarray,
                         fps: float,
                         px_per_mm: float) -> np.ndarray:

    """
    Calculate sliding Euclidean displacement of a body-part point over time windows.

    .. image:: _static/img/sliding_displacement.png
       :width: 600
       :align: center

    :param np.ndarray x: An array of shape (n, 2) representing the time-series sequence of 2D points.
    :param np.ndarray time_windows: Array of time windows (in seconds).
    :param float fps: The sample rate (frames per second) of the sequence.
    :param float px_per_mm: Pixels per millimeter conversion factor.
    :return np.ndarray: 1D array containing the calculated displacements.

    :example:
    >>> x = np.random.randint(0, 50, (100, 2)).astype(np.int32)
    >>> sliding_displacement(x=x, time_windows=np.array([1.0]), fps=1.0, px_per_mm=1.0)
    """

    results = np.full((x.shape[0], time_windows.shape[0]), -1.0)
    for i in range(time_windows.shape[0]):
        w = int(time_windows[i] * fps)
        for j in range(w, x.shape[0]):
            c, s = x[j], x[j-w]
            results[j, i] = (np.sqrt((s[0] - c[0]) ** 2 + (s[1] - c[1]) ** 2)) / px_per_mm
    return results.astype(np.float32)






