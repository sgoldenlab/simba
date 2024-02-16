from typing import Optional

import numpy as np
from numba import float32, int64, njit, prange, types
from numba.typed import Dict, List
from typing_extensions import Literal


# @njit("(float32[:], int64, int64,)", fastmath=True)
@njit(
    [
        "(float32[:, :], float64, int64, float64, types.unicode_type)",
        '(float32[:, :], float64, int64, float64, types.misc.Omitted("mm"))',
        "(float32[:, :], float64, int64, types.misc.Omitted(1), types.unicode_type)",
        '(float32[:, :], float64, int64, types.misc.Omitted(1), types.misc.Omitted("mm"))',
    ]
)
def acceleration(
    data: np.ndarray,
    pixels_per_mm: float,
    fps: int,
    time_window: float = 1,
    unit: Literal["mm", "cm", "dm", "m"] = "mm",
) -> np.ndarray:
    """
    Compute acceleration.
    Computes acceleration from a sequence of body-part coordinates over time. It calculates the difference in velocity between consecutive frames and provides an array of accelerations.
    The computation is based on the formula:
    .. math::
       \\text{{Acceleration}}(t) = \\frac{{\\text{{Norm}}(\\text{{Shift}}(\\text{{data}}[t], t, t-1) - \\text{{data}}[t])}}{{\\text{{pixels\\_per\\_mm}}}}
    where :math:`\\text{{Norm}}` calculates the Euclidean norm, :math:`\\text{{Shift}}(\\text{{array}}, t, t-1)` shifts the array by :math:`t-1` frames, and :math:`\\text{{pixels\\_per\\_mm}}` is the conversion factor from pixels to millimeters.
    .. note::
       By default, acceleration is calculated as change in velocity at millimeters/s. To change the denomitator, modify the ``time_window`` argument. To change the nominator, modify the ``unit`` argument (accepted ``mm``, cm``, ``dm``, ``mm``)
    .. image:: _static/img/acceleration.png
       :width: 700
       :align: center
    :param np.ndarray data: 2D array of size len(frames) x 2 with body-part coordinates.
    :param float pixels_per_mm: Pixels per millimeter of the recorded video.
    :param int fps: Frames per second (FPS) of the recorded video.
    :param float time_window: Rolling time window in seconds. Default is 1.0 representing 1 second.
    :param Literal['mm', 'cm', 'dm', 'm'] unit:  If acceleration should be presented as millimeter, centimeters, decimeter, or meter. Default millimeters.
    :return: Array of accelerations corresponding to each frame.
    :example:
    >>> data = np.array([1, 2, 3, 4, 5, 5, 5, 5, 5, 6]).astype(np.float32)
    >>> TimeseriesFeatureMixin().acceleration(data=data, pixels_per_mm=1.0, fps=2, time_window=1.0)
    >>> [ 0.,  0.,  0.,  0., -1., -1.,  0.,  0.,  1.,  1.]
    """
    results, velocity = np.full((data.shape[0]), 0.0), np.full((data.shape[0]), 0.0)
    size, pv = int(time_window * fps), None
    data_split = np.split(data, list(range(size, data.shape[0], size)))
    print(data_split)
    for i in range(len(data_split)):
        wS = int(size * i)
        wE = int(wS + size)
        v = np.diff(np.ascontiguousarray(data_split[i]))[0] / pixels_per_mm
        if unit == "cm":
            v = v / 10
        elif unit == "dm":
            v = v / 100
        elif unit == "m":
            v = v / 1000
        if i == 0:
            results[wS:wE] = 0
        else:
            results[wS:wE] = v - pv
        pv = v
    return results


data = np.array([[1, 2], [1, 1], [2, 1], [1, 1]]).astype(np.float32)
acceleration(data=data, pixels_per_mm=1.0, fps=2, time_window=1.0)
