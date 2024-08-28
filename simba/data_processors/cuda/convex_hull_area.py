from typing import Optional

import cupy as cp
import numpy as np

from simba.utils.checks import check_float, check_valid_array
from simba.utils.enums import Formats


def poly_area(data: np.ndarray,
              pixels_per_mm: Optional[float] = 1.0,
              batch_size: Optional[int] = int(0.5e+7)) -> np.ndarray:

    """
    Compute the area of a polygon using GPU acceleration.

    This function calculates the area of polygons defined by sets of points in a 3D array.
    Each 2D slice along the first dimension represents a polygon, with each row corresponding
    to a point in the polygon and each column representing the x and y coordinates.

    The computation is done in batches to handle large datasets efficiently.

    :param data: A 3D numpy array of shape (N, M, 2), where N is the number of polygons, M is the number of points per polygon, and 2 represents the x and y coordinates.
    :param pixels_per_mm: Optional scaling factor to convert the area from pixels squared  to square millimeters. Default is 1.0.
    :param batch_size: Optional batch size for processing the data in chunks to fit in memory. Default is 0.5e+7.
    :return: A 1D numpy array of shape (N,) containing the computed area of each polygon in square millimeters.
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
        results[l:r]  = (0.5 * cp.abs(dot_xy_roll_y - dot_y_roll_x)) / pixels_per_mm

    return results.get()
