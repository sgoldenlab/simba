__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import math

import numpy as np
from numba import cuda

from simba.utils.read_write import read_df

THREADS_PER_BLOCK = 256
@cuda.jit
def _get_3pt_angle_kernel(x_dev, y_dev, z_dev, results):
    i = cuda.grid(1)

    if i >= x_dev.shape[0]:
        return
    if i < x_dev.shape[0]:
        x_x, x_y = x_dev[i][0], x_dev[i][1]
        y_x, y_y = y_dev[i][0], y_dev[i][1]
        z_x, z_y = z_dev[i][0], z_dev[i][1]
        D = math.degrees(math.atan2(z_y - y_y, z_x - y_x) - math.atan2(x_y - y_y, x_x - y_x))
        if D < 0:
            D += 360
        results[i] = D

def get_3pt_angle(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Computes the angle formed by three points in 2D space for each corresponding row in the input arrays using
    GPU. The points x, y, and z represent the coordinates of three points in space, and the angle is calculated
    at point `y` between the line segments `xy` and `yz`.

    .. image:: _static/img/get_3pt_angle_cuda.png
       :width: 500
       :align: center

    :param x:  A numpy array of shape (n, 2) representing the first point (e.g., nose) coordinates.
    :param y: A numpy array of shape (n, 2) representing the second point (e.g., center) coordinates, where the angle is computed.
    :param z: A numpy array of shape (n, 2) representing the second point (e.g., center) coordinates, where the angle is computed.
    :return: A numpy array of shape (n, 1) containing the calculated angles (in degrees) for each row.

    :example:
    >>> video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0514.mp4"
    >>> data_path = r"/mnt/c/troubleshooting/mitra/project_folder/csv/outlier_corrected_movement_location/501_MA142_Gi_CNO_0514 - test.csv"
    >>> df = read_df(file_path=data_path, file_type='csv')
    >>> y = df[['Center_x', 'Center_y']].values
    >>> x = df[['Nose_x', 'Nose_y']].values
    >>> z = df[['Tail_base_x', 'Tail_base_y']].values
    >>> angle_x = get_3pt_angle(x=x, y=y, z=z)
    """


    x = np.ascontiguousarray(x).astype(np.float32)
    y = np.ascontiguousarray(y).astype(np.float32)
    n, m = x.shape
    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    z_dev = cuda.to_device(z)
    results = cuda.device_array((n, m), dtype=np.int32)
    bpg = (n + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _get_3pt_angle_kernel[bpg, THREADS_PER_BLOCK](x_dev, y_dev, z_dev, results)
    results = results.copy_to_host()
    cuda.current_context().memory_manager.deallocations.clear()
    return results
