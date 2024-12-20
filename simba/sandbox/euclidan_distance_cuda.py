__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import math

import numpy as np
from numba import cuda

from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.read_write import read_df

THREADS_PER_BLOCK = 128

@cuda.jit
def _euclidean_distance_kernel(x_dev, y_dev, results):
    i = cuda.grid(1)
    if i < x_dev.shape[0]:
        p = (math.sqrt((x_dev[i][0] - y_dev[i][0]) ** 2 + (x_dev[i][1] - y_dev[i][1]) ** 2))
        results[i] = p

def get_euclidean_distance_cuda(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the Euclidean distance between two sets of points using CUDA for GPU acceleration.

    .. image:: _static/img/get_euclidean_distance_cuda.png
       :width: 500
       :align: center

    :param np.ndarray x: A 2D array of shape (n, m) representing n points in m-dimensional space. Each row corresponds to a point.
    :param np.ndarray y: A 2D array of shape (n, m) representing n points in m-dimensional space. Each row corresponds to a point.
    :return np.ndarray: A 1D array of shape (n,) where each element represents the Euclidean distance  between the corresponding points in `x` and `y`.

    :example:
    >>> video_path = r"/mnt/c/troubleshooting/mitra/project_folder/videos/501_MA142_Gi_CNO_0514.mp4"
    >>> data_path = r"/mnt/c/troubleshooting/mitra/project_folder/csv/outlier_corrected_movement_location/501_MA142_Gi_CNO_0514 - test.csv"
    >>> df = read_df(file_path=data_path, file_type='csv')[['Center_x', 'Center_y']]
    >>> shifted_df = FeatureExtractionMixin.create_shifted_df(df=df, periods=1)
    >>> x = shifted_df[['Center_x', 'Center_y']].values
    >>> y = shifted_df[['Center_x_shifted', 'Center_y_shifted']].values
    >>> get_euclidean_distance_cuda(x=x, y=y)
    """

    x = np.ascontiguousarray(x).astype(np.int32)
    y = np.ascontiguousarray(y).astype(np.int32)
    n, m = x.shape
    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    results = cuda.device_array((n, m), dtype=np.int32)
    bpg = (n + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _euclidean_distance_kernel[bpg, THREADS_PER_BLOCK](x_dev, y_dev, results)
    results = results.copy_to_host().astype(np.int32)
    return results
