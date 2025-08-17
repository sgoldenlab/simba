__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"


import math
from itertools import combinations
from typing import Optional, Tuple

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
from numba import cuda
from scipy.spatial import ConvexHull

from simba.utils.read_write import get_unique_values_in_iterable, read_df
from simba.utils.warnings import GPUToolsWarning

try:
    import cupy as cp
    from cuml.metrics import kl_divergence as kl_divergence_gpu
    from cuml.metrics.cluster.adjusted_rand_index import adjusted_rand_score
    from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
    from cupyx.scipy.spatial.distance import cdist
except:
    GPUToolsWarning(msg='GPU tools not detected, reverting to CPU')
    import numpy as cp
    from scipy.spatial.distance import cdist
    from scipy.stats import entropy as kl_divergence_gpu
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics import silhouette_score as cython_silhouette_score

try:
   from cuml.cluster import KMeans
except:
    from sklearn.cluster import KMeans

from simba.data_processors.cuda.utils import _cuda_are_rows_equal
from simba.mixins.statistics_mixin import Statistics
from simba.utils.checks import (check_int, check_str, check_valid_array,
                                check_valid_tuple)
from simba.utils.data import bucket_data
from simba.utils.enums import Formats

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

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/get_3pt_angle.csv
       :widths: 10, 90
       :align: center
       :header-rows: 1

    .. image:: _static/img/get_3pt_angle.webp
       :width: 300
       :align: center

    .. seealso::
       For CPU function see :func:`~simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.angle3pt` and
       For CPU function see :func:`~simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.angle3pt_serialized`.


    :param x:  A numpy array of shape (n, 2) representing the first point (e.g., nose) coordinates.
    :param y: A numpy array of shape (n, 2) representing the second point (e.g., center) coordinates, where the angle is computed.
    :param z: A numpy array of shape (n, 2) representing the second point (e.g., center) coordinates, where the angle is computed.
    :return: A numpy array of shape (n, 1) containing the calculated angles (in degrees) for each row.
    :rtype: np.ndarray

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


@cuda.jit
def _get_values_in_range_kernel(values, ranges, results):
    i = cuda.grid(1)
    if i >= values.shape[0]:
        return
    v = values[i]
    for j in range(ranges.shape[0]):
        l, u = ranges[j][0], ranges[j][1]
        cnt = 0
        for k in v:
            if k <= u and k >= l:
                cnt += 1
        results[i, j] = cnt


def count_values_in_ranges(x: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Counts the number of values in each feature within specified ranges for each row in a 2D array using CUDA.


    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/count_values_in_ranges.csv
       :widths: 10, 90
       :align: center
       :header-rows: 1

    .. image:: _static/img/count_ranges_2.png
       :width: 600
       :align: center

    .. seealso::
       For CPU function see :func:`~simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.count_values_in_range`.

    :param np.ndarray x: 2d array with feature values.
    :param np.ndarray r: 2d array with lower and upper boundaries.
    :return: 2d array of size len(x) x len(r) with the counts of values in each feature range (inclusive).
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(1, 11, (10, 10)).astype(np.int8)
    >>> r = np.array([[1, 6], [6, 11]])
    >>> r_x = count_values_in_ranges(x=x, r=r)
    """

    x = np.ascontiguousarray(x).astype(np.float32)
    r = np.ascontiguousarray(r).astype(np.float32)
    n, m = x.shape[0], r.shape[0]
    values_dev = cuda.to_device(x)
    ranges_dev = cuda.to_device(r)
    results = cuda.device_array((n, m), dtype=np.int32)
    bpg = (n + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _get_values_in_range_kernel[bpg, THREADS_PER_BLOCK](values_dev, ranges_dev, results)
    results = results.copy_to_host()
    cuda.current_context().memory_manager.deallocations.clear()
    return results

@cuda.jit
def _euclidean_distance_kernel(x_dev, y_dev, results):
    i = cuda.grid(1)
    if i < x_dev.shape[0]:
        p = (math.sqrt((x_dev[i][0] - y_dev[i][0]) ** 2 + (x_dev[i][1] - y_dev[i][1]) ** 2))
        results[i] = p

def get_euclidean_distance_cuda(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Computes the Euclidean distance between two sets of points using CUDA for GPU acceleration.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/get_euclidean_distance_cuda.csv
       :widths: 10, 90
       :align: center
       :header-rows: 1

    .. seealso::
       For CPU function see :func:`~simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_euclidean_distance`.
       For CuPY function see :func:`~simba.data_processors.cuda.statistics.get_euclidean_distance_cupy`.


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


def get_euclidean_distance_cupy(x: np.ndarray,
                                y: np.ndarray,
                                batch_size: Optional[int] = int(3.5e10+7)) -> np.ndarray:
    """
    Computes the Euclidean distance between corresponding pairs of points in two 2D arrays
    using CuPy for GPU acceleration. The computation is performed in batches to handle large
    datasets efficiently.


    .. seealso::
       For CPU function see :func:`~simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_euclidean_distance`.
       For CUDA JIT function see :func:`~simba.data_processors.cuda.statistics.get_euclidean_distance_cuda`.

    :param np.ndarray x: A 2D NumPy array with shape (n, 2), where each row represents a point in a 2D space.
    :param np.ndarray y: A 2D NumPy array with shape (n, 2), where each row represents a point in a 2D space. The shape of `y` must match the shape of `x`.
    :param Optional[int] batch_size: The number of points to process in a single batch. This parameter controls memory usage and can be adjusted based on available GPU memory. The default value is large (`3.5e10 + 7`) to maximize GPU utilization, but it can be lowered if memory issues arise.
    :return: A 1D NumPy array of shape (n,) containing the Euclidean distances between corresponding points in `x` and `y`.
    :rtype: np.ndarray

    :example:
    >>> x = np.array([[1, 2], [3, 4], [5, 6]])
    >>> y = np.array([[7, 8], [9, 10], [11, 12]])
    >>> distances = get_euclidean_distance_cupy(x, y)
    """
    check_valid_array(data=x, source=check_valid_array.__name__, accepted_ndims=[2,], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, source=check_valid_array.__name__, accepted_ndims=[2, ], accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_shapes=(x.shape,))
    check_int(name='batch_size', value=batch_size, min_value=1)
    results = cp.full((x.shape[0]), fill_value=cp.nan, dtype=cp.float32)
    for l in range(0, x.shape[0], batch_size):
        r = l + batch_size
        batch_x, batch_y = cp.array(x[l:r]), cp.array(y[l:r])
        results[l:r] = (cp.sqrt((batch_x[:, 0] - batch_y[:, 0]) ** 2 + (batch_x[:, 1] - batch_y[:, 1]) ** 2))
    return results.get()

@cuda.jit(device=True)
def _cuda_sum(x: np.ndarray):
    s = 0
    for i in range(x.shape[0]):
        s += x[i]
    return s

@cuda.jit
def _cuda_sliding_mean(x: np.ndarray, d: np.ndarray, results: np.ndarray):
    r = cuda.grid(1)
    l = np.int32(r - (d[0] - 1))
    if (r >= results.shape[0]) or (l < 0):
        results[r] = -1
    else:
        x_i = x[l:r+1]
        s = _cuda_sum(x_i)
        results[r] = s / x_i.shape[0]

def sliding_mean(x: np.ndarray, time_window: float, sample_rate: int) -> np.ndarray:
    """
    Computes the mean of values within a sliding window over a 1D numpy array `x` using CUDA for acceleration.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_mean.csv
       :widths: 10, 90
       :align: center
       :header-rows: 1

    :param np.ndarray x: The input 1D numpy array of floats. The array over which the sliding window sum is computed.
    :param float time_window: The size of the sliding window in seconds. This window slides over the array `x` to compute the sum.
    :param int sample_rate: The number of samples per second in the array `x`. This is used to convert the time-based window size into the number of samples.
    :return: A numpy array containing the sum of values within each position of the sliding window.
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(1, 11, (100, )).astype(np.float32)
    >>> time_window = 1
    >>> sample_rate = 10
    >>> r_x = sliding_mean(x=x, time_window=time_window, sample_rate=10)
    """
    x = np.ascontiguousarray(x).astype(np.int32)
    window_size = np.array([np.ceil(time_window * sample_rate)])
    x_dev = cuda.to_device(x)
    delta_dev = cuda.to_device(window_size)
    results = cuda.device_array(x.shape, dtype=np.float32)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _cuda_sliding_mean[bpg, THREADS_PER_BLOCK](x_dev, delta_dev, results)
    results = results.copy_to_host()
    return results

def silhouette_score_gpu(x: np.ndarray,
                         y: np.ndarray,
                         metric: Literal["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "sqeuclidean"] =  'euclidean') -> float:
    """
    Compute the Silhouette Score for clustering assignments on GPU using a specified distance metric.

    :param np.ndarray x: Feature matrix of shape (n_samples, n_features) containing numeric data.
    :param np.ndarray y: Cluster labels array of shape (n_samples,) with numeric labels.
    :param Literal["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "sqeuclidean"] metric:  Distance metric to use (default='euclidean'). Must be one of: "cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", or "sqeuclidean".
    :return: Mean silhouette score as a float.
    :rtype: float

    :example:
    >>> x, y = make_blobs(n_samples=50000, n_features=20, centers=5, cluster_std=10, center_box=(-1, 1))
    >>> score_gpu = silhouette_score_gpu(x=x, y=y)
    """
    VALID_METRICS = ["cityblock", "cosine", "euclidean", "l1", "l2", "manhattan", "sqeuclidean"]
    check_str(name=f'{silhouette_score_gpu.__name__} metric', value=metric, options=VALID_METRICS)
    check_valid_array(data=x, source=f'{silhouette_score_gpu.__name__} x', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, source=f'{silhouette_score_gpu.__name__} y', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0]])
    x = cython_silhouette_score(X=x, labels=y, metric=metric)
    return x


@cuda.jit
def _cuda_sliding_min(x: np.ndarray, d: np.ndarray, results: np.ndarray):
    def _cuda_min(a, b):
        return a if a < b else b
    r = cuda.grid(1)
    l = np.int32(r - (d[0]-1))
    if (r > results.shape[0]) or (l < 0):
        results[r] = -1
    else:
        x_i = x[l:r-1]
        local_min = x_i[0]
        for k in range(x_i.shape[0]):
            local_min = _cuda_min(local_min, x_i[k])
        results[r] = local_min

def sliding_min(x: np.ndarray, time_window: float, sample_rate: int) -> np.ndarray:
    """
    Computes the minimum value within a sliding window over a 1D numpy array `x` using CUDA for acceleration.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_min.csv
       :widths: 10, 90
       :align: center
       :header-rows: 1

    :param np.ndarray x: Input 1D numpy array of floats. The array over which the sliding window minimum is computed.
    :param float time_window: The size of the sliding window in seconds.
    :param int sample_rate: The sampling rate of the data, which determines the number of samples per second.
    :return: A numpy array containing the minimum value for each position of the sliding window.
    :rtype: np.ndarray

    :example:
    >>> x = np.arange(0, 10000000)
    >>> time_window = 1
    >>> sample_rate = 10
    >>> sliding_min(x=x, time_window=time_window, sample_rate=sample_rate)
    """

    x = np.ascontiguousarray(x).astype(np.float32)
    window_size = np.array([np.ceil(time_window * sample_rate)])
    x_dev = cuda.to_device(x)
    delta_dev = cuda.to_device(window_size)
    results = cuda.device_array(x.shape, dtype=np.float32)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _cuda_sliding_min[bpg, THREADS_PER_BLOCK](x_dev, delta_dev, results)
    results = results.copy_to_host()
    return results


def sliding_spearmans_rank(x: np.ndarray,
                           y: np.ndarray,
                           time_window: float,
                           sample_rate: int,
                           batch_size: Optional[int] = int(1.6e+7)) -> np.ndarray:
    """
    Computes the Spearman's rank correlation coefficient between two 1D arrays `x` and `y`
    over sliding windows of size `time_window * sample_rate`. The computation is performed
    in batches to optimize memory usage, leveraging GPU acceleration with CuPy.

    .. seealso::

       For CPU function see :func:`~simba.mixins.statistics.StatisticsMixin.sliding_spearman_rank_correlation`.

    :math:`\\rho = 1 - \\frac{6 \\sum d_i^2}{n_w(n_w^2 - 1)}`

    Where:
    - \( \\rho \) is the Spearman's rank correlation coefficient.
    - \( d_i \) is the difference between the ranks of corresponding elements in the sliding window.
    - \( n_w \) is the size of the sliding window.

    :param np.ndarray x: The first 1D array containing the values for Feature 1.
    :param np.ndarray y: The second 1D array containing the values for Feature 2.
    :param float time_window: The size of the sliding window in seconds.
    :param int sample_rate: The sampling rate (samples per second) of the data.
    :param Optional[int] batch_size: The size of each batch to process at a time for memory efficiency. Defaults to 1.6e7.
    :return: A 1D numpy array containing the Spearman's rank correlation coefficient for each sliding window.
    :rtype: np.ndarray

    :example:
    >>> x = np.array([9, 10, 13, 22, 15, 18, 15, 19, 32, 11])
    >>> y = np.array([11, 12, 15, 19, 21, 26, 19, 20, 22, 19])
    >>> sliding_spearmans_rank(x, y, time_window=0.5, sample_rate=2)
    """

    window_size = int(np.ceil(time_window * sample_rate))
    n = x.shape[0]
    results = cp.full(n, -1, dtype=cp.float32)

    for cnt, left in enumerate(range(0, n, batch_size)):
        right = int(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size + 1
        x_batch = cp.asarray(x[left:right])
        y_batch = cp.asarray(y[left:right])
        x_batch = cp.lib.stride_tricks.sliding_window_view(x_batch, window_size)
        y_batch = cp.lib.stride_tricks.sliding_window_view(y_batch, window_size)
        rank_x = cp.argsort(cp.argsort(x_batch, axis=1), axis=1)
        rank_y = cp.argsort(cp.argsort(y_batch, axis=1), axis=1)
        d_squared = cp.sum((rank_x - rank_y) ** 2, axis=1)
        n_w = window_size
        s = 1 - (6 * d_squared) / (n_w * (n_w ** 2 - 1))

        results[left + window_size - 1:right] = s

    return cp.asnumpy(results)



@cuda.jit(device=True)
def _cuda_sum(x: np.ndarray):
    s = 0
    for i in range(x.shape[0]):
        s += x[i]
    return s

@cuda.jit(device=True)
def _cuda_std(x: np.ndarray, x_hat: float):
    std = 0
    for i in range(x.shape[0]):
        std += (x[0] - x_hat) ** 2
    return std

@cuda.jit(device=False)
def _cuda_sliding_std(x: np.ndarray, d: np.ndarray, results: np.ndarray):
    r = cuda.grid(1)
    l = np.int32(r - (d[0] - 1))
    if (r >= results.shape[0]) or (l < 0):
        results[r] = -1
    else:
        x_i = x[l:r + 1]
        s = _cuda_sum(x_i)
        m = s / x_i.shape[0]
        std = _cuda_std(x_i, m)
        results[r] = std

def sliding_std(x: np.ndarray, time_window: float, sample_rate: int) -> np.ndarray:
    """
    :param np.ndarray x: The input 1D numpy array of floats. The array over which the sliding window sum is computed.
    :param float time_window: The size of the sliding window in seconds. This window slides over the array `x` to compute the sum.
    :param int sample_rate: The number of samples per second in the array `x`. This is used to convert the time-based window size into the number of samples.
    :return: A numpy array containing the sum of values within each position of the sliding window.
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(1, 11, (100, )).astype(np.float32)
    >>> time_window = 1
    >>> sample_rate = 10
    >>> r_x = sliding_sum(x=x, time_window=time_window, sample_rate=10)
    """
    x = np.ascontiguousarray(x).astype(np.int32)
    window_size = np.array([np.ceil(time_window * sample_rate)])
    x_dev = cuda.to_device(x)
    delta_dev = cuda.to_device(window_size)
    results = cuda.device_array(x.shape, dtype=np.float32)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _cuda_sliding_std[bpg, THREADS_PER_BLOCK](x_dev, delta_dev, results)
    results = results.copy_to_host()
    return results



@cuda.jit
def _cuda_sliding_sum(x: np.ndarray, d: np.ndarray, results: np.ndarray):
    r = cuda.grid(1)
    l = np.int32(r - (d[0]-1))
    if (r > results.shape[0]) or (l < 0):
        results[r] = -1
    else:
        x_i = x[l:r]
        local_sum = 0
        for k in range(x_i.shape[0]):
            local_sum += x_i[k]
        results[r-1] = local_sum

def sliding_sum(x: np.ndarray, time_window: float, sample_rate: int) -> np.ndarray:
    """
    Computes the sum of values within a sliding window over a 1D numpy array `x` using CUDA for acceleration.

    :param np.ndarray x: The input 1D numpy array of floats. The array over which the sliding window sum is computed.
    :param float time_window: The size of the sliding window in seconds. This window slides over the array `x` to compute the sum.
    :param int sample_rate: The number of samples per second in the array `x`. This is used to convert the time-based window size into the number of samples.
    :return: A numpy array containing the sum of values within each position of the sliding window.
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(1, 11, (100, )).astype(np.float32)
    >>> time_window = 1
    >>> sample_rate = 10
    >>> r_x = sliding_sum(x=x, time_window=time_window, sample_rate=10)
    """
    x = np.ascontiguousarray(x).astype(np.float32)
    window_size = np.array([np.ceil(time_window * sample_rate)])
    x_dev = cuda.to_device(x)
    delta_dev = cuda.to_device(window_size)
    results = cuda.device_array(x.shape, dtype=np.float32)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _cuda_sliding_sum[bpg, THREADS_PER_BLOCK](x_dev, delta_dev, results)
    results = results.copy_to_host()
    return results


def euclidean_distance_to_static_point(data: np.ndarray,
                                       point: Tuple[int, int],
                                       pixels_per_millimeter: Optional[int] = 1,
                                       centimeter: Optional[bool] = False,
                                       batch_size: Optional[int] = int(6.5e+7)) -> np.ndarray:
    """
    Computes the Euclidean distance between each point in a given 2D array `data` and a static point using GPU acceleration.

    :param data: A 2D array of shape (N, 2), where N is the number of points, and each point is represented by its (x, y) coordinates. The array can represent pixel coordinates.
    :param point: A tuple of two integers representing the static point (x, y) in the same space as `data`.
    :param pixels_per_millimeter: A scaling factor that indicates how many pixels correspond to one millimeter. Defaults to 1 if no scaling is necessary.
    :param centimeter:  A flag to indicate whether the output distances should be converted from millimeters to centimeters. If True, the result is divided by 10. Defaults to False (millimeters).
    :param batch_size: The number of points to process in each batch to avoid memory overflow on the GPU. The default  batch size is set to 65 million points (6.5e+7). Adjust this parameter based on GPU memory capacity.
    :param batch_size: The number of points to process in each batch to avoid memory overflow on the GPU. The default  batch size is set to 65 million points (6.5e+7). Adjust this parameter based on GPU memory capacity.
    :return: A 1D array of distances between each point in `data` and the static `point`, either in millimeters or centimeters depending on the `centimeter` flag.
    :rtype: np.ndarray
    """
    check_valid_array(data=data, source=euclidean_distance_to_static_point.__name__, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_1_shape=[2,])
    check_valid_tuple(x=point, source=euclidean_distance_to_static_point.__name__, accepted_lengths=(2,), valid_dtypes=Formats.NUMERIC_DTYPES.value)
    n = data.shape[0]
    results = cp.full((data.shape[0], 1),-1, dtype=np.float32)
    point = cp.array(point).reshape(1, 2)
    for cnt, l in enumerate(range(0, int(n), int(batch_size))):
        r = np.int32(min(l + batch_size, n))
        batch_data = cp.array(data[l:r])
        results[l:r]  = cdist(batch_data, point).astype(np.float32) / pixels_per_millimeter
    if centimeter:
        results = results / 10
    return results.get()


def dunn_index(x: np.ndarray, y: np.ndarray) -> float:

    r"""
    Computes the Dunn Index for clustering quality using GPU acceleration, which is a ratio of the minimum inter-cluster
    distance to the maximum intra-cluster distance. The higher the Dunn Index, the better the separation
    between clusters.

    .. seelalso:
       For CPU-based method, use :func:`simba.mixins.statistics_mixin.Statistics.dunn_index`

    The Dunn Index is given by:

    .. math::
       D = \frac{\min_{i \neq j} \{ \delta(C_i, C_j) \}}{\max_k \{ \Delta(C_k) \}}

    where :math:`\delta(C_i, C_j)` is the distance between clusters :math:`C_i` and :math:`C_j`, and
    :math:`\Delta(C_k)` is the diameter of cluster :math:`C_k`.

    The higher the Dunn Index, the better the clustering, as a higher value indicates that the clusters are well-separated relative to their internal cohesion.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/dunn_index_cuda.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    :param np.ndarray x: The input data points, where each row corresponds to an observation, and columns are features.
    :param np.ndarray y: Cluster labels for the data points. Each label corresponds to a cluster assignment for the respective observation in `x`.
    :return: The Dunn Index, a floating point value that measures the quality of clustering.
    :rtype: float

    :example:
    >>> centers = [[0, 0], [5, 10], [10, 0], [20, 10]]  # Adjust distances between cluster centers
    >>> x, y = make_blobs(n_samples=80_000_000, n_features=10, centers=centers, cluster_std=1, random_state=10)
    >>> v = dunn_index(x=x, y=y)
    """

    y = cp.array(y)
    x = cp.array(x)
    _, y = cp.unique(y, return_inverse=True)
    ys = cp.sort(cp.unique(y)).astype(cp.int64)
    boundaries = {}
    intra_deltas = cp.full(cp.unique(y).shape[0], fill_value=-cp.inf, dtype=cp.float64)
    inter_deltas = cp.full((cp.unique(y).shape[0], cp.unique(y).shape[0]), fill_value=cp.inf, dtype=cp.float64)
    for cnt, k in enumerate(ys):
        k = int(k)
        idx = cp.argwhere((y == k)).flatten()
        current_vals = x[idx, :].get()
        boundaries[k] = cp.array(current_vals[ConvexHull(current_vals).vertices])
        intra_dists = cdist(boundaries[k], boundaries[k])
        intra_deltas[cnt] = cp.max(intra_dists)
    for i, j in combinations(list(boundaries.keys()), 2):
        inter_dists = cdist(boundaries[i], boundaries[j])
        min_inter_dist = cp.min(inter_dists)
        inter_deltas[i, j] = min_inter_dist
        inter_deltas[j, i] = min_inter_dist
    v = cp.min(inter_deltas) / cp.max(intra_deltas)
    return v.get()


def adjusted_rand_gpu(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the Adjusted Rand Index (ARI) between two clusterings.

    The Adjusted Rand Index (ARI) is a measure of the similarity between two clusterings. It considers all pairs of samples and counts pairs that are assigned to the same or different clusters in both the true and predicted clusterings.

    The ARI is defined as:

    .. math::
       ARI = \\frac{TP + TN}{TP + FP + FN + TN}

    where:
        - :math:`TP` (True Positive) is the number of pairs of elements that are in the same cluster in both x and y,
        - :math:`FP` (False Positive) is the number of pairs of elements that are in the same cluster in y but not in x,
        - :math:`FN` (False Negative) is the number of pairs of elements that are in the same cluster in x but not in y,
        - :math:`TN` (True Negative) is the number of pairs of elements that are in different clusters in both x and y.

    The ARI value ranges from -1 to 1. A value of 1 indicates perfect clustering agreement, 0 indicates random clustering, and negative values indicate disagreement between the clusterings.

    .. note::
       Modified from `scikit-learn <https://github.com/scikit-learn/scikit-learn/blob/8721245511de2f225ff5f9aa5f5fadce663cd4a3/sklearn/metrics/cluster/_supervised.py#L353>`_

    .. seealso::
       For CPU call, see :func:`simba.mixins.statistics_mixin.Statistics.adjusted_rand`.


    :param np.ndarray x: 1D array representing the labels of the first model.
    :param np.ndarray y: 1D array representing the labels of the second model.
    :return: A value of 1 indicates perfect clustering agreement, a value of 0 indicates random clustering, and negative values indicate disagreement between the clusterings.
    :rtype: float

    :example:
    >>> x = np.random.randint(low=0, high=55, size=100000000)
    >>> y = np.random.randint(low=0, high=55, size=100000000)
    >>> adjusted_rand_gpu(x=x, y=y)
    """

    check_valid_array(data=x, source=f'{adjusted_rand_gpu.__name__} x', accepted_ndims=(1,), accepted_dtypes=Formats.INTEGER_DTYPES.value, min_axis_0=1)
    check_valid_array(data=y, source=f'{adjusted_rand_gpu.__name__} y', accepted_ndims=(1,), accepted_dtypes=Formats.INTEGER_DTYPES.value, accepted_shapes=[(x.shape[0],)])
    return adjusted_rand_score(x, y)

def davis_bouldin(x: np.ndarray,
                  y: np.ndarray) -> float:
    """
    Computes the Davis-Bouldin Index using GPU acceleration, a clustering evaluation metric that assesses
    the quality of clustering based on the ratio of within-cluster and between-cluster distances.

    The lower the Davis-Bouldin Index, the better the clusters are separated and compact.
    The function calculates the average similarity between each cluster and its most similar cluster.

    .. seealso::
       For CPU implementation, use :func:`simba.mixins.statistics_mixin.Statistics.davis_bouldin`

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/davis_bouldin_cuda.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    :param np.ndarray x: A 2D array of data points where each row corresponds to aan observation and  each column corresponds to a feature.
    :param np.ndarray y: A 1D array containing the cluster labels for each sample in `x`.
    :return: The Davis-Bouldin Index as a float, where lower values indicate better-defined clusters.
    :rtype: float

    :example:
    >>> centers = [[0, 0], [5, 10], [10, 0], [20, 10]]  # Adjust distances between cluster centers
    >>> x, y = make_blobs(n_samples=50000, n_features=4, centers=3, cluster_std=0.1)
    >>> p = davis_bouldin(x, y)
    """


    x, y = cp.array(x), cp.array(y)
    n_labels, labels = cp.unique(y).shape[0], cp.unique(y)
    cluster_centers = cp.full((n_labels, x.shape[1]), cp.nan, dtype=cp.float32)
    intra_center_distances = np.full((n_labels), 0.0)
    for cnt, cluster_id in enumerate(labels):
        cluster_data = x[cp.argwhere(y == cluster_id), :]
        cluster_data = cluster_data.reshape(cluster_data.shape[0], -1)
        cluster_centers[cnt] = cp.mean(cluster_data, axis=0)
        center_dists = cdist(cluster_data, cluster_centers[cnt].reshape(1, cluster_centers[cnt].shape[0])).T[0]
        intra_center_distances[cnt] = cp.mean(center_dists)
    inter_center_distances = cdist(cluster_centers, cluster_centers)
    inter_center_distances[inter_center_distances == 0] = np.inf

    db_index = 0
    for i in range(inter_center_distances.shape[0]):
        max_ratio = -cp.inf
        for j in range(inter_center_distances.shape[1]):
            if i != j:
                ratio = (intra_center_distances[i] + intra_center_distances[j]) / inter_center_distances[i, j]
                max_ratio = max(max_ratio, ratio)
        db_index += max_ratio
    return db_index / n_labels


def kmeans_cuml(data: np.ndarray,
                k: int = 2,
                max_iter: int = 300,
                output_type: Optional[str] = None,
                sample_n: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """CRAP, SLOWER THAN SCIKIT"""

    check_valid_array(data=data, source=f'{kmeans_cuml.__name__} data', accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name=f'{kmeans_cuml.__name__} k', value=k, min_value=1)
    check_int(name=f'{kmeans_cuml.__name__} max_iter', value=max_iter, min_value=1)
    kmeans = KMeans(n_clusters=k, max_iter=max_iter)
    if sample_n is not None:
        check_int(name=f'{kmeans_cuml.__name__} sample', value=sample_n, min_value=1)
        sample = min(sample_n, data.shape[0])
        data_idx = np.random.choice(np.arange(data.shape[0]), sample)
        mdl = kmeans.fit(data[data_idx])
    else:
        mdl = kmeans.fit(data)

    return (mdl.cluster_centers_, mdl.predict(data))



def xie_beni(x: np.ndarray, y: np.ndarray) -> float:
    """
    Computes the Xie-Beni index for clustering evaluation.

    The score is calculated as the ratio between the average intra-cluster variance and the squared minimum distance between cluster centroids. This ensures that the index penalizes both loosely packed clusters and clusters that are too close to each other.

    A lower Xie-Beni index indicates better clustering quality, signifying well-separated and compact clusters.

    .. seealso::
       To compute Xie-Beni on the CPU, use :func:`~simba.mixins.statistics_mixin.Statistics.xie_beni`
       Significant GPU savings detected at about 1m features, 25 clusters.

    :param np.ndarray x: The dataset as a 2D NumPy array of shape (n_samples, n_features).
    :param np.ndarray y: Cluster labels for each data point as a 1D NumPy array of shape (n_samples,).
    :returns: The Xie-Beni score for the dataset.
    :rtype: float

    :example:
    >>> from sklearn.datasets import make_blobs
    >>> X, y = make_blobs(n_samples=100000, centers=40, n_features=600, random_state=0, cluster_std=0.3)
    >>> xie_beni(x=X, y=y)

    :references:
    .. [1] X. L. Xie, G. Beni (1991). A validity measure for fuzzy clustering.
           In: IEEE Transactions on Pattern Analysis and Machine Intelligence 13(8), 841 - 847. DOI: 10.1109/34.85677
    """
    check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0], ])
    _ = get_unique_values_in_iterable(data=y, name=xie_beni.__name__, min=2)
    x, y = cp.array(x), cp.array(y)
    cluster_ids = cp.unique(y)
    centroids = cp.full(shape=(cluster_ids.shape[0], x.shape[1]), fill_value=-1.0, dtype=cp.float32)
    intra_centroid_distances = cp.full(shape=(y.shape[0]), fill_value=-1.0, dtype=cp.float32)
    obs_cnt = 0
    for cnt, cluster_id in enumerate(cluster_ids):
        cluster_obs = x[cp.argwhere(y == cluster_id).flatten()]
        centroids[cnt] = cp.mean(cluster_obs, axis=0)
        intra_dist = cp.linalg.norm(cluster_obs - centroids[cnt], axis=1)
        intra_centroid_distances[obs_cnt: cluster_obs.shape[0] + obs_cnt] = intra_dist
        obs_cnt += cluster_obs.shape[0]
    compactness = cp.mean(cp.square(intra_centroid_distances))
    cluster_dists = cdist(centroids, centroids).flatten()
    d = cp.sqrt(cluster_dists[cp.argwhere(cluster_dists > 0).flatten()])
    separation = cp.min(d)
    xb = compactness / separation
    return xb


def i_index(x: np.ndarray, y: np.ndarray):
    """
    Calculate the I-Index for evaluating clustering quality.

    The I-Index is a metric that measures the compactness and separation of clusters.
    A higher I-Index indicates better clustering with compact and well-separated clusters.

    .. seealso::
       To compute Xie-Beni on the CPU, use :func:`~simba.mixins.statistics_mixin.Statistics.i_index`

    :param np.ndarray x: The dataset as a 2D NumPy array of shape (n_samples, n_features).
    :param np.ndarray y: Cluster labels for each data point as a 1D NumPy array of shape (n_samples,).
    :returns: The I-index score for the dataset.
    :rtype: float

    :references:
        .. [1] Zhao, Q., Xu, M., Fränti, P. (2009). Sum-of-Squares Based Cluster Validity Index and Significance Analysis.
               In: Kolehmainen, M., Toivanen, P., Beliczynski, B. (eds) Adaptive and Natural Computing Algorithms. ICANNGA 2009.
                Lecture Notes in Computer Science, vol 5495. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-04921-7_32

    :example:
    >>> X, y = make_blobs(n_samples=5000, centers=20, n_features=3, random_state=0, cluster_std=0.1)
    >>> i_index(x=X, y=y)
    """
    check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value,
                      accepted_axis_0_shape=[x.shape[0], ])
    _ = get_unique_values_in_iterable(data=y, name=i_index.__name__, min=2)
    x, y = cp.array(x), cp.array(y)
    unique_y = cp.unique(y)
    n_y = unique_y.shape[0]
    global_centroid = cp.mean(x, axis=0)
    sst = cp.sum(cp.linalg.norm(x - global_centroid, axis=1) ** 2)

    swc = 0
    for cluster_cnt, cluster_id in enumerate(unique_y):
        cluster_obs = x[cp.argwhere(y == cluster_id).flatten()]
        cluster_centroid = cp.mean(cluster_obs, axis=0)
        swc += cp.sum(cp.linalg.norm(cluster_obs - cluster_centroid, axis=1) ** 2)

    i_idx = sst / (n_y * swc)

    return i_idx


def kullback_leibler_divergence_gpu(x: np.ndarray, y: np.ndarray, fill_value: int = 1, bucket_method: Literal[
    "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"] = "scott") -> float:
    """
    Compute Kullback-Leibler divergence between two distributions.

    .. note::
       Empty bins (0 observations in bin) in is replaced with passed ``fill_value``.

       Its range is from 0 to positive infinity. When the KL divergence is zero, it indicates that the two distributions are identical. As the KL divergence increases, it signifies an increasing difference between the distributions.

    .. seealso::
       For CPU implementation, see :func:`simba.mixins.statistics_mixin.Statistics.kullback_leibler_divergence`.


    :param ndarray x: First 1d array representing feature values.
    :param ndarray y: Second 1d array representing feature values.
    :param Optional[int] fill_value: Optional pseudo-value to use to fill empty buckets in ``y`` histogram
    :param Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
    :returns: Kullback-Leibler divergence between ``x`` and ``y``
    :rtype: float

    :example:
    >>> x, y = np.random.normal(loc=150, scale=900, size=10000000), np.random.normal(loc=140, scale=900, size=10000000)
    >>> kl = kullback_leibler_divergence_gpu(x=x, y=y)
    """

    bin_width, bin_count = bucket_data(data=x, method=bucket_method)
    r = np.array([np.min(x), np.max(x)])
    x_hist = Statistics._hist_1d(data=x, bin_count=bin_count, range=r)
    y_hist = Statistics._hist_1d(data=y, bin_count=bin_count, range=r)
    y_hist[y_hist == 0] = fill_value
    x_hist, y_hist = x_hist / np.sum(x_hist), y_hist / np.sum(y_hist)
    return kl_divergence_gpu(P=x_hist.astype(np.float32), Q=y_hist.astype(np.float32), convert_dtype=False)


@cuda.jit()
def _hamming_kernel(x, y, w, r):
    """
    Hamming distance kernal called by :func:`simba.data_processors.cuda.statistics.hamming_distance_gpu`
    """
    idx = cuda.grid(1)
    if idx < 0 or idx >= x.shape[0]:
        return
    if not _cuda_are_rows_equal(x, y, idx, idx):
        r[idx] = 1.0 * w[idx]

def hamming_distance_gpu(x: np.ndarray,
                         y: np.ndarray,
                         w: Optional[np.ndarray] = None) -> float:
    """
    Computes the weighted Hamming distance between two arrays using GPU acceleration.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/hamming_distance_gpu.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    .. seealso::
       For jitted CPU method, see :func:`simba.mixins.statistics_mixin.Statistics.hamming_distance`.

    :param ndarray x: A 1D or 2D NumPy array representing the reference data. If 2D, shape should be (n_samples, n_features). Supported dtypes are numeric.
    :param ndarray y: Array of the same shape as `x` representing the data to compare.
    :param ndarray w: A 1D array of shape (n_samples,) representing sample weights. If None, uniform weights are used.
    :returns: The weighted average Hamming distance between corresponding rows of `x` and `y`.
    :rtype: float


    :example:
    >>> x, y = np.random.randint(0, 2, (10, 1)).astype(np.int8), np.random.randint(0, 2, (10, 1)).astype(np.int8)
    >>> gpu_hamming = hamming_distance_gpu(x=x, y=y)
    """

    check_valid_array(data=x, source=f'{hamming_distance_gpu.__name__} x', accepted_ndims=(1, 2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, source=f'{hamming_distance_gpu.__name__} y', accepted_ndims=(x.ndim,), accepted_axis_0_shape=[x.shape[0]], accepted_axis_1_shape=[x.shape[1]] if x.ndim==2 else None, accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    if w is None:
        w = np.ones(x.shape[0]).astype(np.float32)
    check_valid_array(data=w, source=f'{hamming_distance_gpu.__name__} w', accepted_ndims=(1,), accepted_axis_0_shape=[x.shape[0]], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    results = np.full(shape=(x.shape[0],), fill_value=0.0, dtype=np.bool_)
    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    w_dev = cuda.to_device(w)
    results_dev = cuda.to_device(results)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _hamming_kernel[bpg, THREADS_PER_BLOCK](x_dev, y_dev, w_dev, results_dev)
    return np.sum(results_dev.copy_to_host()) / x.shape[0]


@cuda.jit()
def _sokal_sneath_kernel(x, y, w, c):
    idx = cuda.grid(1)
    if idx < 0 or idx >= x.shape[0]:
        return
    if (x[idx] == 1) and (y[idx] == 1):
        cuda.atomic.add(c, 0, 1 * w[idx])
    elif (x[idx] == 1) and (y[idx] == 0):
        cuda.atomic.add(c, 1, 1 * w[idx])
    elif (x[idx] == 0) and (y[idx] == 1):
        cuda.atomic.add(c, 2, 1 * w[idx])


def sokal_sneath_gpu(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """
    Compute the Sokal–Sneath similarity coefficient between two binary vectors using CUDA acceleration.

    .. seealso::
       For CPU method, see :func:`simba.mixins.statistics_mixin.Statistics.sokal_sneath`

    :param ndarray x: First binary vector (1D array of 0s and 1s).
    :param ndarray y: Second binary vector of the same shape as `x`.
    :param ndarray w: A 1D array of shape (n_samples,) representing sample weights. If None, uniform weights are used.
    :returns: The Sokal–Sneath similarity coefficient between `x` and `y`.
    :rtype: float.
    """


    check_valid_array(data=x, source=f'{sokal_sneath_gpu.__name__} x', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, source=f'{sokal_sneath_gpu.__name__} y', accepted_ndims=(x.ndim,), accepted_axis_0_shape=[x.shape[0]], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    if w is None:
        w = np.ones(x.shape[0]).astype(np.float32)
    check_valid_array(data=w, source=f'{sokal_sneath_gpu.__name__} w', accepted_ndims=(1,), accepted_axis_0_shape=[x.shape[0]], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    w_dev = cuda.to_device(w)
    counter = cuda.to_device(np.zeros(3, dtype=np.float32))
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _sokal_sneath_kernel[bpg, THREADS_PER_BLOCK](x_dev, y_dev, w_dev, counter)
    result = counter.copy_to_host()
    a, b, c = result[0], result[1], result[2]
    denom = a + 2 * (b + c)
    return a / denom if denom != 0.0 else 1.0