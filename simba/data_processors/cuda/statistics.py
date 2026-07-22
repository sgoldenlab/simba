__author__ = "Simon Nilsson; sronilsson@gmail.com"


import math
from itertools import combinations
from typing import Optional, Tuple, Union

from simba.utils.printing import SimbaTimer

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
    from cupyx.scipy.spatial.distance import cdist
except Exception as e:
    GPUToolsWarning(msg=f'GPU tools not detected, reverting to CPU: {e.args}')
    import numpy as cp
    from scipy.spatial.distance import cdist
try:
    from cuml.metrics import kl_divergence as kl_divergence_gpu
    from cuml.metrics.cluster.adjusted_rand_index import adjusted_rand_score
    from cuml.metrics.cluster.silhouette_score import cython_silhouette_score
except Exception as e:
    GPUToolsWarning(msg=f'GPU tools not detected, reverting to CPU: {e.args}')
    from scipy.stats import entropy as kl_divergence_gpu
    from sklearn.metrics import adjusted_rand_score
    from sklearn.metrics import silhouette_score as cython_silhouette_score

try:
   from cuml.cluster import KMeans
except:
    from sklearn.cluster import KMeans

from simba.data_processors.cuda.utils import _cuda_are_rows_equal
from simba.mixins.statistics_mixin import Statistics
from simba.utils.checks import (check_float, check_int, check_str,
                                check_valid_array, check_valid_tuple)
from simba.utils.data import bucket_data
from simba.utils.enums import Formats
from simba.utils.errors import SimBAGPUError

MAX_GRID_Y = 65535   # CUDA grid y-dimension limit (windows are launched on grid-y)

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
       :alt: Get 3pt angle
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
       :alt: Count ranges 2
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
                           time_window: Union[float, int],
                           sample_rate: Union[float, int],
                           batch_size: Optional[int] = int(1.6e+7),
                           verbose: bool = False) -> np.ndarray:
    r"""
    Computes the Spearman's rank correlation coefficient between two 1D arrays `x` and `y`
    over sliding windows of size `time_window * sample_rate`. The computation is performed
    in batches to optimize memory usage, leveraging GPU acceleration with CuPy.

    .. seealso::

       For CPU function see :func:`~simba.mixins.statistics.StatisticsMixin.sliding_spearman_rank_correlation`.

    :math:`\rho = 1 - \frac{6 \sum d_i^2}{n_w(n_w^2 - 1)}`

    Where:
    - :math:`\rho` is the Spearman's rank correlation coefficient.
    - :math:`d_i` is the difference between the ranks of corresponding elements in the sliding window.
    - :math:`n_w` is the size of the sliding window.

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

    timer = SimbaTimer(start=True)
    check_valid_array(data=x, source=f'{sliding_spearmans_rank.__name__} x', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, source=f'{sliding_spearmans_rank.__name__} y', accepted_ndims=(1,), accepted_axis_0_shape=(x.shape[0],), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_spearmans_rank.__name__} time_window', value=time_window, allow_zero=False, allow_negative=False, raise_error=True)
    check_float(name=f'{sliding_spearmans_rank.__name__} sample_rate', value=sample_rate, allow_zero=False, allow_negative=False, raise_error=True)
    check_int(name=f'{sliding_spearmans_rank.__name__} batch_size', value=batch_size, allow_zero=False, allow_negative=False, raise_error=True)
    window_size = np.int32(np.ceil(time_window * sample_rate))
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

    r = cp.asnumpy(results)
    timer.stop_timer()
    if verbose: print(f'Sliding Spearmans rank for {x.shape[0]} observations computed (elapsed time: {timer.elapsed_time_str}s)')
    return r




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

    .. seealso::
       For CPU-based distance to static point (ROI center), see :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_euclidean_distance_roi`
       For CPU-based framewise Euclidean distance, see :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_euclidean_distance`
       For GPU CuPy solution for distance between two sets of points, see :func:`simba.data_processors.cuda.statistics.get_euclidean_distance_cupy`
       For GPU numba CUDA solution for distance between two sets of points, see :func:`simba.data_processors.cuda.statistics.get_euclidean_distance_cuda`

    :param data: A 2D array of shape (N, 2), where N is the number of points, and each point is represented by its (x, y) coordinates. The array can represent pixel coordinates.
    :param point: A tuple of two integers representing the static point (x, y) in the same space as `data`.
    :param pixels_per_millimeter: A scaling factor that indicates how many pixels correspond to one millimeter. Defaults to 1 if no scaling is necessary.
    :param centimeter:  A flag to indicate whether the output distances should be converted from millimeters to centimeters. If True, the result is divided by 10. Defaults to False (millimeters).
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
    r"""
    Calculate the Adjusted Rand Index (ARI) between two clusterings.

    The Adjusted Rand Index (ARI) is a measure of the similarity between two clusterings. It considers all pairs of samples and counts pairs that are assigned to the same or different clusters in both the true and predicted clusterings.

    The ARI is defined as:

    .. math::
       ARI = \frac{TP + TN}{TP + FP + FN + TN}

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
    :return: The Xie-Beni score for the dataset.
    :rtype: float

    :example:

    >>> from sklearn.datasets import make_blobs
    >>> X, y = make_blobs(n_samples=100000, centers=40, n_features=600, random_state=0, cluster_std=0.3)
    >>> xie_beni(x=X, y=y)

    References
    ----------
    .. [1] Xie, X. L., & Beni, G. (1991). A validity measure for fuzzy clustering.
           `IEEE Transactions on Pattern Analysis and Machine Intelligence, 13(8), 841–847 <https://doi.org/10.1109/34.85677>`_.
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


def i_index(x: np.ndarray, y: np.ndarray, verbose: bool = False) -> float:
    r"""
    Calculate the I-Index for evaluating clustering quality.

    The I-Index is a metric that measures the compactness and separation of clusters.
    A higher I-Index indicates better clustering with compact and well-separated clusters.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/i_index_cuda.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    The I-Index is calculated as:

    .. math::
        I = \frac{SST}{k \times SWC}

    where:

    - :math:`SST = \sum_{i=1}^{n} \|x_i - \mu\|^2` is the total sum of squares (sum of squared distances from all points to the global centroid)
    - :math:`k` is the number of clusters
    - :math:`SWC = \sum_{c=1}^{k} \sum_{i \in c} \|x_i - \mu_c\|^2` is the within-cluster sum of squares (sum of squared distances from points to their cluster centroids)

    .. seealso::
       To compute Xie-Beni on the CPU, use :func:`~simba.mixins.statistics_mixin.Statistics.i_index`

    :param np.ndarray x: The dataset as a 2D NumPy array of shape (n_samples, n_features).
    :param np.ndarray y: Cluster labels for each data point as a 1D NumPy array of shape (n_samples,).
    :return: The I-index score for the dataset.
    :rtype: float

    References
    ----------
    .. [1] Zhao, Q., Xu, M., & Fränti, P. (2009). Sum-of-squares based cluster validity index and significance analysis.
           In Adaptive and Natural Computing Algorithms (ICANNGA 2009), Lecture Notes in Computer Science, vol. 5495.
           `Springer <https://doi.org/10.1007/978-3-642-04921-7_32>`_.

    :example:

    >>> X, y = make_blobs(n_samples=5000, centers=20, n_features=3, random_state=0, cluster_std=0.1)
    >>> i_index(x=X, y=y)
    """
    timer = SimbaTimer(start=True)
    check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0], ])
    cluster_cnt = get_unique_values_in_iterable(data=y, name=i_index.__name__, min=2)
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
    i_idx = np.float32(i_idx.get()) if hasattr(i_idx, 'get') else np.float32(i_idx)
    timer.stop_timer()
    if verbose: print(f'I-index for {x.shape[0]} observations in {cluster_cnt} clusters computed (elapsed time: {timer.elapsed_time_str}s)')
    return i_idx

def kullback_leibler_divergence_gpu(x: np.ndarray,
                                    y: np.ndarray,
                                    fill_value: int = 1,
                                    bucket_method: Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"] = "scott",
                                    verbose: bool = False) -> float:
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
    :return: Kullback-Leibler divergence between ``x`` and ``y``
    :rtype: float

    :example:

    >>> x, y = np.random.normal(loc=150, scale=900, size=10000000), np.random.normal(loc=140, scale=900, size=10000000)
    >>> kl = kullback_leibler_divergence_gpu(x=x, y=y)
    """

    timer = SimbaTimer(start=True)

    bin_width, bin_count = bucket_data(data=x, method=bucket_method)
    r = np.array([np.min(x), np.max(x)])
    x_hist = Statistics._hist_1d(data=x, bin_count=bin_count, range=r)
    y_hist = Statistics._hist_1d(data=y, bin_count=bin_count, range=r)
    y_hist[y_hist == 0] = fill_value
    x_hist, y_hist = x_hist / np.sum(x_hist), y_hist / np.sum(y_hist)
    r =  kl_divergence_gpu(P=x_hist.astype(np.float32), Q=y_hist.astype(np.float32), convert_dtype=False)
    timer.stop_timer()
    if verbose: print(f'KL divergence performed on {x.shape[0]} observations (elapsed time: {timer.elapsed_time_str}s)')
    return r


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
    :return: The weighted average Hamming distance between corresponding rows of `x` and `y`.
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
    :return: The Sokal–Sneath similarity coefficient between `x` and `y`.
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

@cuda.jit()
def _sliding_autocorr_kernel(data, window_frms, max_lag_frms, results):
    """One thread per window (indexed by its right/last frame)."""
    right = cuda.grid(1)
    n = data.shape[0]
    if right >= n or right < window_frms - 1:
        return
    left = right - window_frms + 1
    N = max_lag_frms
    if N < 2:
        results[right] = 0.0
        return
    sc = 0.0; sy = 0.0; scy = 0.0; scc = 0.0
    for shift in range(N):
        if shift == 0:
            c = 1.0
        else:
            L = window_frms - shift
            if L < 2:
                c = 1.0
            else:
                mx = 0.0; mz = 0.0
                for k in range(L):
                    mx += data[left + k]
                    mz += data[left + shift + k]
                mx /= L; mz /= L
                sxz = 0.0; sxx = 0.0; szz = 0.0
                for k in range(L):
                    dx = data[left + k] - mx
                    dz = data[left + shift + k] - mz
                    sxz += dx * dz; sxx += dx * dx; szz += dz * dz
                denom = math.sqrt(sxx * szz)
                if denom < 1e-12:
                    c = 1.0
                else:
                    c = sxz / denom
        y = float(shift)
        sc += c; sy += y; scy += c * y; scc += c * c
    d = N * scc - sc * sc
    if -1e-12 < d < 1e-12:
        results[right] = 0.0
    else:
        results[right] = (N * scy - sc * sy) / d


def sliding_autocorrelation_cuda(data: np.ndarray,
                                 max_lag: float,
                                 time_window: float,
                                 fps: float) -> np.ndarray:
    """
    Compute, for each sliding window, how quickly the signal's self-correlation decays with lag (the slope of
    autocorrelation vs lag), on the GPU.

    .. note::
       Matches the CPU function to ~1e-3 (the CPU casts to float32 internally). Frames before the first full
       window are ``-1``.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_autocorrelation_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numba) version: :func:`simba.mixins.statistics_mixin.Statistics.sliding_autocorrelation`.

    :param np.ndarray data: 1D array of feature values.
    :param float max_lag: Maximum lag in seconds for the autocorrelation.
    :param float time_window: Sliding window length in seconds.
    :param float fps: Frames per second (converts the second-valued parameters to frames).
    :return: 1D float32 array (len == data) of the sliding autocorrelation slope per window; entries before the first full window are -1.
    :rtype: np.ndarray

    :example:

    >>> data = np.array([0,1,2,3,4,5,6,7,8,1,10,11,12,13,14]).astype(np.float32)
    >>> sliding_autocorrelation_cuda(data=data, max_lag=0.5, time_window=1.0, fps=10)
    """
    check_valid_array(data=data, source=f'{sliding_autocorrelation_cuda.__name__} data', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_autocorrelation_cuda.__name__} max_lag', value=max_lag, min_value=0.0)
    check_float(name=f'{sliding_autocorrelation_cuda.__name__} time_window', value=time_window, min_value=0.0)
    check_float(name=f'{sliding_autocorrelation_cuda.__name__} fps', value=fps, min_value=10e-6)
    max_frm_lag, window_frms = int(max_lag * fps), int(time_window * fps)
    n = data.shape[0]
    data = np.ascontiguousarray(data).astype(np.float32)
    data_dev = cuda.to_device(data)
    results = cuda.to_device(np.full(n, -1.0, dtype=np.float32))
    bpg = (n + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _sliding_autocorr_kernel[bpg, THREADS_PER_BLOCK](data_dev, window_frms, max_frm_lag, results)
    return results.copy_to_host()


@cuda.jit()
def _sliding_pearsons_kernel(s1, s2, window_sizes, results):
    """One thread per (frame idx, time-window wi). ``results`` (N, n_windows)."""
    idx, wi = cuda.grid(2)
    n = s1.shape[0]
    if idx >= n or wi >= window_sizes.shape[0]:
        return
    w = window_sizes[wi]
    if w < 1 or idx < w - 1:
        return
    left = idx - w + 1
    m1 = 0.0; m2 = 0.0
    for k in range(left, idx + 1):
        m1 += s1[k]; m2 += s2[k]
    m1 /= w; m2 /= w
    num = 0.0; d1 = 0.0; d2 = 0.0
    for k in range(left, idx + 1):
        a = s1[k] - m1; b = s2[k] - m2
        num += a * b; d1 += a * a; d2 += b * b
    denom = math.sqrt(d1 * d2)
    if denom != 0.0:
        results[idx, wi] = num / denom


def sliding_pearsons_r_cuda(sample_1: np.ndarray,
                            sample_2: np.ndarray,
                            time_windows: np.ndarray,
                            fps: float) -> np.ndarray:
    """
    Compute Pearson's R between two signals over each sliding window (for every window length and frame), on the GPU.

    .. note::
       Matches the CPU function to ~1e-7. Incomplete windows and zero-denominator windows are -1.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_pearsons_r_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numba) version: :func:`simba.mixins.statistics_mixin.Statistics.sliding_pearsons_r`.

    :param np.ndarray sample_1: First 1D signal.
    :param np.ndarray sample_2: Second 1D signal (same length as ``sample_1``).
    :param np.ndarray time_windows: 1D array of window lengths in seconds.
    :param float fps: Frames per second (converts window lengths in seconds to frames). May be fractional (e.g. 29.97).
    :return: 2D float32 array (len(sample_1), len(time_windows)) of Pearson's R per window; incomplete/zero-denominator windows are -1.
    :rtype: np.ndarray

    :example:

    >>> s1 = np.random.randint(0, 50, (10)).astype(np.float32)
    >>> s2 = np.random.randint(0, 50, (10)).astype(np.float32)
    >>> sliding_pearsons_r_cuda(sample_1=s1, sample_2=s2, time_windows=np.array([0.5]), fps=10)
    """
    check_valid_array(data=sample_1, source=f'{sliding_pearsons_r_cuda.__name__} sample_1', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=sample_2, source=f'{sliding_pearsons_r_cuda.__name__} sample_2', accepted_ndims=(1,), accepted_shapes=[(sample_1.shape[0],)], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=time_windows, source=f'{sliding_pearsons_r_cuda.__name__} time_windows', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_pearsons_r_cuda.__name__} fps', value=fps, min_value=10e-6)
    n, nw = sample_1.shape[0], time_windows.shape[0]
    if nw > MAX_GRID_Y:
        raise SimBAGPUError(msg=f'{sliding_pearsons_r_cuda.__name__} supports at most {MAX_GRID_Y} time_windows (CUDA grid limit); got {nw}.', source=sliding_pearsons_r_cuda.__name__)
    window_sizes = (np.asarray(time_windows) * fps).astype(np.int32)
    s1 = cuda.to_device(np.ascontiguousarray(sample_1).astype(np.float32))
    s2 = cuda.to_device(np.ascontiguousarray(sample_2).astype(np.float32))
    ws_dev = cuda.to_device(window_sizes)
    results = cuda.to_device(np.full((n, nw), -1.0, dtype=np.float32))
    tpb = (128, 1)
    bpg = (math.ceil(n / tpb[0]), math.ceil(nw / tpb[1]))
    _sliding_pearsons_kernel[bpg, tpb](s1, s2, ws_dev, results)
    return results.copy_to_host()


@cuda.jit()
def _sliding_kendall_kernel(s1, s2, window_sizes, results):
    """One thread per (storage-row r, time-window wi). Window = s[r-W : r] (ends at frame r-1, stored at row r)."""
    r, wi = cuda.grid(2)
    n = s1.shape[0]
    if r >= n or wi >= window_sizes.shape[0]:
        return
    w = window_sizes[wi]
    if w < 1 or r < w:
        return
    left = r - w
    conc = 0
    disc = 0
    for p in range(left, r):
        s1p = s1[p]; s2p = s2[p]
        for q in range(p + 1, r):
            if s1p <= s1[q]:
                s1i = s1p; s2j = s2[q]
            else:
                s1i = s1[q]; s2j = s2p
            if s2j > s1i:
                conc += 1
            elif s2j < s1i:
                disc += 1
    d = conc + disc
    if d == 0:
        results[r, wi] = -1.0
    else:
        results[r, wi] = (conc - disc) / d


def sliding_kendall_tau_cuda(sample_1: np.ndarray,
                             sample_2: np.ndarray,
                             time_windows: np.ndarray,
                             fps: float) -> np.ndarray:
    """
    Compute Kendall's Tau rank correlation between two signals over each sliding window (for every window length
    and frame), on the GPU.

    .. note::
       Matches the CPU function. Incomplete windows are 0; zero-denominator windows are -1.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_kendall_tau_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numba) version: :func:`simba.mixins.statistics_mixin.Statistics.sliding_kendall_tau`.

    :param np.ndarray sample_1: First 1D signal.
    :param np.ndarray sample_2: Second 1D signal (same length as ``sample_1``).
    :param np.ndarray time_windows: 1D array of window lengths in seconds.
    :param float fps: Frames per second.
    :return: 2D float32 array (len(sample_1), len(time_windows)); 0 for incomplete windows, -1 for zero-denominator windows.
    :rtype: np.ndarray

    :example:

    >>> s1 = np.random.rand(20).astype(np.float32); s2 = np.random.rand(20).astype(np.float32)
    >>> sliding_kendall_tau_cuda(sample_1=s1, sample_2=s2, time_windows=np.array([0.5]), fps=10)
    """
    check_valid_array(data=sample_1, source=f'{sliding_kendall_tau_cuda.__name__} sample_1', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=sample_2, source=f'{sliding_kendall_tau_cuda.__name__} sample_2', accepted_ndims=(1,), accepted_shapes=[(sample_1.shape[0],)], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=time_windows, source=f'{sliding_kendall_tau_cuda.__name__} time_windows', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_kendall_tau_cuda.__name__} fps', value=fps, min_value=10e-6)
    n, nw = sample_1.shape[0], time_windows.shape[0]
    if nw > MAX_GRID_Y:
        raise SimBAGPUError(msg=f'{sliding_kendall_tau_cuda.__name__} supports at most {MAX_GRID_Y} time_windows (CUDA grid limit); got {nw}.', source=sliding_kendall_tau_cuda.__name__)
    window_sizes = (np.asarray(time_windows) * fps).astype(np.int32)
    s1 = cuda.to_device(np.ascontiguousarray(sample_1).astype(np.float32))
    s2 = cuda.to_device(np.ascontiguousarray(sample_2).astype(np.float32))
    ws_dev = cuda.to_device(window_sizes)
    results = cuda.to_device(np.full((n, nw), 0.0, dtype=np.float32))
    tpb = (128, 1)
    bpg = (math.ceil(n / tpb[0]), math.ceil(nw / tpb[1]))
    _sliding_kendall_kernel[bpg, tpb](s1, s2, ws_dev, results)
    return results.copy_to_host()


@cuda.jit()
def _mahalanobis_kernel(data, w, q, results):
    """One thread per (i, j). results[i,j] = sqrt(q[i] + q[j] - 2 * dot(data[i], w[j]))."""
    i, j = cuda.grid(2)
    n, d = data.shape[0], data.shape[1]
    if i >= n or j >= n:
        return
    dot = 0.0
    for a in range(d):
        dot += data[i, a] * w[j, a]
    s = q[i] + q[j] - 2.0 * dot
    if s < 0.0:
        s = 0.0
    results[i, j] = math.sqrt(s)


def mahalanobis_distance_cdist_cuda(data: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Mahalanobis distance between all observations (rows) of ``data`` - the distance between
    each pair, scaled by the feature covariance - on the GPU.

    .. note::
       Matches the CPU/scipy result to ~1e-6. The output is a dense (n, n) float32 matrix, so memory scales with
       n^2 (n=20,000 is ~1.6 GB; ~50,000 is the ~10 GB in-VRAM ceiling on a 12 GB card) - bounded by GPU memory,
       not compute. Requires a non-singular covariance matrix (raises if features are collinear or n < n_features).

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/mahalanobis_distance_cdist_cuda.csv
       :widths: 20, 20, 20, 20, 20
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numba) version: :func:`simba.mixins.statistics_mixin.Statistics.mahalanobis_distance_cdist`.

    :param np.ndarray data: 2D array (n_observations, n_features).
    :return: (n, n) float32 pairwise Mahalanobis distance matrix.
    :rtype: np.ndarray

    :example:

    >>> data = np.random.randint(0, 50, (1000, 200)).astype(np.float32)
    >>> d = mahalanobis_distance_cdist_cuda(data=data)
    """
    check_valid_array(data=data, source=f'{mahalanobis_distance_cdist_cuda.__name__} data', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    x = np.ascontiguousarray(data).astype(np.float64)
    m = np.linalg.inv(np.cov(x, rowvar=False))
    w = x @ m
    q = np.einsum('ij,ij->i', x, w)
    n = x.shape[0]
    data_dev = cuda.to_device(x)
    w_dev = cuda.to_device(np.ascontiguousarray(w))
    q_dev = cuda.to_device(np.ascontiguousarray(q))
    results = cuda.device_array((n, n), dtype=np.float32)
    tpb = (16, 16)
    bpg = (math.ceil(n / tpb[0]), math.ceil(n / tpb[1]))
    _mahalanobis_kernel[bpg, tpb](data_dev, w_dev, q_dev, results)
    return results.copy_to_host()


@cuda.jit()
def _manhattan_kernel(data, results):
    """One thread per (i, j). results[i,j] = sum_a |data[i,a] - data[j,a]|."""
    i, j = cuda.grid(2)
    n, d = data.shape[0], data.shape[1]
    if i >= n or j >= n:
        return
    s = 0.0
    for a in range(d):
        diff = data[i, a] - data[j, a]
        if diff < 0.0:
            diff = -diff
        s += diff
    results[i, j] = s


def manhattan_distance_cdist_cuda(data: np.ndarray) -> np.ndarray:
    """
    Compute the pairwise Manhattan (L1) distance between all observations (rows) of ``data`` - the sum of
    absolute feature differences between each pair - on the GPU.

    .. note::
       The output is a dense (n, n) float32 matrix, so memory scales with n^2 (n=50,000 ~ 10 GB, the in-VRAM
       ceiling on a 12 GB card). Unlike the CPU version - which builds a full (N, N, D) intermediate and raises
       MemoryError for large inputs (e.g. ~74 GB at n=10,000, d=200) - this only allocates the n x n output, so
       it handles sizes the CPU cannot.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/manhattan_distance_cdist_cuda.csv
       :widths: 20, 20, 20, 20, 20
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numpy) version: :func:`simba.mixins.statistics_mixin.Statistics.manhattan_distance_cdist`.

    :param np.ndarray data: 2D array (n_observations, n_features).
    :return: (n, n) float32 pairwise Manhattan (L1) distance matrix.
    :rtype: np.ndarray

    :example:

    >>> data = np.random.randint(0, 50, (10000, 2)).astype(np.float32)
    >>> d = manhattan_distance_cdist_cuda(data=data)
    """
    check_valid_array(data=data, source=f'{manhattan_distance_cdist_cuda.__name__} data', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    n = data.shape[0]
    data_dev = cuda.to_device(np.ascontiguousarray(data).astype(np.float32))
    results = cuda.device_array((n, n), dtype=np.float32)
    tpb = (16, 16)
    bpg = (math.ceil(n / tpb[0]), math.ceil(n / tpb[1]))
    _manhattan_kernel[bpg, tpb](data_dev, results)
    return results.copy_to_host()
