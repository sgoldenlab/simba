__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"


import math
from typing import Optional

import numpy as np
from numba import cuda

from simba.utils.read_write import read_df

try:
    import cupy as cp
except:
    import numpy as cp

from simba.utils.checks import check_int, check_valid_array
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
       :file: ../../../docs/tables/sliding_resultant_vector_length.csv
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