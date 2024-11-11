from time import perf_counter
from typing import Optional

import numpy as np
from numba import cuda

from simba.data_processors.cuda.utils import (_cuda_available, _cuda_mean,
                                              _cuda_std, _euclid_dist)
from simba.utils.checks import check_float, check_valid_array
from simba.utils.enums import Formats
from simba.utils.errors import SimBAGPUError

THREADS_PER_BLOCK = 1024

@cuda.jit(device=True)
def _count_at_threshold(x: np.ndarray, inverse: int, threshold: float):
    results = 0
    for i in range(x.shape[0]):
        if inverse[0] == 0:
            if x[i] > threshold[0]:
                results += 1
        else:
            if x[i] < threshold[0]:
                results += 1
    return results

@cuda.jit()
def _sliding_crossings_kernal(data, time, threshold, inverse, results):
    r = cuda.grid(1)
    l = int(r - time[0])
    if r > data.shape[0] or r < 0:
        return
    elif l > data.shape[0] or l < 0:
        return
    else:
        sample = data[l:r]
        results[r-1] = _count_at_threshold(sample, inverse, threshold)

def sliding_threshold(data: np.ndarray, time_window: float, sample_rate: float, value: float, inverse: Optional[bool] = False) -> np.ndarray:
    """
    Compute the count of observations above or below threshold crossings over a sliding window using GPU acceleration.

    :param np.ndarray data: Input data array.
    :param float time_window: Size of the sliding window in seconds.
    :param float sample_rate: Number of samples per second in the data.
    :param float value: Threshold value.
    :param Optional[bool] inverse: If False, counts values above the threshold. If True, counts values below.
    :return: Array containing count of threshold crossings per window.
    :rtype: np.ndarray
    """

    check_float(name='sample_rate', value=sample_rate, min_value=10e-6)
    check_float(name='sample_rate', value=sample_rate, min_value=10e-6)
    check_float(name='time_window', value=time_window, min_value=10e-6)
    check_valid_array(data=data, source=sliding_threshold.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)

    data_dev = cuda.to_device(data)
    time_window_frames = np.array([np.ceil(time_window * sample_rate)])
    time_window_frames_dev = cuda.to_device(time_window_frames)
    value = np.array([value])
    invert = np.array([0])
    if inverse: invert[0] = 1
    value_dev = cuda.to_device(value)
    inverse_dev = cuda.to_device(invert)
    bpg = (data.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    results = cuda.device_array(shape=(data.shape), dtype=np.int32)
    _sliding_crossings_kernal[bpg, THREADS_PER_BLOCK](data_dev, time_window_frames_dev, value_dev, inverse_dev, results)
    return results.copy_to_host()

@cuda.jit()
def _sliding_percent_beyond_n_std_kernel(data, time, std_n, results):
    r = cuda.grid(1)
    l = int(r - time[0])
    if r > data.shape[0] or r < 0:
        return
    elif l > data.shape[0] or l < 0:
        return
    else:
        sample = data[l:r]
        m = _cuda_mean(sample)
        std_val = _cuda_std(sample, m) * std_n[0]
        cnt = 0
        for i in range(sample.shape[0]):

            if (sample[i] > (m + std_val)) or (sample[i] < (m - std_val)):
                print(sample[i], m + std_val)
                cnt += 1
        results[r-1] = cnt

def sliding_percent_beyond_n_std(data: np.ndarray, time_window: float, sample_rate: float, value: float) -> np.ndarray:
    """
    Computes the percentage of points in each sliding window of `data` that fall beyond
    `n` standard deviations from the mean of that window.

    This function uses GPU acceleration via CUDA to efficiently compute the result over large datasets.

    :param np.ndarray data: The input 1D data array for which the sliding window computation is to be performed.
    :param float time_window: The length of the time window in seconds.
    :param float sample_rate: The sample rate of the data in Hz (samples per second).
    :param float value: The number of standard deviations beyond which to count data points.
    :return: An array containing the count of data points beyond `n` standard deviations for each window.
    :rtype: np.ndarray

    :example:
    >>> data = np.random.randint(0, 100, (100,))
    >>> results = sliding_percent_beyond_n_std(data=data, time_window=1, sample_rate=10, value=2)
    """

    data_dev = cuda.to_device(data)
    time_window_frames = np.array([np.ceil(time_window * sample_rate)])
    time_window_frames_dev = cuda.to_device(time_window_frames)
    value = np.array([value])
    value_dev = cuda.to_device(value)
    bpg = (data.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    results = cuda.device_array(shape=(data.shape), dtype=np.int32)
    _sliding_percent_beyond_n_std_kernel[bpg, THREADS_PER_BLOCK](data_dev, time_window_frames_dev, value_dev, results)
    return results.copy_to_host()



@cuda.jit()
def _sliding_spatial_density_kernel(x, time_window, radius, results):
    r = cuda.grid(1)
    if r >= x.shape[0] or r < 0:
        return
    l = int(r - time_window[0])
    if l < 0 or l >= r:
        return
    total_neighbors = 0
    n_points = r - l
    if n_points <= 0:
        results[r] = 0
        return
    for i in range(l, r):
        for j in range(l, r):
            if i != j:
                dist = _euclid_dist(x[i], x[j])
                if dist <= radius[0]:
                    total_neighbors += 1

    results[r] = total_neighbors / n_points if n_points > 0 else 0


def sliding_spatial_density_cuda(x: np.ndarray,
                                 radius: float,
                                 pixels_per_mm: float,
                                 window_size: float,
                                 sample_rate: float) -> np.ndarray:
    """
    Computes the spatial density of points within a moving window along a trajectory using CUDA for acceleration.

    This function calculates a spatial density measure for each point along a 2D trajectory path by counting the number
    of neighboring points within a specified radius. The computation is performed within a sliding window that moves
    along the trajectory, using GPU acceleration to handle large datasets efficiently.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_spatial_density_cuda.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    .. seealso::
       :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.spatial_density`, :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_spatial_density`

    :param np.ndarray x: A 2D array of shape (N, 2), where N is the number of points and each point has two spatial coordinates (x, y). The array represents the trajectory path of points in a 2D space (e.g., x and y positions in space).
    :param float radius: The radius (in millimeters) within which to count neighboring points around each trajectory point. Defines the area of interest around each point.
    :param float pixels_per_mm: The scaling factor that converts the physical radius (in millimeters) to pixel units for spatial density calculations.
    :param float window_size: The size of the sliding window (in seconds or points) to compute the density of points. A larger window size will consider more points in each density calculation.
    :param float sample_rate: The rate at which to sample the trajectory points (e.g., frames per second or samples per unit time). It adjusts the granularity of the sliding window.
    :return: A 1D numpy array where each element represents the computed spatial density for the trajectory at the corresponding point in time (or frame). Higher values indicate more densely packed points within the specified radius, while lower values suggest more sparsely distributed points.
    :rtype: np.ndarray

    :example:
    >>> df = pd.read_csv("/mnt/c/troubleshooting/two_black_animals_14bp/project_folder/csv/outlier_corrected_movement_location/Test_3.csv")
    >>> x = df[['Nose_1_x', 'Nose_1_y']].values
    >>> results_cuda = sliding_spatial_density_cuda(x=x, radius=10.0, pixels_per_mm=4.0, window_size=1, sample_rate=20)

    """

    check_valid_array(data=x, source=f'{sliding_spatial_density_cuda.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=[2, ], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_spatial_density_cuda.__name__} radius', value=radius)
    check_float(name=f'{sliding_spatial_density_cuda.__name__} window_size', value=window_size)
    check_float(name=f'{sliding_spatial_density_cuda.__name__} sample_rate', value=sample_rate)
    check_float(name=f'{sliding_spatial_density_cuda.__name__} pixels_per_mm', value=pixels_per_mm)

    x = np.ascontiguousarray(x)
    pixel_radius = np.array([np.ceil(max(1.0, (radius * pixels_per_mm)))]).astype(np.float64)
    time_window_frames = np.array([np.ceil(window_size * sample_rate)])
    x_dev = cuda.to_device(x)
    time_window_frames_dev = cuda.to_device(time_window_frames)
    radius_dev = cuda.to_device(pixel_radius)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    results = cuda.device_array(shape=x.shape[0], dtype=np.float16)
    _sliding_spatial_density_kernel[bpg, THREADS_PER_BLOCK](x_dev, time_window_frames_dev, radius_dev, results)
    return results.copy_to_host()


@cuda.jit()
def _sliding_linearity_index_kernel(x, time_frms, results):
    r = cuda.grid(1)
    if r >= x.shape[0] or r < 0:
        return
    l = int(r - time_frms[0])
    if l < 0 or l >= r:
        return
    sample_x = x[l:r]
    straight_line_distance = _euclid_dist(sample_x[0], sample_x[-1])
    path_dist = 0
    for i in range(1, sample_x.shape[0]):
        path_dist += _euclid_dist(sample_x[i - 1], sample_x[i])
    if path_dist == 0:
        results[r] = 0.0
    else:
        results[r] = straight_line_distance / path_dist


def sliding_linearity_index_cuda(x: np.ndarray,
                                 window_size: float,
                                 sample_rate: float) -> np.ndarray:
    """

    Calculates the straightness (linearity) index of a path using CUDA acceleration.

    The output is a value between 0 and 1, where 1 indicates a perfectly straight path.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_linearity_index_cuda.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    .. seealso::
       :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_linearity_index`, :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.linearity_index`


    :param np.ndarray x: An (N, M) array representing the path, where N is the number of points and M is the number of spatial dimensions (e.g., 2 for 2D or 3 for 3D). Each row represents the coordinates of a point along the path.
    :param float x: The size of the sliding window in seconds. This defines the time window over which the linearity index is calculated. The window size should be specified in seconds.
    :param float sample_rate: The sample rate in Hz (samples per second), which is used to convert the window size from seconds to frames.
    :return: A 1D array of length N, where each element represents the linearity index of the path within a sliding  window. The value is a ratio between the straight-line distance and the actual path length for each window. Values range from 0 to 1, with 1 indicating a perfectly straight path.
    :rtype: np.ndarray

    :example:
    >>> x = np.random.randint(0, 500, (100, 2)).astype(np.float32)
    >>> q = sliding_linearity_index_cuda(x=x, window_size=2, sample_rate=30)
    """

    check_valid_array(data=x, source=f'{sliding_linearity_index_cuda.__name__} x', accepted_ndims=(2,),
                      accepted_axis_1_shape=[2, ], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_linearity_index_cuda.__name__} window_size', value=window_size)
    check_float(name=f'{sliding_linearity_index_cuda.__name__} sample_rate', value=sample_rate)
    x = np.ascontiguousarray(x)
    time_window_frames = np.array([max(1.0, np.ceil(window_size * sample_rate))])
    if not _cuda_available()[0]:
        SimBAGPUError(msg='No GPU found', source=sliding_linearity_index_cuda.__name__)
    x_dev = cuda.to_device(x)
    time_window_frames_dev = cuda.to_device(time_window_frames)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    results = cuda.device_array(shape=x.shape[0], dtype=np.float16)
    _sliding_linearity_index_kernel[bpg, THREADS_PER_BLOCK](x_dev, time_window_frames_dev, results)
    return results.copy_to_host()
