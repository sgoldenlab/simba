import math
from time import perf_counter
from typing import Literal, Optional

import numpy as np
from numba import cuda, float64

from simba.data_processors.cuda.utils import (_cuda_diff, _cuda_mean,
                                              _cuda_median, _cuda_nanvariance,
                                              _cuda_std, _euclid_dist_2d,
                                              _is_cuda_available)
from simba.utils.checks import (check_float, check_int, check_str,
                                check_valid_array, check_valid_boolean)
from simba.utils.enums import Formats
from simba.utils.errors import SimBAGPUError

THREADS_PER_BLOCK = 1024
MAX_HJORTH_WINDOW = 512
MAX_GRID_Y = 65535   # CUDA grid y-dimension limit (windows are launched on grid-y)
MAX_WIN = 512        # per-thread sort-buffer cap (sliding_path_curvature median)
_NAN = np.float32(np.nan)

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

    .. image:: _static/img/sliding_threshold.webp
       :alt: Sliding threshold
       :width: 700
       :align: center

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
                cnt += 1
        results[r-1] = cnt

def sliding_percent_beyond_n_std(data: np.ndarray, time_window: float, sample_rate: float, value: float) -> np.ndarray:
    """
    Computes the percentage of points in each sliding window of `data` that fall beyond
    `n` standard deviations from the mean of that window.

    This function uses GPU acceleration via CUDA to efficiently compute the result over large datasets.

    .. image:: _static/img/simba.data_processors.cuda.timeseries.sliding_percent_beyond_n_std.webp
       :alt: Sliding percent beyond n std
       :width: 700
       :align: center

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
                dist = _euclid_dist_2d(x[i], x[j])
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
    straight_line_distance = _euclid_dist_2d(sample_x[0], sample_x[-1])
    path_dist = 0
    for i in range(1, sample_x.shape[0]):
        path_dist += _euclid_dist_2d(sample_x[i - 1], sample_x[i])
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
    :param float window_size: The size of the sliding window in seconds. This defines the time window over which the linearity index is calculated. The window size should be specified in seconds.
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
    if not _is_cuda_available()[0]:
        SimBAGPUError(msg='No GPU found', source=sliding_linearity_index_cuda.__name__)
    x_dev = cuda.to_device(x)
    time_window_frames_dev = cuda.to_device(time_window_frames)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    results = cuda.device_array(shape=x.shape[0], dtype=np.float16)
    _sliding_linearity_index_kernel[bpg, THREADS_PER_BLOCK](x_dev, time_window_frames_dev, results)
    return results.copy_to_host()


@cuda.jit
def _sliding_hjort_parameters_kernel(x, y, results):
    r_idx, y_idx = cuda.grid(2)
    if r_idx >= x.shape[0] or y_idx >= y.shape[0]:
        return

    win_size = y[y_idx]
    l_idx = int(r_idx - win_size + 1)
    if l_idx < 0:
        return

    x_win = cuda.local.array(MAX_HJORTH_WINDOW, dtype=float64)
    dx = cuda.local.array(MAX_HJORTH_WINDOW, dtype=float64)
    ddx = cuda.local.array(MAX_HJORTH_WINDOW, dtype=float64)

    N = win_size
    for i in range(N):
        x_win[i] = x[l_idx + i]

    _cuda_diff(x, l_idx, r_idx + 1, dx)
    _cuda_diff(dx, 0, N, ddx)

    activity = _cuda_nanvariance(x_win, N)
    dx_var = _cuda_nanvariance(dx, N)
    ddx_var = _cuda_nanvariance(ddx, N)

    if activity == 0 or dx_var == 0:
        return

    mobility = math.sqrt(dx_var / activity)
    complexity = math.sqrt(ddx_var / dx_var) / mobility

    results[0, r_idx, y_idx] = mobility
    results[1, r_idx, y_idx] = complexity
    results[2, r_idx, y_idx] = activity


def sliding_hjort_parameters_gpu(data: np.ndarray, window_sizes: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Compute Hjorth parameters over sliding windows on the GPU.

    .. seealso::
       For CPU implementation, see :`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.hjort_parameters`

    :param np.ndarray data: 1D numeric array of signal data.
    :param np.ndarray window_sizes: 1D numeric array of window sizes (in seconds).
    :param int sample_rate: Sampling rate of the data (samples per second).
    :return: 3D array of shape (3, len(data), len(window_sizes)) containing Hjorth parameters computed for each data point and window size.
    :rtype: np.ndarray

    :example:

    >>> x = np.random.randint(0, 500, (10,)).astype(np.float32)
    >>> window_sizes = np.array([1.0, 0.5]).astype(np.float64)
    >>> sample_rate = 10
    >>> H = sliding_hjort_parameters_gpu(data=x, window_sizes=window_sizes, sample_rate=sample_rate)
    """

    THREADS_PER_BLOCK = (32, 16)
    check_valid_array(data=data, source=f'{sliding_hjort_parameters_gpu.__name__} data', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_axis_0=1)
    check_valid_array(data=window_sizes, source=f'{sliding_hjort_parameters_gpu.__name__} window_sizes', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(name=f'{sliding_hjort_parameters_gpu.__name__} sample_rate', value=sample_rate)
    data = np.ascontiguousarray(data).astype(np.float64)
    results = np.full((3, data.shape[0], window_sizes.shape[0]), -1.0)
    window_sizes = np.ceil(window_sizes * sample_rate).astype(np.float64)
    data_dev = cuda.to_device(data)
    window_sizes_dev = cuda.to_device(window_sizes)
    results_dev = cuda.to_device(results)
    grid_x = (data.shape[0] + THREADS_PER_BLOCK[0] -1) // THREADS_PER_BLOCK[0]
    grid_y = (window_sizes.shape[0] + THREADS_PER_BLOCK[1] -1) // THREADS_PER_BLOCK[1]
    bpg = (grid_x, grid_y)
    _sliding_hjort_parameters_kernel[bpg, THREADS_PER_BLOCK](data_dev, window_sizes_dev, results_dev)
    return results_dev.copy_to_host()

@cuda.jit()
def _sliding_xcorr_kernel(x, y, window_sizes, lag, normalize, results):
    """One thread per (storage-row ri, window wi). Window ends at frame ri (x[ri-W+1 : ri+1]);
    the y-window is shifted back by ``lag`` frames."""
    ri, wi = cuda.grid(2)
    N = x.shape[0]
    if ri >= N or wi >= window_sizes.shape[0]:
        return
    W = window_sizes[wi]
    if W < 1 or ri < W - 1:
        return
    l1 = ri - W + 1
    l2 = l1 - lag
    if l2 < 0:
        l2 = 0
    sx = 0.0; sxx = 0.0; sy = 0.0; syy = 0.0; sxy = 0.0
    for k in range(W):
        xv = x[l1 + k]
        yv = y[l2 + k]
        sx += xv; sxx += xv * xv
        sy += yv; syy += yv * yv
        sxy += xv * yv
    if normalize:
        mx = sx / W; my = sy / W
        varx = sxx / W - mx * mx
        vary = syy / W - my * my
        stdx = math.sqrt(varx) if varx > 0.0 else 0.0
        stdy = math.sqrt(vary) if vary > 0.0 else 0.0
        denom = stdx * W * stdy
        results[ri, wi] = 0.0 if denom == 0.0 else (sxy - W * mx * my) / denom
    else:
        results[ri, wi] = sxy


def sliding_two_signal_crosscorrelation_cuda(x: np.ndarray, y: np.ndarray, windows: np.ndarray,
                                             sample_rate: float, normalize: bool = True, lag: float = 0.0) -> np.ndarray:
    """
    Compute the lagged cross-correlation between two signals over each sliding window (for every window length
    and frame) - an x-window against a ``lag``-shifted y-window - on the GPU.

    .. note::
       Matches the CPU function to float precision. With ``normalize=True`` each window is z-normalized before
       correlating. Frames before the first full window are 0.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_two_signal_crosscorrelation_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numba) version: :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_two_signal_crosscorrelation`.

    :param np.ndarray x: First 1D signal.
    :param np.ndarray y: Second 1D signal (same length as ``x``).
    :param np.ndarray windows: 1D array of window lengths in seconds.
    :param float sample_rate: Sampling rate (Hz / FPS).
    :param bool normalize: If True, z-normalize each window before correlating (normalized cross-correlation). Default True.
    :param float lag: Time lag (seconds) applied to ``y``. 0.0 = no lag. Default 0.0.
    :return: 2D float32 array (len(x), len(windows)); rows before the first full window are 0.
    :rtype: np.ndarray

    :example:

    >>> sliding_two_signal_crosscorrelation_cuda(x, y, windows=np.array([1.0, 1.2]), sample_rate=10, normalize=True, lag=0.0)
    """
    check_valid_array(data=x, source=f'{sliding_two_signal_crosscorrelation_cuda.__name__} x', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, source=f'{sliding_two_signal_crosscorrelation_cuda.__name__} y', accepted_ndims=(1,), accepted_shapes=[(x.shape[0],)], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=windows, source=f'{sliding_two_signal_crosscorrelation_cuda.__name__} windows', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_boolean(value=[normalize], source=f'{sliding_two_signal_crosscorrelation_cuda.__name__} normalize')
    n, nw = x.shape[0], windows.shape[0]
    if nw > MAX_GRID_Y:
        raise SimBAGPUError(msg=f'{sliding_two_signal_crosscorrelation_cuda.__name__} supports at most {MAX_GRID_Y} windows (CUDA grid limit); got {nw}.', source=sliding_two_signal_crosscorrelation_cuda.__name__)
    window_sizes = (np.asarray(windows) * sample_rate).astype(np.int32)
    lag_frm = int(sample_rate * lag)
    x_dev = cuda.to_device(np.ascontiguousarray(x).astype(np.float32))
    y_dev = cuda.to_device(np.ascontiguousarray(y).astype(np.float32))
    ws_dev = cuda.to_device(window_sizes)
    results = cuda.to_device(np.zeros((n, nw), dtype=np.float32))
    tpb = (128, 1)
    bpg = (math.ceil(n / tpb[0]), math.ceil(nw / tpb[1]))
    _sliding_xcorr_kernel[bpg, tpb](x_dev, y_dev, ws_dev, lag_frm, normalize, results)
    return results.copy_to_host()


@cuda.jit()
def _sliding_line_length_kernel(data, window_sizes, results):
    """One thread per (frame idx, window wi). results[idx,wi] = sum of |consecutive diffs| in the window ending at idx."""
    idx, wi = cuda.grid(2)
    n = data.shape[0]
    if idx >= n or wi >= window_sizes.shape[0]:
        return
    w = window_sizes[wi]
    if w < 1 or idx < w - 1:
        return
    left = idx - w + 1
    s = 0.0
    for k in range(left, idx):
        d = data[k + 1] - data[k]
        if d < 0.0:
            d = -d
        s += d
    results[idx, wi] = s


def sliding_line_length_cuda(data: np.ndarray, window_sizes: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Compute the sliding line length (sum of absolute consecutive differences) of a 1D signal, on the GPU.

    .. note::
       Output is (n_frames, n_window_sizes) float32; frames before the first full window are -1. Matches the CPU
       version exactly (float64 accumulation).

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_line_length_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numba) version: :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_line_length`.

    :param np.ndarray data: 1D signal.
    :param np.ndarray window_sizes: 1D array of window sizes in seconds.
    :param float sample_rate: Samples per second (may be fractional).
    :return: (n_frames, n_window_sizes) float32 array of sliding line-length values.
    :rtype: np.ndarray

    :example:

    >>> data = np.array([1, 4, 2, 3, 5, 6, 8, 7, 9, 10]).astype(np.float32)
    >>> sliding_line_length_cuda(data=data, window_sizes=np.array([1.0]), sample_rate=2.0)
    """
    check_valid_array(data=data, source=f'{sliding_line_length_cuda.__name__} data', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=window_sizes, source=f'{sliding_line_length_cuda.__name__} window_sizes', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_line_length_cuda.__name__} sample_rate', value=sample_rate, min_value=10e-6)
    n, nw = data.shape[0], window_sizes.shape[0]
    if nw > MAX_GRID_Y:
        raise SimBAGPUError(msg=f'{sliding_line_length_cuda.__name__} supports at most {MAX_GRID_Y} window_sizes (CUDA grid limit); got {nw}.', source=sliding_line_length_cuda.__name__)
    ws = (np.asarray(window_sizes) * sample_rate).astype(np.int32)
    data_dev = cuda.to_device(np.ascontiguousarray(data).astype(np.float64))
    ws_dev = cuda.to_device(ws)
    results = cuda.to_device(np.full((n, nw), -1.0, dtype=np.float32))
    tpb = (128, 1)
    bpg = (math.ceil(n / tpb[0]), math.ceil(nw / tpb[1]))
    _sliding_line_length_kernel[bpg, tpb](data_dev, ws_dev, results)
    return results.copy_to_host()


@cuda.jit()
def _sliding_msj_kernel(A, n_full, frame_step, results):
    """One thread per output row r. Mean squared jerk = mean over the window of sum_m (A[i+1,m]-A[i,m])^2."""
    r = cuda.grid(1)
    n = results.shape[0]
    m = A.shape[1]
    if r >= n or r < frame_step:
        return
    left = r - frame_step
    a_end = r if r < n_full else n_full
    njerk = (a_end - left) - 1
    if njerk <= 0:
        results[r] = 0.0
        return
    s = 0.0
    for i in range(left, a_end - 1):
        for c in range(m):
            d = A[i + 1, c] - A[i, c]
            s += d * d
    results[r] = s / njerk


def sliding_mean_squared_jerk_cuda(x: np.ndarray, window_size: float, sample_rate: float) -> np.ndarray:
    """
    Compute the sliding mean squared jerk (rate of change of acceleration) of a 2D path, on the GPU.

    Jerk is the derivative of acceleration; high values indicate abrupt motion changes.

    .. note::
       Output is a 1D float64 array of length n_frames; frames before the first full window are -1. Matches the CPU
       version to floating-point summation order (~1e-9).

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_mean_squared_jerk_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numba) version: :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_mean_squared_jerk`.

    :param np.ndarray x: 2D array (n_frames, n_dims) of positions.
    :param float window_size: Sliding-window size in seconds.
    :param float sample_rate: Samples per second (may be fractional).
    :return: (n_frames,) float64 array of mean squared jerk per window.
    :rtype: np.ndarray

    :example:

    >>> x = np.random.randint(0, 500, (5000, 2)).astype(np.float32)
    >>> sliding_mean_squared_jerk_cuda(x=x, window_size=1.0, sample_rate=30.0)
    """
    check_valid_array(data=x, source=f'{sliding_mean_squared_jerk_cuda.__name__} x', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_mean_squared_jerk_cuda.__name__} window_size', value=window_size, min_value=10e-6)
    check_float(name=f'{sliding_mean_squared_jerk_cuda.__name__} sample_rate', value=sample_rate, min_value=10e-6)
    n = x.shape[0]
    frame_step = int(max(1.0, window_size * sample_rate))
    A = np.diff(np.diff(np.ascontiguousarray(x).astype(np.float64), axis=0), axis=0)
    n_full = A.shape[0]
    A_dev = cuda.to_device(np.ascontiguousarray(A))
    results = cuda.to_device(np.full(n, -1.0, dtype=np.float64))
    bpg = math.ceil(n / THREADS_PER_BLOCK)
    _sliding_msj_kernel[bpg, THREADS_PER_BLOCK](A_dev, n_full, frame_step, results)
    return results.copy_to_host()


@cuda.jit()
def _sliding_path_curvature_kernel(x, frame_step, agg, results):
    """One thread per output row. Aggregated (0=mean,1=median,2=max) path curvature over the window ending at the row."""
    o = cuda.grid(1)
    n = results.shape[0]
    if o >= n or o < frame_step - 1:
        return
    left = o + 1 - frame_step
    buf = cuda.local.array(shape=512, dtype=np.float64)
    cnt = 0
    s = 0.0
    mx = -1.0e30
    for k in range(frame_step - 2):
        x0 = x[left + k, 0]; x1 = x[left + k + 1, 0]; x2 = x[left + k + 2, 0]
        y0 = x[left + k, 1]; y1 = x[left + k + 1, 1]; y2 = x[left + k + 2, 1]
        xp = x1 - x0
        yp = y1 - y0
        xpp = (x2 - x1) - (x1 - x0)
        ypp = (y2 - y1) - (y1 - y0)
        den = (xp * xp + yp * yp) ** 1.5
        if den != 0.0:
            c = math.fabs(xp * ypp - yp * xpp) / den
            s += c
            if c > mx:
                mx = c
            if agg == 1:
                buf[cnt] = c
            cnt += 1
    if cnt == 0:
        results[o] = _NAN
        return
    if agg == 0:
        results[o] = s / cnt
    elif agg == 2:
        results[o] = mx
    else:
        results[o] = _cuda_median(buf[:cnt])


def sliding_path_curvature_cuda(x: np.ndarray,
                                agg_type: Literal['mean', 'median', 'max'],
                                window_size: float,
                                sample_rate: float) -> np.ndarray:
    """
    Compute the aggregated path curvature over sliding windows of a 2D path, on the GPU.

    Higher values indicate sharper/more frequent directional changes within the window.

    .. note::
       Output is a 1D float32 array of length n_frames; frames before the first full window are NaN, and windows
       whose curvature is undefined everywhere (zero velocity) are NaN. For ``agg_type='median'`` the window must
       be <= 514 frames (per-thread sort buffer). Matches the CPU version.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_path_curvature_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU version: :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_path_curvature`.

    :param np.ndarray x: 2D array (n_frames, 2) of path (x, y) coordinates.
    :param Literal['mean','median','max'] agg_type: Aggregation of curvature within each window.
    :param float window_size: Window size in seconds.
    :param float sample_rate: Samples per second (may be fractional).
    :return: (n_frames,) float32 array of aggregated curvature per window.
    :rtype: np.ndarray

    :example:

    >>> x = np.random.randint(0, 500, (91, 2)).astype(np.float32)
    >>> sliding_path_curvature_cuda(x=x, agg_type='mean', window_size=1.0, sample_rate=30.0)
    """
    check_valid_array(data=x, source=f'{sliding_path_curvature_cuda.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_str(name=f'{sliding_path_curvature_cuda.__name__} agg_type', value=agg_type, options=('mean', 'median', 'max'))
    check_float(name=f'{sliding_path_curvature_cuda.__name__} window_size', value=window_size, min_value=10e-6)
    check_float(name=f'{sliding_path_curvature_cuda.__name__} sample_rate', value=sample_rate, min_value=10e-6)
    n = x.shape[0]
    frame_step = int(max(1.0, window_size * sample_rate))
    agg = {'mean': 0, 'median': 1, 'max': 2}[agg_type]
    if agg == 1 and (frame_step - 2) > MAX_WIN:
        raise SimBAGPUError(msg=f"{sliding_path_curvature_cuda.__name__} agg_type='median' supports windows up to {MAX_WIN + 2} frames; got {frame_step}.", source=sliding_path_curvature_cuda.__name__)
    x_dev = cuda.to_device(np.ascontiguousarray(x).astype(np.float64))
    results = cuda.to_device(np.full(n, np.nan, dtype=np.float32))
    bpg = math.ceil(n / THREADS_PER_BLOCK)
    _sliding_path_curvature_kernel[bpg, THREADS_PER_BLOCK](x_dev, frame_step, agg, results)
    return results.copy_to_host()


@cuda.jit()
def _sliding_aspect_ratio_kernel(x, window_frm, px_per_mm, results):
    """One thread per output row. Bounding-box (w/h)*px_per_mm over the window ending at the row; -1 if degenerate."""
    o = cuda.grid(1)
    n = results.shape[0]
    if o >= n or o < window_frm - 1:
        return
    left = o + 1 - window_frm
    xmin = x[left, 0]; xmax = x[left, 0]
    ymin = x[left, 1]; ymax = x[left, 1]
    for k in range(left + 1, o + 1):
        vx = x[k, 0]; vy = x[k, 1]
        if vx < xmin:
            xmin = vx
        if vx > xmax:
            xmax = vx
        if vy < ymin:
            ymin = vy
        if vy > ymax:
            ymax = vy
    w = xmax - xmin
    h = ymax - ymin
    if w == 0.0 or h == 0.0:
        results[o] = -1.0
    else:
        results[o] = (w / h) * px_per_mm


def sliding_path_aspect_ratio_cuda(x: np.ndarray,
                                   window_size: float,
                                   sample_rate: float,
                                   px_per_mm: float) -> np.ndarray:
    """
    Compute the sliding bounding-box aspect ratio (width/height) of a 2D path, on the GPU.

    .. note::
       Output is a 1D float32 array of length n_frames; frames before the first full window are NaN, and windows
       with zero width or height are -1. Matches the CPU version (float64 division).

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_path_aspect_ratio_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU version: :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_path_aspect_ratio`.

    :param np.ndarray x: 2D array (n_frames, 2) of path (x, y) coordinates.
    :param float window_size: Window size in seconds (converted to frames via ``ceil(window_size * sample_rate)``).
    :param float sample_rate: Samples per second (may be fractional).
    :param float px_per_mm: Pixels-per-millimeter conversion factor.
    :return: (n_frames,) float32 array of aspect ratios per window.
    :rtype: np.ndarray

    :example:

    >>> x = np.random.randint(0, 500, (10, 2)).astype(np.float32)
    >>> sliding_path_aspect_ratio_cuda(x=x, window_size=1.0, sample_rate=2.0, px_per_mm=1.0)
    """
    check_valid_array(data=x, source=f'{sliding_path_aspect_ratio_cuda.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_path_aspect_ratio_cuda.__name__} window_size', value=window_size, min_value=10e-6)
    check_float(name=f'{sliding_path_aspect_ratio_cuda.__name__} sample_rate', value=sample_rate, min_value=10e-6)
    check_float(name=f'{sliding_path_aspect_ratio_cuda.__name__} px_per_mm', value=px_per_mm, min_value=10e-6)
    n = x.shape[0]
    window_frm = int(math.ceil(window_size * sample_rate))
    x_dev = cuda.to_device(np.ascontiguousarray(x).astype(np.float64))
    results = cuda.to_device(np.full(n, np.nan, dtype=np.float32))
    bpg = math.ceil(n / THREADS_PER_BLOCK)
    _sliding_aspect_ratio_kernel[bpg, THREADS_PER_BLOCK](x_dev, window_frm, float(px_per_mm), results)
    return results.copy_to_host()


@cuda.jit()
def _sliding_ake_kernel(x, mass, window_frm, inv_dt2, results):
    """One thread per output row. 0.5 * mean(mass) * mean(speed^2) over the window ending at the row."""
    o = cuda.grid(1)
    n = results.shape[0]
    if o >= n or o < window_frm - 1:
        return
    left = o + 1 - window_frm
    msum = 0.0
    for k in range(left, o + 1):
        msum += mass[k]
    mass_mean = msum / window_frm
    ndiff = window_frm - 1
    if ndiff <= 0:
        results[o] = _NAN
        return
    ssq = 0.0
    for k in range(left, o):
        dx = x[k + 1, 0] - x[k, 0]
        dy = x[k + 1, 1] - x[k, 1]
        ssq += dx * dx + dy * dy
    mean_speed_sq = (ssq * inv_dt2) / ndiff
    results[o] = 0.5 * mass_mean * mean_speed_sq


def sliding_avg_kinetic_energy_cuda(x: np.ndarray, mass: np.ndarray, sample_rate: float, time_window: float) -> np.ndarray:
    """
    Compute the sliding average kinetic energy of a moving 2D point over time windows, on the GPU.

    Kinetic energy is ``0.5 * mean(mass) * mean(speed^2)`` within each window, where ``speed = |delta position| * sample_rate``.

    .. note::
       Output is a 1D float32 array of length n_frames; frames before the first full window are -1. The window
       length is ``ceil(sample_rate * time_window)``. Matches the CPU version (float64 computation).

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_avg_kinetic_energy_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU version: :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_avg_kinetic_energy`.

    :param np.ndarray x: 2D array (n_frames, 2) of point (x, y) coordinates.
    :param np.ndarray mass: 1D array (n_frames,) of per-frame mass (e.g. hull area).
    :param float sample_rate: Samples per second (may be fractional).
    :param float time_window: Window size in seconds.
    :return: (n_frames,) float32 array of average kinetic energy per window.
    :rtype: np.ndarray

    :example:

    >>> x = np.random.randint(0, 500, (5000, 2)).astype(np.float32)
    >>> mass = np.random.randint(10, 100, (5000,)).astype(np.float32)
    >>> sliding_avg_kinetic_energy_cuda(x=x, mass=mass, sample_rate=30.0, time_window=1.0)
    """
    check_valid_array(data=x, source=f'{sliding_avg_kinetic_energy_cuda.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=mass, source=f'{sliding_avg_kinetic_energy_cuda.__name__} mass', accepted_ndims=(1,), accepted_shapes=[(x.shape[0],)], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_avg_kinetic_energy_cuda.__name__} sample_rate', value=sample_rate, min_value=10e-6)
    check_float(name=f'{sliding_avg_kinetic_energy_cuda.__name__} time_window', value=time_window, min_value=10e-6)
    n = x.shape[0]
    window_frm = int(math.ceil(sample_rate * time_window))
    inv_dt2 = float(sample_rate) * float(sample_rate)
    x_dev = cuda.to_device(np.ascontiguousarray(x).astype(np.float64))
    mass_dev = cuda.to_device(np.ascontiguousarray(mass).astype(np.float64))
    results = cuda.to_device(np.full(n, -1.0, dtype=np.float32))
    bpg = math.ceil(n / THREADS_PER_BLOCK)
    _sliding_ake_kernel[bpg, THREADS_PER_BLOCK](x_dev, mass_dev, window_frm, inv_dt2, results)
    return results.copy_to_host()


@cuda.jit()
def _sliding_momentum_kernel(x, mass, window_frm, inv_dt, results):
    """One thread per output row. mean(mass) * mean(speed) over the window ending at the row."""
    o = cuda.grid(1)
    n = results.shape[0]
    if o >= n or o < window_frm - 1:
        return
    left = o + 1 - window_frm
    msum = 0.0
    for k in range(left, o + 1):
        msum += mass[k]
    mass_mean = msum / window_frm
    ndiff = window_frm - 1
    if ndiff <= 0:
        results[o] = _NAN
        return
    dsum = 0.0
    for k in range(left, o):
        dx = x[k + 1, 0] - x[k, 0]
        dy = x[k + 1, 1] - x[k, 1]
        dsum += math.sqrt(dx * dx + dy * dy)
    speed = (dsum * inv_dt) / ndiff
    results[o] = mass_mean * speed


def sliding_momentum_magnitude_cuda(x: np.ndarray, mass: np.ndarray, sample_rate: float, time_window: float) -> np.ndarray:
    """
    Compute the sliding momentum magnitude of a moving 2D point over time windows, on the GPU.

    Momentum magnitude is ``mean(mass) * mean(speed)`` within each window, where ``speed = |delta position| * sample_rate``.

    .. note::
       Output is a 1D float32 array of length n_frames; frames before the first full window are -1. The window
       length is ``ceil(sample_rate * time_window)``. Matches the CPU version (float64 computation).

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_momentum_magnitude_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU version: :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_momentum_magnitude`.

    :param np.ndarray x: 2D array (n_frames, 2) of point (x, y) coordinates.
    :param np.ndarray mass: 1D array (n_frames,) of per-frame mass.
    :param float sample_rate: Samples per second (may be fractional).
    :param float time_window: Window size in seconds.
    :return: (n_frames,) float32 array of momentum magnitude per window.
    :rtype: np.ndarray

    :example:

    >>> x = np.random.randint(0, 500, (5000, 2)).astype(np.float32)
    >>> mass = np.random.randint(10, 100, (5000,)).astype(np.float32)
    >>> sliding_momentum_magnitude_cuda(x=x, mass=mass, sample_rate=30.0, time_window=1.0)
    """
    check_valid_array(data=x, source=f'{sliding_momentum_magnitude_cuda.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=mass, source=f'{sliding_momentum_magnitude_cuda.__name__} mass', accepted_ndims=(1,), accepted_shapes=[(x.shape[0],)], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_momentum_magnitude_cuda.__name__} sample_rate', value=sample_rate, min_value=10e-6)
    check_float(name=f'{sliding_momentum_magnitude_cuda.__name__} time_window', value=time_window, min_value=10e-6)
    n = x.shape[0]
    window_frm = int(math.ceil(sample_rate * time_window))
    inv_dt = float(sample_rate)
    x_dev = cuda.to_device(np.ascontiguousarray(x).astype(np.float64))
    mass_dev = cuda.to_device(np.ascontiguousarray(mass).astype(np.float64))
    results = cuda.to_device(np.full(n, -1.0, dtype=np.float32))
    bpg = math.ceil(n / THREADS_PER_BLOCK)
    _sliding_momentum_kernel[bpg, THREADS_PER_BLOCK](x_dev, mass_dev, window_frm, inv_dt, results)
    return results.copy_to_host()


@cuda.jit()
def _sliding_variance_kernel(data, window_sizes, results):
    """One thread per (frame idx, window wi). results[idx,wi] = population variance of the window ending at idx."""
    idx, wi = cuda.grid(2)
    n = data.shape[0]
    if idx >= n or wi >= window_sizes.shape[0]:
        return
    w = window_sizes[wi]
    if w < 1 or idx < w - 1:
        return
    left = idx - w + 1
    m = 0.0
    for k in range(left, idx + 1):
        m += data[k]
    m /= w
    s = 0.0
    for k in range(left, idx + 1):
        d = data[k] - m
        s += d * d
    results[idx, wi] = s / w


def sliding_variance_cuda(data: np.ndarray, window_sizes: np.ndarray, sample_rate: float) -> np.ndarray:
    """
    Compute the sliding population variance of a 1D signal over one or more window sizes, on the GPU.

    .. note::
       Output is (n_frames, n_window_sizes) float32; frames before the first full window are -1. Population
       variance (ddof=0), matching ``np.var``.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_variance_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numba) version: :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_variance`.

    :param np.ndarray data: 1D signal.
    :param np.ndarray window_sizes: 1D array of window sizes in seconds.
    :param float sample_rate: Samples per second (may be fractional).
    :return: (n_frames, n_window_sizes) float32 array of sliding variance values.
    :rtype: np.ndarray

    :example:

    >>> data = np.array([1, 2, 3, 1, 2, 9, 17, 2, 10, 4]).astype(np.float32)
    >>> sliding_variance_cuda(data=data, window_sizes=np.array([0.5]), sample_rate=10.0)
    """
    check_valid_array(data=data, source=f'{sliding_variance_cuda.__name__} data', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=window_sizes, source=f'{sliding_variance_cuda.__name__} window_sizes', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_variance_cuda.__name__} sample_rate', value=sample_rate, min_value=10e-6)
    n, nw = data.shape[0], window_sizes.shape[0]
    if nw > MAX_GRID_Y:
        raise SimBAGPUError(msg=f'{sliding_variance_cuda.__name__} supports at most {MAX_GRID_Y} window_sizes (CUDA grid limit); got {nw}.', source=sliding_variance_cuda.__name__)
    ws = (np.asarray(window_sizes) * sample_rate).astype(np.int32)
    data_dev = cuda.to_device(np.ascontiguousarray(data).astype(np.float64))
    ws_dev = cuda.to_device(ws)
    results = cuda.to_device(np.full((n, nw), -1.0, dtype=np.float32))
    tpb = (128, 1)
    bpg = (math.ceil(n / tpb[0]), math.ceil(nw / tpb[1]))
    _sliding_variance_kernel[bpg, tpb](data_dev, ws_dev, results)
    return results.copy_to_host()
