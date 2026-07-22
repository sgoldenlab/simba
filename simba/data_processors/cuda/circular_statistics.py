__author__ = "Simon Nilsson; sronilsson@gmail.com"

import math
from typing import Optional, Tuple

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
from numba import cuda, int32

try:
    import cupy as cp
except:
    import numpy as cp

from simba.data_processors.cuda.utils import _cuda_bubble_sort
from simba.utils.checks import check_float, check_int, check_valid_array
from simba.utils.enums import Formats
from simba.utils.errors import SimBAGPUError

THREADS_PER_BLOCK = 1024
MAX_GRID_Y = 65535
MAX_WIN = 512        # per-thread sort-buffer cap (sliding_kuipers_two_sample_test)

@cuda.jit()
def _cuda_direction_from_two_bps(x, y, results):
    i = cuda.grid(1)
    if i > x.shape[0]:
        return
    else:
        a = math.atan2(x[i][0] - y[i][0], y[i][1] - x[i][1]) * (180 / math.pi)
        a = int32(a + 360 if a < 0 else a)
        results[i] = a


def direction_from_two_bps(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute the directionality in degrees from two body-parts. E.g., ``nape`` and ``nose``,
    or ``swim_bladder`` and ``tail`` with GPU acceleration.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/direction_two_bps.csv
       :widths: 10, 90
       :align: center
       :header-rows: 1

    .. seealso::
       For CPU function see :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.direction_two_bps`.

    .. image:: _static/img/angle_from_2_bps.png
       :alt: Angle from 2 bps
       :width: 600
       :align: center

    :param np.ndarray x: Size len(frames) x 2 representing x and y coordinates for first body-part.
    :param np.ndarray y: Size len(frames) x 2 representing x and y coordinates for second body-part.
    :return: Frame-wise directionality in degrees.
    :rtype: np.ndarray.

    """
    x = np.ascontiguousarray(x).astype(np.int32)
    y = np.ascontiguousarray(y).astype(np.int32)
    x_dev = cuda.to_device(x)
    y_dev = cuda.to_device(y)
    results = cuda.device_array((x.shape[0]), dtype=np.int32)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _cuda_direction_from_two_bps[bpg, THREADS_PER_BLOCK](x_dev, y_dev, results)
    results = results.copy_to_host()
    return results


def sliding_circular_hotspots(x: np.ndarray,
                              time_window: float,
                              sample_rate: float,
                              bins: np.ndarray,
                              batch_size: Optional[int] = int(3.5e+7)) -> np.ndarray:
    """
    Calculate the proportion of data points falling within specified circular bins over a sliding time window using GPU

    This function processes time series data representing angles (in degrees) and calculates the proportion of data
    points within specified angular bins over a sliding window. The calculations are performed in batches to
    accommodate large datasets efficiently.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_circular_hotspots.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    .. image:: _static/img/sliding_circular_hotspot.png
       :alt: Sliding circular hotspot
       :width: 300
       :align: center

    .. seealso::
       For CPU function see :func:`~simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_hotspots`.


    :param np.ndarray x: The input time series data in degrees. Should be a 1D numpy array.
    :param float time_window: The size of the sliding window in seconds.
    :param float sample_rate: The sample rate of the time series data (i.e., hz, fps).
    :param ndarray bins: 2D array of shape representing circular bins defining [start_degree, end_degree] inclusive.
    :param Optional[int] batch_size: The size of each batch for processing the data. Default is 5e+7 (50m).
    :return: A 2D numpy array where each row corresponds to a time point in `data`, and each column represents a circular bin. The values in the array represent the proportion of data points within each bin at each time point. The first column represents the first bin.
    :rtype: np.ndarray
    """

    n = x.shape[0]
    x = cp.asarray(x, dtype=cp.float16)
    results = cp.full((x.shape[0], bins.shape[0]), dtype=cp.float16, fill_value=-1)
    window_size = int(cp.ceil(time_window * sample_rate))
    for cnt, left in enumerate(range(0, n, batch_size)):
        right = int(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size + 1
        x_batch = x[left:right]
        x_batch = cp.lib.stride_tricks.sliding_window_view(x_batch, window_size).astype(cp.float16)
        batch_results = cp.full((x_batch.shape[0], bins.shape[0]), dtype=cp.float16, fill_value=-1)
        for bin_cnt in range(bins.shape[0]):
            if bins[bin_cnt][0] > bins[bin_cnt][1]:
                mask = ((x_batch >= bins[bin_cnt][0]) & (x_batch <= 360)) | ((x_batch >= 0) & (x_batch <= bins[bin_cnt][1]))
            else:
                mask = (x_batch >= bins[bin_cnt][0]) & (x_batch <= bins[bin_cnt][1])
            count_per_row = cp.array(mask.sum(axis=1) / window_size).reshape(-1, )
            batch_results[:, bin_cnt] = count_per_row
        results[left + window_size - 1:right, ] = batch_results
    return results.get()

def sliding_circular_mean(x: np.ndarray,
                          time_window: float,
                          sample_rate: int,
                          batch_size: Optional[int] = 3e+7) -> np.ndarray:

    r"""
    Calculate the sliding circular mean over a time window for a series of angles.

    This function computes the circular mean of angles in the input array `x` over a specified sliding window.
    The circular mean is a measure of the average direction for angles, which is especially useful for angular data
    where traditional averaging would not be meaningful due to the circular nature of angles (e.g., 359° and 1° should average to 0°).

    The calculation is performed using a sliding window approach, where the circular mean is computed for each window
    of angles. The function leverages GPU acceleration via CuPy for efficiency when processing large datasets.

    The circular mean :math:`\mu` for a set of angles is calculated using the following formula:

    .. math::

        \mu = \text{atan2}\left(\frac{1}{N} \sum_{i=1}^{N} \sin(\theta_i), \frac{1}{N} \sum_{i=1}^{N} \cos(\theta_i)\right)

    - :math:`\theta_i` are the angles in radians within the sliding window
    - :math:`N` is the number of samples in the window


    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_circular_mean.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    .. seealso::
       For CPU function see :func:`~simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_mean`.

    :param np.ndarray x: Input array containing angle values in degrees. The array should be 1-dimensional.
    :param float time_window: Time duration for the sliding window, in seconds. This determines the number of samples in each window  based on the `sample_rate`.
    :param int sample_rate: The number of samples per second (i.e., FPS). This is used to calculate the window size in terms of array indices.
    :param Optional[int] batch_size: The maximum number of elements to process in each batch. This is used to handle large arrays by processing them in chunks to avoid memory overflow. Defaults to 3e+7 (30 million elements).
    :return np.ndarray: A 1D numpy array of the same length as `x`, containing the circular mean for each sliding window.  Values before the window is fully populated will be set to -1.

    :example:

    >>> x = np.random.randint(0, 361, (i, )).astype(np.int32)
    >>> results = sliding_circular_mean(x, 1, 10)
    """


    window_size = np.ceil(time_window * sample_rate).astype(np.int64)
    n = x.shape[0]
    results = cp.full(x.shape[0], -1, dtype=np.int32)
    for cnt, left in enumerate(range(0, int(n), int(batch_size))):
        right = np.int32(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size+1
        x_batch = cp.asarray(x[left:right])
        x_batch = cp.lib.stride_tricks.sliding_window_view(x_batch, window_size)
        x_batch = np.deg2rad(x_batch)
        cos, sin = cp.cos(x_batch).astype(np.float32), cp.sin(x_batch).astype(np.float32)
        r = cp.rad2deg(cp.arctan2(cp.mean(sin, axis=1), cp.mean(cos, axis=1)))
        r = cp.where(r < 0, r + 360, r)
        results[left + window_size - 1:right] = r
    return results.get()



def sliding_circular_range(x: np.ndarray,
                          time_window: float,
                          sample_rate: float,
                          batch_size: Optional[int] = int(5e+7)) -> np.ndarray:
    r"""
    Computes the sliding circular range of a time series data array using GPU.

    This function calculates the circular range of a time series data array using a sliding window approach.
    The input data is assumed to be in degrees, and the function handles the circular nature of the data
    by considering the circular distance between angles.

    .. math::

       R = \min \left( \text{max}(\Delta \theta) - \text{min}(\Delta \theta), \, 360 - \text{max}(\Delta \theta) + \text{min}(\Delta \theta) \right)

    where:

    - :math:`\Delta \theta` is the difference between angles within the window,
    - :math:`360` accounts for the circular nature of the data (i.e., wrap-around at 360 degrees).

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_circular_range.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    .. seealso::
       For CPU function see :func:`~simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_range`.

    :param np.ndarray x: The input time series data in degrees. Should be a 1D numpy array.
    :param float time_window: The size of the sliding window in seconds.
    :param float sample_rate: The sample rate of the time series data (i.e., hz, fps).
    :param Optional[int] batch_size: The size of each batch for processing the data. Default is 5e+7 (50m).
    :return: A numpy array containing the sliding circular range values.
    :rtype: np.ndarray

    :example:

    >>> x = np.random.randint(0, 361, (19, )).astype(np.int32)
    >>> p = sliding_circular_range(x, 1, 10)
    """

    n = x.shape[0]
    x = cp.asarray(x, dtype=cp.float16)
    results = cp.zeros_like(x, dtype=cp.int16)
    x = cp.deg2rad(x).astype(cp.float16)
    window_size = int(cp.ceil(time_window * sample_rate))
    for cnt, left in enumerate(range(0, n, batch_size)):
        right = int(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size + 1
        x_batch = x[left:right]
        x_batch = cp.lib.stride_tricks.sliding_window_view(x_batch, window_size).astype(cp.float16)
        x_batch = cp.sort(x_batch)
        results[left + window_size - 1:right] = cp.abs(cp.rint(cp.rad2deg(cp.amin(cp.vstack([x_batch[:, -1] - x_batch[:, 0], 2 * cp.pi - cp.max(cp.diff(x_batch), axis=1)]).T, axis=1))))
    return results.get()




def sliding_circular_std(x: np.ndarray,
                         time_window: float,
                         sample_rate: float,
                         batch_size: Optional[int] = int(5e+7)) -> np.ndarray:

    r"""
    Calculate the sliding circular standard deviation of a time series data on GPU.

    This function computes the circular standard deviation over a sliding window for a given time series array.
    The time series data is assumed to be in degrees, and the function converts it to radians for computation.
    The sliding window approach is used to handle large datasets efficiently, processing the data in batches.

    The circular standard deviation (σ) is computed using the formula:

    .. math::

       \sigma = \sqrt{-2 \cdot \log \left|\text{mean}\left(\exp(i \cdot x_{\text{batch}})\right)\right|}

    where :math:`x_{\text{batch}}` is the data within the current sliding window, and :math:`\text{mean}` and
    :math:`\log` are computed in the circular (complex plane) domain.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_circular_std.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    .. seealso::
       For CPU function see :func:`~simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_std`.

    :param np.ndarray x: The input time series data in degrees. Should be a 1D numpy array.
    :param float time_window: The size of the sliding window in seconds.
    :param float sample_rate: The sample rate of the time series data (i.e., hz, fps).
    :param Optional[int] batch_size: The size of each batch for processing the data. Default is 5e+7 (50m).

    :return: A numpy array containing the sliding circular standard deviation values.
    :rtype: np.ndarray
    """


    n = x.shape[0]
    x = cp.asarray(x, dtype=cp.float16)
    results = cp.zeros_like(x, dtype=cp.float16)
    x = np.deg2rad(x).astype(cp.float16)
    window_size = int(np.ceil(time_window * sample_rate))
    for cnt, left in enumerate(range(0, n, batch_size)):
        right = int(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size + 1
        x_batch = x[left:right]
        x_batch = cp.lib.stride_tricks.sliding_window_view(x_batch, window_size).astype(cp.float16)
        m = cp.log(cp.abs(cp.mean(cp.exp(1j * x_batch), axis=1)))
        stdev = cp.rad2deg(cp.sqrt(-2 * m))
        results[left + window_size - 1:right] = stdev

    return results.get()


def sliding_rayleigh_z(x: np.ndarray,
                       time_window: float,
                       sample_rate: float,
                       batch_size: Optional[int] = int(5e+7)) -> Tuple[np.ndarray, np.ndarray]:

    r"""
    Computes the Rayleigh Z-statistic over a sliding window for a given time series of angles

    This function calculates the Rayleigh Z-statistic, which tests the null hypothesis that the population of angles
    is uniformly distributed around the circle. The calculation is performed over a sliding window across the input
    time series, and results are computed in batches for memory efficiency.

    Data is processed using GPU acceleration via CuPy, which allows for faster computation compared to a CPU-based approach.

    .. note::
        Adapted from ``pingouin.circular.circ_rayleigh`` and ``pycircstat.tests.rayleigh``.


    **Rayleigh Z-statistic:**

    The Rayleigh Z-statistic is given by:

    .. math::

       R = \frac{1}{n} \sqrt{\left(\sum_{i=1}^{n} \cos(\theta_i)\right)^2 + \left(\sum_{i=1}^{n} \sin(\theta_i)\right)^2}

    where:
    - :math:`\theta_i` are the angles in the window.
    - :math:`n` is the number of angles in the window.


    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_rayleigh_z.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    .. seealso::
       For CPU function see :func:`~simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_rayleigh_z`.


    :param np.ndarray x: Input array of angles in degrees. Should be a 1D numpy array.
    :param float time_window: The size of the sliding window in time units (e.g., seconds).
    :param float sample_rate: The sampling rate of the input time series in samples per time unit (e.g., Hz, fps).
    :param Optional[int] batch_size: The number of samples to process in each batch. Default is 5e7 (50m). Reducing this value may save memory at the cost of longer computation time.
    :return:
       A tuple containing two numpy arrays:
       - **z_results**: Rayleigh Z-statistics for each position in the input array where the window was fully applied.
       - **p_results**: Corresponding p-values for the Rayleigh Z-statistics.
    :rtype: Tuple[np.ndarray, np.ndarray]
    """

    n = x.shape[0]
    x = cp.asarray(x, dtype=cp.float16)
    z_results = cp.zeros_like(x, dtype=cp.float16)
    p_results = cp.zeros_like(x, dtype=cp.float16)
    x = np.deg2rad(x).astype(cp.float16)
    window_size = int(np.ceil(time_window * sample_rate))
    for cnt, left in enumerate(range(0, n, batch_size)):
        right = int(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size + 1
        x_batch = x[left:right]
        x_batch = cp.lib.stride_tricks.sliding_window_view(x_batch, window_size).astype(cp.float16)
        cos_sums = cp.nansum(cp.cos(x_batch), axis=1) ** 2
        sin_sums = cp.nansum(cp.sin(x_batch), axis=1) ** 2
        R = cp.sqrt(cos_sums + sin_sums) / window_size
        Z = window_size * (R**2)
        P = cp.exp(np.sqrt(1 + 4 * window_size + 4 * (window_size ** 2 - R ** 2)) - (1 + 2 * window_size))
        z_results[left + window_size - 1:right] = Z
        p_results[left + window_size - 1:right] = P

    return z_results.get(), p_results.get()


def sliding_resultant_vector_length(x: np.ndarray,
                                    time_window: float,
                                    sample_rate: int,
                                    batch_size: Optional[int] = 3e+7) -> np.ndarray:

    r"""
    Calculate the sliding resultant vector length over a time window for a series of angles.

    This function computes the resultant vector length (R) for each window of angles in the input array `x`.
    The resultant vector length is a measure of the concentration of angles, and it ranges from 0 to 1, where 1
    indicates all angles point in the same direction, and 0 indicates uniform distribution of angles.

    For a given sliding window of angles, the resultant vector length :math:`R` is calculated using the following formula:

    .. math::

        R = \frac{1}{N} \sqrt{\left(\sum_{i=1}^{N} \cos(\theta_i)\right)^2 + \left(\sum_{i=1}^{N} \sin(\theta_i)\right)^2}

    where:

    - :math:`\theta_i` are the angles in radians within the sliding window
    - :math:`N` is the number of samples in the window

    The computation is performed in a sliding window manner over the entire array, utilizing GPU acceleration
    with CuPy for efficiency, especially on large datasets.


    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_resultant_vector_length.csv
       :widths: 10, 10, 80
       :align: center
       :header-rows: 1

    .. seealso::
       For CPU function see :func:`~simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_resultant_vector_length`.

    :param np.ndarray x: Input array containing angle values in degrees. The array should be 1-dimensional.
    :param float time_window: Time duration for the sliding window, in seconds. This determines the number of samples in each window  based on the `sample_rate`.
    :param int sample_rate: The number of samples per second (i.e., FPS). This is used to calculate the window size in terms of array indices.
    :param Optional[int] batch_size: The maximum number of elements to process in each batch. This is used to handle large arrays by processing them in chunks to avoid memory overflow. Defaults to 3e+7 (30 million elements).
    :return np.ndarray: A 1D numpy array of the same length as `x`, containing the resultant vector length for each sliding window. Values before the window is fully populated will be set to -1.

    :example:

    >>> x = np.random.randint(0, 361, (5000, )).astype(np.int32)
    >>> results = sliding_resultant_vector_length(x, 1, 10)
    """

    window_size = np.ceil(time_window * sample_rate).astype(np.int64)
    n = x.shape[0]
    results = cp.full(x.shape[0], -1, dtype=np.float32)
    for cnt, left in enumerate(range(0, int(n), int(batch_size))):
        right = np.int32(min(left + batch_size, n))
        if cnt > 0:
            left = left - window_size+1
        x_batch = cp.asarray(x[left:right])
        x_batch = cp.lib.stride_tricks.sliding_window_view(x_batch, window_size)
        x_batch = np.deg2rad(x_batch)
        cos, sin = cp.cos(x_batch).astype(np.float32), cp.sin(x_batch).astype(np.float32)
        cos_sum, sin_sum = cp.sum(cos, axis=1), cp.sum(sin, axis=1)
        r = np.sqrt(cos_sum ** 2 + sin_sum ** 2) / window_size
        results[left+window_size-1:right] = r
    return results.get()


def direction_from_three_bps(x: np.ndarray,
                             y: np.ndarray,
                             z: np.ndarray,
                             batch_size: Optional[int] = int(1.5e+7)) -> np.ndarray:

    """
    Calculate the direction angle based on the coordinates of three body points using GPU acceleration.

    This function computes the mean direction angle (in degrees) for a batch of coordinates
    provided in the form of NumPy arrays. The calculation is based on the arctangent of the
    difference in x and y coordinates between pairs of points. The result is a value in
    the range [0, 360) degrees.

    .. seealso::
       * More CPU function, see :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.direction_three_bps`

    .. image:: _static/img/angle_from_3_bps.png
       :alt: Angle from 3 bps
       :width: 300
       :align: center

    :param np.ndarray x: A 2D array of shape (N, 2) containing the x-coordinates of the first body part  (nose)
    :param np.ndarray y: A 2D array of shape (N, 2) containing the coordinates of the second body part (left ear).
    :param np.ndarray z: A 2D array of shape (N, 2) containing the coordinates of the second body part (right ear).
    :param Optional[int] batch_size: The size of the batch to be processed in each iteration. Default is 15 million.
    :return: An array of shape (N,) containing the computed direction angles in degrees.
    :rtype: np.ndarray
    """

    check_valid_array(data=x, source=direction_from_three_bps.__name__, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=y, source=direction_from_three_bps.__name__, accepted_shapes=(x.shape,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=z, source=direction_from_three_bps.__name__, accepted_shapes=(x.shape,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_int(value=batch_size, name=direction_from_three_bps.__name__, min_value=1)
    results = cp.full((x.shape[0]), fill_value=-1, dtype=np.int16)

    for l in range(0, x.shape[0], batch_size):
        r = l + batch_size
        x_batch = cp.array(x[l:r])
        y_batch = cp.array(y[l:r])
        z_batch = cp.array(z[l:r])
        left_ear_to_nose = cp.arctan2(x_batch[:, 0] - y_batch[:, 0], y_batch[:, 1] - x_batch[:,1])
        right_ear_nose = cp.arctan2(x_batch[:, 0] - z_batch[:, 0], z_batch[:, 1] - x_batch[:, 1])
        mean_angle_rad = cp.arctan2(cp.sin(left_ear_to_nose) + cp.sin(right_ear_nose), cp.cos(left_ear_to_nose) + cp.cos(right_ear_nose))
        results[l:r] = (cp.degrees(mean_angle_rad) + 360) % 360

    return results.get()


@cuda.jit()
def _instantaneous_angular_velocity(x, stride, results):
    r = cuda.grid(1)
    l = np.int32(r - (stride[0]))
    if (r > results.shape[0]) or (l < 0):
        results[r] = -1
    else:
        d = math.pi - (abs(math.pi - abs(x[l] - x[r])))
        results[r] = d * (180 / math.pi)


def instantaneous_angular_velocity(x: np.ndarray, stride: Optional[int] = 1) -> np.ndarray:
    r"""
    Calculate the instantaneous angular velocity between angles in a given array.

    This function uses CUDA to perform parallel computations on the GPU.

    The angular velocity is computed using the difference in angles between
    the current and previous values (with a specified stride) in the array.
    The result is returned in degrees per unit time.

    .. image:: _static/img/instantaneous_angular_velocity.png
       :alt: Instantaneous angular velocity
       :width: 400
       :align: center

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/instantaneous_angular_velocity.csv
       :widths: 10, 90
       :align: center
       :header-rows: 1

    .. math::
       \omega = \frac{{\Delta \theta}}{{\Delta t}} = \frac{{180}}{{\pi}} \times \left( \pi - \left| \pi - \left| \theta_r - \theta_l \right| \right| \right)

    where:
    - :math:`\theta_r` is the current angle.
    - :math:`\theta_l` is the angle at the specified stride before the current angle.
    - :math:`\Delta t` is the time difference between the two angles.


    .. seealso::
       For CPU function, see :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.instantaneous_angular_velocity`

    :param np.ndarray x: Array of angles in degrees, for which the instantaneous angular velocity will be calculated.
    :param Optional[int] stride: The stride or lag (in frames) to use when calculating the difference in angles. Defaults to 1.
    :return: Array of instantaneous angular velocities corresponding to the input angles. Velocities are in degrees per unit time.
    :rtype: np.ndarray
    """

    x = np.deg2rad(x).astype(np.int16)
    stride = np.array([stride]).astype(np.int64)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    x_dev = cuda.to_device(x)
    stride_dev = cuda.to_device(stride)
    results = cuda.device_array(x.shape[0], dtype=np.float32)
    _instantaneous_angular_velocity[bpg, THREADS_PER_BLOCK](x_dev, stride_dev, results)
    return results.copy_to_host()


@cuda.jit(device=True)
def _rad2deg(x):
    return x * (180/math.pi)


@cuda.jit()
def _sliding_bearing(x, stride, results):
    r = cuda.grid(1)
    l = np.int32(r - (stride[0]))
    if (r > results.shape[0]-1) or (l < 0):
        results[r] = -1
    else:
        x1, y1 = x[l, 0], x[l, 1]
        x2, y2 = x[r, 0], x[r, 1]
        bearing = _rad2deg(math.atan2(x2 - x1, y2 - y1))
        results[r] = (bearing + 360) % 360


def sliding_bearing(x: np.ndarray,
                    stride: Optional[float] = 1,
                    sample_rate: Optional[float] = 1) -> np.ndarray:
    """
    Compute the bearing between consecutive points in a 2D coordinate array using a sliding window approach using GPU acceleration.

    This function calculates the angle (bearing) in degrees between each point and a point a certain number of
    steps ahead (defined by `stride`) in the 2D coordinate array `x`. The bearing is calculated using the
    arctangent of the difference in coordinates, converted from radians to degrees.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_bearing.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1

    .. image:: _static/img/sliding_bearing.png
       :alt: Sliding bearing
       :width: 300
       :align: center

    .. seealso::
       :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_bearing`

    :param np.ndarray x: A 2D array of shape `(n, 2)` where each row represents a point with `x` and `y` coordinates. The array must be numeric.
    :param Optional[float] stride: The time (multiplied by `sample_rate`) to look ahead when computing the bearing in seconds. Defaults to 1.
    :param Optional[float] sample_rate: A multiplier applied to the `stride` value to determine the actual step size for calculating the bearing. E.g., frames per second. Defaults to 1. If the resulting stride is less than 1, it is automatically set to 1.
    :return:A 1D array of shape `(n,)` containing the calculated bearings in degrees. Values outside the valid range (i.e., where the stride exceeds array bounds) are set to -1.
    :rtype: np.ndarray
    """

    check_valid_array(data=x, source=f'{sliding_bearing.__name__} x', accepted_ndims=(2,), accepted_axis_1_shape=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_bearing.__name__} stride', value=stride, min_value=10e-6, max_value=x.shape[0]-1)
    check_float(name=f'{sliding_bearing.__name__} sample_rate', value=sample_rate, min_value=10e-6, max_value=x.shape[0]-1)
    stride = int(stride * sample_rate)
    if stride < 1:
        stride = 1
    stride = np.array([stride]).astype(np.int64)
    bpg = (x.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    x_dev = cuda.to_device(x)
    stride_dev = cuda.to_device(stride)
    results = cuda.device_array(x.shape[0], dtype=np.float32)
    _sliding_bearing[bpg, THREADS_PER_BLOCK](x_dev, stride_dev, results)
    return results.copy_to_host()


@cuda.jit(device=True)
def _rad2deg(x):
    return x * (180 / math.pi)


@cuda.jit()
def _sliding_angular_diff(data, strides, results):
    x, y = cuda.grid(2)
    if (x > data.shape[0] - 1) or (y > strides.shape[0] - 1):
        return
    else:
        stride = int(strides[y])
        if x - stride < 0:
            return
        a_2 = data[x]
        a_1 = data[x - stride]
        distance = math.pi - abs(math.pi - abs(a_1 - a_2))
        distance = abs(int(_rad2deg(distance)) + 1)
        results[x][y] = distance


def sliding_angular_diff(x: np.ndarray,
                         time_windows: np.ndarray,
                         fps: float) -> np.ndarray:
    r"""
    Calculate the sliding angular differences for a given time window using GPU acceleration.


    This function computes the angular differences between each angle in `x`
    and the corresponding angle located at a distance determined by the time window
    and frame rate (fps). The results are returned as a 2D array where each row corresponds
    to a position in `x`, and each column corresponds to a different time window.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_angular_diff.csv
       :widths: 10, 45, 45
       :align: center
       :header-rows: 1


    .. seealso::
       * For CPU function, see :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_angular_diff`

    .. math::
       \text{difference} = \pi - |\pi - |a_1 - a_2||

    Where:
    - :math:`a_1` is the angle at position `x`.
    - :math:`a_2` is the angle at position `x - \text{stride}`.

    :param np.ndarray x: 1D array of angles in degrees.
    :param np.ndarray time_windows: 1D array of time windows in seconds to determine the stride (distance in frames) between angles.
    :param float fps: Frame rate (frames per second) used to convert time windows to strides.
    :return: 2D array of angular differences. Each row corresponds to an angle in `x`, and each column corresponds to a time window.
    :rtype: np.ndarray
    """

    x = np.deg2rad(x)
    strides = np.zeros(time_windows.shape[0])
    for i in range(time_windows.shape[0]):
        strides[i] = np.ceil(time_windows[i] * fps).astype(np.int32)
    x_dev = cuda.to_device(x)
    stride_dev = cuda.to_device(strides)
    results = cuda.device_array((x.shape[0], time_windows.shape[0]))
    grid_x = (x.shape[0] + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    grid_y = (strides.shape[0] + THREADS_PER_BLOCK - 1)
    blocks_per_grid = (grid_x, grid_y)
    _sliding_angular_diff[blocks_per_grid, THREADS_PER_BLOCK](x_dev, stride_dev, results)
    results = results.copy_to_host().astype(np.int32)
    return results

@cuda.jit()
def _rotational_direction(data, stride, results):
    r = cuda.grid(1)
    l = int(r - stride[0])
    if (r < 0) or (r > data.shape[0] - 1):
        return
    elif (l < 0):
        return
    else:
        l_val, r_val = data[l], data[r]
        angle_diff = r_val - l_val
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        if angle_diff == 0:
            results[r] = 0
        elif angle_diff > 0:
            results[r] = 1
        else:
            results[r] = 2

def rotational_direction(data: np.ndarray, stride: Optional[int] = 1) -> np.ndarray:
    """
    Computes the rotational direction between consecutive data points in a circular space, where the angles wrap
    around at 360 degrees. The function uses GPU acceleration via CUDA to process the data in parallel.

    The result array contains values:

    * `0` where there is no change between points.
    * `1` where the angle has increased in the positive direction.
    * `2` where the angle has decreased in the negative direction.

    .. seealso::
       :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.rotational_direction` for jitted CPU method.

    .. image:: _static/img/rotational_direction.png
       :alt: Rotational direction
       :width: 400
       :align: center

    :param np.ndarray data: 1D array of angular data (in degrees) to analyze. The data will be internally converted to radians and wrapped between [0, 360) degrees before processing.
    :param Optional[int] stride: The stride or gap between data points for which the rotational direction is calculated. Default is 1.
    :return: A 1D array of integers of the same length as `data`, where each element indicates the rotational direction between the current and previous point based on the stride. The first `stride` elements in the result will  be initialized to -1 since they cannot be compared.
    :rtype: np.ndarray

    :example:

    >>> data = np.random.randint(0, 365, (100))
    >>> p = rotational_direction(data=data)
    """
    data = np.deg2rad(data % 360)
    results = np.full((data.shape[0]), fill_value=-1, dtype=np.int16)
    results_dev = cuda.to_device(results)
    stride = np.array([stride])
    stride_dev = cuda.to_device(stride)
    data_dev = cuda.to_device(data)
    bpg = (data.shape[0] + (THREADS_PER_BLOCK - 1)) // THREADS_PER_BLOCK
    _rotational_direction[bpg, THREADS_PER_BLOCK](data_dev, stride_dev, results_dev)
    return results_dev.copy_to_host()


@cuda.jit()
def _sliding_circular_corr_kernel(s1, s2, window_sizes, results):
    """One thread per (frame idx, time-window wi). ``results`` (N, n_windows) = |circular cross-correlation| per window."""
    idx, wi = cuda.grid(2)
    n = s1.shape[0]
    if idx >= n or wi >= window_sizes.shape[0]:
        return
    w = window_sizes[wi]
    if w < 1 or idx < w - 1:
        return
    left = idx - w + 1
    ss1 = 0.0; sc1 = 0.0; ss2 = 0.0; sc2 = 0.0
    for k in range(left, idx + 1):
        ss1 += math.sin(s1[k]); sc1 += math.cos(s1[k])
        ss2 += math.sin(s2[k]); sc2 += math.cos(s2[k])
    m1 = math.atan2(ss1, sc1)
    m2 = math.atan2(ss2, sc2)
    num = 0.0; d1 = 0.0; d2 = 0.0
    for k in range(left, idx + 1):
        a = math.sin(s1[k] - m1); b = math.sin(s2[k] - m2)
        num += a * b; d1 += a * a; d2 += b * b
    denom = math.sqrt(d1 * d2)
    if denom != 0.0:
        val = num / denom
        if val < 0.0:
            val = -val
        results[idx, wi] = val


def sliding_circular_correlation_cuda(sample_1: np.ndarray,
                                      sample_2: np.ndarray,
                                      time_windows: np.ndarray,
                                      fps: float) -> np.ndarray:
    """
    Compute the circular correlation coefficient between two angular signals over each sliding window
    (for every window length and frame), on the GPU.

    .. note::
       Angular inputs are in degrees. The result at each frame is the absolute circular cross-correlation of
       the window ending at that frame; incomplete windows (before the first full window) and zero-denominator
       windows are 0.

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_circular_correlation_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numba) version: :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_correlation`.

    :param np.ndarray sample_1: First 1D angular signal in degrees.
    :param np.ndarray sample_2: Second 1D angular signal in degrees (same length as ``sample_1``).
    :param np.ndarray time_windows: 1D array of window lengths in seconds.
    :param float fps: Frames per second (converts window lengths in seconds to frames). May be fractional (e.g. 29.97).
    :return: 2D float32 array (len(sample_1), len(time_windows)) of circular correlation per window.
    :rtype: np.ndarray

    :example:

    >>> s1 = np.random.randint(0, 361, (200,)).astype(np.float32)
    >>> s2 = np.random.randint(0, 361, (200,)).astype(np.float32)
    >>> sliding_circular_correlation_cuda(sample_1=s1, sample_2=s2, time_windows=np.array([0.5, 1.0]), fps=10.0)
    """
    check_valid_array(data=sample_1, source=f'{sliding_circular_correlation_cuda.__name__} sample_1', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=sample_2, source=f'{sliding_circular_correlation_cuda.__name__} sample_2', accepted_ndims=(1,), accepted_shapes=[(sample_1.shape[0],)], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=time_windows, source=f'{sliding_circular_correlation_cuda.__name__} time_windows', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_circular_correlation_cuda.__name__} fps', value=fps, min_value=10e-6)
    n, nw = sample_1.shape[0], time_windows.shape[0]
    if nw > MAX_GRID_Y:
        raise SimBAGPUError(msg=f'{sliding_circular_correlation_cuda.__name__} supports at most {MAX_GRID_Y} time_windows (CUDA grid limit); got {nw}.', source=sliding_circular_correlation_cuda.__name__)
    window_sizes = (np.asarray(time_windows) * fps).astype(np.int32)
    s1 = cuda.to_device(np.deg2rad(np.ascontiguousarray(sample_1).astype(np.float32)))
    s2 = cuda.to_device(np.deg2rad(np.ascontiguousarray(sample_2).astype(np.float32)))
    ws_dev = cuda.to_device(window_sizes)
    results = cuda.to_device(np.zeros((n, nw), dtype=np.float32))
    tpb = (128, 1)
    bpg = (math.ceil(n / tpb[0]), math.ceil(nw / tpb[1]))
    _sliding_circular_corr_kernel[bpg, tpb](s1, s2, ws_dev, results)
    return results.copy_to_host()


@cuda.jit(device=True)
def _cuda_bisect_left(arr, n, val):
    """Device: count of arr[0:n] strictly less than val (== np.searchsorted(side='left')). arr must be sorted."""
    lo = 0; hi = n
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < val:
            lo = mid + 1
        else:
            hi = mid
    return lo


@cuda.jit()
def _sliding_kuipers_kernel(s1, s2, window_sizes, results):
    """One thread per (frame idx, window wi). Kuiper's V for window s[idx-w:idx], stored at row idx."""
    idx, wi = cuda.grid(2)
    n = s1.shape[0]
    if idx >= n or wi >= window_sizes.shape[0]:
        return
    w = window_sizes[wi]
    if w < 1 or idx < w:
        return
    buf1 = cuda.local.array(shape=512, dtype=np.float32)
    buf2 = cuda.local.array(shape=512, dtype=np.float32)
    left = idx - w
    for i in range(w):
        buf1[i] = s1[left + i]
        buf2[i] = s2[left + i]
    _cuda_bubble_sort(buf1[:w])
    _cuda_bubble_sort(buf2[:w])
    max1 = -1.0e30
    for j in range(w):
        v = _cuda_bisect_left(buf2, w, buf1[j]) / w - j / w
        if v > max1:
            max1 = v
    max2 = -1.0e30
    for j in range(w):
        v = _cuda_bisect_left(buf1, w, buf2[j]) / w - j / w
        if v > max2:
            max2 = v
    results[idx, wi] = max1 + max2


def sliding_kuipers_two_sample_test_cuda(sample_1: np.ndarray,
                                         sample_2: np.ndarray,
                                         time_windows: np.ndarray,
                                         fps: float) -> np.ndarray:
    """
    Compute Kuiper's two-sample test statistic between two circular signals over sliding windows, on the GPU.

    .. note::
       Each window length (``int(time_window * fps)``) must be <= 512 (per-thread sort buffers). Following the CPU
       version, the statistic for window ``sample[i-w:i]`` is stored at row ``i``, and rows before the first full
       window are -1. Output is (n_frames, n_time_windows) float64. Runtime grows with window length (per-thread sort).

    .. csv-table::
       :header: EXPECTED RUNTIMES
       :file: ../../../docs/tables/sliding_kuipers_two_sample_test_cuda.csv
       :widths: 25, 25, 25, 25
       :align: center
       :class: simba-table
       :header-rows: 1

    .. seealso::
       CPU (numba) version: :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_kuipers_two_sample_test`.

    :param np.ndarray sample_1: First 1D circular signal in degrees.
    :param np.ndarray sample_2: Second 1D circular signal in degrees (same length as ``sample_1``).
    :param np.ndarray time_windows: 1D array of window sizes in seconds.
    :param float fps: Sampling rate of the signal (may be fractional).
    :return: (n_frames, n_time_windows) float64 array of Kuiper's V statistics.
    :rtype: np.ndarray

    :example:

    >>> s1 = np.random.randint(0, 360, (5000,)).astype(np.float32)
    >>> s2 = np.random.randint(0, 360, (5000,)).astype(np.float32)
    >>> sliding_kuipers_two_sample_test_cuda(sample_1=s1, sample_2=s2, time_windows=np.array([0.5, 1.0]), fps=30.0)
    """
    check_valid_array(data=sample_1, source=f'{sliding_kuipers_two_sample_test_cuda.__name__} sample_1', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=sample_2, source=f'{sliding_kuipers_two_sample_test_cuda.__name__} sample_2', accepted_ndims=(1,), accepted_shapes=[(sample_1.shape[0],)], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_valid_array(data=time_windows, source=f'{sliding_kuipers_two_sample_test_cuda.__name__} time_windows', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
    check_float(name=f'{sliding_kuipers_two_sample_test_cuda.__name__} fps', value=fps, min_value=10e-6)
    n, nw = sample_1.shape[0], time_windows.shape[0]
    if nw > MAX_GRID_Y:
        raise SimBAGPUError(msg=f'{sliding_kuipers_two_sample_test_cuda.__name__} supports at most {MAX_GRID_Y} time_windows (CUDA grid limit); got {nw}.', source=sliding_kuipers_two_sample_test_cuda.__name__)
    window_sizes = (np.asarray(time_windows) * fps).astype(np.int32)
    if window_sizes.max() > MAX_WIN:
        raise SimBAGPUError(msg=f'{sliding_kuipers_two_sample_test_cuda.__name__} max window ({int(window_sizes.max())} frames) exceeds the max of {MAX_WIN}; reduce time_windows or fps.', source=sliding_kuipers_two_sample_test_cuda.__name__)
    s1 = cuda.to_device(np.deg2rad(np.ascontiguousarray(sample_1).astype(np.float32)))
    s2 = cuda.to_device(np.deg2rad(np.ascontiguousarray(sample_2).astype(np.float32)))
    ws_dev = cuda.to_device(window_sizes)
    results = cuda.to_device(np.full((n, nw), -1.0, dtype=np.float64))
    tpb = (128, 1)
    bpg = (math.ceil(n / tpb[0]), math.ceil(nw / tpb[1]))
    _sliding_kuipers_kernel[bpg, tpb](s1, s2, ws_dev, results)
    return results.copy_to_host()



