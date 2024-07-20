__author__ = "Simon Nilsson"

import itertools
import multiprocessing

import numpy as np
import pandas as pd
from numba import (boolean, float32, float64, int64, jit, njit, prange, typed,
                   types)
from numba.typed import Dict, List
from numpy.lib.stride_tricks import as_strided
from statsmodels.tsa.stattools import (adfuller, grangercausalitytests, kpss,
                                       zivot_andrews)

from simba.utils.errors import InvalidInputError

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import typing
from typing import get_type_hints

from simba.utils.checks import (check_float, check_instance, check_int,
                                check_str, check_that_column_exist,
                                check_valid_array, check_valid_lst)
from simba.utils.read_write import find_core_cnt


class TimeseriesFeatureMixin(object):
    """
    Time-series methods focused on signal complexity in sliding windows. Mainly in time-domain - fft methods (through e.g. scipy)
    I've found so far has not been fast enough for rolling windows in large datasets.

    .. image:: _static/img/ts_runtimes.png
       :width: 1200
       :align: center

    .. note::
       Many method has numba typed `signatures <https://numba.pydata.org/numba-doc/latest/reference/types.html>`_ to decrease
       compilation time through reduced type inference. Make sure to pass the correct dtypes as indicated by signature decorators.

    .. important::
       See references for mature packages computing more extensive timeseries measurements

       .. [1] `cesium <https://github.com/cesium-ml/cesium>`_.
       .. [2] `eeglib <https://github.com/Xiul109/eeglib>`_.
       .. [3] `antropy <https://github.com/raphaelvallat/antropy>`_.
       .. [4] `tsfresh <https://tsfresh.readthedocs.io>`_.
       .. [5] `pycaret <https://github.com/pycaret/pycaret>`_.
    """

    def __init__(self):
        pass

    @staticmethod
    @njit("(float32[:],)")
    def hjort_parameters(data: np.ndarray) -> (float, float, float):
        """
        Jitted compute of Hjorth parameters for a given time series data. Hjorth parameters describe
        mobility, complexity, and activity of a time series.

        :param numpy.ndarray data: A 1-dimensional numpy array containing the time series data.
        :return: tuple
            A tuple containing the following Hjorth parameters:
            - activity (float): The activity of the time series, which is the variance of the input data.
            - mobility (float): The mobility of the time series, calculated as the square root of the variance
              of the first derivative of the input data divided by the variance of the input data.
            - complexity (float): The complexity of the time series, calculated as the square root of the variance
              of the second derivative of the input data divided by the variance of the first derivative, and then
              divided by the mobility.

        :example:
        >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        >>> TimeseriesFeatureMixin().hjort_parameters(data)
        >>> (2.5, 0.5, 0.4082482904638631)

        :math:`mobility = \sqrt{\\frac{dx_{var}}{x_{var}}}`

        :math:`complexity = \sqrt{\\frac{ddx_{var}}{dx_{var}} / mobility}`
        """
        dx = np.diff(np.ascontiguousarray(data))
        ddx = np.diff(np.ascontiguousarray(dx))
        x_var, dx_var = np.var(data), np.var(dx)
        if (x_var <= 0) or (dx_var <= 0):
            return 0, 0, 0

        ddx_var = np.var(ddx)
        mobility = np.sqrt(dx_var / x_var)
        complexity = np.sqrt(ddx_var / dx_var) / mobility
        activity = np.var(data)

        return activity, mobility, complexity

    @staticmethod
    @njit("(float32[:], float64[:], int64)")
    def sliding_hjort_parameters(data: np.ndarray, window_sizes: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Jitted compute of Hjorth parameters, including mobility, complexity, and activity, for
        sliding windows of varying sizes applied to the input data array.

        :param np.ndarray data: Input data array.
        :param np.ndarray window_sizes: Array of window sizes (in seconds).
        :param int sample_rate: Sampling rate of the data in samples per second.
        :return np.ndarray: An array containing Hjorth parameters for each window size and data point.
                   The shape of the result array is (3, data.shape[0], window_sizes.shape[0]).
                   The three parameters are stored in the first dimension (0 - mobility, 1 - complexity,
                   2 - activity), and the remaining dimensions correspond to data points and window sizes.

        """
        results = np.full((3, data.shape[0], window_sizes.shape[0]), -1.0)
        for i in range(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(
                prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)
            ):
                sample = data[l:r]
                dx = np.diff(np.ascontiguousarray(sample))
                ddx = np.diff(np.ascontiguousarray(dx))
                x_var, dx_var = np.var(sample), np.var(dx)
                if (x_var <= 0) or (dx_var <= 0):
                    results[0, r + 1, i] = 0
                    results[1, r + 1, i] = 0
                    results[2, r + 1, i] = 0
                else:
                    ddx_var = np.var(ddx)
                    mobility = np.sqrt(dx_var / x_var)
                    complexity = np.sqrt(ddx_var / dx_var) / mobility
                    activity = np.var(sample)
                    results[0, r + 1, i] = mobility
                    results[1, r + 1, i] = complexity
                    results[2, r + 1, i] = activity

        return results.astype(np.float32)

    @staticmethod
    @njit([(float32[:], boolean), (float32[:], types.misc.Omitted(True))])
    def local_maxima_minima(data: np.ndarray, maxima: bool = True) -> np.ndarray:
        """
        Jitted compute of the local maxima or minima defined as values which are higher or lower than immediately preceding and proceeding time-series neighbors, repectively.
        Returns 2D np.ndarray with columns representing idx and values of local maxima.

        .. image:: _static/img/local_maxima_minima.png
           :width: 600
           :align: center


        :param np.ndarray data: Time-series data.
        :param bool maxima: If True, returns maxima. Else, minima.
        :return np.ndarray: 2D np.ndarray with columns representing idx in input data in first column and values of local maxima in second column

        :example:
        >>> data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
        >>> TimeseriesFeatureMixin().local_maxima_minima(data=data, maxima=True)
        >>> [[1, 7.5], [4, 7.5], [9, 9.5]]
        >>> TimeseriesFeatureMixin().local_maxima_minima(data=data, maxima=False)
        >>> [[0, 3.9], [2, 4.2], [5, 3.9]]

        """
        if not maxima:
            data = -data
        results = np.full((data.shape[0], 2), -1.0)
        if data[0] >= data[1]:
            if not maxima:
                results[0, :] = np.array([0, -data[0]])
            else:
                results[0, :] = np.array([0, data[0]])
        if data[-1] >= data[-2]:
            if not maxima:
                results[-1, :] = np.array([data.shape[0] - 1, -data[-1]])
            else:
                results[-1, :] = np.array([data.shape[0] - 1, data[-1]])
        for i in prange(1, data.shape[0] - 1):
            if data[i - 1] < data[i] > data[i + 1]:
                if not maxima:
                    results[i, :] = np.array([i, -data[i]])
                else:
                    results[i, :] = np.array([i, data[i]])

        return results[np.argwhere(results[:, 0].T != -1).flatten()].astype(np.float32)

    @staticmethod
    @njit("(float32[:], float64)")
    def crossings(data: np.ndarray, val: float) -> int:
        """
        Jitted compute of the count in time-series where sequential values crosses a defined value.

        .. image:: _static/img/crossings.png
           :width: 600
           :align: center

        :param np.ndarray data: Time-series data.
        :param float val: Cross value. E.g., to count the number of zero-crossings, pass `0`.
        :return int: Count of events where sequential values crosses ``val``.

        :example:
        >>> data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
        >>> TimeseriesFeatureMixin().crossings(data=data, val=7)
        >>> 5
        """

        cnt, last_val = 0, -1
        if data[0] > val:
            last_val = 1
        for i in prange(1, data.shape[0]):
            current_val = -1
            if data[i] > val:
                current_val = 1
            if last_val != current_val:
                cnt += 1
            last_val = current_val

        return cnt

    @staticmethod
    @njit("(float32[:], float64,  float64[:], int64,)")
    def sliding_crossings(data: np.ndarray, val: float, time_windows: np.ndarray, fps: int) -> np.ndarray:
        """
        Compute the number of crossings over sliding windows in a data array.

        Computes the number of times a value in the data array crosses a given threshold
        value within sliding windows of varying sizes. The number of crossings is computed for each
        window size and stored in the result array where columns represents time windows.

        .. note::
           For frames occurring before a complete time window, -1.0 is returned.

        .. image:: _static/img/sliding_crossings.png
           :width: 1500
           :align: center

        :param np.ndarray data: Input data array.
        :param float val: Threshold value for crossings.
        :param np.ndarray time_windows: Array of window sizes (in seconds).
        :param int sample_rate: Sampling rate of the data in samples per second.
        :return np.ndarray: An array containing the number of crossings for each window size and data point. The shape of the result array is (data.shape[0], window_sizes.shape[0]).

        :example:
        >>> data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
        >>> results = TimeseriesFeatureMixin().sliding_crossings(data=data, time_windows=np.array([1.0]), fps=2.0, val=7.0)
        """

        results = np.full((data.shape[0], time_windows.shape[0]), -1.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            for l, r in zip(
                prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)
            ):
                sample = data[l:r]
                cnt, last_val = 0, -1
                if sample[0] > val:
                    last_val = 1
                for j in prange(1, sample.shape[0]):
                    current_val = -1
                    if sample[j] > val:
                        current_val = 1
                    if last_val != current_val:
                        cnt += 1
                    last_val = current_val
                results[r - 1, i] = cnt

        return results.astype(np.int32)

    @staticmethod
    @njit("(float32[:], int64, int64, )", cache=True, fastmath=True)
    def percentile_difference(data: np.ndarray, upper_pct: int, lower_pct: int) -> float:
        """
        Jitted compute of the difference between the ``upper`` and ``lower`` percentiles of the data as
        a percentage of the median value. Helps understanding the spread or variability of the data within specified percentiles.

        .. note::
           Adapted from `cesium <https://github.com/cesium-ml/cesium>`_.

        .. image:: _static/img/percentile_difference.png
           :width: 600
           :align: center

        :parameter np.ndarray data: 1D array of representing time-series.
        :parameter int upper_pct: Upper-boundary percentile.
        :parameter int lower_pct: Lower-boundary percentile.
        :returns float: The difference between the ``upper`` and ``lower`` percentiles of the data as a percentage of the median value.

        :examples:
        >>> data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
        >>> TimeseriesFeatureMixin().percentile_difference(data=data, upper_pct=95, lower_pct=5)
        >>> 0.7401574764125177

        """
        upper_val, lower_val = np.percentile(data, upper_pct), np.percentile(
            data, lower_pct
        )
        return np.abs(upper_val - lower_val) / np.median(data)

    @staticmethod
    @njit("(float32[:], int64, int64, float64[:], int64, )", cache=True, fastmath=True)
    def sliding_percentile_difference(
        data: np.ndarray,
        upper_pct: int,
        lower_pct: int,
        window_sizes: np.ndarray,
        fps: int,
    ) -> np.ndarray:
        """
        Jitted computes the difference between the upper and lower percentiles within a sliding window for each position
        in the time series using various window sizes. It returns a 2D array where each row corresponds to a position in the time series,
        and each column corresponds to a different window size. The results are calculated as the absolute difference between
        upper and lower percentiles divided by the median of the window.

        :param np.ndarray data: The input time series data.
        :param int upper_pct: The upper percentile value for the window (e.g., 95 for the 95th percentile).
        :param int lower_pct: The lower percentile value for the window (e.g., 5 for the 5th percentile).
        :param np.ndarray window_sizes: An array of window sizes (in seconds) to use for the sliding calculation.
        :param int sample_rate: The sampling rate (samples per second) of the time series data.
        :return np.ndarray: A 2D array containing the difference between upper and lower percentiles for each window size.

        """
        results = np.full((data.shape[0], window_sizes.shape[0]), -1.0)
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * fps)
            for l, r in zip(
                prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)
            ):
                sample = data[l:r]
                upper_val, lower_val = np.percentile(sample, upper_pct), np.percentile(
                    sample, lower_pct
                )
                median = np.median(sample)
                if median != 0:
                    results[r - 1, i] = np.abs(upper_val - lower_val) / median
                else:
                    results[r - 1, i] = -1.0

        return results.astype(np.float32)

    @staticmethod
    @njit("(float64[:], float64,)")
    def percent_beyond_n_std(data: np.ndarray, n: float) -> float:
        """
        Jitted compute of the ratio of values in time-series more than N standard deviations from the mean of the time-series.

        :parameter np.ndarray data: 1D array representing time-series.
        :parameter float n: Standard deviation cut-off.
        :returns float: Ratio of values in ``data`` that fall more than ``n`` standard deviations from mean of ``data``.

        .. note::
           Adapted from `cesium <https://github.com/cesium-ml/cesium>`_.

           Oddetity: mean calculation is incorrect if passing float32 data but correct if passing float64.

        .. image:: _static/img/percent_beyond_n_std.png
           :width: 600
           :align: center

        :examples:
        >>> data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
        >>> TimeseriesFeatureMixin().percent_beyond_n_std(data=data, n=1)
        >>> 0.1

        """
        m = np.mean(data)
        std = np.std(data)
        target = m + (std * n)
        return np.argwhere(data > target).shape[0] / data.shape[0]

    @staticmethod
    @njit("(float64[:], float64, float64[:], int64,)", cache=True, fastmath=True)
    def sliding_percent_beyond_n_std(
        data: np.ndarray, n: float, window_sizes: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Computed the percentage of data points that exceed 'n' standard deviations from the mean for each position in
        the time series using various window sizes. It returns a 2D array where each row corresponds to a position in the time series,
        and each column corresponds to a different window size. The results are given as a percentage of data points beyond the threshold.

        :param np.ndarray data: The input time series data.
        :param float n: The number of standard deviations to determine the threshold.
        :param np.ndarray window_sizes: An array of window sizes (in seconds) to use for the sliding calculation.
        :param int sample_rate: The sampling rate (samples per second) of the time series data.
        :return np.ndarray: A 2D array containing the percentage of data points beyond the specified 'n' standard deviations for each window size.
        """

        results = np.full((data.shape[0], window_sizes.shape[0]), -1.0)
        target = (np.std(data) * n) + np.mean(data)
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(
                prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)
            ):
                sample = data[l:r]
                results[r - 1, i] = (
                    np.argwhere(np.abs(sample) > target).shape[0] / sample.shape[0]
                )

        return results.astype(np.float32)

    @staticmethod
    @njit(
        [
            (float32[:], float64[:], int64),
            (int64[:], float64[:], int64),
        ]
    )
    def sliding_unique(x: np.ndarray, time_windows: np.ndarray, fps: int):
        results = np.full((x.shape[0], time_windows.shape[0]), -1)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            for l, r in zip(
                range(0, x.shape[0] + 1), range(window_size, x.shape[0] + 1)
            ):
                sample = x[l:r]
                unique_cnt = np.unique(sample)
                results[r - 1, i] = unique_cnt.shape[0]
        return results

    @staticmethod
    @njit("(float32[:], int64, int64, )", fastmath=True)
    def percent_in_percentile_window(data: np.ndarray, upper_pct: int, lower_pct: int):
        """
        Jitted compute of the ratio of values in time-series that fall between the ``upper`` and ``lower`` percentile.

        :parameter np.ndarray data: 1D array of representing time-series.
        :parameter int upper_pct: Upper-boundary percentile.
        :parameter int lower_pct: Lower-boundary percentile.
        :returns float: Ratio of values in ``data`` that fall within ``upper_pct`` and ``lower_pct`` percentiles.

        .. note::
           Adapted from `cesium <https://github.com/cesium-ml/cesium>`_.

        .. image:: _static/img/percent_in_percentile_window.png
           :width: 600
           :align: center

        :example:
        >>> data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
        >>> TimeseriesFeatureMixin().percent_in_percentile_window(data, upper_pct=70, lower_pct=30)
        >>> 0.4
        """

        upper_val, lower_val = np.percentile(data, upper_pct), np.percentile(
            data, lower_pct
        )
        return (
            np.argwhere((data <= upper_val) & (data >= lower_val)).flatten().shape[0]
            / data.shape[0]
        )

    @staticmethod
    @njit("(float32[:], int64, int64, float64[:], int64)", cache=True, fastmath=True)
    def sliding_percent_in_percentile_window(
        data: np.ndarray,
        upper_pct: int,
        lower_pct: int,
        window_sizes: np.ndarray,
        sample_rate: int,
    ):
        """
        Jitted compute of the percentage of data points falling within a percentile window in a sliding manner.

        The function computes the percentage of data points within the specified percentile window for each position in the time series
        using various window sizes. It returns a 2D array where each row corresponds to a position in the time series, and each column
        corresponds to a different window size. The results are given as a percentage of data points within the percentile window.

        :param np.ndarray data: The input time series data.
        :param int upper_pct: The upper percentile value for the window (e.g., 95 for the 95th percentile).
        :param int lower_pct: The lower percentile value for the window (e.g., 5 for the 5th percentile).
        :param np.ndarray window_sizes: An array of window sizes (in seconds) to use for the sliding calculation.
        :param int sample_rate: The sampling rate (samples per second) of the time series data.
        :return np.ndarray: A 2D array containing the percentage of data points within the percentile window for each window size.

        """
        results = np.full((data.shape[0], window_sizes.shape[0]), -1.0)
        upper_val, lower_val = np.percentile(data, upper_pct), np.percentile(
            data, lower_pct
        )
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(
                prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)
            ):
                sample = data[l:r]
                results[r - 1, i] = (
                    np.argwhere((sample <= upper_val) & (sample >= lower_val))
                    .flatten()
                    .shape[0]
                    / sample.shape[0]
                )

        return results.astype(np.float32)

    @staticmethod
    @njit("(float32[:],)", fastmath=True, cache=True)
    def petrosian_fractal_dimension(data: np.ndarray) -> float:
        """
        Calculate the Petrosian Fractal Dimension (PFD) of a given time series data. The PFD is a measure of the
        irregularity or self-similarity of a time series. Larger values indicate higher complexity. Lower values indicate lower complexity.

        .. note::
           The PFD is computed based on the number of sign changes in the first derivative of the time series. If the input data is empty or no sign changes are found, the PFD is returned as -1.0. Adapted from `eeglib <https://github.com/Xiul109/eeglib/>`_.

        .. math::
           PFD = \\frac{\\log_{10}(N)}{\\log_{10}(N) + \\log_{10}\\left(\\frac{N}{N + 0.4 \\cdot zC}\\right)}

        :parameter np.ndarray data: A 1-dimensional numpy array containing the time series data.
        :returns float: The Petrosian Fractal Dimension of the input time series.

        :examples:
        >>> t = np.linspace(0, 50, int(44100 * 2.0), endpoint=False)
        >>> sine_wave = 1.0 * np.sin(2 * np.pi * 1.0 * t).astype(np.float32)
        >>> TimeseriesFeatureMixin().petrosian_fractal_dimension(data=sine_wave)
        >>> 1.0000398187022719
        >>> np.random.shuffle(sine_wave)
        >>> TimeseriesFeatureMixin().petrosian_fractal_dimension(data=sine_wave)
        >>> 1.0211625348743218
        """

        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        derivative = data[1:] - data[:-1]
        if derivative.shape[0] == 0:
            return -1.0
        zC, last_val = 0, -1
        if derivative[0] > 0.0:
            last_val = 1
        for i in prange(1, derivative.shape[0]):
            current_val = -1
            if derivative[i] > 0.0:
                current_val = 1
            if last_val != current_val:
                zC += 1
            last_val = current_val
        if zC == 0:
            return -1.0

        return np.log10(data.shape[0]) / (
            np.log10(data.shape[0])
            + np.log10(data.shape[0] / (data.shape[0] + 0.4 * zC))
        )

    @staticmethod
    @njit("(float32[:], float64[:], int64)", fastmath=True, cache=True)
    def sliding_petrosian_fractal_dimension(
        data: np.ndarray, window_sizes: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Jitted compute of Petrosian Fractal Dimension over sliding windows in a data array.

        This method computes the Petrosian Fractal Dimension for sliding windows of varying sizes applied
        to the input data array. The Petrosian Fractal Dimension is a measure of signal complexity.

        :param np.ndarray data: Input data array.
        :param np.ndarray window_sizes: Array of window sizes (in seconds).
        :param int sample_rate: Sampling rate of the data in samples per second.
        :return np.ndarray: An array containing Petrosian Fractal Dimension values for each window size and data
                            point. The shape of the result array is (data.shape[0], window_sizes.shape[0]).

        """

        results = np.full((data.shape[0], window_sizes.shape[0]), -1.0)
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(
                prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)
            ):
                sample = (data[l:r] - np.min(data[l:r])) / (
                    np.max(data[l:r]) - np.min(data[l:r])
                )
                derivative = sample[1:] - sample[:-1]
                if derivative.shape[0] == 0:
                    results[r - 1, i] = -1.0
                else:
                    zC, last_val = 0, -1
                    if derivative[0] > 0.0:
                        last_val = 1
                    for j in prange(1, derivative.shape[0]):
                        current_val = -1
                        if derivative[j] > 0.0:
                            current_val = 1
                        if last_val != current_val:
                            zC += 1
                        last_val = current_val
                    if zC == 0:
                        results[r - 1, i] = -1.0
                    else:
                        results[r - 1, i] = np.log10(sample.shape[0]) / (
                            np.log10(sample.shape[0])
                            + np.log10(sample.shape[0] / (sample.shape[0] + 0.4 * zC))
                        )

        return results.astype(np.float32)

    @staticmethod
    @njit("(float32[:], int64)")
    def higuchi_fractal_dimension(data: np.ndarray, kmax: int = 10):
        """
        Jitted compute of the Higuchi Fractal Dimension of a given time series data. The Higuchi Fractal Dimension provides a measure of the fractal
        complexity of a time series.

        The maximum value of k used in the calculation. Increasing kmax considers longer sequences
            of data, providing a more detailed analysis of fractal complexity. Default is 10.

        :parameter np.ndarray data: A 1-dimensional numpy array containing the time series data.
        :parameter int kmax: The maximum value of k used in the calculation. Increasing kmax considers longer sequences of data, providing a more detailed analysis of fractal complexity. Default is 10.
        :returns float: The Higuchi Fractal Dimension of the input time series.

        .. note::
           - Adapted from `eeglib <https://github.com/Xiul109/eeglib/>`_.

        .. math::
           HFD = \\frac{\\log(N)}{\\log(N) + \\log\\left(\\frac{N}{N + 0.4 \\cdot zC}\\right)}

        :example:
        >>> t = np.linspace(0, 50, int(44100 * 2.0), endpoint=False)
        >>> sine_wave = 1.0 * np.sin(2 * np.pi * 1.0 * t).astype(np.float32)
        >>> sine_wave = (sine_wave - np.min(sine_wave)) / (np.max(sine_wave) - np.min(sine_wave))
        >>> TimeseriesFeatureMixin().higuchi_fractal_dimension(data=data, kmax=10)
        >>> 1.0001506805419922
        >>> np.random.shuffle(sine_wave)
        >>> TimeseriesFeatureMixin().higuchi_fractal_dimension(data=data, kmax=10)
        >>> 1.9996402263641357
        """

        L, N = np.zeros(kmax - 1), len(data)
        x = np.hstack(
            (
                -np.log(np.arange(2, kmax + 1)).reshape(-1, 1).astype(np.float32),
                np.ones(kmax - 1).reshape(-1, 1).astype(np.float32),
            )
        )
        for k in prange(2, kmax + 1):
            Lk = np.zeros(k)
            for m in range(0, k):
                Lmk = 0
                for i in range(1, (N - m) // k):
                    Lmk += abs(data[m + i * k] - data[m + i * k - k])
                denominator = ((N - m) // k) * k * k
                if denominator == 0:
                    return -1
                Lk[m] = Lmk * (N - 1) / (((N - m) // k) * k * k)
            Laux = np.mean(Lk)
            Laux = 0.01 / k if Laux == 0 else Laux
            L[k - 2] = np.log(Laux)

        return np.linalg.lstsq(x, L.astype(np.float32))[0][0]

    @staticmethod
    @njit("(float32[:], int64, int64,)", fastmath=True)
    def permutation_entropy(data: np.ndarray, dimension: int, delay: int) -> float:
        """
        Calculate the permutation entropy of a time series.

        Permutation entropy is a measure of the complexity of a time series data by quantifying
        the irregularity and unpredictability of its order patterns. It is computed based on the
        frequency of unique order patterns of a given dimension in the time series data.

        The permutation entropy (PE) is calculated using the following formula:

        .. math::
            PE = - \\sum(p_i \\log(p_i))

        where:
        - PE is the permutation entropy.
        - p_i is the probability of each unique order pattern.

        :param numpy.ndarray data: The time series data for which permutation entropy is calculated.
        :param int dimension: It specifies the length of the order patterns to be considered.
        :param int delay: Time delay between elements in an order pattern.
        :return float: The permutation entropy of the time series, indicating its complexity and predictability. A higher permutation entropy value indicates higher complexity and unpredictability in the time series.

        :example:
        >>> t = np.linspace(0, 50, int(44100 * 2.0), endpoint=False)
        >>> sine_wave = 1.0 * np.sin(2 * np.pi * 1.0 * t).astype(np.float32)
        >>> TimeseriesFeatureMixin().permutation_entropy(data=sine_wave, dimension=3, delay=1)
        >>> 0.701970058666407
        >>> np.random.shuffle(sine_wave)
        >>> TimeseriesFeatureMixin().permutation_entropy(data=sine_wave, dimension=3, delay=1)
        >>> 1.79172449934604
        """

        n, permutations, counts = len(data), List(), List()
        for i in prange(n - (dimension - 1) * delay):
            indices = np.arange(i, i + dimension * delay, delay)
            permutation = List(np.argsort(data[indices]))
            is_unique = True
            for j in range(len(permutations)):
                p = permutations[j]
                if len(p) == len(permutation):
                    is_equal = True
                    for k in range(len(p)):
                        if p[k] != permutation[k]:
                            is_equal = False
                            break
                    if is_equal:
                        is_unique = False
                        counts[j] += 1
                        break
            if is_unique:
                permutations.append(permutation)
                counts.append(1)

        total_permutations = len(permutations)
        probs = np.empty(total_permutations, dtype=types.float64)
        for i in prange(total_permutations):
            probs[i] = counts[i] / (n - (dimension - 1) * delay)

        return -np.sum(probs * np.log(probs))

    @staticmethod
    @njit("(float32[:],)", fastmath=True)
    def line_length(data: np.ndarray) -> float:
        """
        Calculate the line length of a 1D array.

        Line length is a measure of signal complexity and is computed by summing the absolute
        differences between consecutive elements of the input array. Used in EEG
        analysis and other signal processing applications to quantify variations in the signal.

        :param numpy.ndarray data: The 1D array for which the line length is to be calculated.
        :return float: The line length of the input array, indicating its complexity.

        .. math::

            LL = \sum_{i=1}^{N-1} |x[i] - x[i-1]|

        where:
        - LL is the line length.
        - N is the number of elements in the input data array.
        - x[i] represents the value of the data at index i.


        .. image:: _static/img/line_length.png
           :width: 600
           :align: center

        :example:
        >>> data = np.array([1, 4, 2, 3, 5, 6, 8, 7, 9, 10]).astype(np.float32)
        >>> TimeseriesFeatureMixin().line_length(data=data)
        >>> 12.0
        """

        diff = np.abs(np.diff(data.astype(np.float64)))
        return np.sum(diff)

    @staticmethod
    @njit("(float32[:], float64[:], int64)", fastmath=True)
    def sliding_line_length(
        data: np.ndarray, window_sizes: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Jitted compute of  sliding line length for a given time series using different window sizes.

        The function computes line length for the input data using various window sizes. It returns a 2D array where each row
        corresponds to a position in the time series, and each column corresponds to a different window size. The line length
        is calculated for each window, and the results are returned as a 2D array of float32 values.

        :param np.ndarray data: 1D array input data.
        :param window_sizes: An array of window sizes (in seconds) to use for line length calculation.
        :param sample_rate: The sampling rate (samples per second) of the time series data.
        :return np.ndarray: A 2D array containing line length values for each window size at each position in the time series.

        .. image:: _static/img/sliding_line_length.png
           :width: 600
           :align: center

        :examples:
        >>> data = np.array([1, 4, 2, 3, 5, 6, 8, 7, 9, 10]).astype(np.float32)
        >>> TimeseriesFeatureMixin().sliding_line_length(data=data, window_sizes=np.array([1.0]), sample_rate=2)
        """

        results = np.full((data.shape[0], window_sizes.shape[0]), -1.0)
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(
                prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)
            ):
                sample = data[l:r]
                results[r - 1, i] = np.sum(np.abs(np.diff(sample.astype(np.float64))))
        return results.astype(np.float32)

    @staticmethod
    @njit("(float32[:], float64[:], int64)", fastmath=True, cache=True)
    def sliding_variance(data: np.ndarray, window_sizes: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Jitted compute of the variance of data within sliding windows of varying sizes applied to
        the input data array. Variance is a measure of data dispersion or spread.

        .. image:: _static/img/sliding_variance.png
           :width: 600
           :align: center

        :param data: 1d input data array.
        :param window_sizes: Array of window sizes (in seconds).
        :param sample_rate: Sampling rate of the data in samples per second.
        :return: Variance values for each window size and data point. The shape of the result array is (data.shape[0], window_sizes.shape[0]).

        :example:
        >>> data = np.array([1, 2, 3, 1, 2, 9, 17, 2, 10, 4]).astype(np.float32)
        >>> TimeseriesFeatureMixin().sliding_variance(data=data, window_sizes=np.array([0.5]), sample_rate=10)
        >>> [[-1.],[-1.],[-1.],[-1.],[ 0.56],[ 8.23],[35.84],[39.20],[34.15],[30.15]])
        """

        results = np.full((data.shape[0], window_sizes.shape[0]), -1.0)
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(
                prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)
            ):
                sample = data[l:r]
                results[r - 1, i] = np.var(sample)

        return results.astype(np.float32)

    @staticmethod
    @njit(
        "(float32[:], float64[:], int64, types.ListType(types.unicode_type))",
        fastmath=True,
        cache=True,
    )
    def sliding_descriptive_statistics(data: np.ndarray, window_sizes: np.ndarray, sample_rate: int, statistics: Literal["var", "max", "min", "std", "median", "mean", "mad", "sum", "mac", "rms", "absenergy"]) -> np.ndarray:
        """
        Jitted compute of descriptive statistics over sliding windows in 1D data array.

        Computes various descriptive statistics (e.g., variance, maximum, minimum, standard deviation,
        median, mean, median absolute deviation) for sliding windows of varying sizes applied to the input data array.

        :param np.ndarray data: 1D input data array.
        :param np.ndarray window_sizes: Array of window sizes (in seconds).
        :param int sample_rate: Sampling rate of the data in samples per second.
        :param types.ListType(types.unicode_type) statistics: List of statistics to compute. Options: 'var', 'max', 'min', 'std', 'median', 'mean', 'mad', 'sum', 'mac', 'rms', 'abs_energy'.
        :return np.ndarray: Array containing the selected descriptive statistics for each window size, data point, and statistic type. The shape of the result array is (len(statistics), data.shape[0], window_sizes.shape[0).

        .. note::
           - The `statistics` parameter should be a list containing one or more of the following statistics:
           'var' (variance), 'max' (maximum), 'min' (minimum), 'std' (standard deviation), 'median' (median),
           'mean' (mean), 'mad' (median absolute deviation), 'sum' (sum), 'mac' (mean absolute change),
           'rms' (root mean square), 'absenergy' (absolute energy).
           - If the statistics list is ['var', 'max', 'mean'], the
           3rd dimension order in the result array will be: [variance, maximum, mean]

        :example:
        >>> data = np.array([1, 4, 2, 3, 5, 6, 8, 7, 9, 10]).astype(np.float32)
        >>> results = TimeseriesFeatureMixin().sliding_descriptive_statistics(data=data, window_sizes=np.array([1.0, 5.0]), sample_rate=2, statistics=typed.List(['var', 'max']))
        """

        results = np.full((len(statistics), data.shape[0], window_sizes.shape[0]), -1.0)
        for j in prange(len(statistics)):
            for i in prange(window_sizes.shape[0]):
                window_size = int(window_sizes[i] * sample_rate)
                for l, r in zip(prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)):
                    sample = data[l:r]
                    if statistics[j] == "var":
                        results[j, r - 1, i] = np.var(sample)
                    elif statistics[j] == "max":
                        results[j, r - 1, i] = np.max(sample)
                    elif statistics[j] == "min":
                        results[j, r - 1, i] = np.min(sample)
                    elif statistics[j] == "std":
                        results[j, r - 1, i] = np.std(sample)
                    elif statistics[j] == "median":
                        results[j, r - 1, i] = np.median(sample)
                    elif statistics[j] == "mean":
                        results[j, r - 1, i] = np.mean(sample)
                    elif statistics[j] == "mad":
                        results[j, r - 1, i] = np.median(np.abs(sample - np.median(sample)))
                    elif statistics[j] == "sum":
                        results[j, r - 1, i] = np.sum(sample)
                    elif statistics[j] == "mac":
                        results[j, r - 1, i] = np.mean(np.abs(sample[1:] - sample[:-1]))
                    elif statistics[j] == "rms":
                        results[j, r - 1, i] = np.sqrt(np.mean(sample**2))
                    elif statistics[j] == "absenergy":
                        results[j, r - 1, i] = np.sqrt(np.sum(sample**2))

        return results.astype(np.float32)

    @staticmethod
    def dominant_frequencies(
        data: np.ndarray,
        fps: float,
        k: int,
        window_function: Literal["Hann", "Hamming", "Blackman"] = None,
    ):
        """Find the K dominant frequencies within a feature vector"""

        if window_function == "Hann":
            data = data * np.hanning(len(data))
        elif window_function == "Hamming":
            data = data * np.hamming(len(data))
        elif window_function == "Blackman":
            data = data * np.blackman(len(data))
        fft_result = np.fft.fft(data)
        frequencies = np.fft.fftfreq(data.shape[0], 1 / fps)
        magnitude = np.abs(fft_result)

        return frequencies[np.argsort(magnitude)[-(k + 1) : -1]]

    @staticmethod
    @njit(
        [
            (float32[:], float64, boolean),
            (float32[:], float64, types.misc.Omitted(True)),
        ]
    )
    def longest_strike(data: np.ndarray, threshold: float, above: bool = True) -> int:
        """
        Jitted compute of the length of the longest consecutive sequence of values in the input data that either exceed
        or fall below a specified threshold.

        .. image:: _static/img/longest_strike.png
           :width: 700
           :align: center

        :param np.ndarray data: The input 1D NumPy array containing the values to be analyzed.
        :param float threshold: The threshold value used for the comparison.
        :param bool above: If True, the function looks for strikes where values are above or equal to the threshold. If False, it looks for strikes where values are below or equal to the threshold.
        :return int: The length of the longest strike that satisfies the condition.

        :example:
        >>> data = np.array([1, 8, 2, 10, 8, 6, 8, 1, 1, 1]).astype(np.float32)
        >>> TimeseriesFeatureMixin().longest_strike(data=data, threshold=7, above=True)
        >>> 2
        >>> TimeseriesFeatureMixin().longest_strike(data=data, threshold=7, above=False)
        >>> 3
        """

        result, l, r, cnt = -np.inf, 0, 0, 0
        while l < data.shape[0]:
            if above:
                if data[l] >= threshold:
                    cnt, r = cnt + 1, r + 1
                    while data[r] >= threshold and r < data.shape[0]:
                        cnt, r = cnt + 1, r + 1
            else:
                if data[l] <= threshold:
                    cnt, r = cnt + 1, r + 1
                    while data[r] <= threshold and r < data.shape[0]:
                        cnt, r = cnt + 1, r + 1

            l += 1
            if cnt > result:
                result = cnt
            if data.shape[0] - l < result:
                break
            r, cnt = l, 0

        return int(result)

    @staticmethod
    @njit(
        [
            (float32[:], float64, float64[:], int64, boolean),
            (float32[:], float64, float64[:], int64, types.misc.Omitted(True)),
        ]
    )
    def sliding_longest_strike(
        data: np.ndarray,
        threshold: float,
        time_windows: np.ndarray,
        sample_rate: int,
        above: bool,
    ) -> np.ndarray:
        """
        Jitted compute of the length of the longest strike of values within sliding time windows that satisfy a given condition.

        Calculates the length of the longest consecutive sequence of values in a 1D NumPy array, where each
        sequence is determined by a sliding time window. The condition is specified by a threshold, and
        you can choose whether to look for values above or below the threshold.

        .. image:: _static/img/sliding_longest_strike.png
           :width: 700
           :align: center

        :param np.ndarray data: The input 1D NumPy array containing the values to be analyzed.
        :param float threshold: The threshold value used for the comparison.
        :param np.ndarray time_windows: An array containing the time window sizes in seconds.
        :param int sample_rate: The sample rate in samples per second.
        :param bool above: If True, the function looks for strikes where values are above or equal to the threshold. If False, it looks for strikes where values are below or equal to the threshold.
        :return np.ndarray: A 2D NumPy array with dimensions (data.shape[0], time_windows.shape[0]). Each element in the array represents the length of the longest strike that satisfies the condition for the
        corresponding time window.

        :example:
        >>> data = np.array([1, 8, 2, 10, 8, 6, 8, 1, 1, 1]).astype(np.float32)
        >>> TimeseriesFeatureMixin().sliding_longest_strike(data=data, threshold=7, above=True, time_windows=np.array([1.0]), sample_rate=2)
        >>> [[-1.][ 1.][ 1.][ 1.][ 2.][ 1.][ 1.][ 1.][ 0.][ 0.]]
        >>> TimeseriesFeatureMixin().sliding_longest_strike(data=data, threshold=7, above=True, time_windows=np.array([1.0]), sample_rate=2)
        >>> [[-1.][ 1.][ 1.][ 1.][ 0.][ 1.][ 1.][ 1.][ 2.][ 2.]]
        """

        results = np.full((data.shape[0], time_windows.shape[0]), -1.0)

        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * sample_rate)
            for l1, r1 in zip(
                prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)
            ):
                sample = data[l1:r1]
                result, l, r, cnt = -np.inf, 0, 0, 0

                while l < sample.shape[0]:
                    if above:
                        if sample[l] >= threshold:
                            cnt, r = cnt + 1, r + 1
                            while sample[r] >= threshold and r < sample.shape[0]:
                                cnt, r = cnt + 1, r + 1
                    else:
                        if sample[l] <= threshold:
                            cnt, r = cnt + 1, r + 1
                            while sample[r] <= threshold and r < sample.shape[0]:
                                cnt, r = cnt + 1, r + 1

                    l += 1
                    if cnt > result:
                        result = cnt
                    if data.shape[0] - l < result:
                        results[r - 1, i] = result
                        break
                    r, cnt = l, 0

                results[r1 - 1, i] = result

            return results

    @staticmethod
    @njit(
        [
            (float32[:], float64, int64, boolean),
            (float32[:], float64, int64, types.misc.Omitted(True)),
        ]
    )
    def time_since_previous_threshold(
        data: np.ndarray, threshold: float, fps: int, above: bool
    ) -> np.ndarray:
        """
        Jitted compute of the time (in seconds) that has elapsed since the last occurrence of a value above (or below)
        a specified threshold in a time series. The time series is assumed to have a constant sample rate.

        .. image:: _static/img/time_since_previous_threshold.png
           :width: 600
           :align: center

        :param np.ndarray data: The input 1D array containing the time series data.
        :param int threshold: The threshold value used for the comparison.
        :param int fps: The sample rate of the time series in samples per second.
        :param bool above: If True, the function looks for values above or equal to the threshold. If False, it looks for values below or equal to the threshold.
        :return np.ndarray: A 1D array of the same length as the input data. Each element represents the time elapsed (in seconds) since the last occurrence of the threshold value. If no threshold value is found before the current data point, the corresponding result is set to -1.0.

        :examples:
        >>> data = np.array([1, 8, 2, 10, 8, 6, 8, 1, 1, 1]).astype(np.float32)
        >>> TimeseriesFeatureMixin().time_since_previous_threshold(data=data, threshold=7.0, above=True, sample_rate=2.0)
        >>> [-1. ,  0. ,  0.5,  0. ,  0. ,  0.5,  0. ,  0.5,  1. ,  1.5]
        >>> TimeseriesFeatureMixin().time_since_previous_threshold(data=data, threshold=7.0, above=False, sample_rate=2.0)
        >>> [0. , 0.5, 0. , 0.5, 1. , 0. , 0.5, 0. , 0. , 0. ]
        """

        results = np.full((data.shape[0]), -1.0)
        if above:
            criterion_idx = np.argwhere(data > threshold).flatten()
        else:
            criterion_idx = np.argwhere(data < threshold).flatten()

        for i in prange(data.shape[0]):
            if above and (data[i] > threshold):
                results[i] = 0.0
            elif not above and (data[i] < threshold):
                results[i] = 0.0
            else:
                x = criterion_idx[np.argwhere(criterion_idx < i).flatten()]
                if len(x) > 0:
                    results[i] = (i - x[-1]) / fps

        return results

    @staticmethod
    @njit(
        [
            (float32[:], float64, int64, boolean),
            (float32[:], float64, int64, types.misc.Omitted(True)),
        ]
    )
    def time_since_previous_target_value(
        data: np.ndarray, value: float, fps: int, inverse: bool = False
    ) -> np.ndarray:
        """
        Calculate the time duration (in seconds) since the previous occurrence of a specific value in a data array.

        Calculates the time duration, in seconds, between each data point and the previous occurrence
        of a specific value within the data array.

        .. image:: _static/img/time_since_previous_target_value.png
           :width: 700
           :align: center

        :param np.ndarray data: The input 1D array containing the time series data.
        :param float value: The specific value to search for in the data array.
        :param int sample_rate: The sampling rate which data points were collected. It is used to calculate the time duration in seconds.
        :param bool inverse: If True, the function calculates the time since the previous value that is NOT equal to the specified 'value'. If False, it calculates the time since the previous occurrence of the specified 'value'.
        :returns np.ndarray: A 1D NumPy array containing the time duration (in seconds) since the previous occurrence of the specified 'value' for each data point.

        :example:
        >>> data = np.array([8, 8, 2, 10, 8, 6, 8, 1, 1, 1]).astype(np.float32)
        >>> TimeseriesFeatureMixin().time_since_previous_target_value(data=data, value=8.0, inverse=False, sample_rate=2.0)
        >>> [0. , 0. , 0.5, 1. , 0. , 0.5, 0. , 0.5, 1. , 1.5])
        >>> TimeseriesFeatureMixin().time_since_previous_target_value(data=data, value=8.0, inverse=True, sample_rate=2.0)
        >>> [-1. , -1. ,  0. ,  0. ,  0.5,  0. ,  0.5,  0. ,  0. ,  0. ]
        """

        results = np.full((data.shape[0]), -1.0)
        if not inverse:
            criterion_idx = np.argwhere(data == value).flatten()
        else:
            criterion_idx = np.argwhere(data != value).flatten()
        if criterion_idx.shape[0] == 0:
            return np.full((data.shape[0]), -1.0)
        for i in prange(data.shape[0]):
            if not inverse and (data[i] == value):
                results[i] = 0
            elif inverse and (data[i] != value):
                results[i] = 0
            else:
                x = criterion_idx[np.argwhere(criterion_idx < i).flatten()]
                if len(x) > 0:
                    results[i] = (i - x[-1]) / fps
        return results

    @staticmethod
    @njit("(float32[:],)")
    def benford_correlation(data: np.ndarray) -> float:
        """
        Jitted compute of the correlation between the Benford's Law distribution and the first-digit distribution of given data.

        Benford's Law describes the expected distribution of leading (first) digits in many real-life datasets. This function
        calculates the correlation between the expected Benford's Law distribution and the actual distribution of the
        first digits in the provided data.

        .. image:: _static/img/benford_correlation.png
           :width: 600
           :align: center

        .. note::
           Adapted from `tsfresh <https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#benford_correlation>`_.

           The returned correlation values are calculated using Pearson's correlation coefficient.

        :param np.ndarray data: The input 1D array containing the time series data.
        :return float: The correlation coefficient between the Benford's Law distribution and the first-digit distribution in the input data. A higher correlation value suggests that the data follows the expected distribution more closely.

        :examples:
        >>> data = np.array([1, 8, 2, 10, 8, 6, 8, 1, 1, 1]).astype(np.float32)
        >>> TimeseriesFeatureMixin().benford_correlation(data=data)
        >>> 0.6797500374831786
        """

        data = np.abs(data)
        benford_distribution = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])
        first_vals, digit_ratio = np.full((data.shape[0]), np.nan), np.full(9, np.nan)
        for i in prange(data.shape[0]):
            first_vals[i] = data[i] // 10 ** (int(np.log10(data[i])) - 1 + 1)

        for i in range(1, 10):
            digit_ratio[i - 1] = np.argwhere(first_vals == i).shape[0] / data.shape[0]

        return np.corrcoef(benford_distribution, digit_ratio)[0, 1]

    @staticmethod
    @njit("(float32[:], float64[:], int64)")
    def sliding_benford_correlation(
        data: np.ndarray, time_windows: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Calculate the sliding Benford's Law correlation coefficient for a given dataset within
        specified time windows.

        Benford's Law is a statistical phenomenon where the leading digits of many datasets follow a
        specific distribution pattern. This function calculates the correlation between the observed
        distribution of leading digits in a dataset and the ideal Benford's Law distribution.

        .. note::
           Adapted from `tsfresh <https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#benford_correlation>`_.

           The returned correlation values are calculated using Pearson's correlation coefficient.

        The correlation coefficient is calculated between the observed leading digit distribution and
        the ideal Benford's Law distribution.

        .. math::

            P(d) = \\log_{10}\\left(1 + \\frac{1}{d}\\right) \\quad \\text{for } d \\in \{1, 2, \\ldots, 9\}

        :param np.ndarray data: The input 1D array containing the time series data.
        :param np.ndarray time_windows: A 1D array containing the time windows (in seconds) for which the correlation will be calculated at different points in the dataset.
        :param int sample_rate: The sample rate, indicating how many data points are collected per second.
        :return np.ndarray: 2D array containing the correlation coefficient values for each time window. With time window lenths represented by different columns.

        :examples:
        >>> data = np.array([1, 8, 2, 10, 8, 6, 8, 1, 1, 1]).astype(np.float32)
        >>> TimeseriesFeatureMixin.sliding_benford_correlation(data=data, time_windows=np.array([1.0]), sample_rate=2)
        >>> [[ 0.][0.447][0.017][0.877][0.447][0.358][0.358][0.447][0.864][0.864]]
        """

        data = np.abs(data)
        benford_distribution = np.array([np.log10(1 + 1 / n) for n in range(1, 10)])
        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * sample_rate)
            for l, r in zip(
                prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)
            ):
                first_vals, digit_ratio = np.full((data.shape[0]), np.nan), np.full(
                    9, np.nan
                )
                sample = data[l:r]
                for k in range(sample.shape[0]):
                    first_vals[k] = sample[k] // 10 ** (
                        int(np.log10(sample[k])) - 1 + 1
                    )
                for k in range(1, 10):
                    digit_ratio[k - 1] = (
                        np.argwhere(first_vals == k).shape[0] / sample.shape[0]
                    )
                results[r - 1, i] = np.corrcoef(benford_distribution, digit_ratio)[0, 1]

        return results.astype(np.float32)

    @staticmethod
    @njit("(float32[:], float64, float64, float64, float64, float64)", fastmath=True)
    def spike_finder(
        data: np.ndarray,
        sample_rate: int,
        baseline: float,
        min_spike_amplitude: float,
        min_fwhm: float = -np.inf,
        min_half_width: float = -np.inf,
    ) -> float:
        """
        Identify and characterize spikes in a given time-series data sequence. This method identifies spikes in the input data based on the specified criteria and characterizes
        each detected spike by computing its amplitude, full-width at half maximum (FWHM), and half-width.

        :param np.ndarray data: A 1D array containing the input data sequence to analyze.
        :param int sample_rate: The sample rate, indicating how many data points are collected per second.
        :param float baseline: The baseline value used to identify spikes. Any data point above (baseline + min_spike_amplitude) is considered part of a spike.
        :param float min_spike_amplitude: The minimum amplitude (above baseline) required for a spike to be considered.
        :param Optional[float] min_fwhm: The minimum full-width at half maximum (FWHM) for a spike to be included. If not specified, it defaults to negative infinity, meaning it is not considered for filtering.
        :param Optional[float] min_half_width: The minimum half-width required for a spike to be included. If not specified, it defaults to negative infinity, meaning it is not considered for filtering.
        :return tuple: A tuple containing three elements:
            - spike_idx (List[np.ndarray]): A list of 1D arrays, each representing the indices of the data points belonging to a detected spike.
            - spike_vals (List[np.ndarray]): A list of 1D arrays, each containing the values of the data points within a detected spike.
            - spike_dict (Dict[int, Dict[str, float]]): A dictionary where the keys are spike indices, and the values are dictionaries containing spike characteristics including 'amplitude' (spike amplitude), 'fwhm' (FWHM), and 'half_width' (half-width).

        .. note::
           - The function uses the Numba JIT (Just-In-Time) compilation for optimized performance. Without fastmath=True there is no runtime improvement over standard numpy.

        :example:
        >>> data = np.array([0.1, 0.1, 0.3, 0.1, 10, 10, 8, 0.1, 0.1, 0.1, 10, 10, 8, 99, 0.1, 99, 99, 0.1]).astype(np.float32)
        >>> spike_idx, spike_vals, spike_stats = TimeseriesFeatureMixin().spike_finder(data=data, baseline=1, min_spike_amplitude=5, sample_rate=2, min_fwhm=-np.inf, min_half_width=0.0002)
        """

        spike_idxs = np.argwhere(data >= baseline + min_spike_amplitude).flatten()
        spike_idx = np.split(
            spike_idxs, np.argwhere(spike_idxs[1:] - spike_idxs[:-1] != 1).flatten() + 1
        )
        spike_dict = Dict.empty(
            key_type=types.int64,
            value_type=Dict.empty(
                key_type=types.unicode_type, value_type=types.float64
            ),
        )
        spike_vals = []
        for i in prange(len(spike_idx)):
            spike_data = data[spike_idx[i]]
            spike_amplitude = np.max(spike_data) - baseline
            half_width_idx = np.argwhere(spike_data > spike_amplitude / 2).flatten()
            spike_dict[i] = {
                "amplitude": np.max(spike_data) - baseline,
                "fwhm": (half_width_idx[-1] - half_width_idx[0]) / sample_rate,
                "half_width": half_width_idx.shape[0] / sample_rate,
            }
            spike_vals.append(spike_data)

        remove_idx = []
        for k, v in spike_dict.items():
            if (v["fwhm"] < min_fwhm) or (v["half_width"] < min_half_width):
                remove_idx.append(k)
        for idx in remove_idx:
            spike_dict.pop(idx)
        spike_idx = [i for j, i in enumerate(spike_idx) if j not in remove_idx]
        spike_vals = [i for j, i in enumerate(spike_vals) if j not in remove_idx]

        return spike_idx, spike_vals, spike_dict

    # @njit("(float32[:], types.List(types.Array(types.int64, 1, 'C')), int64, float64, float64)", fastmath=True)
    @staticmethod
    @jit(nopython=True, fastmath=True)
    def spike_train_finder(
        data: np.ndarray,
        spike_idx: list,
        sample_rate: float,
        min_spike_train_length: float = np.inf,
        max_spike_train_separation: float = np.inf,
    ):
        """
        Identify and analyze spike trains from a list of spike indices.

        This function takes spike indices and additional information, such as the data, sample rate,
        minimum spike train length, and maximum spike train separation, to identify and analyze
        spike trains in the data.

        .. note::
           - The function may return an empty dictionary if no spike trains meet the criteria.
           - A required input is ``spike_idx``, which is returned by :func:`~timeseries_features_mixin.TimeseriesFeatureMixin.spike_finder`.


        :param np.ndarray data: The data from which spike trains are extracted.
        :param types.List(types.Array(types.int64, 1, 'C')) data: A list of spike indices, typically as integer timestamps.
        :param float sample_rate: The sample rate of the data.
        :param Optional[float] min_spike_train_length: The minimum length a spike train must have to be considered. Default is set to positive infinity, meaning no minimum length is enforced.
        :param Optional[float] max_spike_train_separation: The maximum allowable separation between spikes in the same train. Default is set to positive infinity, meaning no maximum separation is enforced.
        :return DictType[int64,DictType[unicode_type,float64]]: A dictionary containing information about identified spike trains.

        Each entry in the returned dictionary is indexed by an integer, and contains the following information:
            - 'train_start_time': Start time of the spike train in seconds.
            - 'train_end_time': End time of the spike train in seconds.
            - 'train_start_obs': Start time index in observations.
            - 'train_end_obs': End time index in observations.
            - 'spike_cnt': Number of spikes in the spike train.
            - 'train_length_obs_cnt': Length of the spike train in observations.
            - 'train_length_obs_s': Length of the spike train in seconds.
            - 'train_spike_mean_lengths_s': Mean length of individual spikes in seconds.
            - 'train_spike_std_length_obs': Standard deviation of spike lengths in observations.
            - 'train_spike_std_length_s': Standard deviation of spike lengths in seconds.
            - 'train_spike_max_length_obs': Maximum spike length in observations.
            - 'train_spike_max_length_s': Maximum spike length in seconds.
            - 'train_spike_min_length_obs': Minimum spike length in observations.
            - 'train_spike_min_length_s': Minimum spike length in seconds.
            - 'train_mean_amplitude': Mean amplitude of the spike train.
            - 'train_std_amplitude': Standard deviation of spike amplitudes.
            - 'train_min_amplitude': Minimum spike amplitude.
            - 'train_max_amplitude': Maximum spike amplitude.

        :example:
        >>> data = np.array([0.1, 0.1, 0.3, 0.1, 10, 10, 8, 0.1, 0.1, 0.1, 10, 10, 8, 99, 0.1, 99, 99, 0.1]).astype(np.float32)
        >>> spike_idx, _, _ = TimeseriesFeatureMixin().spike_finder(data=data, baseline=0.3, min_spike_amplitude=0.2, sample_rate=2, min_fwhm=-np.inf, min_half_width=-np.inf)
        >>> results = TimeseriesFeatureMixin().spike_train_finder(data=data, spike_idx=typed.List(spike_idx), sample_rate=2.0, min_spike_train_length=2.0, max_spike_train_separation=2.0)
        """

        (
            l,
            r,
        ) = (
            0,
            1,
        )
        train_data_idx = []
        train_spikes_idx = []
        max_spike_train_separation = int(max_spike_train_separation * sample_rate)
        min_spike_train_length = int(min_spike_train_length * sample_rate)
        while l < len(spike_idx):
            current_train, current_spike_idx = spike_idx[l], [l]
            while r < len(spike_idx) and (
                (spike_idx[r][0] - current_train[-1]) <= max_spike_train_separation
            ):
                current_train = np.hstack((current_train, spike_idx[r]))
                current_spike_idx.append(r)
                r += 1
            l, r = r, r + 1
            train_data_idx.append(current_train)
            train_spikes_idx.append(current_spike_idx)

        spike_dict = Dict.empty(
            key_type=types.int64,
            value_type=Dict.empty(
                key_type=types.unicode_type, value_type=types.float64
            ),
        )
        for i in prange(len(train_data_idx)):
            if train_data_idx[i].shape[0] >= min_spike_train_length:
                spike_train_amps = data[train_data_idx[i]]
                spike_train_idx = [
                    k for j, k in enumerate(spike_idx) if j in train_spikes_idx[i]
                ]
                train_spike_lengths = np.array(([len(j) for j in spike_train_idx]))
                spike_dict[int(i)] = {
                    "train_start_time": float(train_data_idx[i][0] * sample_rate),
                    "train_end_time": train_data_idx[i][-1] * sample_rate,
                    "train_start_obs": train_data_idx[i][0],
                    "train_end_obs": train_data_idx[i][-1],
                    "spike_cnt": len(train_spikes_idx[i]),
                    "train_length_obs_cnt": len(spike_train_amps),
                    "train_length_obs_s": len(spike_train_amps) / sample_rate,
                    "train_spike_mean_lengths_s": np.mean(train_spike_lengths)
                    * sample_rate,
                    "train_spike_std_length_obs": np.mean(train_spike_lengths),
                    "train_spike_std_length_s": np.mean(train_spike_lengths)
                    * sample_rate,
                    "train_spike_max_length_obs": np.max(train_spike_lengths),
                    "train_spike_max_length_s": np.max(train_spike_lengths)
                    * sample_rate,
                    "train_spike_min_length_obs": np.min(train_spike_lengths),
                    "train_spike_min_length_s": np.min(train_spike_lengths)
                    * sample_rate,
                    "train_mean_amplitude": np.mean(spike_train_amps),
                    "train_std_amplitude": np.std(spike_train_amps),
                    "train_min_amplitude": np.min(spike_train_amps),
                    "train_max_amplitude": np.max(spike_train_amps),
                }

        return spike_dict

    @staticmethod
    def _adf_executor(data: np.ndarray) -> tuple:
        """
        Helper function to execute Augmented Dickey-Fuller (ADF) test on a data segment.
        Called by :meth:`timeseries_features_mixin.TimeseriesFeatureMixin.sliding_stationary_test_test`.
        """

        adfuller_results = adfuller(data)
        return adfuller_results[0], adfuller_results[1]

    @staticmethod
    def _kpss_executor(data: np.ndarray) -> tuple:
        """
        Helper function to execute Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test on a data segment.
        Called by :meth:`timeseries_features_mixin.TimeseriesFeatureMixin.sliding_stationary_test_test`.
        """

        kpss_results = kpss(data)
        return kpss_results[0], kpss_results[1]

    @staticmethod
    def _zivotandrews_executor(data: np.ndarray) -> tuple:
        """
        Helper function to execute Zivot-Andrews structural-break unit-root test on a data segment.
        Called by :meth:`timeseries_features_mixin.TimeseriesFeatureMixin.sliding_stationary_test_test`.
        """
        try:
            za_results = zivot_andrews(data)
            return za_results[0], za_results[1]
        except (np.linalg.LinAlgError, ValueError):
            return 0, 0

    @staticmethod
    def sliding_stationary(
        data: np.ndarray,
        time_windows: np.ndarray,
        sample_rate: int,
        test: Literal["ADF", "KPSS", "ZA"] = "adf",
    ) -> (np.ndarray, np.ndarray):
        """
        Perform the Augmented Dickey-Fuller (ADF), Kwiatkowski-Phillips-Schmidt-Shin (KPSS), or Zivot-Andrews test on sliding windows of time series data.
        Parallel processing using all available cores is used to accelerate computation.

        .. note::
           - ADF: A high p-value suggests non-stationarity, while a low p-value indicates stationarity.
           - KPSS: A high p-value suggests stationarity, while a low p-value indicates non-stationarity.
           - ZA: A high p-value suggests non-stationarity, while a low p-value indicates stationarity.

        :param np.ndarray data: 1-D NumPy array containing the time series data to be tested.
        :param np.ndarray time_windows: A 1-D NumPy array containing the time window sizes in seconds.
        :param np.ndarray sample_rate: The sample rate of the time series data (samples per second).
        :param Literal test: Test to perfrom: Options: 'ADF' (Augmented Dickey-Fuller), 'KPSS' (Kwiatkowski-Phillips-Schmidt-Shin), 'ZA' (Zivot-Andrews).
        :return (np.ndarray, np.ndarray): A tuple of two 2-D NumPy arrays containing test statistics and p-values. - The first array (stat) contains the ADF test statistics. - The second array (p_vals) contains the corresponding p-values

        :example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> TimeseriesFeatureMixin().sliding_stationary(data=data, time_windows=np.array([2.0]), test='KPSS', sample_rate=2)
        """

        stat = np.full((data.shape[0], time_windows.shape[0]), -1.0)
        p_vals = np.full((data.shape[0], time_windows.shape[0]), -1.0)
        if test == "ADF":
            test_func = TimeseriesFeatureMixin()._adf_executor
        elif test == "KPSS":
            test_func = TimeseriesFeatureMixin()._kpss_executor
        elif test == "ZA":
            test_func = TimeseriesFeatureMixin()._zivotandrews_executor
        else:
            type_hints = get_type_hints(TimeseriesFeatureMixin().sliding_stationary)
            raise InvalidInputError(
                msg=f'Test {test} not recognized. Options: {type_hints["test"]}'
            )
        for i in range(time_windows.shape[0]):
            window_size = int(time_windows[i] * sample_rate)
            strided_data = as_strided(
                data,
                shape=(data.shape[0] - window_size + 1, window_size),
                strides=(data.strides[0] * 1, data.strides[0]),
            )
            with multiprocessing.Pool(find_core_cnt()[0], maxtasksperchild=10) as pool:
                for cnt, result in enumerate(
                    pool.imap(test_func, strided_data, chunksize=1)
                ):
                    stat[cnt + window_size - 1, i] = result[0]
                    p_vals[cnt + window_size - 1, i] = result[1]

        return stat, p_vals

    @staticmethod
    @njit(
        [
            "(float32[:], float64, int64, float64, types.unicode_type)",
            '(float32[:], float64, int64, float64, types.misc.Omitted("mm"))',
            "(float32[:], float64, int64, types.misc.Omitted(1), types.unicode_type)",
            '(float32[:], float64, int64, types.misc.Omitted(1), types.misc.Omitted("mm"))',
        ]
    )
    def acceleration(data: np.ndarray,
                     pixels_per_mm: float,
                     fps: int,
                     time_window: float = 1,
                     unit: Literal["mm", "cm", "dm", "m"] = "mm") -> np.ndarray:

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

        :param np.ndarray data: 1D array of framewise euclidean distances.
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

    @staticmethod
    def granger_tests(
        data: pd.DataFrame,
        variables: typing.List[str],
        lag: int,
        test: Literal[
            "ssr_ftest", "ssr_chi2test", "lrtest", "params_ftest"
        ] = "ssr_chi2test",
    ) -> pd.DataFrame:
        """
        Perform Granger causality tests between pairs of variables in a DataFrame.

        This function computes Granger causality tests between pairs of variables in a DataFrame
        using the statsmodels library. The Granger causality test assesses whether one time series
        variable (predictor) can predict another time series variable (outcome). This test can help
        determine the presence of causal relationships between variables.

        .. note::
           Modified from `Selva Prabhakaran <https://www.machinelearningplus.com/time-series/granger-causality-test-in-python/>`_.

        :example:
        >>> x = np.random.randint(0, 50, (100, 2))
        >>> data = pd.DataFrame(x, columns=['r', 'k'])
        >>> TimeseriesFeatureMixin.granger_tests(data=data, variables=['r', 'k'], lag=4, test='ssr_chi2test')
        >>>     r           k
        >>>     r  1.0000  0.4312
        >>>     k  0.3102  1.0000
        """
        check_instance(
            source=TimeseriesFeatureMixin.granger_tests.__name__,
            instance=data,
            accepted_types=(pd.DataFrame,),
        )
        check_valid_lst(
            data=variables,
            source=TimeseriesFeatureMixin.granger_tests.__name__,
            valid_dtypes=(str,),
            min_len=2,
        )
        check_that_column_exist(df=data, column_name=variables, file_name="")
        check_str(
            name=TimeseriesFeatureMixin.granger_tests.__name__,
            value=test,
            options=("ssr_ftest", "ssr_chi2test", "lrtest", "params_ftest"),
        )
        check_int(
            name=TimeseriesFeatureMixin.granger_tests.__name__, value=lag, min_value=1
        )
        df = pd.DataFrame(
            np.zeros((len(variables), len(variables))),
            columns=variables,
            index=variables,
        )
        for c, r in itertools.product(df.columns, df.index):
            result = grangercausalitytests(data[[r, c]], maxlag=[lag], verbose=False)
            p_val = min([round(result[lag][0][test][1], 4) for i in range(1)])
            df.loc[r, c] = p_val
        return df

    @staticmethod
    @njit("(int32[:,:], float64[:], float64, float64)")
    def sliding_displacement(
        x: np.ndarray, time_windows: np.ndarray, fps: float, px_per_mm: float
    ) -> np.ndarray:
        """
        Calculate sliding Euclidean displacement of a body-part point over time windows.

        .. image:: _static/img/sliding_displacement.png
           :width: 600
           :align: center

        :param np.ndarray x: An array of shape (n, 2) representing the time-series sequence of 2D points.
        :param np.ndarray time_windows: Array of time windows (in seconds).
        :param float fps: The sample rate (frames per second) of the sequence.
        :param float px_per_mm: Pixels per millimeter conversion factor.
        :return np.ndarray: 1D array containing the calculated displacements.

        :example:
        >>> x = np.random.randint(0, 50, (100, 2)).astype(np.int32)
        >>> TimeseriesFeatureMixin.sliding_displacement(x=x, time_windows=np.array([1.0]), fps=1.0, px_per_mm=1.0)
        """

        results = np.full((x.shape[0], time_windows.shape[0]), -1.0)
        for i in range(time_windows.shape[0]):
            w = int(time_windows[i] * fps)
            for j in range(w, x.shape[0]):
                c, s = x[j], x[j - w]
                results[j, i] = (
                    np.sqrt((s[0] - c[0]) ** 2 + (s[1] - c[1]) ** 2)
                ) / px_per_mm
        return results.astype(np.float32)

    @staticmethod
    @njit("(float64[:], float64[:], float64[:], float64, boolean, float64)")
    def sliding_two_signal_crosscorrelation(
        x: np.ndarray,
        y: np.ndarray,
        windows: np.ndarray,
        sample_rate: float,
        normalize: bool,
        lag: float,
    ) -> np.ndarray:
        """
        Calculate sliding (lagged) cross-correlation between two signals, e.g., the movement and velocity of two animals.

        .. note::
            If no lag needed, pass lag 0.0.

        :param np.ndarray x: The first input signal.
        :param np.ndarray y: The second input signal.
        :param np.ndarray windows: Array of window lengths in seconds.
        :param float sample_rate: Sampling rate of the signals (in Hz or FPS).
        :param bool normalize: If True, normalize the signals before computing the correlation.
        :param float lag: Time lag between the signals in seconds.

        :return: 2D array of sliding cross-correlation values. Each row corresponds to a time index, and each column corresponds to a window size specified in the `windows` parameter.

        :example:
        >>> x = np.random.randint(0, 10, size=(20,))
        >>> y = np.random.randint(0, 10, size=(20,))
        >>> TimeseriesFeatureMixin.sliding_two_signal_crosscorrelation(x=x, y=y, windows=np.array([1.0, 1.2]), sample_rate=10, normalize=True, lag=0.0)
        """

        results = np.full((x.shape[0], windows.shape[0]), 0.0)
        lag = int(sample_rate * lag)
        for i in prange(windows.shape[0]):
            W_s = int(windows[i] * sample_rate)
            for cnt, (l1, r1) in enumerate(
                zip(range(0, x.shape[0] + 1), range(W_s, x.shape[0] + 1))
            ):
                l2 = l1 - lag
                if l2 < 0:
                    l2 = 0
                r2 = r1 - lag
                if r2 - l2 < W_s:
                    r2 = l2 + W_s
                X_w = x[l1:r1]
                Y_w = y[l2:r2]
                if normalize:
                    X_w = (X_w - np.mean(X_w)) / (np.std(X_w) * X_w.shape[0])
                    Y_w = (Y_w - np.mean(Y_w)) / np.std(Y_w)
                v = np.correlate(a=X_w, v=Y_w)[0]
                if np.isnan(v):
                    results[r1 - 1, i] = 0.0
                else:
                    results[int(r1 - 1), i] = v
        return results.astype(np.float32)

    @staticmethod
    def sliding_pct_in_top_n(
        x: np.ndarray, windows: np.ndarray, n: int, fps: float
    ) -> np.ndarray:
        """
        Compute the percentage of elements in the top 'n' frequencies in sliding windows of the input array.

        .. note::
          To compute percentage of elements in the top 'n' frequencies in entire array, use ``simba.mixins.statistics_mixin.Statistics.pct_in_top_n``.

        :param np.ndarray x: Input 1D array.
        :param np.ndarray windows: Array of window sizes in seconds.
        :param int n: Number of top frequencies.
        :param float fps: Sampling frequency for time convesrion.
        :return np.ndarray: 2D array of computed percentages of elements in the top 'n' frequencies for each sliding window.

        :example:
        >>> x = np.random.randint(0, 10, (100000,))
        >>> results = TimeseriesFeatureMixin.sliding_pct_in_top_n(x=x, windows=np.array([1.0]), n=4, fps=10)
        """

        check_valid_array(
            data=x,
            source=f"{TimeseriesFeatureMixin.sliding_pct_in_top_n.__name__} x",
            accepted_ndims=(1,),
            accepted_dtypes=(np.float32, np.float64, np.int64, np.int32, int, float),
        )
        check_valid_array(
            data=windows,
            source=f"{TimeseriesFeatureMixin.sliding_pct_in_top_n.__name__} windows",
            accepted_ndims=(1,),
            accepted_dtypes=(np.float32, np.float64, np.int64, np.int32, int, float),
        )
        check_int(
            name=f"{TimeseriesFeatureMixin.sliding_pct_in_top_n.__name__} n",
            value=n,
            min_value=1,
        )
        check_float(
            name=f"{TimeseriesFeatureMixin.sliding_pct_in_top_n.__name__} fps",
            value=n,
            min_value=10e-6,
        )
        results = np.full((x.shape[0], windows.shape[0]), -1.0)
        for i in range(windows.shape[0]):
            W_s = int(windows[i] * fps)
            for cnt, (l, r) in enumerate(
                zip(range(0, x.shape[0] + 1), range(W_s, x.shape[0] + 1))
            ):
                sample = x[l:r]
                cnts = np.sort(np.unique(sample, return_counts=True)[1])[-n:]
                results[int(r - 1), i] = np.sum(cnts) / sample.shape[0]
        return results
