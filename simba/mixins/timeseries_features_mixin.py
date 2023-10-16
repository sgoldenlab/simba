from numba import njit, prange, types, typed
from numba.typed import List
import numpy as np
try:
    from typing import Literal
except:
    from typing_extensions import Literal

class TimeseriesFeatureMixin(object):

    """
    Time-series methods focused on signal complexity in sliding windows.

    .. note::
       Many method has numba typed `signatures <https://numba.pydata.org/numba-doc/latest/reference/types.html>`_ to decrease
       compilation time. Make sure to pass the correct dtypes as indicated by signature decorators.

    .. important::
       See references for mature packages computing more extensive circular measurements

       .. [1] `cesium <https://github.com/cesium-ml/cesium>`_.
       .. [2] `eeglib <https://github.com/Xiul109/eeglib>`_.
       .. [3] `antropy <https://github.com/raphaelvallat/antropy>`_.
    """

    def __init__(self):
        pass

    @staticmethod
    @njit('(float32[:],)')
    def hjort_parameters(data: np.ndarray):
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

        def diff(x):
            return x[1:] - x[:-1]

        dx = diff(data)
        ddx = diff(dx)
        x_var, dx_var = np.var(data), np.var(dx)
        ddx_var = np.var(ddx)

        mobility = np.sqrt(dx_var / x_var)
        complexity = np.sqrt(ddx_var / dx_var) / mobility
        activity = np.var(data)

        return activity, mobility, complexity

    @staticmethod
    @njit('(float32[:], float64[:], int64)')
    def sliding_hjort_parameters(data: np.ndarray,
                                 window_sizes: np.ndarray,
                                 sample_rate: int) -> np.ndarray:
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
            for l, r in zip(prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)):
                sample = data[l:r]
                dx = sample[1:] - sample[:-1]
                ddx = dx[1:] - dx[:-1]
                x_var, dx_var = np.var(sample), np.var(dx)
                ddx_var = np.var(ddx)
                mobility = np.sqrt(dx_var / x_var)
                complexity = np.sqrt(ddx_var / dx_var) / mobility
                activity = np.var(sample)
                results[0, r + 1, i] = mobility
                results[1, r + 1, i] = complexity
                results[2, r + 1, i] = activity

        return results.astype(np.float32)

    @staticmethod
    @njit('(float32[:], boolean)')
    def local_maxima_minima(data: np.ndarray, maxima: bool) -> np.ndarray:
        """
        Jitted compute of the local maxima or minima defined as values which are higher or lower than immediately preceding and proceeding time-series neighbors, repectively.
        Returns 2D np.ndarray with columns representing idx and values of local maxima.

        :param np.ndarray data: Time-series data.
        :param bool maxima: If True, returns maxima. Else, minima.
        :return np.ndarray: 2D np.ndarray with columns representing idx in input data in first column and values of local maxima in second column

        :example:
        >>> data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
        >>> TimeseriesFeatureMixin().local_maxima_minima(data=data, maxima=True)
        >>> [[1, 7.5], [4, 7.5], [9, 9.5]]
        >>> TimeseriesFeatureMixin().local_maxima_minima(data=data, maxima=False)
        >>> [[0, 3.9], [2, 4.2], [5, 3.9]]

        .. image:: _static/img/local_maxima_minima.png
           :width: 600
           :align: center

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

        return results[np.argwhere(results[:, 0].T != -1).flatten()]

    @staticmethod
    @njit('(float32[:], float64)')
    def crossings(data: np.ndarray, val: float) -> int:
        """
        Jitted compute of the count in time-series where sequential values crosses a defined value.

        :param np.ndarray data: Time-series data.
        :param float val: Cross value. E.g., to count the number of zero-crossings, pass `0`.
        :return int: Count of events where sequential values crosses ``val``.

        :example:
        >>> data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
        >>> TimeseriesFeatureMixin().crossings(data=data, val=7)
        >>> 5

        .. image:: _static/img/crossings.png
           :width: 600
           :align: center
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
    @njit('(float32[:], float64,  float64[:], int64,)')
    def sliding_crossings(data: np.ndarray,
                          val: float,
                          window_sizes: np.ndarray,
                          sample_rate: int) -> np.ndarray:
        """
        Compute the number of crossings over sliding windows in a data array.

        Computes the number of times a value in the data array crosses a given threshold
        value within sliding windows of varying sizes. The number of crossings is computed for each
        window size and stored in the result array.

        :param np.ndarray data: Input data array.
        :param float val: Threshold value for crossings.
        :param np.ndarray window_sizes: Array of window sizes (in seconds).
        :param int sample_rate: Sampling rate of the data in samples per second.
        :return np.ndarray: An array containing the number of crossings for each window size and data point. The shape of the result array is (data.shape[0], window_sizes.shape[0]).
        """
        results = np.full((data.shape[0], window_sizes.shape[0]), -1.0)
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)):
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
    @njit('(float32[:], int64, int64, )', cache=True, fastmath=True)
    def percentile_difference(data: np.ndarray, upper_pct: int, lower_pct: int) -> float:
        """
        Jitted compute of the difference between the ``upper`` and ``lower`` percentiles of the data as
        a percentage of the median value.

        :parameter np.ndarray data: 1D array of representing time-series.
        :parameter int upper_pct: Upper-boundary percentile.
        :parameter int lower_pct: Lower-boundary percentile.
        :returns float: The difference between the ``upper`` and ``lower`` percentiles of the data as a percentage of the median value.

        .. note::
           Adapted from `cesium <https://github.com/cesium-ml/cesium>`_.

        :examples:
        >>> data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
        >>> TimeseriesFeatureMixin().percentile_difference(data=data, upper_pct=95, lower_pct=5)
        >>> 0.7401574764125177

        .. image:: _static/img/percentile_difference.png
           :width: 600
           :align: center

        """

        upper_val, lower_val = np.percentile(data, upper_pct), np.percentile(data, lower_pct)
        return np.abs(upper_val - lower_val) / np.median(data)

    @staticmethod
    @njit('(float32[:], int64, int64, float64[:], int64, )', cache=True, fastmath=True)
    def sliding_percentile_difference(data: np.ndarray,
                                      upper_pct: int,
                                      lower_pct: int,
                                      window_sizes: np.ndarray,
                                      sample_rate: int) -> np.ndarray:
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
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)):
                sample = data[l:r]
                upper_val, lower_val = np.percentile(sample, upper_pct), np.percentile(sample, lower_pct)
                median = np.median(sample)
                if median != 0:
                    results[r - 1, i] = np.abs(upper_val - lower_val) / median
                else:
                    results[r - 1, i] = -1.0

        return results.astype(np.float32)

    @staticmethod
    @njit('(float32[:], float64,)', cache=True, fastmath=True)
    def percent_beyond_n_std(data: np.ndarray, n: float) -> float:
        """
        Jitted compute of the ratio of values in time-series more than N standard deviations from the mean of the time-series.

        :parameter np.ndarray data: 1D array representing time-series.
        :parameter float n: Standard deviation cut-off.
        :returns float: Ratio of values in ``data`` that fall more than ``n`` standard deviations from mean of ``data``.

        .. note::
           Adapted from `cesium <https://github.com/cesium-ml/cesium>`_.

        :examples:
        >>> data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
        >>> TimeseriesFeatureMixin().percent_beyond_n_std(data=data, n=1)
        >>> 0.1

        .. image:: _static/img/percent_beyond_n_std.png
           :width: 600
           :align: center

        """

        target = (np.std(data) * n) + np.mean(data)
        return np.argwhere(np.abs(data) > target).shape[0] / data.shape[0]

    @staticmethod
    @njit('(float32[:], float64, float64[:], int64,)', cache=True, fastmath=True)
    def sliding_percent_beyond_n_std(data: np.ndarray,
                                     n: float,
                                     window_sizes: np.ndarray,
                                     sample_rate: int) -> np.ndarray:

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
            for l, r in zip(prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)):
                sample = data[l:r]
                results[r - 1, i] = np.argwhere(np.abs(sample) > target).shape[0] / sample.shape[0]

        return results.astype(np.float32)

    @staticmethod
    @njit('(float32[:], int64, int64, )', cache=True, fastmath=True)
    def percent_in_percentile_window(data: np.ndarray, upper_pct: int, lower_pct: int):
        """
        Jitted compute of the ratio of values in time-series that fall between the ``upper`` and ``lower`` percentile.

        :parameter np.ndarray data: 1D array of representing time-series.
        :parameter int upper_pct: Upper-boundary percentile.
        :parameter int lower_pct: Lower-boundary percentile.
        :returns float: Ratio of values in ``data`` that fall within ``upper_pct`` and ``lower_pct`` percentiles.

        .. note::
           Adapted from `cesium <https://github.com/cesium-ml/cesium>`_.

        :example:
        >>> data = np.array([3.9, 7.5,  4.2, 6.2, 7.5, 3.9, 6.2, 6.5, 7.2, 9.5]).astype(np.float32)
        >>> TimeseriesFeatureMixin().percent_in_percentile_window(data, upper_pct=70, lower_pct=30)
        >>> 0.4


        .. image:: _static/img/percent_in_percentile_window.png
           :width: 600
           :align: center
        """

        upper_val, lower_val = np.percentile(data, upper_pct), np.percentile(data, lower_pct)
        return np.argwhere((data <= upper_val) & (data >= lower_val)).flatten().shape[0] / data.shape[0]

    @staticmethod
    @njit('(float32[:], int64, int64, float64[:], int64)', cache=True, fastmath=True)
    def sliding_percent_in_percentile_window(data: np.ndarray,
                                             upper_pct: int,
                                             lower_pct: int,
                                             window_sizes: np.ndarray,
                                             sample_rate: int):
        """
        Jitted compute of the percentage of data points falling within a percentile window in a sliding manner.

        The function computes the percentage of data points within the specified percentile window for each position in the time series
        using various window sizes. It returns a 2D array where each row corresponds to a position in the time series, and each column
        corresponds to a different window size. The results are given as a percentage of data points within the percentile window.

        :param np.ndarray data : The input time series data.
        :param int upper_pct: The upper percentile value for the window (e.g., 95 for the 95th percentile).
        :param int lower_pct (int): The lower percentile value for the window (e.g., 5 for the 5th percentile).
        :param np.ndarray window_sizes: An array of window sizes (in seconds) to use for the sliding calculation.
        :param int sample_rate: The sampling rate (samples per second) of the time series data.
        :return np.ndarray: A 2D array containing the percentage of data points within the percentile window for each window size.

        """
        results = np.full((data.shape[0], window_sizes.shape[0]), -1.0)
        upper_val, lower_val = np.percentile(data, upper_pct), np.percentile(data, lower_pct)
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)):
                sample = data[l:r]
                results[r - 1, i] = np.argwhere((sample <= upper_val) & (sample >= lower_val)).flatten().shape[0] / sample.shape[0]

        return results.astype(np.float32)

    @staticmethod
    @njit('(float32[:],)', fastmath=True, cache=True)
    def petrosian_fractal_dimension(data: np.ndarray) -> float:
        """
        Calculate the Petrosian Fractal Dimension (PFD) of a given time series data. The PFD is a measure of the
        irregularity or self-similarity of a time series. Larger values indicate higher complexity. Lower values indicate lower complexity.

        :parameter np.ndarray data: A 1-dimensional numpy array containing the time series data.
        :returns float: The Petrosian Fractal Dimension of the input time series.

        .. note::
           - The PFD is computed based on the number of sign changes in the first derivative of the time series.
           - If the input data is empty or no sign changes are found, the PFD is returned as -1.0.
           - Adapted from `eeglib <https://github.com/Xiul109/eeglib/>`_.

        .. math::
           PFD = \\frac{\\log_{10}(N)}{\\log_{10}(N) + \\log_{10}\\left(\\frac{N}{N + 0.4 \\cdot zC}\\right)}

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

        return np.log10(data.shape[0]) / (np.log10(data.shape[0]) + np.log10(data.shape[0] / (data.shape[0] + 0.4 * zC)))

    @staticmethod
    @njit('(float32[:], float64[:], int64)', fastmath=True, cache=True)
    def sliding_petrosian_fractal_dimension(data: np.ndarray,
                                            window_sizes: np.ndarray,
                                            sample_rate: int) -> np.ndarray:
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
            for l, r in zip(prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)):
                sample = (data[l:r] - np.min(data[l:r])) / (np.max(data[l:r]) - np.min(data[l:r]))
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
                        results[r - 1, i] = np.log10(sample.shape[0]) / (np.log10(sample.shape[0]) + np.log10(
                            sample.shape[0] / (sample.shape[0] + 0.4 * zC)))

        return results.astype(np.float32)

    @staticmethod
    @njit('(float32[:], int64)')
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
        x = np.hstack((-np.log(np.arange(2, kmax + 1)).reshape(-1, 1).astype(np.float32),
                       np.ones(kmax - 1).reshape(-1, 1).astype(np.float32)))
        for k in prange(2, kmax + 1):
            Lk = np.zeros(k)
            for m in range(0, k):
                Lmk = 0
                for i in range(1, (N - m) // k):
                    Lmk += abs(data[m + i * k] - data[m + i * k - k])
                Lk[m] = Lmk * (N - 1) / (((N - m) // k) * k * k)
            Laux = np.mean(Lk)
            Laux = 0.01 / k if Laux == 0 else Laux
            L[k - 2] = np.log(Laux)

        return np.linalg.lstsq(x, L.astype(np.float32))[0][0]

    @staticmethod
    @njit('(float32[:], int64, int64,)', fastmath=True)
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
    @njit('(float32[:],)', fastmath=True)
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
        return np.sum(diff[1:])

    @staticmethod
    @njit('(float32[:], float64[:], int64)', fastmath=True)
    def sliding_line_length(data: np.ndarray,
                            window_sizes: np.ndarray,
                            sample_rate: int) -> np.ndarray:

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
            for l, r in zip(prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)):
                sample = data[l:r]
                results[r - 1, i] = np.sum(np.abs(np.diff(sample.astype(np.float64))))
        return results.astype(np.float32)
    
    @staticmethod
    @njit('(float32[:], float64[:], int64)', fastmath=True, cache=True)
    def sliding_variance(data: np.ndarray,
                         window_sizes: np.ndarray,
                         sample_rate: int):
        """
        Jitted compute of the variance of data within sliding windows of varying sizes applied to
        the input data array. Variance is a measure of data dispersion or spread.

        :param data: 1d input data array.
        :param window_sizes: Array of window sizes (in seconds).
        :param sample_rate: Sampling rate of the data in samples per second.
        :return: Variance values for each window size and data point. The shape of the result array is (data.shape[0], window_sizes.shape[0]).
        """

        results = np.full((data.shape[0], window_sizes.shape[0]), -1.0)
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(prange(0, data.shape[0] + 1), prange(window_size, data.shape[0] + 1)):
                sample = (data[l:r] - np.min(data[l:r])) / (np.max(data[l:r]) - np.min(data[l:r]))
                results[r - 1, i] = np.var(sample)

        return results.astype(np.float32)

    @staticmethod
    @njit('(float32[:], float64[:], int64, types.ListType(types.unicode_type))', fastmath=True, cache=True)
    def sliding_descriptive_statistics(data: np.ndarray,
                                       window_sizes: np.ndarray,
                                       sample_rate: int,
                                       statistics: Literal['var', 'max', 'min', 'std', 'median', 'mean', 'mad']):

        """
        Jitted compute of descriptive statistics over sliding windows in 1D data array.

        Computes various descriptive statistics (e.g., variance, maximum, minimum, standard deviation,
        median, mean, median absolute deviation) for sliding windows of varying sizes applied to the input data array.

        :param np.ndarray data: 1D input data array.
        :param np.ndarray window_sizes: Array of window sizes (in seconds).
        :param int sample_rate: Sampling rate of the data in samples per second.
        :param types.ListType(types.unicode_type) statistics: List of statistics to compute. Options: 'var', 'max', 'min', 'std', 'median', 'mean', 'mad'.
        :return np.ndarray: Array containing the selected descriptive statistics for each window size, data point, and statistic type. The shape of the result array is (len(statistics), data.shape[0], window_sizes.shape[0).

        .. note::
           - The `statistics` parameter should be a list containing one or more of the following statistics:
           'var' (variance), 'max' (maximum), 'min' (minimum), 'std' (standard deviation), 'median' (median),
           'mean' (mean), 'mad' (median absolute deviation).
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
                    if statistics[j] == 'var':
                        results[j, r - 1, i] = np.var(sample)
                    elif statistics[j] == 'max':
                        results[j, r - 1, i] = np.max(sample)
                    elif statistics[j] == 'min':
                        results[j, r - 1, i] = np.min(sample)
                    elif statistics[j] == 'std':
                        results[j, r - 1, i] = np.std(sample)
                    elif statistics[j] == 'median':
                        results[j, r - 1, i] = np.median(sample)
                    elif statistics[j] == 'mean':
                        results[j, r - 1, i] = np.mean(sample)
                    elif statistics[j] == 'mad':
                        results[j, r - 1, i] = np.median(np.abs(sample - np.median(sample)))

        return results.astype(np.float32)

    @staticmethod
    def dominant_frequencies(data: np.ndarray,
                             fps: float,
                             k: int,
                             window_function: Literal['Hann', 'Hamming', 'Blackman'] = None):

        """ Find the K dominant frequencies within a feature vector """

        if window_function == 'Hann':
            data = data * np.hanning(len(data))
        elif window_function == 'Hamming':
            data = data * np.hamming(len(data))
        elif window_function == 'Blackman':
            data = data * np.blackman(len(data))
        fft_result = np.fft.fft(data)
        frequencies = np.fft.fftfreq(data.shape[0], 1 / fps)
        magnitude = np.abs(fft_result)

        return frequencies[np.argsort(magnitude)[-(k + 1):-1]]

# t = np.linspace(0, 50, int(44100 * 2.0), endpoint=False)
# sine_wave = 1.0 * np.sin(2 * np.pi * 1.0 * t).astype(np.float32)
# TimeseriesFeatureMixin().petrosian_fractal_dimension(data=sine_wave)
# #1.0000398187022719
# np.random.shuffle(sine_wave)
# TimeseriesFeatureMixin().petrosian_fractal_dimension(data=sine_wave)
# #1.0211625348743218