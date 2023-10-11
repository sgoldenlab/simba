from numba import njit, prange, types
from numba.typed import List
import numpy as np

class TimeseriesFeatureMixin(object):

    """
    Time-series methods.
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
        print(target, np.mean(data), target / 2)
        return np.argwhere(np.abs(data) > target).shape[0] / data.shape[0]

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

# t = np.linspace(0, 50, int(44100 * 2.0), endpoint=False)
# sine_wave = 1.0 * np.sin(2 * np.pi * 1.0 * t).astype(np.float32)
# TimeseriesFeatureMixin().petrosian_fractal_dimension(data=sine_wave)
# #1.0000398187022719
# np.random.shuffle(sine_wave)
# TimeseriesFeatureMixin().petrosian_fractal_dimension(data=sine_wave)
# #1.0211625348743218