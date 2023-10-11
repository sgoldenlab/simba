import numpy as np
from numba import njit, prange


class TimeseriesFeatureMixin(object):

    """
    Time-series methods.
    """

    def __init__(self):
        pass

    @staticmethod
    @njit("(float32[:],)")
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
        
        :math:`complexity = \sqrt{\\frac{ddx_{var}}{dx_{var}} / mobility}
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
    @njit("(float32[:], boolean)")
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
    @njit("(float32[:], float64)")
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
    @njit("(float32[:], int64, int64, )", cache=True, fastmath=True)
    def percentile_difference(
        data: np.ndarray, upper_pct: int, lower_pct: int
    ) -> float:
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

        upper_val, lower_val = np.percentile(data, upper_pct), np.percentile(
            data, lower_pct
        )
        return np.abs(upper_val - lower_val) / np.median(data)

    @staticmethod
    @njit("(float32[:], float64,)", cache=True, fastmath=True)
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
    @njit("(float32[:], int64, int64, )", cache=True, fastmath=True)
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

        upper_val, lower_val = np.percentile(data, upper_pct), np.percentile(
            data, lower_pct
        )
        return (
            np.argwhere((data <= upper_val) & (data >= lower_val)).flatten().shape[0]
            / data.shape[0]
        )
