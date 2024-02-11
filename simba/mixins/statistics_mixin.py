__author__ = "Simon Nilsson"

from typing import Optional, Tuple, Union

from sklearn.neighbors import LocalOutlierFactor

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
from numba import (bool_, float32, float64, int8, jit, njit, objmode, optional,
                   prange, typed, types)
from scipy import stats

from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import (check_float, check_instance, check_int,
                                check_str, check_valid_array)
from simba.utils.data import bucket_data, fast_mean_rank, fast_minimum_rank
from simba.utils.enums import Options
from simba.utils.errors import CountError


class Statistics(FeatureExtractionMixin):
    """
    Primarily frequentist statistics methods used for feature extraction or drift assessment.

    .. note::

       Most methods implemented using `numba parallelization <https://numba.pydata.org/>`_ for improved run-times. See
       `line graph <https://github.com/sgoldenlab/simba/blob/master/docs/_static/img/statistics_runtimes.png>`_ below for expected run-times for a few methods included in this class.

       Most method has numba typed `signatures <https://numba.pydata.org/numba-doc/latest/reference/types.html>`_ to decrease
       compilation time. Make sure to pass the correct dtypes as indicated by signature decorators. If dtype is not specified at
       array creation, it will typically be ``float64`` or ``int64``. As most methods here use ``float32`` for the input data argument,
       make sure to downcast.

    .. image:: _static/img/statistics_runtimes.png
       :width: 1200
       :align: center

    """

    def __init__(self):
        FeatureExtractionMixin.__init__(self)

    @staticmethod
    @jit(nopython=True)
    def _hist_1d(data: np.ndarray, bin_count: int, range: np.ndarray):
        """
        Jitted helper to compute 1D histograms with counts.

        .. note::
           For non-heuristic rules for bin counts and bin ranges, see ``simba.data.freedman_diaconis`` or simba.data.bucket_data``.

        :parameter np.ndarray data: 1d array containing feature values.
        :parameter int bins: The number of bins.
        :parameter: np.ndarray range: 1d array with two values representing minimum and maximum value to bin.
        """

        hist = np.histogram(data, bin_count, (range[0], range[1]))[0]
        return hist

    @staticmethod
    @njit("(float32[:], float64, float64)", cache=True)
    def rolling_independent_sample_t(
        data: np.ndarray, time_window: float, fps: float
    ) -> np.ndarray:
        """
        Jitted compute independent-sample t-statistics for sequentially binned values in a time-series.
        E.g., compute t-test statistics when comparing ``Feature N`` in the current 1s
        time-window, versus ``Feature N`` in the previous 1s time-window.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter int group_size_s: The size of the buckets in seconds.
        :parameter int fps: Frame-rate of recorded video.

        .. image:: _static/img/independent_t_tests.png
           :width: 700
           :align: center

        .. attention::
           Each window is compared to the prior window. Output for the windows without a prior window (the first window) is ``-1``.

        :example:
        >>> data_1, data_2 = np.random.normal(loc=10, scale=2, size=10), np.random.normal(loc=20, scale=2, size=10)
        >>> data = np.hstack([data_1, data_2])
        >>> Statistics().rolling_independent_sample_t(data, time_window=1, fps=10)
        >>> [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389])

        """

        results = np.full((data.shape[0]), -1.0)
        window_size = int(time_window * fps)
        data = np.split(data, list(range(window_size, data.shape[0], window_size)))
        for cnt, i in enumerate(prange(1, len(data))):
            start, end = int((cnt + 1) * window_size), int(
                ((cnt + 1) * window_size) + window_size
            )
            mean_1, mean_2 = np.mean(data[i - 1]), np.mean(data[i])
            stdev_1, stdev_2 = np.std(data[i - 1]), np.std(data[i])
            pooled_std = np.sqrt(
                ((len(data[i - 1]) - 1) * stdev_1**2 + (len(data[i]) - 1) * stdev_2**2)
                / (len(data[i - 1]) + len(data[i]) - 2)
            )
            results[start:end] = (mean_1 - mean_2) / (
                pooled_std * np.sqrt(1 / len(data[i - 1]) + 1 / len(data[i]))
            )
        return results

    @staticmethod
    @njit(
        [
            (float32[:], float32[:], float64[:, :]),
            (float32[:], float32[:], types.misc.Omitted(None)),
        ]
    )
    def independent_samples_t(
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        critical_values: Optional[np.ndarray] = None,
    ) -> (float, Union[None, bool]):
        """
        Jitted compute independent-samples t-test statistic and boolean significance between two distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter ndarray critical_values: 2d array where the first column represents degrees of freedom and second column represents critical values.
        :returns (float Union[None, bool]) t_statistic, p_value: Representing t-statistic and associated probability value. p_value is ``None`` if critical_values is None. Else True or False with True representing significant.

        .. note::
           Critical values are stored in simba.assets.lookups.critical_values_**.pickle

        :example:
        >>> sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
        >>> sample_2 = np.array([2, 5, 10, 4, 8, 10, 7, 10, 7, 10, 10])
        >>> Statistics().independent_samples_t(sample_1=sample_1, sample_2=sample_2)
        >>> (-2.5266046804590183, None)
        >>> critical_values = pickle.load(open("simba/assets/lookups/critical_values_05.pickle","rb"))['independent_t_test']['one_tail'].values
        >>> Statistics().independent_samples_t(sample_1=sample_1, sample_2=sample_2, critical_values=critical_values)
        >>> (-2.5266046804590183, True)
        """

        significance_bool = None
        m1, m2 = np.mean(sample_1), np.mean(sample_2)
        std_1 = np.sqrt(np.sum((sample_1 - m1) ** 2) / (len(sample_1) - 1))
        std_2 = np.sqrt(np.sum((sample_2 - m2) ** 2) / (len(sample_2) - 1))
        pooled_std = np.sqrt(
            ((len(sample_1) - 1) * std_1**2 + (len(sample_2) - 1) * std_2**2)
            / (len(sample_1) + len(sample_2) - 2)
        )
        t_statistic = (m1 - m2) / (
            pooled_std * np.sqrt(1 / len(sample_1) + 1 / len(sample_2))
        )
        if critical_values is not None:
            dof = (sample_1.shape[0] + sample_2.shape[0]) - 2
            critical_value = np.interp(
                dof, critical_values[:, 0], critical_values[:, 1]
            )
            if critical_value < abs(t_statistic):
                significance_bool = True
            else:
                significance_bool = False

        return t_statistic, significance_bool

    @staticmethod
    @njit("(float64[:], float64[:])", cache=True)
    def cohens_d(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Jitted compute of Cohen's d between two distributions

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns float: Cohens D statistic.

        :example:
        >>> sample_1 = [2, 4, 7, 3, 7, 35, 8, 9]
        >>> sample_2 = [4, 8, 14, 6, 14, 70, 16, 18]
        >>> Statistics().cohens_d(sample_1=sample_1, sample_2=sample_2)
        >>> -0.5952099775170546
        """
        return (np.mean(sample_1) - np.mean(sample_2)) / (
            np.sqrt((np.std(sample_1) ** 2 + np.std(sample_2) ** 2) / 2)
        )

    @staticmethod
    @njit("(float64[:], float64[:], float64)", cache=True)
    def rolling_cohens_d(
        data: np.ndarray, time_windows: np.ndarray, fps: float
    ) -> np.ndarray:
        """
        Jitted compute of rolling Cohen's D statistic comparing the current time-window of
        size N to the preceding window of size N.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter np.ndarray[ints] time_window: Time windows to compute ANOVAs for in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :returns np.ndarray: Array of size data.shape[0] x window_sizes.shape[1] with Cohens D.

        :example:
        >>> sample_1, sample_2 = np.random.normal(loc=10, scale=1, size=4), np.random.normal(loc=11, scale=2, size=4)
        >>> sample = np.hstack((sample_1, sample_2))
        >>> Statistics().rolling_cohens_d(data=sample, window_sizes=np.array([1]), fps=4)
        >>> [[0.],[0.],[0.],[0.],[0.14718302],[0.14718302],[0.14718302],[0.14718302]])
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(
                data, list(range(window_size, data.shape[0], window_size))
            )
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[
                    j
                ].astype(np.float32)
                d = (np.mean(sample_1) - np.mean(sample_2)) / (
                    np.sqrt((np.std(sample_1) ** 2 + np.std(sample_2) ** 2) / 2)
                )
                results[window_start:window_end, i] = d
        return results

    @staticmethod
    @njit("(float32[:], float64, float64)")
    def rolling_two_sample_ks(
        data: np.ndarray, time_window: float, fps: float
    ) -> np.ndarray:
        """
        Jitted compute Kolmogorov two-sample statistics for sequentially binned values in a time-series.
        E.g., compute KS statistics when comparing ``Feature N`` in the current 1s time-window, versus ``Feature N`` in the previous 1s time-window.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter float time_window: The size of the buckets in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :return np.ndarray: Array of size data.shape[0] with KS statistics

        :example:
        >>> data = np.random.randint(low=0, high=100, size=(200)).astype('float32')
        >>> results = Statistics().rolling_two_sample_ks(data=data, time_window=1, fps=30)
        """

        window_size, results = int(time_window * fps), np.full((data.shape[0]), -1.0)
        data = np.split(data, list(range(window_size, data.shape[0], window_size)))
        for cnt, i in enumerate(prange(1, len(data))):
            start, end = int((cnt + 1) * window_size), int(
                ((cnt + 1) * window_size) + window_size
            )
            sample_1, sample_2 = data[i - 1], data[i]
            combined_samples = np.sort(np.concatenate((sample_1, sample_2)))
            ecdf_sample_1 = np.searchsorted(
                sample_1, combined_samples, side="right"
            ) / len(sample_1)
            ecdf_sample_2 = np.searchsorted(
                sample_2, combined_samples, side="right"
            ) / len(sample_2)
            ks = np.max(np.abs(ecdf_sample_1 - ecdf_sample_2))
            results[start:end] = ks
        return results

    @staticmethod
    @njit(
        [
            (float32[:], float32[:], float64[:, :]),
            (float32[:], float32[:], types.misc.Omitted(None)),
        ]
    )
    def two_sample_ks(
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        critical_values: Optional[float64[:, :]] = None,
    ) -> (float, Union[bool, None]):
        """
        Compute the two-sample Kolmogorov-Smirnov (KS) test statistic and, optionally, test for statistical significance.

        The two-sample KS test is a non-parametric test that compares the cumulative distribution functions (ECDFs) of two independent samples to assess whether they come from the same distribution.

        :param np.ndarray data: The first sample array for the KS test.
        :param np.ndarray data: The second sample array for the KS test.
        :param Optional[bool] time_windows: An array of critical values for the KS test. If provided, the function will also check the significance of the KS statistic against the critical values. Default: None.
        :returns (float Union[bool, None]): Returns a tuple containing the KS statistic and a boolean indicating whether the test is statistically significant.

        :example:
        >>> sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10]).astype(np.float32)
        >>> sample_2 = np.array([10, 5, 10, 4, 8, 10, 7, 10, 7, 10, 10]).astype(np.float32)
        >>> critical_values = pickle.load(open("simba/assets/lookups/critical_values_5.pickle", "rb"))['two_sample_KS']['one_tail'].values
        >>> two_sample_ks(sample_1=sample_1, sample_2=sample_2, critical_values=critical_values)
        >>> (0.7272727272727273, True)
        """
        significance_bool = None
        combined_samples = np.sort(np.concatenate((sample_1, sample_2)))
        ecdf_sample_1 = np.searchsorted(sample_1, combined_samples, side="right") / len(
            sample_1
        )
        ecdf_sample_2 = np.searchsorted(sample_2, combined_samples, side="right") / len(
            sample_2
        )
        ks = np.max(np.abs(ecdf_sample_1 - ecdf_sample_2))
        if critical_values is not None:
            combined_sample_size = len(sample_1) + len(sample_2)
            critical_value = np.interp(
                combined_sample_size, critical_values[:, 0], critical_values[:, 1]
            )
            if critical_value < abs(ks):
                significance_bool = True
            else:
                significance_bool = False
        return (ks, significance_bool)

    @staticmethod
    @jit(nopython=True)
    def one_way_anova(
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        critical_values: Optional[np.ndarray] = None,
    ) -> (float, float):
        """
        Jitted compute of one-way ANOVA F statistics and associated p-value for two distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns (float float): Representing ANOVA F statistic and associated probability value.

        :example:
        >>> sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
        >>> sample_2 = np.array([8, 5, 5, 8, 8, 9, 10, 1, 7, 10, 10])
        >>> Statistics().one_way_anova(sample_1=sample_2, sample_2=sample_1)
        """

        significance_bool = None
        n1, n2 = len(sample_1), len(sample_2)
        m1, m2 = np.mean(sample_1), np.mean(sample_2)
        ss_between = (
            n1 * (m1 - np.mean(np.concatenate((sample_1, sample_2)))) ** 2
            + n2 * (m2 - np.mean(np.concatenate((sample_1, sample_2)))) ** 2
        )
        ss_within = np.sum((sample_1 - m1) ** 2) + np.sum((sample_2 - m2) ** 2)
        df_between, df_within = 1, n1 + n2 - 2
        ms_between, ms_within = ss_between / df_between, ss_within / df_within
        f = ms_between / ms_within
        if critical_values is not None:
            critical_values = critical_values[:, np.array([0, df_between])]
            critical_value = np.interp(
                df_within, critical_values[:, 0], critical_values[:, 1]
            )
            if f > critical_value:
                significance_bool = True
            else:
                significance_bool = False

        return (f, significance_bool)

    @staticmethod
    @njit("(float32[:], float64[:], float64)", cache=True)
    def rolling_one_way_anova(
        data: np.ndarray, time_windows: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Jitted compute of rolling one-way ANOVA F-statistic comparing the current time-window of
        size N to the preceding window of size N.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter np.ndarray[ints] time_windows: Time windows to compute ANOVAs for in seconds.
        :parameter int fps: Frame-rate of recorded video.

        .. image:: _static/img/rolling_anova.png
           :width: 600
           :align: center

        :example:
        >>> sample = np.random.normal(loc=10, scale=1, size=10).astype(np.float32)
        >>> Statistics().rolling_one_way_anova(data=sample, time_windows=np.array([1.0]), fps=2)
        >>> [[0.00000000e+00][0.00000000e+00][2.26221263e-06][2.26221263e-06][5.39119950e-03][5.39119950e-03][1.46725486e-03][1.46725486e-03][1.16392111e-02][1.16392111e-02]]
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(
                data, list(range(window_size, data.shape[0], window_size))
            )
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[
                    j
                ].astype(np.float32)
                n1, n2 = len(sample_1), len(sample_2)
                m1, m2 = np.mean(sample_1), np.mean(sample_2)
                ss_between = (
                    n1 * (m1 - np.mean(np.concatenate((sample_1, sample_2)))) ** 2
                    + n2 * (m2 - np.mean(np.concatenate((sample_1, sample_2)))) ** 2
                )
                ss_within = np.sum((sample_1 - m1) ** 2) + np.sum((sample_2 - m2) ** 2)
                df_between, df_within = 1, n1 + n2 - 2
                ms_between, ms_within = ss_between / df_between, ss_within / df_within
                f = ms_between / ms_within
                results[window_start:window_end, i] = f

        return results

    def kullback_leibler_divergence(
        self,
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        fill_value: int = 1,
        bucket_method: Literal[
            "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"
        ] = "auto",
    ) -> float:
        """
        Compute Kullback-Leibler divergence between two distributions.

        .. note::
           Empty bins (0 observations in bin) in is replaced with ``fill_value``.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns float: Kullback-Leibler divergence between ``sample_1`` and ``sample_2``
        """

        bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
        sample_1_hist = self._hist_1d(
            data=sample_1,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
        )
        sample_2_hist = self._hist_1d(
            data=sample_2,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
        )
        sample_1_hist[sample_1_hist == 0] = fill_value
        sample_2_hist[sample_2_hist == 0] = fill_value
        sample_1_hist, sample_2_hist = sample_1_hist / np.sum(
            sample_1_hist
        ), sample_2_hist / np.sum(sample_2_hist)
        return stats.entropy(pk=sample_1_hist, qk=sample_2_hist)

    def rolling_kullback_leibler_divergence(
        self,
        data: np.ndarray,
        time_windows: np.ndarray,
        fps: int,
        fill_value: int = 1,
        bucket_method: Literal[
            "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"
        ] = "auto",
    ) -> np.ndarray:
        """
        Compute rolling Kullback-Leibler divergence comparing the current time-window of
        size N to the preceding window of size N.

        .. note::
           Empty bins (0 observations in bin) in is replaced with ``fill_value``.

        :parameter ndarray sample_1: 1d array representing feature values.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :parameter np.ndarray[floats] time_windows: Time windows to compute JS for in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :returns np.ndarray: Size data.shape[0] x window_sizes.shape with Kullback-Leibler divergence. Columns represents different tiem windows.

        :example:
        >>> sample_1, sample_2 = np.random.normal(loc=10, scale=700, size=5), np.random.normal(loc=50, scale=700, size=5)
        >>> data = np.hstack((sample_1, sample_2))
        >>> Statistics().rolling_kullback_leibler_divergence(data=data, time_windows=np.array([1]), fps=2)
        """

        check_valid_array(data=data, source=self.__class__.__name__, accepted_sizes=[1])
        check_valid_array(
            data=time_windows, source=self.__class__.__name__, accepted_sizes=[1]
        )
        check_int(name=f"{self.__class__.__name__} fps", value=fps, min_value=1)
        check_str(
            name=f"{self.__class__.__name__} bucket_method",
            value=bucket_method,
            options=Options.BUCKET_METHODS.value,
        )

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(
                data, list(range(window_size, data.shape[0], window_size))
            )
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[
                    j
                ].astype(np.float32)
                bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
                sample_1_hist = self._hist_1d(
                    data=sample_1,
                    bin_count=bin_count,
                    range=np.array([0, int(bin_width * bin_count)]),
                )
                sample_2_hist = self._hist_1d(
                    data=sample_2,
                    bin_count=bin_count,
                    range=np.array([0, int(bin_width * bin_count)]),
                )
                sample_1_hist[sample_1_hist == 0] = fill_value
                sample_2_hist[sample_2_hist == 0] = fill_value
                sample_1_hist, sample_2_hist = sample_1_hist / np.sum(
                    sample_1_hist
                ), sample_2_hist / np.sum(sample_2_hist)
                kl = stats.entropy(pk=sample_1_hist, qk=sample_2_hist)
                results[window_start:window_end, i] = kl
        return results

    def jensen_shannon_divergence(
        self,
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        bucket_method: Literal[
            "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"
        ] = "auto",
    ) -> float:
        """
        Compute Jensen-Shannon divergence between two distributions. Useful for (i) measure drift in datasets, and (ii) featurization of distribution shifts across
        sequential time-bins.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators.
        :returns float: Jensen-Shannon divergence between ``sample_1`` and ``sample_2``

        :example:
        >>> sample_1, sample_2 = np.array([1, 2, 3, 4, 5, 10, 1, 2, 3]), np.array([1, 5, 10, 9, 10, 1, 10, 6, 7])
        >>> Statistics().jensen_shannon_divergence(sample_1=sample_1, sample_2=sample_2, bucket_method='fd')
        >>> 0.30806541358219786
        """

        bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
        sample_1_hist = self._hist_1d(
            data=sample_1,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
        )
        sample_2_hist = self._hist_1d(
            data=sample_2,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
        )
        mean_hist = np.mean([sample_1_hist, sample_2_hist], axis=0)
        kl_sample_1, kl_sample_2 = stats.entropy(
            pk=sample_1_hist, qk=mean_hist
        ), stats.entropy(pk=sample_2_hist, qk=mean_hist)
        return (kl_sample_1 + kl_sample_2) / 2

    def rolling_jensen_shannon_divergence(
        self,
        data: np.ndarray,
        time_windows: np.ndarray,
        fps=int,
        bucket_method: Literal[
            "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"
        ] = "auto",
    ) -> np.ndarray:
        """
        Compute rolling Jensen-Shannon divergence comparing the current time-window of size N to the preceding window of size N.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter np.ndarray[ints] time_windows: Time windows to compute JS for in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        """

        check_valid_array(data=data, source=self.__class__.__name__, accepted_sizes=[1])
        check_valid_array(
            data=time_windows, source=self.__class__.__name__, accepted_sizes=[1]
        )
        check_int(name=f"{self.__class__.__name__} fps", value=fps, min_value=1)
        check_str(
            name=f"{self.__class__.__name__} bucket_method",
            value=bucket_method,
            options=Options.BUCKET_METHODS.value,
        )
        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(
                data, list(range(window_size, data.shape[0], window_size))
            )
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[
                    j
                ].astype(np.float32)
                bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
                sample_1_hist = self._hist_1d(
                    data=sample_1,
                    bin_count=bin_count,
                    range=np.array([0, int(bin_width * bin_count)]),
                )
                sample_2_hist = self._hist_1d(
                    data=sample_2,
                    bin_count=bin_count,
                    range=np.array([0, int(bin_width * bin_count)]),
                )
                sample_1_hist, sample_2_hist = sample_1_hist / np.sum(
                    sample_1_hist
                ), sample_2_hist / np.sum(sample_2_hist)
                mean_hist = np.mean([sample_1_hist, sample_2_hist], axis=0)
                kl_sample_1, kl_sample_2 = stats.entropy(
                    pk=sample_1_hist, qk=mean_hist
                ), stats.entropy(pk=sample_2_hist, qk=mean_hist)
                js = (kl_sample_1 + kl_sample_2) / 2
                results[window_start:window_end, i] = js
        return results

    def wasserstein_distance(
        self,
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        bucket_method: Literal[
            "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"
        ] = "auto",
    ) -> float:
        """
        Compute Wasserstein distance between two distributions.

        .. note::
           Uses ``stats.wasserstein_distance``. I have tried to move ``stats.wasserstein_distance`` to jitted method extensively,
           but this doesn't give significant runtime improvement. Rate-limiter appears to be the _hist_1d.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns float: Wasserstein distance between ``sample_1`` and ``sample_2``

        :example:

        >>> sample_1 = np.random.normal(loc=10, scale=2, size=10)
        >>> sample_2 = np.random.normal(loc=10, scale=3, size=10)
        >>> Statistics().wasserstein_distance(sample_1=sample_1, sample_2=sample_2)
        >>> 0.020833333333333332
        """

        bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
        sample_1_hist = self._hist_1d(
            data=sample_1,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
        )
        sample_2_hist = self._hist_1d(
            data=sample_2,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
        )
        sample_1_hist, sample_2_hist = sample_1_hist / np.sum(
            sample_1_hist
        ), sample_2_hist / np.sum(sample_2_hist)
        return stats.wasserstein_distance(
            u_values=sample_1_hist, v_values=sample_2_hist
        )

    def rolling_wasserstein_distance(
        self,
        data: np.ndarray,
        time_windows: np.ndarray,
        fps: int,
        bucket_method: Literal[
            "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"
        ] = "auto",
    ) -> np.ndarray:
        """
        Compute rolling Wasserstein distance comparing the current time-window of size N to the preceding window of size N.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter np.ndarray[ints] time_windows: Time windows to compute JS for in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns np.ndarray: Size data.shape[0] x window_sizes.shape with Wasserstein distance. Columns represent different time windows.

        :example:
        >>> data = np.random.randint(0, 100, (100,))
        >>> Statistics().rolling_wasserstein_distance(data=data, time_windows=np.array([1, 2]), fps=30)
        """

        check_valid_array(data=data, source=self.__class__.__name__, accepted_sizes=[1])
        check_valid_array(
            data=time_windows, source=self.__class__.__name__, accepted_sizes=[1]
        )
        check_int(name=f"{self.__class__.__name__} fps", value=fps, min_value=1)
        check_str(
            name=f"{self.__class__.__name__} bucket_method",
            value=bucket_method,
            options=Options.BUCKET_METHODS.value,
        )
        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(
                data, list(range(window_size, data.shape[0], window_size))
            )
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[
                    j
                ].astype(np.float32)
                bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
                sample_1_hist = self._hist_1d(
                    data=sample_1,
                    bin_count=bin_count,
                    range=np.array([0, int(bin_width * bin_count)]),
                )
                sample_2_hist = self._hist_1d(
                    data=sample_2,
                    bin_count=bin_count,
                    range=np.array([0, int(bin_width * bin_count)]),
                )
                sample_1_hist, sample_2_hist = sample_1_hist / np.sum(
                    sample_1_hist
                ), sample_2_hist / np.sum(sample_2_hist)
                w = stats.wasserstein_distance(
                    u_values=sample_1_hist, v_values=sample_2_hist
                )
                results[window_start:window_end, i] = w

        return results

    def population_stability_index(
        self,
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        fill_value: Optional[int] = 1,
        bucket_method: Optional[
            Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]
        ] = "auto",
    ) -> float:
        """
        Compute Population Stability Index (PSI) comparing two distributions.

        .. note::
           Empty bins (0 observations in bin) in is replaced with ``fill_value``.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter Optional[int] fill_value: Empty bins (0 observations in bin) in is replaced with ``fill_value``. Default 1.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns float: PSI distance between ``sample_1`` and ``sample_2``

        :example:
        >>> sample_1, sample_2 = np.random.randint(0, 100, (100,)), np.random.randint(0, 10, (100,))
        >>> Statistics().population_stability_index(sample_1=sample_1, sample_2=sample_2, fill_value=1, bucket_method='auto')
        >>> 3.9657026867553817
        """

        check_valid_array(
            data=sample_1, source=self.__class__.__name__, accepted_sizes=[1]
        )
        check_valid_array(
            data=sample_2, source=self.__class__.__name__, accepted_sizes=[1]
        )
        check_int(name=self.__class__.__name__, value=fill_value)
        check_str(
            name=self.__class__.__name__,
            value=bucket_method,
            options=Options.BUCKET_METHODS.value,
        )
        bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
        sample_1_hist = self._hist_1d(
            data=sample_1,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
        )
        sample_2_hist = self._hist_1d(
            data=sample_2,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
        )
        sample_1_hist[sample_1_hist == 0] = fill_value
        sample_2_hist[sample_2_hist == 0] = fill_value
        sample_1_hist, sample_2_hist = sample_1_hist / np.sum(
            sample_1_hist
        ), sample_2_hist / np.sum(sample_2_hist)
        samples_diff = sample_2_hist - sample_1_hist
        log = np.log(sample_2_hist / sample_1_hist)
        return np.sum(samples_diff * log)

    def rolling_population_stability_index(
        self,
        data: np.ndarray,
        time_windows: np.ndarray,
        fps: int,
        fill_value: int = 1,
        bucket_method: Literal[
            "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"
        ] = "auto",
    ) -> np.ndarray:
        """
        Compute rolling Population Stability Index (PSI) comparing the current time-window of
        size N to the preceding window of size N.

        .. note::
           Empty bins (0 observations in bin) in is replaced with ``fill_value``.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter int fill_value: Empty bins (0 observations in bin) in is replaced with ``fill_value``.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns np.ndarray: PSI data of size len(data) x len(time_windows).
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(
                data, list(range(window_size, data.shape[0], window_size))
            )
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[
                    j
                ].astype(np.float32)
                bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
                sample_1_hist = self._hist_1d(
                    data=sample_1,
                    bin_count=bin_count,
                    range=np.array([0, int(bin_width * bin_count)]),
                )
                sample_2_hist = self._hist_1d(
                    data=sample_2,
                    bin_count=bin_count,
                    range=np.array([0, int(bin_width * bin_count)]),
                )
                sample_1_hist[sample_1_hist == 0] = fill_value
                sample_2_hist[sample_2_hist == 0] = fill_value
                sample_1_hist, sample_2_hist = sample_1_hist / np.sum(
                    sample_1_hist
                ), sample_2_hist / np.sum(sample_2_hist)
                samples_diff = sample_2_hist - sample_1_hist
                log = np.log(sample_2_hist / sample_1_hist)
                psi = np.sum(samples_diff * log)
                results[window_start:window_end, i] = psi

        return results

    @staticmethod
    @njit("(float64[:], float64[:])", cache=True)
    def kruskal_wallis(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Jitted compute of Kruskal-Wallis H between two distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns float: Kruskal-Wallis H.

        :example:
        >>> sample_1 = np.array([1, 1, 3, 4, 5]).astype(np.float64)
        >>> sample_2 = np.array([6, 7, 8, 9, 10]).astype(np.float64)
        >>> Statistics().kruskal_wallis(sample_1=sample_1, sample_2=sample_2)
        >>> 39.4
        """

        # sample_1 = np.concatenate((np.zeros((sample_1.shape[0], 1)), sample_1.reshape(-1, 1)), axis=1)
        # sample_2 = np.concatenate((np.ones((sample_2.shape[0], 1)), sample_2.reshape(-1, 1)), axis=1)
        data = np.vstack((sample_1, sample_2))
        ranks = fast_mean_rank(data=data[:, 1], descending=False)
        data = np.hstack((data, ranks.reshape(-1, 1)))
        sample_1_summed_rank = np.sum(data[0 : sample_1.shape[0], 2].flatten())
        sample_2_summed_rank = np.sum(data[sample_1.shape[0] :, 2].flatten())
        h1 = 12 / (data.shape[0] * (data.shape[0] + 1))
        h2 = (np.square(sample_1_summed_rank) / sample_1.shape[0]) + (
            np.square(sample_2_summed_rank) / sample_2.shape[0]
        )
        h3 = 3 * (data.shape[0] + 1)
        return h1 * h2 - h3

    @staticmethod
    @njit("(float64[:], float64[:])", cache=True)
    def mann_whitney(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Jitted compute of Mann-Whitney U between two distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns float: Mann-Whitney U.

        :references:
            `Modified from James Webber gist on GitHub <https://gist.github.com/jamestwebber/38ab26d281f97feb8196b3d93edeeb7b>`__.

        :example:
        >>> sample_1 = np.array([1, 1, 3, 4, 5])
        >>> sample_2 = np.array([6, 7, 8, 9, 10])
        >>> results = Statistics().mann_whitney(sample_1=sample_1, sample_2=sample_2)
        """

        n1, n2 = sample_1.shape[0], sample_2.shape[0]
        ranked = fast_mean_rank(np.concatenate((sample_1, sample_2)))
        u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(ranked[:n1], axis=0)
        u2 = n1 * n2 - u1
        return min(u1, u2)

    @staticmethod
    @jit(nopython=True, cache=True)
    def levenes(
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        critical_values: Optional[np.ndarray] = None,
    ) -> (float, Union[bool, None]):
        """
        Jitted compute of two-sample Leven's W.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter ndarray critical_values: 2D array with where first column represent dfn first row dfd with values represent critical values.
                                            Can be found in ``simba.assets.critical_values_05.pickle``

        :returns float: Leven's W.

        :examples:
        >>> sample_1 = np.array(list(range(0, 50)))
        >>> sample_2 = np.array(list(range(25, 100)))
        >>> Statistics().levenes(sample_1=sample_1, sample_2=sample_2)
        >>> 12.63909108903254
        >>> critical_values = pickle.load(open("simba/assets/lookups/critical_values_5.pickle","rb"))['f']['one_tail'].values
        >>> Statistics().levenes(sample_1=sample_1, sample_2=sample_2, critical_values=critical_values)
        >>> (12.63909108903254, True)
        """

        significance_bool = None
        Ni_x, Ni_y = len(sample_1), len(sample_2)
        Yci_x, Yci_y = np.median(sample_1), np.median(sample_2)
        Ntot = Ni_x + Ni_y
        Zij_x, Zij_y = np.abs(sample_1 - Yci_x).astype(np.float32), np.abs(
            sample_2 - Yci_y
        ).astype(np.float32)
        Zbari_x, Zbari_y = np.mean(Zij_x), np.mean(Zij_y)
        Zbar = ((Zbari_x * Ni_x) + (Zbari_y * Ni_y)) / Ntot
        numer = (Ntot - 2) * np.sum(
            np.array([Ni_x, Ni_y]) * (np.array([Zbari_x, Zbari_y]) - Zbar) ** 2
        )
        dvar = np.sum((Zij_x - Zbari_x) ** 2) + np.sum((Zij_y - Zbari_y) ** 2)
        denom = (2 - 1.0) * dvar
        l_statistic = numer / denom

        if critical_values is not None:
            dfn, dfd = 1, (Ni_x + Ni_y) - 2
            idx = (np.abs(critical_values[0][1:] - dfd)).argmin() + 1
            critical_values = critical_values[1:, np.array([0, idx])]
            critical_value = np.interp(
                dfd, critical_values[:, 0], critical_values[:, 1]
            )
            if l_statistic >= critical_value:
                significance_bool = True
            else:
                significance_bool = False

        return (l_statistic, significance_bool)

    @staticmethod
    @njit("(float64[:], float64[:], float64)", cache=True)
    def rolling_levenes(
        data: np.ndarray, time_windows: np.ndarray, fps: float
    ) -> float:
        """
        Jitted compute of rolling Levene's W comparing the current time-window of size N to the preceding window of size N.

        .. note::
           First time bin (where has no preceding time bin) will have fill value ``0``

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns np.ndarray: Levene's W data of size len(data) x len(time_windows).

        :example:
        >>> data = np.random.randint(0, 50, (100)).astype(np.float64)
        >>> Statistics().rolling_levenes(data=data, time_windows=np.array([1]).astype(np.float64), fps=5.0)
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(
                data, list(range(window_size, data.shape[0], window_size))
            )
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[
                    j
                ].astype(np.float32)
                Ni_x, Ni_y = len(sample_1), len(sample_2)
                Yci_x, Yci_y = np.median(sample_1), np.median(sample_2)
                Ntot = Ni_x + Ni_y
                Zij_x, Zij_y = np.abs(sample_1 - Yci_x).astype(np.float32), np.abs(
                    sample_2 - Yci_y
                ).astype(np.float32)
                Zbari_x, Zbari_y = np.mean(Zij_x), np.mean(Zij_y)
                Zbar = ((Zbari_x * Ni_x) + (Zbari_y * Ni_y)) / Ntot
                numer = (Ntot - 2) * np.sum(
                    np.array([Ni_x, Ni_y]) * (np.array([Zbari_x, Zbari_y]) - Zbar) ** 2
                )
                dvar = np.sum((Zij_x - Zbari_x) ** 2) + np.sum((Zij_y - Zbari_y) ** 2)
                denom = (2 - 1.0) * dvar
                w = numer / denom
                results[window_start:window_end, i] = w
        return results

    @staticmethod
    @jit(nopython=True, cache=True)
    def brunner_munzel(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Jitted compute of Brunner-Munzel W between two distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns float: Brunner-Munzel W.

        .. note::
           Modified from ``scipy.stats.brunnermunzel``.

        :example:
        >>> sample_1, sample_2 = np.random.normal(loc=10, scale=2, size=10), np.random.normal(loc=20, scale=2, size=10)
        >>> Statistics().brunner_munzel(sample_1=sample_1, sample_2=sample_2)
        >>> 0.5751408161437165
        """
        nx, ny = len(sample_1), len(sample_2)
        rankc = fast_mean_rank(np.concatenate((sample_1, sample_2)))
        rankcx, rankcy = rankc[0:nx], rankc[nx : nx + ny]
        rankcx_mean, rankcy_mean = np.mean(rankcx), np.mean(rankcy)
        rankx, ranky = fast_mean_rank(sample_1), fast_mean_rank(sample_2)
        rankx_mean, ranky_mean = np.mean(rankx), np.mean(ranky)
        Sx = np.sum(np.power(rankcx - rankx - rankcx_mean + rankx_mean, 2.0)) / nx - 1
        Sy = np.sum(np.power(rankcy - ranky - rankcy_mean + ranky_mean, 2.0)) / ny - 1
        wbfn = nx * ny * (rankcy_mean - rankcx_mean)
        wbfn /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)
        return -wbfn

    @staticmethod
    @njit("(float32[:], float64[:], float64)", cache=True)
    def rolling_barletts_test(
        data: np.ndarray, time_windows: np.ndarray, fps: float
    ) -> float:

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(
                data, list(range(window_size, data.shape[0], window_size))
            )
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[
                    j
                ].astype(np.float32)
                n_1 = len(sample_1)
                n_2 = len(sample_2)
                N = n_1 + n_2
                mean_variance_1 = np.sum((sample_1 - np.mean(sample_1)) ** 2) / (
                    n_1 - 1
                )
                mean_variance_2 = np.sum((sample_2 - np.mean(sample_2)) ** 2) / (
                    n_2 - 1
                )
                numerator = (N - 2) * (
                    np.log(mean_variance_1) + np.log(mean_variance_2)
                )
                denominator = 1 / (n_1 - 1) + 1 / (n_2 - 1)
                u = numerator / denominator
                results[window_start:window_end, i] = u

        return results

    @staticmethod
    @njit("(float32[:], float32[:])")
    def pearsons_r(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Calculate the Pearson correlation coefficient (Pearson's r) between two numeric samples.

        Pearson's r is a measure of the linear correlation between two sets of data points. It quantifies the strength and
        direction of the linear relationship between the two variables. The coefficient varies between -1 and 1, with
        -1 indicating a perfect negative linear relationship, 1 indicating a perfect positive linear relationship, and 0
        indicating no linear relationship.

        :example:
        >>> sample_1 = np.array([7, 2, 9, 4, 5, 6, 7, 8, 9]).astype(np.float32)
        >>> sample_2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]).astype(np.float32)
        >>> Statistics().pearsons_r(sample_1=sample_1, sample_2=sample_2)
        >>> 0.47
        """

        m1, m2 = np.mean(sample_1), np.mean(sample_2)
        numerator = np.sum((sample_1 - m1) * (sample_2 - m2))
        denominator = np.sqrt(
            np.sum((sample_1 - m1) ** 2) * np.sum((sample_2 - m2) ** 2)
        )
        r = numerator / denominator
        return r

    @staticmethod
    @njit("(float32[:], float32[:])")
    def spearman_rank_correlation(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        :example:
        >>> sample_1 = np.array([7, 2, 9, 4, 5, 6, 7, 8, 9]).astype(np.float32)
        >>> sample_2 = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]).astype(np.float32)
        >>> Statistics().spearman_rank_correlation(sample_1=sample_1, sample_2=sample_2)
        >>> 0.0003979206085205078
        """

        rank_x, rank_y = np.argsort(np.argsort(sample_1)), np.argsort(
            np.argsort(sample_2)
        )
        d_squared = np.sum((rank_x - rank_y) ** 2)
        return 1 - (6 * d_squared) / (len(sample_1) * (len(sample_2) ** 2 - 1))

    @staticmethod
    @njit("(float32[:], float32[:], float64[:], int64)")
    def sliding_pearsons_r(
        sample_1: np.ndarray, sample_2: np.ndarray, time_windows: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Given two 1D arrays of size N, create sliding window of size time_windows[i] * fps and return Pearson's R
        between the values in the two 1D arrays in each window. Address "what is the correlation between Feature 1 and
        Feature 2 in the current X.X seconds of the video".

        .. image:: _static/img/sliding_pearsons.png
           :width: 600
           :align: center

        :parameter ndarray sample_1: First 1D array with feature values.
        :parameter ndarray sample_1: Second 1D array with feature values.
        :parameter float time_windows: The length of the sliding window in seconds.
        :parameter int fps: The fps of the recorded video.
        :returns np.ndarray: 2d array of Pearsons R of size len(sample_1) x len(time_windows). Note, if sliding window is 10 frames, the first 9 entries will be filled with 0.

        :example:
        >>> sample_1 = np.random.randint(0, 50, (10)).astype(np.float32)
        >>> sample_2 = np.random.randint(0, 50, (10)).astype(np.float32)
        >>> Statistics().sliding_pearsons_r(sample_1=sample_1, sample_2=sample_2, time_windows=np.array([0.5]), fps=10)
        >>> [[-1.][-1.][-1.][-1.][0.227][-0.319][-0.196][0.474][-0.061][0.713]]
        """

        results = np.full((sample_1.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            for left, right in zip(
                prange(0, sample_1.shape[0] + 1),
                prange(window_size, sample_1.shape[0] + 1),
            ):
                s1, s2 = sample_1[left:right], sample_2[left:right]
                m1, m2 = np.mean(s1), np.mean(s2)
                numerator = np.sum((s1 - m1) * (s2 - m2))
                denominator = np.sqrt(np.sum((s1 - m1) ** 2) * np.sum((s2 - m2) ** 2))
                if denominator != 0:
                    r = numerator / denominator
                    results[right - 1, i] = r
                else:
                    results[right - 1, i] = -1.0

        return results

    @staticmethod
    @njit(
        [
            "(float32[:], float32[:], float64[:,:], types.unicode_type)",
            '(float32[:], float32[:], float64[:,:], types.misc.Omitted("goodness_of_fit"))',
            "(float32[:], float32[:], types.misc.Omitted(None), types.unicode_type)",
            '(float32[:], float32[:], types.misc.Omitted(None), types.misc.Omitted("goodness_of_fit"))',
        ]
    )
    def chi_square(
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        critical_values: Optional[np.ndarray] = None,
        type: Optional[Literal["goodness_of_fit", "independence"]] = "goodness_of_fit",
    ) -> Tuple[float, Union[bool, None]]:
        """
        Jitted compute of chi square between two categorical distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter ndarray critical_values: 2D array with where indexes represent degrees of freedom and values
                                            represent critical values. Can be found in ``simba.assets.critical_values_05.pickle``

        .. note::
           Requires sample_1 and sample_2 has to be numeric. if working with strings, convert to
           numeric category values before using chi_square.

        .. warning:
           Non-overlapping values (i.e., categories exist in sample_1 that does not exist in sample2) or small values may cause inflated chi square values.
           If small contingency table small values, consider TODO Fisher's exact test

        :example:
        >>> sample_1 = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5]).astype(np.float32)
        >>> sample_2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        >>> critical_values = pickle.load(open("simba/assets/lookups/critical_values_5.pickle", "rb"))['chi_square']['one_tail'].values
        >>> chi_square(sample_1=sample_2, sample_2=sample_1, critical_values=critical_values, type='goodness_of_fit')
        >>> (8.333, False)
        >>>
        """

        chi_square, significance_bool = 0.0, None
        unique_categories = np.unique(np.concatenate((sample_1, sample_2)))
        sample_1_counts = np.zeros(len(unique_categories), dtype=np.int64)
        sample_2_counts = np.zeros(len(unique_categories), dtype=np.int64)

        for i in prange(len(unique_categories)):
            sample_1_counts[i], sample_2_counts[i] = np.sum(
                sample_1 == unique_categories[i]
            ), np.sum(sample_2 == unique_categories[i])

        for i in prange(len(unique_categories)):
            count_1, count_2 = sample_1_counts[i], sample_2_counts[i]
            if count_2 > 0:
                chi_square += ((count_1 - count_2) ** 2) / count_2
            else:
                chi_square += ((count_1 - count_2) ** 2) / (count_2 + 1)

        if critical_values is not None:
            if type == "goodness_of_fit":
                df = unique_categories.shape[0] - 1
            else:
                df = (len(sample_1_counts) - 1) * (len(sample_2_counts) - 1)

            critical_value = np.interp(df, critical_values[:, 0], critical_values[:, 1])
            if chi_square >= critical_value:
                significance_bool = True
            else:
                significance_bool = False

        return chi_square, significance_bool

    @staticmethod
    @njit("(float32[:], float32, float32, float32[:,:], float32)")
    def sliding_independent_samples_t(
        data: np.ndarray,
        time_window: float,
        slide_time: float,
        critical_values: np.ndarray,
        fps: float,
    ) -> np.ndarray:
        """
        Jitted compute of sliding independent sample t-test. Compares the feature values in current time-window
        to prior time-windows to find the length in time to the most recent time-window where a significantly different
        feature value distribution is detected.

        .. image:: _static/img/sliding_statistics.png
           :width: 1500
           :align: center

        :parameter ndarray data: 1D array with feature values.
        :parameter float time_window: The sizes of the two feature value windows being compared in seconds.
        :parameter float slide_time: The slide size of the second window.
        :parameter ndarray critical_values: 2D array with where indexes represent degrees of freedom and values
                                            represent critical T values. Can be found in ``simba.assets.critical_values_05.pickle``.
        :parameter int fps: The fps of the recorded video.
        :returns np.ndarray: 1D array of size len(data) with values representing time to most recent significantly different feature distribution.

        :example:
        >>> data = np.random.randint(0, 50, (10)).astype(np.float32)
        >>> critical_values = pickle.load(open("simba/assets/lookups/critical_values_05.pickle", "rb"))['independent_t_test']['one_tail'].values.astype(np.float32)
        >>> results = Statistics().sliding_independent_samples_t(data=data, time_window=0.5, fps=5.0, critical_values=critical_values, slide_time=0.30)
        """

        results = np.full((data.shape[0]), 0.0)
        window_size, slide_size = int(time_window * fps), int(slide_time * fps)
        for i in range(1, data.shape[0]):
            sample_1_left, sample_1_right = i, i + window_size
            sample_2_left, sample_2_right = (
                sample_1_left - slide_size,
                sample_1_right - slide_size,
            )
            sample_1 = data[sample_1_left:sample_1_right]
            dof, steps_taken = (sample_1.shape[0] + sample_1.shape[0]) - 2, 1
            while sample_2_left >= 0:
                sample_2 = data[sample_2_left:sample_2_right]
                t_statistic = (np.mean(sample_1) - np.mean(sample_2)) / np.sqrt(
                    (np.std(sample_1) / sample_1.shape[0])
                    + (np.std(sample_2) / sample_1.shape[0])
                )
                critical_val = critical_values[dof - 1][1]
                if t_statistic >= critical_val:
                    break
                else:
                    sample_2_left -= 1
                    sample_2_right -= 1
                    steps_taken += 1
                if sample_2_left < 0:
                    steps_taken = -1
            if steps_taken == -1:
                results[i + window_size] = -1
            else:
                results[i + window_size] = steps_taken * slide_time

        return results

    @staticmethod
    @njit("(float32[:], float64[:], float32)")
    def rolling_mann_whitney(data: np.ndarray, time_windows: np.ndarray, fps: float):
        """
        Jitted compute of rolling Mann-Whitney U comparing the current time-window of
        size N to the preceding window of size N.

        .. note::
           First time bin (where has no preceding time bin) will have fill value ``0``

           `Modified from James Webber gist <https://gist.github.com/jamestwebber/38ab26d281f97feb8196b3d93edeeb7b>`__.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns np.ndarray: Mann-Whitney U data of size len(data) x len(time_windows).

        :examples:
        >>> data = np.random.randint(0, 4, (200)).astype(np.float32)
        >>> results = Statistics().rolling_mann_whitney(data=data, time_windows=np.array([1.0]), fps=1)
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(
                data, list(range(window_size, data.shape[0], window_size))
            )
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[
                    j
                ].astype(np.float32)
                n1, n2 = sample_1.shape[0], sample_2.shape[0]
                ranked = fast_mean_rank(np.concatenate((sample_1, sample_2)))
                u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(ranked[:n1], axis=0)
                u2 = n1 * n2 - u1
                u = min(u1, u2)
                results[window_start:window_end, i] = u

        return results

    def chow_test(self):
        pass

    @staticmethod
    @njit("(float32[:], float32[:], float64[:], int64)")
    def sliding_spearman_rank_correlation(
        sample_1: np.ndarray, sample_2: np.ndarray, time_windows: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Given two 1D arrays of size N, create sliding window of size time_windows[i] * fps and return Spearman's rank correlation
        between the values in the two 1D arrays in each window. Address "what is the correlation between Feature 1 and
        Feature 2 in the current X.X seconds of the video.

        .. image:: _static/img/sliding_spearman.png
           :width: 600
           :align: center

        :parameter ndarray sample_1: First 1D array with feature values.
        :parameter ndarray sample_1: Second 1D array with feature values.
        :parameter float time_windows: The length of the sliding window in seconds.
        :parameter int fps: The fps of the recorded video.
        :returns np.ndarray: 2d array of Soearman's ranks of size len(sample_1) x len(time_windows). Note, if sliding window is 10 frames, the first 9 entries will be filled with 0. The 10th value represents the correlation in the first 10 frames.

        :example:
        >>> sample_1 = np.array([9,10,13,22,15,18,15,19,32,11]).astype(np.float32)
        >>> sample_2 = np.array([11, 12, 15, 19, 21, 26, 19, 20, 22, 19]).astype(np.float32)
        >>> Statistics().sliding_spearman_rank_correlation(sample_1=sample_1, sample_2=sample_2, time_windows=np.array([0.5]), fps=10)
        """

        results = np.full((sample_1.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            for left, right in zip(
                prange(0, sample_1.shape[0] + 1),
                prange(window_size, sample_1.shape[0] + 1),
            ):
                s1, s2 = sample_1[left:right], sample_2[left:right]
                rank_x, rank_y = np.argsort(np.argsort(s1)), np.argsort(np.argsort(s2))
                d_squared = np.sum((rank_x - rank_y) ** 2)
                s = 1 - (6 * d_squared) / (len(s1) * (len(s2) ** 2 - 1))
                results[right - 1, i] = s

        return results

    @staticmethod
    @njit("(float32[:], float64, float64, float64)")
    def sliding_autocorrelation(
        data: np.ndarray, max_lag: float, time_window: float, fps: float
    ):
        """
        Jitted compute of sliding auto-correlations (the correlation of a feature with itself using lagged windows).

        :example:
        >>> data = np.array([0,1,2,3,4, 5,6,7,8,1,10,11,12,13,14]).astype(np.float32)
        >>> Statistics().sliding_autocorrelation(data=data, max_lag=0.5, time_window=1.0, fps=10)
        >>> [ 0., 0., 0.,  0.,  0., 0., 0.,  0. ,  0., -3.686, -2.029, -1.323, -1.753, -3.807, -4.634]
        """

        max_frm_lag, time_window_frms = int(max_lag * fps), int(time_window * fps)
        results = np.full((data.shape[0]), 0.0)
        for right in prange(time_window_frms - 1, data.shape[0]):
            left = right - time_window_frms + 1
            w_data = data[left : right + 1]
            corrcfs = np.full((max_frm_lag), np.nan)
            corrcfs[0] = 1
            for shift in range(1, max_frm_lag):
                corrcfs[shift] = np.corrcoef(w_data[:-shift], w_data[shift:])[0][1]
            mat_ = np.zeros(shape=(corrcfs.shape[0], 2))
            const = np.ones_like(corrcfs)
            mat_[:, 0] = const
            mat_[:, 1] = corrcfs
            det_ = np.linalg.lstsq(
                mat_.astype(np.float32), np.arange(0, max_frm_lag).astype(np.float32)
            )[0]
            results[right] = det_[::-1][0]
        return results

    @staticmethod
    def sliding_dominant_frequencies(
        data: np.ndarray,
        fps: float,
        k: int,
        time_windows: np.ndarray,
        window_function: Literal["Hann", "Hamming", "Blackman"] = None,
    ):
        """Find the K dominant frequencies within a feature vector using sliding windows"""

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for time_window_cnt in range(time_windows.shape[0]):
            window_size = int(time_windows[time_window_cnt] * fps)
            for left, right in zip(
                range(0, data.shape[0] + 1), range(window_size, data.shape[0] + 1)
            ):
                window_data = data[left:right]
                if window_function == "Hann":
                    window_data = window_data * np.hanning(len(window_data))
                elif window_function == "Hamming":
                    window_data = window_data * np.hamming(len(window_data))
                elif window_function == "Blackman":
                    window_data = window_data * np.blackman(len(window_data))
                fft_result = np.fft.fft(window_data)
                frequencies = np.fft.fftfreq(window_data.shape[0], 1 / fps)
                magnitude = np.abs(fft_result)
                top_k_frequency = frequencies[np.argsort(magnitude)[-(k + 1) : -1]]
                results[right - 1][time_window_cnt] = top_k_frequency[0]
        return results

    @staticmethod
    @njit("(float32[:], float32[:])")
    def kendall_tau(sample_1: np.ndarray, sample_2: np.ndarray) -> Tuple[float, float]:
        """
        Jitted Kendall Tau (rank correlation coefficient). Non-parametric method for computing correlation
        between two time-series features. Returns tau and associated z-score.

        :parameter ndarray sample_1: First 1D array with feature values.
        :parameter ndarray sample_1: Second 1D array with feature values.
        :returns Tuple[float, float]: Kendall Tau and associated z-score.

        :examples:
        >>> sample_1 = np.array([4, 2, 3, 4, 5, 7]).astype(np.float32)
        >>> sample_2 = np.array([1, 2, 3, 4, 5, 7]).astype(np.float32)
        >>> Statistics().kendall_tau(sample_1=sample_1, sample_2=sample_2)
        >>> (0.7333333333333333, 2.0665401605809928)

        References
        ----------
        .. [1] `Stephanie Glen, "Kendall’s Tau (Kendall Rank Correlation Coefficient)"  <https://www.statisticshowto.com/kendalls-tau/>`__.
        """

        rnks = np.argsort(sample_1)
        s1_rnk, s2_rnk = sample_1[rnks], sample_2[rnks]
        cncrdnt_cnts, dscrdnt_cnts = np.full((s1_rnk.shape[0] - 1), np.nan), np.full(
            (s1_rnk.shape[0] - 1), np.nan
        )
        for i in range(s2_rnk.shape[0] - 1):
            cncrdnt_cnts[i] = (
                np.argwhere(s2_rnk[i + 1 :] > s2_rnk[i]).flatten().shape[0]
            )
            dscrdnt_cnts[i] = (
                np.argwhere(s2_rnk[i + 1 :] < s2_rnk[i]).flatten().shape[0]
            )
        t = (np.sum(cncrdnt_cnts) - np.sum(dscrdnt_cnts)) / (
            np.sum(cncrdnt_cnts) + np.sum(dscrdnt_cnts)
        )
        z = (
            3
            * t
            * (np.sqrt(s1_rnk.shape[0] * (s1_rnk.shape[0] - 1)))
            / np.sqrt(2 * ((2 * s1_rnk.shape[0]) + 5))
        )

        return t, z

    @staticmethod
    @njit("(float32[:], float32[:], float64[:], int64)")
    def sliding_kendall_tau(
        sample_1: np.ndarray, sample_2: np.ndarray, time_windows: np.ndarray, fps: float
    ) -> np.ndarray:
        """
        Compute sliding Kendall's Tau correlation coefficient.

        Calculates Kendall's Tau correlation coefficient between two samples over sliding time windows. Kendall's Tau is a measure of correlation between two ranked datasets.

        The computation is based on the formula:

        .. math::

           \\tau = \\frac{{\\text{{concordant pairs}} - \\text{{discordant pairs}}}}{{\\text{{concordant pairs}} + \\text{{discordant pairs}}}}

        where concordant pairs are pairs of elements with the same order in both samples, and discordant pairs are pairs with different orders.

        References
        ----------
        .. [1] `Stephanie Glen, "Kendall’s Tau (Kendall Rank Correlation Coefficient)"  <https://www.statisticshowto.com/kendalls-tau/>`__.


        :param np.ndarray sample_1: First sample for comparison.
        :param np.ndarray sample_2: Second sample for comparison.
        :param np.ndarray time_windows: Rolling time windows in seconds.
        :param float fps: Frames per second (FPS) of the recorded video.
        :return: Array of Kendall's Tau correlation coefficients corresponding to each time window.
        """

        results = np.full((sample_1.shape[0], time_windows.shape[0]), 0.0)
        for time_window_cnt in range(time_windows.shape[0]):
            window_size = int(time_windows[time_window_cnt] * fps)
            for left, right in zip(
                range(0, sample_1.shape[0] + 1),
                range(window_size, sample_1.shape[0] + 1),
            ):
                sliced_sample_1, sliced_sample_2 = (
                    sample_1[left:right],
                    sample_2[left:right],
                )
                rnks = np.argsort(sliced_sample_1)
                s1_rnk, s2_rnk = sliced_sample_1[rnks], sliced_sample_2[rnks]
                cncrdnt_cnts, dscrdnt_cnts = np.full(
                    (s1_rnk.shape[0] - 1), np.nan
                ), np.full((s1_rnk.shape[0] - 1), np.nan)
                for i in range(s2_rnk.shape[0] - 1):
                    cncrdnt_cnts[i] = (
                        np.argwhere(s2_rnk[i + 1 :] > s2_rnk[i]).flatten().shape[0]
                    )
                    dscrdnt_cnts[i] = (
                        np.argwhere(s2_rnk[i + 1 :] < s2_rnk[i]).flatten().shape[0]
                    )
                results[right][time_window_cnt] = (
                    np.sum(cncrdnt_cnts) - np.sum(dscrdnt_cnts)
                ) / (np.sum(cncrdnt_cnts) + np.sum(dscrdnt_cnts))

        return results

    @staticmethod
    def local_outlier_factor(
        data: np.ndarray, k: Union[int, float] = 5, contamination: float = 1e-10
    ) -> np.ndarray:
        """
        Compute the local outlier factor of each observation.

        .. note::
           Method calls ``sklearn.neighbors.LocalOutlierFactor`` directly. Previously called using own implementation JIT'ed method,
           but runtime was 3x-ish slower than ``sklearn.neighbors.LocalOutlierFactor``.

        :parameter ndarray data: 2D array with feature values where rows represent frames and columns represent features.
        :parameter Union[int, float] sample_1: Number of neighbors to evaluate for each observation. If float, then interpreted as the ratio of data.shape[0].
        :parameter float contamination: Small pseudonumber to avoid DivisionByZero error.
        :returns np.ndarray: Array of size data.shape[0] with local outlier scores.

        :example:
        >>> data = np.random.normal(loc=45, scale=1, size=100).astype(np.float32)
        >>> for i in range(5): data = np.vstack([data, np.random.normal(loc=45, scale=1, size=100).astype(np.float32)])
        >>> for i in range(2): data = np.vstack([data, np.random.normal(loc=90, scale=1, size=100).astype(np.float32)])
        >>> Statistics().local_outlier_factor(data=data, k=5)
        >>> [1.004, 1.007, 0.986, 1.018, 0.986, 0.996, 24.067, 24.057]
        """

        check_valid_array(
            source=f"{Statistics.__class__.__name__} local_outlier_factor",
            data=data,
            accepted_sizes=[2],
        )
        if k is not None:
            check_float(
                name=f"{Statistics.__class__.__name__} k",
                value=k,
                min_value=1,
                max_value=data.shape[0],
            )
        check_float(
            name=f"{Statistics.__class__.__name__} contamination",
            value=contamination,
            min_value=0.0,
        )
        if isinstance(k, float):
            k = int(data.shape[0] * k)
        lof_model = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
        _ = lof_model.fit_predict(data)
        return -lof_model.negative_outlier_factor_.astype(np.float32)

    @staticmethod
    @jit(nopython=True)
    def _hbos_compute(
        data: np.ndarray, histograms: typed.Dict, histogram_edges: typed.Dict
    ) -> np.ndarray:
        """
        Jitted helper to compute Histogram-based Outlier Score (HBOS) called by ``simba.mixins.statistics_mixin.Statistics.hbos``.

        :parameter np.ndarray data: 2d array with frames represented by rows and columns representing feature values.
        :parameter typed.Dict histograms: Numba typed.Dict with integer keys (representing order of feature) and 1d arrays as values representing observation bin counts.
        :parameter: typed.Dict histogram_edges: Numba typed.Dict with integer keys (representing order of feature) and 1d arrays as values representing bin edges.
        :return np.ndarray: Array of size data.shape[0] representing outlier scores, with higher values representing greater outliers.
        """

        results = np.full((data.shape[0]), np.nan)
        data = data.astype(np.float32)
        for i in prange(data.shape[0]):
            score = 0.0
            for j in prange(data.shape[1]):
                value, bin_idx = data[i][j], np.nan
                for k in np.arange(histogram_edges[j].shape[0], 0, -1):
                    bin_max, bin_min = histogram_edges[j][k], histogram_edges[j][k - 1]
                    if (value <= bin_max) and (value > bin_min):
                        bin_idx = k
                        continue
                if np.isnan(bin_idx):
                    bin_idx = 0
                score += -np.log(histograms[j][int(bin_idx) - 1] + 1e-10)
            results[i] = score
        return results

    def hbos(
        self,
        data: np.ndarray,
        bucket_method: Literal[
            "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"
        ] = "auto",
    ) -> np.ndarray:
        """
        Jitted compute of Histogram-based Outlier Scores (HBOS). HBOS quantifies the abnormality of data points based on the densities of their feature values
        within their respective buckets over all feature values.

        :parameter np.ndarray data: 2d array with frames represented by rows and columns representing feature values.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators.
        :return np.ndarray: Array of size data.shape[0] representing outlier scores, with higher values representing greater outliers.

        .. image:: _static/img/hbos.png
           :width: 1200
           :align: center

        :example:
        >>> sample_1 = np.random.random_integers(low=1, high=2, size=(10, 50)).astype(np.float64)
        >>> sample_2 = np.random.random_integers(low=7, high=20, size=(2, 50)).astype(np.float64)
        >>> data = np.vstack([sample_1, sample_2])
        >>> Statistics().hbos(data=data)
        """

        check_valid_array(
            data=data,
            source=f"{Statistics.__class__.__name__} data",
            accepted_sizes=[2],
        )
        check_str(
            name=f"{Statistics.__class__.__name__} bucket_method",
            value=bucket_method,
            options=Options.BUCKET_METHODS.value,
        )
        min_vals, max_vals = np.min(data, axis=0), np.max(data, axis=0)
        data = (data - min_vals) / (max_vals - min_vals) * (1 - 0) + 0
        histogram_edges = typed.Dict.empty(
            key_type=types.int64, value_type=types.float64[:]
        )
        histograms = typed.Dict.empty(key_type=types.int64, value_type=types.int64[:])
        for i in range(data.shape[1]):
            bin_width, bin_count = bucket_data(data=data, method=bucket_method)
            histograms[i] = self._hist_1d(
                data=data,
                bin_count=bin_count,
                range=np.array([0, int(bin_width * bin_count)]),
            )
            histogram_edges[i] = np.arange(0, 1 + bin_width, bin_width).astype(
                np.float64
            )

        results = self._hbos_compute(
            data=data, histograms=histograms, histogram_edges=histogram_edges
        )
        return results.astype(np.float32)

    def rolling_shapiro_wilks(
        self, data: np.ndarray, time_window: float, fps: int
    ) -> np.ndarray:
        """
        Compute Shapiro-Wilks normality statistics for sequentially binned values in a time-series. E.g., compute
        the normality statistics of ``Feature N`` in each window of ``time_window`` seconds.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter int time_window: The size of the buckets in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :return np.ndarray: Array of size data.shape[0] with Shapiro-Wilks normality statistics

        :example:
        >>> data = np.random.randint(low=0, high=100, size=(200)).astype('float32')
        >>> results = self.rolling_shapiro_wilks(data=data, time_window=1, fps=30)
        """

        check_valid_array(
            data=data,
            source=f"{Statistics.__class__.__name__} data",
            accepted_sizes=[1],
        )
        check_float(
            name=f"{Statistics.__class__.__name__} data",
            value=time_window,
            min_value=0.1,
        )
        check_int(
            name=f"{Statistics.__class__.__name__} data", value=time_window, min_value=1
        )
        window_size, results = int(time_window * fps), np.full((data.shape[0]), -1.0)
        data = np.split(data, list(range(window_size, data.shape[0], window_size)))
        for cnt, i in enumerate(prange(1, len(data))):
            start, end = int((cnt + 1) * window_size), int(
                ((cnt + 1) * window_size) + window_size
            )
            results[start:end] = stats.shapiro(data[i])[0]
        return results

    @staticmethod
    @njit("(float32[:], float64[:], int64,)")
    def sliding_z_scores(
        data: np.ndarray, time_windows: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Calculate sliding Z-scores for a given data array over specified time windows.

        This function computes sliding Z-scores for a 1D data array over different time windows. The sliding Z-score
        is a measure of how many standard deviations a data point is from the mean of the surrounding data within
        the specified time window. This can be useful for detecting anomalies or variations in time-series data.

        :parameter ndarray data: 1D NumPy array containing the time-series data.
        :parameter ndarray time_windows: 1D NumPy array specifying the time windows in seconds over which to calculate the Z-scores.
        :parameter int time_windows: Frames per second, used to convert time windows from seconds to the corresponding number of data points.
        :returns np.ndarray: A 2D NumPy array containing the calculated Z-scores. Each row corresponds to the Z-scores calculated for a specific time window. The time windows are represented by the columns.

        :example:
        >>> data = np.random.randint(0, 100, (1000,)).astype(np.float32)
        >>> z_scores = Statistics().sliding_z_scores(data=data, time_windows=np.array([1.0, 2.5]), fps=10)
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in range(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            for right in range(window_size - 1, data.shape[0]):
                left = right - window_size + 1
                sample_data = data[left : right + 1]
                m, s = np.mean(sample_data), np.std(sample_data)
                vals = (sample_data - m) / s
                results[left : right + 1, i] = vals
        return results

    @staticmethod
    @njit("(int64[:, :],)")
    def phi_coefficient(data: np.ndarray) -> float:
        """
        Compute the phi coefficient for a 2x2 contingency table derived from binary data.

        The phi coefficient is a measure of association for binary data in a 2x2 contingency table. It quantifies the
        degree of association or correlation between two binary variables (e.g., binary classification targets).

        :param np.ndarray data: A NumPy array containing binary data organized in two columns. Each row represents a pair of binary values for two variables. Columns represent two features or two binary classification results.
        :param float: The calculated phi coefficient, a value between 0 and 1. A value of 0 indicates no association between the variables, while 1 indicates a perfect association.


        :example:
        >>> data = np.array([[0, 1], [1, 0], [1, 0], [1, 1]]).astype(np.int64)
        >>> Statistics().phi_coefficient(data=data)
        >>> 0.8164965809277261
        """
        cnt_0_0 = len(np.argwhere((data[:, 0] == 0) & (data[:, 1] == 0)).flatten())
        cnt_0_1 = len(np.argwhere((data[:, 0] == 0) & (data[:, 1] == 1)).flatten())
        cnt_1_0 = len(np.argwhere((data[:, 0] == 1) & (data[:, 1] == 0)).flatten())
        cnt_1_1 = len(np.argwhere((data[:, 0] == 1) & (data[:, 1] == 1)).flatten())

        BC, AD = cnt_1_1 * cnt_0_0, cnt_1_0 * cnt_0_1
        nominator = BC - AD
        denominator = np.sqrt(
            (cnt_1_0 + cnt_1_1)
            * (cnt_0_0 + cnt_0_1)
            * (cnt_1_0 + cnt_0_0)
            * (cnt_1_1 * cnt_0_1)
        )
        if nominator == 0 or denominator == 0:
            return 1.0
        else:
            return np.abs(
                (BC - AD)
                / np.sqrt(
                    (cnt_1_0 + cnt_1_1)
                    * (cnt_0_0 + cnt_0_1)
                    * (cnt_1_0 + cnt_0_0)
                    * (cnt_1_1 * cnt_0_1)
                )
            )

    @staticmethod
    @njit("(int32[:, :], float64[:], int64)")
    def sliding_phi_coefficient(
        data: np.ndarray, window_sizes: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Calculate sliding phi coefficients for a 2x2 contingency table derived from binary data.

        Computes sliding phi coefficients for a 2x2 contingency table derived from binary data over different
        time windows. The phi coefficient is a measure of association between two binary variables, and sliding phi
        coefficients can reveal changes in association over time.

        :param np.ndarray data: A 2D NumPy array containing binary data organized in two columns. Each row represents a pair of binary values for two variables.
        :param np.ndarray window_sizes: 1D NumPy array specifying the time windows (in seconds) over which to calculate the sliding phi coefficients.
        :param int sample_rate: The sampling rate or time interval (in samples per second, e.g., fps) at which data points were collected.
        :returns np.ndarray: A 2D NumPy array containing the calculated sliding phi coefficients. Each row corresponds to the phi coefficients calculated for a specific time point, the columns correspond to time-windows.

        :example:
        >>> data = np.random.randint(0, 2, (200, 2))
        >>> Statistics().sliding_phi_coefficient(data=data, window_sizes=np.array([1.0, 4.0]), sample_rate=10)
        """

        results = np.full((data.shape[0], window_sizes.shape[0]), -1.0)
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(
                range(0, data.shape[0] + 1), range(window_size, data.shape[0] + 1)
            ):
                sample = data[l:r, :]
                cnt_0_0 = len(
                    np.argwhere((sample[:, 0] == 0) & (sample[:, 1] == 0)).flatten()
                )
                cnt_0_1 = len(
                    np.argwhere((sample[:, 0] == 0) & (sample[:, 1] == 1)).flatten()
                )
                cnt_1_0 = len(
                    np.argwhere((sample[:, 0] == 1) & (sample[:, 1] == 0)).flatten()
                )
                cnt_1_1 = len(
                    np.argwhere((sample[:, 0] == 1) & (sample[:, 1] == 1)).flatten()
                )
                BC, AD = cnt_1_1 * cnt_0_0, cnt_1_0 * cnt_0_1
                nominator = BC - AD
                denominator = np.sqrt(
                    (cnt_1_0 + cnt_1_1)
                    * (cnt_0_0 + cnt_0_1)
                    * (cnt_1_0 + cnt_0_0)
                    * (cnt_1_1 * cnt_0_1)
                )
                if nominator == 0 or denominator == 0:
                    results[r - 1, i] = 0.0
                else:
                    results[r - 1, i] = np.abs(
                        (BC - AD)
                        / np.sqrt(
                            (cnt_1_0 + cnt_1_1)
                            * (cnt_0_0 + cnt_0_1)
                            * (cnt_1_0 + cnt_0_0)
                            * (cnt_1_1 * cnt_0_1)
                        )
                    )

        return results.astype(np.float32)

    @staticmethod
    @njit("int64[:], int64[:],")
    def cohens_h(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Jitted compute Cohen's h effect size for two samples of binary [0, 1] values. Cohen's h is a measure of effect size
        for comparing two independent samples based on the differences in proportions of the two samples.

        .. note:
           Modified from `DABEST <https://github.com/ACCLAB/DABEST-python/blob/fa7df50d20ab1c9cc687c66dd8bddf55d9a9dce3/dabest/_stats_tools/effsize.py#L216>`_
           `Cohen's h wiki <https://en.wikipedia.org/wiki/Cohen%27s_h>`_

        .. math::

           \\text{Cohen's h} = 2 \\arcsin\\left(\\sqrt{\\frac{\\sum{sample_1}}{N_1}}\\right) - 2 \\arcsin\left(\\sqrt{\frac{\\sum{sample_2}}{N_2}}\\right)

        Where N_1 and N_2 are the sample sizes of sample_1 and sample_2, respectively.

        :param np.ndarray sample_1: 1D array with binary [0, 1] values (e.g., first classifier inference values).
        :param np.ndarray sample_2: 1D array with binary [0, 1] values (e.g., second classifier inference values).
        :return float: Cohen's h effect size.

        :example:
        >>> sample_1 = np.array([1, 0, 0, 1])
        >>> sample_2 = np.array([1, 1, 1, 0])
        >>> Statistics().cohens_h(sample_1=sample_1, sample_2=sample_2)
        >>> -0.5235987755982985
        """

        sample_1_proportion = np.sum(sample_1) / sample_1.shape[0]
        sample_2_proportion = np.sum(sample_2) / sample_2.shape[0]
        phi_sample_1 = 2 * np.arcsin(np.sqrt(sample_1_proportion))
        phi_sample_2 = 2 * np.arcsin(np.sqrt(sample_2_proportion))

        return phi_sample_1 - phi_sample_2

    @staticmethod
    @jit("(float32[:], float64[:], int64,)")
    def sliding_skew(
        data: np.ndarray, time_windows: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Compute the skewness of a 1D array within sliding time windows.

        :param np.ndarray data: Input data array.
        :param np.ndarray data: 1D array of time window durations in seconds.
        :param np.ndarray data: Sampling rate of the data in samples per second.
        :return np.ndarray: 2D array of skewness`1 values with rows corresponding to data points and columns corresponding to time windows.

        :example:
        >>> data = np.random.randint(0, 100, (10,))
        >>> skewness = Statistics().sliding_skew(data=data.astype(np.float32), time_windows=np.array([1.0, 2.0]), sample_rate=2)
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * sample_rate)
            for j in range(window_size, data.shape[0] + 1):
                sample = data[j - window_size : j]
                mean, std = np.mean(sample), np.std(sample)
                results[j - 1][i] = (1 / sample.shape[0]) * np.sum(
                    ((data - mean) / std) ** 3
                )

        return results

    @staticmethod
    @jit("(float32[:], float64[:], int64,)")
    def sliding_kurtosis(
        data: np.ndarray, time_windows: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        Compute the kurtosis of a 1D array within sliding time windows.

        :param np.ndarray data: Input data array.
        :param np.ndarray time_windows: 1D array of time window durations in seconds.
        :param np.ndarray sample_rate: Sampling rate of the data in samples per second.
        :return np.ndarray: 2D array of skewness`1 values with rows corresponding to data points and columns corresponding to time windows.

        :example:
        >>> data = np.random.randint(0, 100, (10,))
        >>> kurtosis = Statistics().sliding_kurtosis(data=data.astype(np.float32), time_windows=np.array([1.0, 2.0]), sample_rate=2)
        """
        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = time_windows[i] * sample_rate
            for j in range(window_size, data.shape[0] + 1):
                sample = data[j - window_size : j]
                mean, std = np.mean(sample), np.std(sample)
                results[j - 1][i] = np.mean(((data - mean) / std) ** 4) - 3
        return results

    @staticmethod
    @jit(nopython=True)
    # @njit([(float32[:], float64, int64, boolean),])
    def kmeans_1d(
        data: np.ndarray, k: int, max_iters: int, calc_medians: bool
    ) -> Tuple[np.ndarray, np.ndarray, Union[None, types.DictType]]:
        """
        Perform k-means clustering on a 1-dimensional dataset.

        :parameter np.ndarray data: 1d array containing feature values.
        :parameter int k: Number of clusters.
        :parameter int max_iters: Maximum number of iterations for the k-means algorithm.
        :parameter bool calc_medians: Flag indicating whether to calculate cluster medians.
        :returns Tuple: Tuple of three elements. Final centroids of the clusters. Labels assigned to each data point based on clusters. Cluster medians (if calc_medians is True), otherwise None.

        :example:
        >>> data_1d = np.array([1, 2, 3, 55, 65, 40, 43, 40]).astype(np.float64)
        >>> centroids, labels, medians = Statistics().kmeans_1d(data_1d, 2, 1000, True)
        """

        data = np.ascontiguousarray(data)
        X = data.reshape((data.shape[0], 1))
        labels, medians = None, None
        centroids = X[np.random.choice(data.shape[0], k, replace=False)].copy()
        for _ in range(max_iters):
            labels = np.zeros(X.shape[0], dtype=np.int64)
            for i in range(X.shape[0]):
                min_dist = np.inf
                for j in range(k):
                    dist = np.abs(X[i] - centroids[j])
                    dist_sum = np.sum(dist)
                    if dist_sum < min_dist:
                        min_dist = dist_sum
                        labels[i] = j

            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k, dtype=np.int64)
            for i in range(X.shape[0]):
                cluster = labels[i]
                new_centroids[cluster] += X[i]
                counts[cluster] += 1

            for j in range(k):
                if counts[j] > 0:
                    new_centroids[j] /= counts[j]

            if np.array_equal(centroids, new_centroids):
                break
            else:
                centroids = new_centroids

        if calc_medians:
            labels, medians = labels.astype(np.int64), {}
            for i in prange(0, k, 1):
                medians[i] = np.median(data[np.argwhere(labels == i).flatten()])

        return centroids, labels, medians

    @staticmethod
    @njit(
        [
            (int8[:], int8[:], types.misc.Omitted(value=False), float32[:]),
            (
                int8[:],
                int8[:],
                types.misc.Omitted(value=False),
                types.misc.Omitted(None),
            ),
            (int8[:], int8[:], bool_, float32[:]),
            (int8[:], int8[:], bool_, types.misc.Omitted(None)),
        ]
    )
    def hamming_distance(
        x: np.ndarray,
        y: np.ndarray,
        sort: Optional[bool] = False,
        w: Optional[np.ndarray] = None,
    ) -> float:
        """
        Jitted calculate of the Hamming similarity between two vectors.

        .. note::
           Adapted from `pynndescent <https://pynndescent.readthedocs.io/en/latest/>`_.

        :parameter np.ndarray x: First binary vector.
        :parameter np.ndarray x: Second binary vector.
        :parameter Optional[np.ndarray] w: Optional weights for each element. Can be classification probabilities. If not provided, equal weights are assumed.
        :parameter Optional[bool] sort: If True, sorts x and y prior to hamming distance calculation. Default, False.

        :example:
        >>> x, y = np.random.randint(0, 2, (10,)).astype(np.int8), np.random.randint(0, 2, (10,)).astype(np.int8)
        >>> Statistics().hamming_distance(x=x, y=y)
        >>> 0.91
        """
        # pass
        if w is None:
            w = np.ones(x.shape[0]).astype(np.float32)

        results = 0.0
        if sort:
            x, y = np.sort(x), np.sort(y)
        for i in prange(x.shape[0]):
            if x[i] != y[i]:
                results += 1.0 * w[i]
        return results / x.shape[0]

    @staticmethod
    @njit(
        [(int8[:], int8[:], float32[:]), (int8[:], int8[:], types.misc.Omitted(None))]
    )
    def yule_coef(
        x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
    ) -> float64:
        """
        Jitted calculate of the yule coefficient between two binary vectors (e.g., to classified behaviors). 0 represent independence, 2 represents
        complete interdependence.

        .. note::
           Adapted from `pynndescent <https://pynndescent.readthedocs.io/en/latest/>`_.

        :parameter np.ndarray x: First binary vector.
        :parameter np.ndarray x: Second binary vector.
        :parameter Optional[np.ndarray] w: Optional weights for each element. Can be classification probabilities. If not provided, equal weights are assumed.

        :example:
        >>> x = np.random.randint(0, 2, (50,)).astype(np.int8)
        >>> y = x ^ 1
        >>> Statistics().yule_coef(x=x, y=y)
        >>> 2
        >>> random_indices = np.random.choice(len(x), size=len(x)//2, replace=False)
        >>> y = np.copy(x)
        >>> y[random_indices] = 1 - y[random_indices]
        >>> Statistics().yule_coef(x=x, y=y)
        >>> 0.99
        """
        if w is None:
            w = np.ones(x.shape[0]).astype(np.float32)

        f_f, t_t, t_f, f_t = 0.0, 0.0, 0.0, 0.0
        for i in prange(x.shape[0]):
            if (x[i] == 1) and (y[i] == 1):
                t_t += 1 * w[i]
            if (x[i] == 0) and (y[i] == 0):
                f_f += 1 * w[i]
            if (x[i] == 0) and (y[i] == 1):
                f_t += 1 * w[i]
            if (x[i] == 1) and (y[i] == 0):
                t_f += 1 * w[i]
        if t_f == 0.0 or f_t == 0.0:
            return 0.0
        else:
            return (2.0 * t_f * f_t) / (t_t * f_f + t_f * f_t)

    @staticmethod
    @njit(
        [(int8[:], int8[:], types.misc.Omitted(None)), (int8[:], int8[:], float32[:])]
    )
    def sokal_sneath(
        x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
    ) -> float64:
        """
        Jitted calculate of the sokal sneath coefficient between two binary vectors (e.g., to classified behaviors). 0 represent independence, 1 represents complete interdependence.

        :parameter np.ndarray x: First binary vector.
        :parameter np.ndarray x: Second binary vector.
        :parameter Optional[np.ndarray] w: Optional weights for each element. Can be classification probabilities. If not provided, equal weights are assumed.

        .. note::
           Adapted from `pynndescent <https://pynndescent.readthedocs.io/en/latest/>`_.

        :example:
        >>> x = np.array([0, 1, 0, 0, 1]).astype(np.int8)
        >>> y = np.array([1, 0, 1, 1, 0]).astype(np.int8)
        >>> Statistics().sokal_sneath(x, y)
        >>> 0.0
        """
        if w is None:
            w = np.ones(x.shape[0]).astype(float32)
        t_cnt, f_cnt, t_f, f_t = 0.0, 0.0, 0.0, 0.0
        for i in prange(x.shape[0]):
            if (x[i] == 1) and (y[i] == 1):
                t_cnt += 1.0 * w[i]
            elif (x[i] == 0) and (y[i] == 0):
                f_cnt += 1.0 * w[i]
            elif (x[i] == 0) and (y[i] == 1):
                f_t += 1.0 * w[i]
            elif (x[i] == 1) and (y[i] == 0):
                t_f += 1.0 * w[i]

        if t_f + f_t == 0.0:
            return 0.0
        else:
            return (f_t + t_f) / (2 * (t_cnt + f_cnt) + f_t + t_f)

    @staticmethod
    @njit([(float32[:, :], float32[:, :]), (float32[:, :], types.misc.Omitted(None))])
    def bray_curtis_dissimilarity(
        x: np.ndarray, w: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Jitted calculate of the Bray-Curtis dissimilarity matrix between samples based on feature values.

        Useful for finding similar frames based on behavior.

        .. note::
           Adapted from `pynndescent <https://pynndescent.readthedocs.io/en/latest/>`_.

        :parameter np.ndarray x: 2d array with likely normalized feature values.
        :parameter Optional[np.ndarray] w: Optional 2d array with weights of same size as x. Default None and all observations will have the same weight.
        :returns np.ndarray: 2d array with same size as x representing dissimilarity values. 0 and the observations are identical and at 1 the observations are completly disimilar.

        :example:
        >>> x = np.array([[1, 1, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]).astype(np.float32)
        >>> Statistics().bray_curtis_dissimilarity(x=x)
        >>> [[0, 1., 1., 0.], [1., 0., 0., 1.], [1., 0., 0., 1.], [0., 1., 1., 0.]]
        """
        if w is None:
            w = np.ones((x.shape[0], x.shape[0])).astype(np.float32)
        results = np.full((x.shape[0], x.shape[0]), 0.0)
        for i in prange(x.shape[0]):
            for j in range(i + 1, x.shape[0]):
                s1, s2, num, den = x[i], x[j], 0.0, 0.0
                for k in range(s1.shape[0]):
                    num += np.abs(s1[k] - s2[k])
                    den += np.abs(s1[k] + s2[k])
                if den == 0.0:
                    val = 0.0
                else:
                    val = (float(num) / den) * w[i, j]
                results[i, j] = val
                results[j, i] = val
        return results.astype(float32)

    @staticmethod
    @njit((float32[:], float32[:]))
    def _hellinger_helper(x: np.ndarray, y: np.ndarray):
        """Jitted helper for computing Hellinger distances from ``hellinger_distance``"""
        result, norm_x, norm_y = 0.0, 0.0, 0.0
        for i in range(x.shape[0]):
            result += np.sqrt(x[i] * y[i])
            norm_x += x[i]
            norm_y += y[i]
        if norm_x == 0 and norm_y == 0:
            return 0.0
        elif norm_x == 0 or norm_y == 0:
            return 1.0
        else:
            return np.sqrt(1 - result / np.sqrt(norm_x * norm_y))

    def hellinger_distance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        bucket_method: Optional[
            Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]
        ] = "auto",
    ) -> float:
        """
        Compute the Hellinger distance between two vector distribitions.

        :param np.ndarray x: First 1D array representing a probability distribution.
        :param np.ndarray y: Second 1D array representing a probability distribution.
        :param Optional[Literal['fd', 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', 'sqrt']] bucket_method: Method for computing histogram bins. Default is 'auto'.
        :returns float: Hellinger distance between the two input probability distributions.

        :example:
        >>> x = np.random.randint(0, 9000, (500000,))
        >>> y = np.random.randint(0, 9000, (500000,))
        >>> Statistics().hellinger_distance(x=x, y=y, bucket_method='auto')
        """

        check_instance(
            source=f"{Statistics().hellinger_distance.__name__} x",
            instance=x,
            accepted_types=np.ndarray,
        )
        check_instance(
            source=f"{Statistics().hellinger_distance.__name__} y",
            instance=y,
            accepted_types=np.ndarray,
        )
        if (x.ndim != 1) or (y.ndim != 1):
            raise CountError(
                msg=f"x and y are not 1D arrays, got {x.ndim} and {y.ndim}",
                source=Statistics().hellinger_distance.__name__,
            )
        check_str(
            name=f"{Statistics().hellinger_distance} method",
            value=bucket_method,
            options=Options.BUCKET_METHODS.value,
        )
        bin_width, bin_count = bucket_data(data=x, method=bucket_method)
        s1_h = self._hist_1d(
            data=x, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)])
        )
        s2_h = self._hist_1d(
            data=y, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)])
        )
        return self._hellinger_helper(
            x=s1_h.astype(np.float32), y=s2_h.astype(np.float32)
        )
