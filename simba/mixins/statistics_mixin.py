__author__ = "Simon Nilsson"

from itertools import combinations, permutations
from typing import List, Optional, Tuple, Union

import pandas as pd
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score,
                             fowlkes_mallows_score)
from sklearn.neighbors import LocalOutlierFactor

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
from numba import (bool_, float32, float64, int8, jit, njit, prange, typed,
                   types)
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats.distributions import chi2
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from statsmodels.stats.libqsturng import psturng
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import (check_float, check_int, check_str,
                                check_valid_array, check_valid_dataframe,
                                check_valid_lst)
from simba.utils.data import bucket_data, fast_mean_rank, get_confusion_matrix
from simba.utils.enums import ENV_VARS, Formats, Options
from simba.utils.errors import CountError, InvalidInputError
from simba.utils.read_write import get_unique_values_in_iterable, read_sys_env

NUMBA_PRECOMPILE = read_sys_env()[ENV_VARS.NUMBA_PRECOMPILE.value]

def dynamic_numba_decorator(dtypes, cache=True, fastmath=False):
    def decorator(func):
        if NUMBA_PRECOMPILE:
            return njit(dtypes, cache=cache, fastmath=fastmath)(func)
        else:
            return jit(nopython=True, cache=cache, fastmath=fastmath)(func)
    return decorator

class Statistics(FeatureExtractionMixin):
    """
    Statistics methods used for feature extraction, drift assessment, distance computations, distribution comparisons in sliding and static windows.

    .. note::

       Most methods implemented using `numba parallelization <https://numba.pydata.org/>`_ for improved run-times. See
       `line graph <https://github.com/sgoldenlab/simba/blob/master/docs/_static/img/statistics_runtimes.png>`_ below for expected run-times for a few methods included in this class.

       Most method has numba typed `signatures <https://numba.pydata.org/numba-doc/latest/reference/types.html>`_ to decrease
       compilation time through reduced type inference. Make sure to pass the correct dtypes as indicated by signature decorators. If dtype is not specified at
       array creation, it will typically be ``float64`` or ``int64``. As most methods here use ``float32`` for the input data argument,
       make sure to downcast.

       This class contains a few probability distribution comparison methods. These are being moved to ``simba.sandbox.distances`` (05.24).

    .. image:: _static/img/statistics_runtimes.png
       :width: 1200
       :align: center


    :references:
       .. [1]  Bernard Desgraupes - https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf
       .. [2]  Ikotun, A. M., Habyarimana, F., & Ezugwu, A. E. (2025). Cluster validity indices for automatic clustering: A comprehensive review. Heliyon, 11(2), e41953. https://doi.org/10.1016/j.heliyon.2025.e41953
       .. [3]  Hassan, B. A., Tayfor, N. B., Hassan, A. A., Ahmed, A. M., Rashid, T. A., & Abdalla, N. N. (2024). From A-to-Z review of clustering validation indices. arXiv. https://doi.org/10.48550/arXiv.2407.20246
       .. [4]  Leland McInnes - `pynndescent <https://pynndescent.readthedocs.io/en/latest/>`_.

    """

    def __init__(self):
        FeatureExtractionMixin.__init__(self)

    @staticmethod
    @jit(nopython=True)
    def _hist_1d(data: np.ndarray,
                  bin_count: int,
                  range: np.ndarray,
                  normalize: Optional[bool] = False) -> np.ndarray:
        """
        Jitted helper to compute 1D histograms with counts or rations (if normalize is True)

        .. note::
           For non-heuristic rules for bin counts and bin ranges, see ``simba.data.freedman_diaconis`` or simba.data.bucket_data``.

        :parameter np.ndarray data: 1d array containing feature values.
        :parameter int bin_count: The number of bins.
        :parameter: np.ndarray range: 1d array with two values representing minimum and maximum value to bin.
        :parameter: Optional[bool] normalize: If True, then the counts are returned as a ratio of all values. If False, then the raw counts. Pass normalize as True if the datasets are unequal counts. Default: True.
        """

        hist = np.histogram(data, bin_count, (range[0], range[1]))[0]
        if normalize:
            total_sum = np.sum(hist)
            if total_sum == 0:
                pass
            else:
                return hist / total_sum
        return hist.astype(np.float64)

    @staticmethod
    @dynamic_numba_decorator(dtypes="(float32[:], float64, float64)", cache=True, fastmath=False)
    def rolling_independent_sample_t(data: np.ndarray, time_window: float, fps: float) -> np.ndarray:

        r"""
        Jitted compute independent-sample t-statistics for sequentially binned values in a time-series.
        E.g., compute t-test statistics when comparing ``Feature N`` in the current 1s
        time-window, versus ``Feature N`` in the previous 1s time-window.

        :param ndarray data: 1D array of size len(frames) representing feature values.
        :param int group_size_s: The size of the buckets in seconds.
        :param fps: Frame-rate of recorded video.
        :rtype: int

        .. image:: _static/img/independent_t_tests.png
           :width: 700
           :align: center

        .. attention::
           Each window is compared to the prior window. Output for the windows without a prior window (the first window) is ``-1``.

        .. seealso::
           For single non-timeseries independent t, see :func:`simba.mixins.statistics_mixin.Statistics.independent_samples_t`

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
    @dynamic_numba_decorator(dtypes=[(float32[:], float32[:], float64[:, :]), (float32[:], float32[:], types.misc.Omitted(None))], cache=True, fastmath=False)
    def independent_samples_t(
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        critical_values: Optional[np.ndarray] = None,
    ) -> Tuple[float, Union[None, bool]]:

        r"""
        Jitted compute independent-samples t-test statistic and boolean significance between two distributions.

        .. note::
           Critical values are stored in simba.assets.lookups.critical_values_**.pickle


        The t-statistic for independent samples t-test is calculated using the following formula:

        .. math::

           t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}

        where:
        - :math:`\bar{x}_1` and :math:`\bar{x}_2` are the means of the two samples,
        - :math:`s_p` is the pooled standard deviation,
        - :math:`n_1` and :math:`n_2` are the sizes of the two samples.

        .. seealso::
           :func:`simba.mixins.statistics_mixin.Statistics.rolling_independent_sample_t`

        :param ndarray sample_1: First 1d array representing feature values.
        :param ndarray sample_2: Second 1d array representing feature values.
        :param ndarray critical_values: 2d array where the first column represents degrees of freedom and second column represents critical values.
        :returns t_statistic, p_value: Size-2 tuple representing t-statistic and associated probability value. p_value is ``None`` if critical_values is None. Else True or False with True representing significant.
        :rtype: Tuple[float, Union[None, bool]]

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
    @dynamic_numba_decorator(dtypes="(float64[:], float64[:])", cache=True, fastmath=False)
    def cohens_d(sample_1: np.ndarray, sample_2: np.ndarray) -> float:

        r"""
        Jitted compute of Cohen's d between two distributions.

        Cohen's d is a measure of effect size that quantifies the difference between the means of two distributions in terms of their standard deviation. It is calculated as the difference between the means of the two distributions divided by the pooled standard deviation.

        Higher values indicate a larger effect size, with 0.2 considered a small effect, 0.5 a medium effect, and 0.8 or above a large effect. Negative values indicate that the mean of sample 2 is larger than the mean of sample 1.

        .. seealso::
           For time-series based method, see :func:`simba.mixins.statistics_mixin.Statistics.rolling_cohens_d`

        .. math::
           d = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2 + s_2^2}{2}}}

        where:
           - :math:`\bar{x}_1` and :math:`\bar{x}_2` are the means of sample_1 and sample_2 respectively,
           - :math:`s_1` and :math:`s_2` are the standard deviations of sample_1 and sample_2 respectively.

        :param np.ndarray sample_1: First 1d array representing feature values.
        :param np.ndarray sample_2: Second 1d array representing feature values.
        :return: Cohen's D statistic.
        :rtype: float

        :example:
        >>> sample_1 = [2, 4, 7, 3, 7, 35, 8, 9]
        >>> sample_2 = [4, 8, 14, 6, 14, 70, 16, 18]
        >>> Statistics().cohens_d(sample_1=sample_1, sample_2=sample_2)
        -0.5952099775170546
        """

        return (np.mean(sample_1) - np.mean(sample_2)) / (
            np.sqrt((np.std(sample_1) ** 2 + np.std(sample_2) ** 2) / 2)
        )

    @staticmethod
    @dynamic_numba_decorator(dtypes="(float64[:], float64[:], float64)", cache=True, fastmath=False)
    def rolling_cohens_d(data: np.ndarray, time_windows: np.ndarray, fps: float) -> np.ndarray:
        """
        Jitted compute of rolling Cohen's D statistic comparing the current time-window of
        size N to the preceding window of size N.

        .. seealso::
           For single non-timeseries comparison, see :func:`simba.mixins.statistics_mixin.Statistics.cohens_d`

        :param ndarray data: 1D array of size len(frames) representing feature values.
        :param np.ndarray[ints] time_window: Time windows to compute ANOVAs for in seconds.
        :param int fps: Frame-rate of recorded video.
        :returns: Array of size data.shape[0] x window_sizes.shape[1] with Cohens D.
        :rtype: np.ndarray

        :example:
        >>> sample_1, sample_2 = np.random.normal(loc=10, scale=1, size=4), np.random.normal(loc=11, scale=2, size=4)
        >>> sample = np.hstack((sample_1, sample_2))
        >>> Statistics().rolling_cohens_d(data=sample, window_sizes=np.array([1]), fps=4)
        >>> [[0.],[0.],[0.],[0.],[0.14718302],[0.14718302],[0.14718302],[0.14718302]])
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(data, list(range(window_size, data.shape[0], window_size)))
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
    #@njit("(float32[:], float64, float64)")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float64, float64)", cache=True, fastmath=False)
    def rolling_two_sample_ks(data: np.ndarray, time_window: float, fps: float) -> np.ndarray:
        """
        Jitted compute Kolmogorov two-sample statistics for sequentially binned values in a time-series.
        E.g., compute KS statistics when comparing ``Feature N`` in the current 1s time-window, versus ``Feature N`` in the previous 1s time-window.

        .. seealso::
           For single non-timeseries based comparison, see :func:`simba.mixins.statistics_mixin.Statistics.two_sample_ks`

        :param ndarray data: 1D array of size len(frames) representing feature values.
        :param float time_window: The size of the buckets in seconds.
        :param int fps: Frame-rate of recorded video.
        :return: Array of size data.shape[0] with KS statistics
        :rtype: np.ndarray

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
    # @njit(
    #     [
    #         (float32[:], float32[:], float64[:, :]),
    #         (float32[:], float32[:], types.misc.Omitted(None)),
    #     ]
    # )
    # @jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes=[(float32[:], float32[:], float64[:, :]), (float32[:], float32[:], types.misc.Omitted(None))], cache=True, fastmath=False)
    def two_sample_ks(
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        critical_values: Optional[float64[:, :]] = None,
    ) -> Tuple[float, Union[bool, None]]:
        """
        Jitted compute the two-sample Kolmogorov-Smirnov (KS) test statistic and, optionally, test for statistical significance.

        The two-sample KS test is a non-parametric test that compares the cumulative distribution functions (ECDFs) of two independent samples to assess whether they come from the same distribution.

        KS statistic (D) is calculated as the maximum absolute difference between the empirical cumulative distribution functions (ECDFs) of the two samples.

        .. math::
           D = \\max(| ECDF_1(x) - ECDF_2(x) |)

        If `critical_values` are provided, the function checks the significance of the KS statistic against the critical values.

        .. seealso::
           For rolling timeseries based comparison, see :func:`simba.mixins.statistics_mixin.Statistics.rolling_two_sample_ks`

        :param np.ndarray data: The first sample array for the KS test.
        :param np.ndarray data: The second sample array for the KS test.
        :param Optional[float64[:, :]] critical_values: An array of critical values for the KS test. If provided, the function will also check the significance of the KS statistic against the critical values. Default: None.
        :returns (float Union[bool, None]): Returns a tuple containing the KS statistic and a boolean indicating whether the test is statistically significant.

        :example:
        >>> sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10]).astype(np.float32)
        >>> sample_2 = np.array([10, 5, 10, 4, 8, 10, 7, 10, 7, 10, 10]).astype(np.float32)
        >>> critical_values = pickle.load(open("simba/assets/lookups/critical_values_5.pickle", "rb"))['two_sample_KS']['one_tail'].values
        >>> Statistics.two_sample_ks(sample_1=sample_1, sample_2=sample_2, critical_values=critical_values)
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
        critical_values: Optional[np.ndarray] = None) -> Tuple[float, float]:

        r"""
        Compute the one-way ANOVA F-statistic and associated p-value for two distributions.

        This method calculates the F-statistic to determine if there is a significant difference
        between the means of the two samples, based on their variances. The F-statistic is computed as:

        .. math::
          F = \frac{MS_{\text{between}}}{MS_{\text{within}}}

        where:
        - :math:`SS_{\text{between}}` is the sum of squares between the groups.
        - :math:`SS_{\text{within}}` is the sum of squares within each group.
        - :math:`MS_{\text{between}} = \frac{SS_{\text{between}}}{df_{\text{between}}}`
        - :math:`MS_{\text{within}} = \frac{SS_{\text{within}}}{df_{\text{within}}}`

        .. seealso::
           For rolling comparisons in a timeseries, see :func:`simba.mixins.statistics_mixin.Statistics.rolling_one_way_anova`

        :param np.ndarray sample_1: First 1d array representing feature values.
        :param np.ndarray sample_2: Second 1d array representing feature values.
        :return: Tuple representing ANOVA F statistic and associated probability value.
        :rtype: Tuple[float, float]

        :example:
        >>> sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
        >>> sample_2 = np.array([8, 5, 5, 8, 8, 9, 10, 1, 7, 10, 10])
        >>> Statistics().one_way_anova(sample_1=sample_2, sample_2=sample_1)
        """

        significance_bool = None
        n1, n2 = len(sample_1), len(sample_2)
        m1, m2 = np.mean(sample_1), np.mean(sample_2)
        ss_between = (n1 * (m1 - np.mean(np.concatenate((sample_1, sample_2)))) ** 2 + n2 * (m2 - np.mean(np.concatenate((sample_1, sample_2)))) ** 2)
        ss_within = np.sum((sample_1 - m1) ** 2) + np.sum((sample_2 - m2) ** 2)
        df_between, df_within = 1, n1 + n2 - 2
        ms_between, ms_within = ss_between / df_between, ss_within / df_within
        f = ms_between / ms_within
        if critical_values is not None:
            critical_values = critical_values[:, np.array([0, df_between])]
            critical_value = np.interp(df_within, critical_values[:, 0], critical_values[:, 1])
            if f > critical_value:
                significance_bool = True
            else:
                significance_bool = False

        return (f, significance_bool)

    @staticmethod
    @njit("(float32[:], )")
    def sliding_cumulative_mean(x: np.ndarray):

        """
        Compute a sliding cumulative mean over a 1D

        :param np.ndarray x: A 1D NumPy array of type float32
        :return: A 1D float32 array of the same shape as `x`, containing the cumulative mean at each index, ignoring NaNs.
        :rtype: np.ndarray
        """

        results = np.empty(x.shape[0], dtype=np.float32)
        total, count = 0.0, 0
        for i in prange(x.shape[0]):
            val = x[i]
            if not np.isnan(val):
                total += val
                count += 1
            if count > 0:
                results[i] = total / count
            else:
                results[i] = np.nan
        return results

    @staticmethod
    #@njit("(float32[:], float64[:], float64)", cache=True)
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float64[:], float64)", cache=True, fastmath=False)
    def rolling_one_way_anova(data: np.ndarray, time_windows: np.ndarray, fps: int) -> np.ndarray:
        """
        Jitted compute of rolling one-way ANOVA F-statistic comparing the current time-window of
        size N to the preceding window of size N.

        .. image:: _static/img/rolling_anova.png
           :width: 600
           :align: center

        .. seealso::
           For single comparison, see :func:`simba.mixins.statistics_mixin.Statistics.one_way_anova`

        :param ndarray data: 1D array of size len(frames) representing feature values.
        :param np.ndarray[ints] time_windows: Time windows to compute ANOVAs for in seconds.
        :param int fps: Frame-rate of recorded video.
        :returns: 2D numpy array with F values comparing the current time-window to the immedidatly preceeding time-window.
        :rtype: np.ndarray



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
        fill_value: Optional[int] = 1,
        bucket_method: Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"] = "auto") -> float:

        r"""
        Compute Kullback-Leibler divergence between two distributions.

        .. note::
           Empty bins (0 observations in bin) in is replaced with passed ``fill_value``.

           Its range is from 0 to positive infinity. When the KL divergence is zero, it indicates that the two distributions are identical. As the KL divergence increases, it signifies an increasing difference between the distributions.

        .. math::
           \text{KL}(P || Q) = \sum{P(x) \log{\left(\frac{P(x)}{Q(x)}\right)}}

        .. seealso::
           For rolling comparisons in a timeseries, see :func:`simba.mixins.statistics_mixin.Statistics.rolling_kullback_leibler_divergence`
           For GPU implementation, see :func:`simba.data_processors.cuda.statistics.kullback_leibler_divergence_gpu`.

        :param ndarray sample_1: First 1d array representing feature values.
        :param ndarray sample_2: Second 1d array representing feature values.
        :param Optional[int] fill_value: Optional pseudo-value to use to fill empty buckets in ``sample_2`` histogram
        :param Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns: Kullback-Leibler divergence between ``sample_1`` and ``sample_2``
        :rtype: float
        """
        check_valid_array(data=sample_1, source=Statistics.kullback_leibler_divergence.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=sample_2, source=Statistics.kullback_leibler_divergence.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_str(name=f"{self.__class__.__name__} bucket_method", value=bucket_method, options=Options.BUCKET_METHODS.value)
        check_int(name=f"{self.__class__.__name__} fill value", value=fill_value, min_value=1)
        bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
        r = np.array([np.min(sample_1), np.max(sample_1)])
        sample_1_hist = self._hist_1d(data=sample_1, bin_count=bin_count, range=r)
        sample_2_hist = self._hist_1d(data=sample_2, bin_count=bin_count, range=r)
        sample_1_hist[sample_1_hist == 0] = fill_value
        sample_2_hist[sample_2_hist == 0] = fill_value
        sample_1_hist, sample_2_hist = sample_1_hist / np.sum(sample_1_hist), sample_2_hist / np.sum(sample_2_hist)
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

        .. seealso::
           For single comparison between two distributions, see :func:`simba.mixins.statistics_mixin.Statistics.kullback_leibler_divergence`

        :param ndarray sample_1: 1d array representing feature values.
        :param Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :param np.ndarray[floats] time_windows: Time windows to compute JS for in seconds.
        :param int fps: Frame-rate of recorded video.
        :returns: Size data.shape[0] x window_sizes.shape with Kullback-Leibler divergence. Columns represents different tiem windows.
        :rtype: np.ndarray

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

        r"""
        Compute Jensen-Shannon divergence between two distributions. Useful for (i) measure drift in datasets, and (ii) featurization of distribution shifts across
        sequential time-bins.

        .. note::
           JSD = 0: Indicates that the two distributions are identical.
           0 < JSD < 1: Indicates a degree of dissimilarity between the distributions, with values closer to 1 indicating greater dissimilarity.
           JSD = 1: Indicates that the two distributions are maximally dissimilar.

        .. math::
           JSD = \frac{KL(P_1 || M) + KL(P_2 || M)}{2}

        .. seealso::
           For rolling comparisons in a timeseries, see :func:`simba.mixins.statistics_mixin.Statistics.rolling_jensen_shannon_divergence`

        :param ndarray sample_1: First 1d array representing feature values.
        :param ndarray sample_2: Second 1d array representing feature values.
        :param Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators.
        :returns: Jensen-Shannon divergence between ``sample_1`` and ``sample_2``
        :rtype: float

        :example:
        >>> sample_1, sample_2 = np.array([1, 2, 3, 4, 5, 10, 1, 2, 3]), np.array([1, 5, 10, 9, 10, 1, 10, 6, 7])
        >>> Statistics().jensen_shannon_divergence(sample_1=sample_1, sample_2=sample_2, bucket_method='fd')
        >>> 0.30806541358219786
        """

        check_valid_array(
            data=sample_1,
            source=Statistics.jensen_shannon_divergence.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=Formats.NUMERIC_DTYPES.value,
        )
        check_valid_array(
            data=sample_2,
            source=Statistics.jensen_shannon_divergence.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=Formats.NUMERIC_DTYPES.value,
        )
        check_str(
            name=f"{self.__class__.__name__} bucket_method",
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
        mean_hist = np.mean([sample_1_hist, sample_2_hist], axis=0)
        kl_sample_1, kl_sample_2 = stats.entropy(
            pk=sample_1_hist, qk=mean_hist
        ), stats.entropy(pk=sample_2_hist, qk=mean_hist)
        return (kl_sample_1 + kl_sample_2) / 2

    def rolling_jensen_shannon_divergence(
        self,
        data: np.ndarray,
        time_windows: np.ndarray,
        fps: int,
        bucket_method: Literal[
            "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"
        ] = "auto",
    ) -> np.ndarray:
        """
        Compute rolling Jensen-Shannon divergence comparing the current time-window of size N to the preceding window of size N.

        .. seealso::
           For simple two distribution comparison, see :func:`simba.mixins.statistics_mixin.Statistics.jensen_shannon_divergence`

        :param ndarray data: 1D array of size len(frames) representing feature values.
        :param np.ndarray[ints] time_windows: Time windows to compute JS for in seconds.
        :param int fps: Frame-rate of recorded video.
        :param Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns: Array of size data.shape[0] x window_sizes.shape[0] with Jensen-Shannon divergence. Columns represents different time windows.
        :rtype: np.ndarray
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

        .. seealso::
           For time-series based comparisons, see :func:`simba.mixins.statistics_mixin.Statistics.rolling_wasserstein_distance`

        :param ndarray sample_1: First 1d array representing feature values.
        :param ndarray sample_2: Second 1d array representing feature values.
        :param Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns: Wasserstein distance between ``sample_1`` and ``sample_2``
        :rtype: float

        :example:
        >>> sample_1 = np.random.normal(loc=10, scale=2, size=10)
        >>> sample_2 = np.random.normal(loc=10, scale=3, size=10)
        >>> Statistics().wasserstein_distance(sample_1=sample_1, sample_2=sample_2)
        >>> 0.020833333333333332
        """
        check_valid_array(
            data=sample_1,
            source=Statistics.jensen_shannon_divergence.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=Formats.NUMERIC_DTYPES.value,
        )
        check_valid_array(
            data=sample_2,
            source=Statistics.jensen_shannon_divergence.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=Formats.NUMERIC_DTYPES.value,
        )
        check_str(
            name=f"{self.__class__.__name__} bucket_method",
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
        bucket_method: Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"] = "auto") -> np.ndarray:

        """
        Compute rolling Wasserstein distance comparing the current time-window of size N to the preceding window of size N.

        .. seealso::
           For simple two distribution earth mover comparison, see :func:`simba.mixins.statistics_mixin.Statistics.wasserstein_distance`

        :param ndarray data: 1D array of size len(frames) representing feature values.
        :param np.ndarray[ints] time_windows: Time windows to compute JS for in seconds.
        :param int fps: Frame-rate of recorded video.
        :param Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns: Size data.shape[0] x window_sizes.shape with Wasserstein distance. Columns represent different time windows.
        :rtype: np.ndarray

        :example:
        >>> data = np.random.randint(0, 100, (100,))
        >>> Statistics().rolling_wasserstein_distance(data=data, time_windows=np.array([1, 2]), fps=30)
        """

        check_valid_array(data=data, source=self.__class__.__name__, accepted_sizes=[1])
        check_valid_array(data=time_windows, source=self.__class__.__name__, accepted_sizes=[1])
        check_int(name=f"{self.__class__.__name__} fps", value=fps, min_value=1)
        check_str(name=f"{self.__class__.__name__} bucket_method", value=bucket_method, options=Options.BUCKET_METHODS.value)
        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(data, list(range(window_size, data.shape[0], window_size)))
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[j].astype(np.float32)
                bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
                sample_1_hist = self._hist_1d(data=sample_1, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
                sample_2_hist = self._hist_1d(data=sample_2, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
                sample_1_hist, sample_2_hist = sample_1_hist / np.sum(sample_1_hist), sample_2_hist / np.sum(sample_2_hist)
                w = stats.wasserstein_distance(u_values=sample_1_hist, v_values=sample_2_hist)
                results[window_start:window_end, i] = w

        return results

    @staticmethod
    def total_variation_distance(
        x: np.ndarray,
        y: np.ndarray,
        bucket_method: Optional[
            Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]
        ] = "auto",
    ):
        """
        Calculate the total variation distance between two probability distributions.

        :param np.ndarray x: A 1-D array representing the first sample.
        :param np.ndarray y: A 1-D array representing the second sample.
        :param Optional[str] bucket_method: The method used to determine the number of bins for histogram computation. Supported methods are 'fd' (Freedman-Diaconis), 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', and 'sqrt'. Defaults to 'auto'.
        :return: The total variation distance between the two distributions.
        :rtype: float

        .. math::

           TV(P, Q) = 0.5 \sum_i |P_i - Q_i|

        where :math:`P_i` and :math:`Q_i` are the probabilities assigned by the distributions :math:`P` and :math:`Q`
        to the same event :math:`i`, respectively.

        :example:
        >>> total_variation_distance(x=np.array([1, 5, 10, 20, 50]), y=np.array([1, 5, 10, 100, 110]))
        >>> 0.3999999761581421
        """

        check_valid_array(data=x, source=Statistics.total_variation_distance.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, source=Statistics.total_variation_distance.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_str(
            name=f"{Statistics.total_variation_distance.__name__} method",
            value=bucket_method,
            options=Options.BUCKET_METHODS.value,
        )
        bin_width, bin_count = bucket_data(data=x, method=bucket_method)
        s1_h = Statistics._hist_1d(
            data=x,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
            normalize=True,
        )
        s2_h = Statistics._hist_1d(
            data=y,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
            normalize=True,
        )
        return 0.5 * np.sum(np.abs(s1_h - s2_h))

    def population_stability_index(
        self,
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        fill_value: Optional[int] = 1,
        bucket_method: Optional[
            Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]
        ] = "auto",
    ) -> float:

        r"""
        Compute Population Stability Index (PSI) comparing two distributions.

        The Population Stability Index (PSI) is a measure of the difference in distribution
        patterns between two groups of data. A low PSI value indicates a minimal or negligible change in the distribution patterns between the two samples.
        A high PSI value suggests a significant difference in the distribution patterns between the two samples.

        .. note::
           Empty bins (0 observations in bin) in is replaced with ``fill_value``. The PSI value ranges from 0 to positive infinity.

        The Population Stability Index (PSI) is calculated as:

        .. math::

           PSI = \\sum \\left(\\frac{{p_2 - p_1}}{{ln(p_2 / p_1)}}\\right)

        where:
            - \( p_1 \) and \( p_2 \) are the proportions of observations in the bins for sample 1 and sample 2 respectively.


        .. seealso::
           For time-series based rolling comparisons, see :func:`simba.mixins.statistics_mixin.Statistics.rolling_population_stability_index`

        :param ndarray sample_1: First 1d array representing feature values.
        :param ndarray sample_2: Second 1d array representing feature values.
        :param Optional[int] fill_value: Empty bins (0 observations in bin) in is replaced with ``fill_value``. Default 1.
        :param Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns: PSI distance between ``sample_1`` and ``sample_2``
        :rtype: float

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
        sample_1_hist, sample_2_hist = sample_1_hist / np.sum(sample_1_hist), sample_2_hist / np.sum(sample_2_hist)
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

        .. seealso::
           For simple two-distribution comparisons, see :func:`simba.mixins.statistics_mixin.Statistics.population_stability_index`.

        :param ndarray sample_1: First 1d array representing feature values.
        :param ndarray sample_2: Second 1d array representing feature values.
        :param int fill_value: Empty bins (0 observations in bin) in is replaced with ``fill_value``.
        :param Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns: PSI data of size len(data) x len(time_windows).
        :rtype: np.ndarray
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(data, list(range(window_size, data.shape[0], window_size)))
            for j in range(1, len(data_split)):
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
                sample_1_hist, sample_2_hist = sample_1_hist / np.sum(sample_1_hist), sample_2_hist / np.sum(sample_2_hist)
                samples_diff = sample_2_hist - sample_1_hist
                log = np.log(sample_2_hist / sample_1_hist)
                psi = np.sum(samples_diff * log)
                results[window_start:window_end, i] = psi

        return results

    @staticmethod
    #@njit("(float64[:], float64[:])", cache=True)
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float64[:], float64[:])", cache=True, fastmath=False)
    def kruskal_wallis(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Compute the Kruskal-Wallis H statistic between two distributions.

        The Kruskal-Wallis test is a non-parametric method for testing whether samples originate from the same distribution.
        It ranks all the values from the combined samples, then calculates the H statistic based on the ranks.

        .. math::

           H = \\frac{{12}}{{n(n + 1)}} \\left(\\frac{{(\\sum R_{\text{sample1}})^2}}{{n_1}} + \\frac{{(\\sum R_{\text{sample2}})^2}}{{n_2}}\\right) - 3(n + 1)

        where:
        - :math:`n` is the total number of observations,
        - :math:`n_1` and :math:`n_2` are the number of observations in sample 1 and sample 2 respectively,
        - :math:`R_{\text{sample1}}` and :math:`R_{\text{sample2}}` are the sums of ranks for sample 1 and sample 2 respectively.

        :param ndarray sample_1: First 1d array representing feature values.
        :param ndarray sample_2: Second 1d array representing feature values.
        :returns: Kruskal-Wallis H statistic.
        :rtype: float

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
    def pct_in_top_n(x: np.ndarray, n: float) -> float:
        """
        Compute the percentage of elements in the top 'n' frequencies in the input array.

        This function calculates the percentage of elements that belong to the 'n' most
        frequent categories in the input array 'x'.

        :param np.ndarray x: Input array.
        :param float n: Number of top frequencies.
        :return: Percentage of elements in the top 'n' frequencies.
        :rtype: float

        :example:
        >>> x = np.random.randint(0, 10, (100,))
        >>> Statistics.pct_in_top_n(x=x, n=5)
        """

        check_valid_array(
            data=x, accepted_ndims=(1,), source=Statistics.pct_in_top_n.__name__
        )
        check_int(name=Statistics.pct_in_top_n.__name__, value=n, max_value=x.shape[0])
        cnts = np.sort(np.unique(x, return_counts=True)[1])[-n:]
        return np.sum(cnts) / x.shape[0]

    @staticmethod
    #@njit("(float64[:], float64[:])", cache=True)
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float64[:], float64[:])", cache=True, fastmath=False)
    def mann_whitney(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Jitted compute of Mann-Whitney U between two distributions.

        The Mann-Whitney U test is used to assess whether the distributions of two groups
        are the same or different based on their ranks. It is commonly used as an alternative
        to the t-test when the assumptions of normality and equal variances are violated.

        .. math::
           U = \\min(U_1, U_2)

        Where:
              - :math:`U` is the Mann-Whitney U statistic,
              - :math:`U_1` is the sum of ranks for sample 1,
              - :math:`U_2` is the sum of ranks for sample 2.

        :param ndarray sample_1: First 1d array representing feature values.
        :param ndarray sample_2: Second 1d array representing feature values.
        :returns : The Mann-Whitney U statistic.
        :rtype: float

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
    ) -> Tuple[float, Union[bool, None]]:
        """
        Compute Levene's W statistic, a test for the equality of variances between two samples.

        Levene's test is a statistical test used to determine whether two or more groups have equal variances. It is often
        used as an alternative to the Bartlett test when the assumption of normality is violated. The function computes the
        Levene's W statistic, which measures the degree of difference in variances between the two samples.

        .. seealso::
           For time-series based rolling comparisons, see :func:`simba.mixins.statistics_mixin.Statistics.rolling_levenes`

        :param ndarray sample_1: First 1d array representing feature values.
        :param ndarray sample_2: Second 1d array representing feature values.
        :param ndarray critical_values: 2D array with where first column represent dfn first row dfd with values represent critical values. Can be found in ``simba.assets.critical_values_05.pickle``
        :returns: Levene's W statistic and a boolean indicating whether the test is statistically significant (if critical values is not None).
        :rtype: Tuple[float, Union[bool, None]]

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
    #@njit("(float64[:], float64[:], float64)", cache=True)
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float64[:], float64[:], float64)", cache=True, fastmath=False)
    def rolling_levenes(data: np.ndarray, time_windows: np.ndarray, fps: float) -> np.ndarray:
        """
        Jitted compute of rolling Levene's W comparing the current time-window of size N to the preceding window of size N.

        .. note::
           First time bin (where has no preceding time bin) will have fill value ``0``

        .. seealso::
           For simple two-sample comparison, see :func:`simba.mixins.statistics_mixin.Statistics.levenes`

        :param ndarray sample_1: First 1d array representing feature values.
        :param ndarray sample_2: Second 1d array representing feature values.
        :returns: Levene's W data of size len(data) x len(time_windows).
        :rtype: np.ndarray

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

        r"""
        Jitted compute of Brunner-Munzel W between two distributions.

        The Brunner-Munzel W statistic compares the central tendency and the spread of two independent samples. It is useful
        for comparing the distribution of a continuous variable between two groups, especially when the assumptions of
        parametric tests like the t-test are violated.

        .. note::
           Modified from `scipy.stats.brunnermunzel <https://github.com/scipy/scipy/blob/7dcd8c59933524986923cde8e9126f5fc2e6b30b/scipy/stats/_stats_py.py#L9387>`_

        .. math::

           W = -\frac{n_x \cdot n_y \cdot (\bar{R}_y - \bar{R}_x)}{(n_x + n_y) \cdot \sqrt{n_x \cdot S_x + n_y \cdot S_y}}

        where:
           - :math:`n_x` and :math:`n_y` are the sizes of sample_1 and sample_2 respectively,
           - :math:`\bar{R}_x` and :math:`\bar{R}_y` are the mean ranks of sample_1 and sample_2, respectively.
           - :math:`S_x` and :math:`S_y` are the dispersion statistics of sample_1 and sample_2 respectively.

        :param np.ndarray sample_1: First 1d array representing feature values.
        :param np.ndarray sample_2: Second 1d array representing feature values.
        :return: Brunner-Munzel W statistic.
        :rtype: float

        :example:
        >>> sample_1, sample_2 = np.random.normal(loc=10, scale=2, size=10), np.random.normal(loc=20, scale=2, size=10)
        >>> Statistics().brunner_munzel(sample_1=sample_1, sample_2=sample_2)
        0.5751408161437165
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
    #@njit("(float32[:], float64[:], float64)", cache=True)
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float64[:], float64)", cache=True, fastmath=False)

    def rolling_barletts_test(data: np.ndarray, time_windows: np.ndarray, fps: float) -> np.ndarray:

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
    #@njit("(float32[:], float32[:])")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float32[:])", cache=True, fastmath=False)

    def pearsons_r(sample_1: np.ndarray, sample_2: np.ndarray) -> float:

        r"""
        Calculate the Pearson correlation coefficient (Pearson's r) between two numeric samples.

        Pearson's r is a measure of the linear correlation between two sets of data points. It quantifies the strength and
        direction of the linear relationship between the two variables. The coefficient varies between -1 and 1, with
        -1 indicating a perfect negative linear relationship, 1 indicating a perfect positive linear relationship, and 0
        indicating no linear relationship.

        Pearson's r is calculated using the formula:

        .. math::

           r = \frac{\sum{(x_i - \bar{x})(y_i - \bar{y})}}{\sqrt{\sum{(x_i - \bar{x})^2}\sum{(y_i - \bar{y})^2}}}

        where:
           - :math:`x_i` and :math:`y_i` are individual data points in sample_1 and sample_2, respectively.
           - :math:`\bar{x}` and :math:`\bar{y}` are the means of sample_1 and sample_2, respectively.

        .. seealso::
           For timeseries-based sliding comparison, see :func:`simba.mixins.statistics_mixin.Statistics.sliding_pearsons_r`

        :param np.ndarray sample_1: First numeric sample.
        :param np.ndarray sample_2: Second numeric sample.
        :return: Pearson's correlation coefficient between the two samples.
        :rtype: float

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
    #@njit("(float32[:], float32[:])")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float32[:])", cache=True, fastmath=False)

    def spearman_rank_correlation(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Jitted compute of Spearman's rank correlation coefficient between two samples.

        Spearman's rank correlation coefficient assesses how well the relationship between two variables can be described using a monotonic function.
        It computes the strength and direction of the monotonic relationship between ranked variables.

        .. seealso::
           For time-series based sliding comparisons, see :func:`simba.mixins.statistics.StatisticsMixin.sliding_spearman_rank_correlation`
           For time-series based sliding comparisons with GPU acceleration, see :func:`simba.data_processors.cuda.statistics.sliding_spearman_rank_correlation`,

        :param np.ndarray sample_1: First 1D array containing feature values.
        :param np.ndarray sample_2: Second 1D array containing feature values.
        :return: Spearman's rank correlation coefficient.
        :rtype: float

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
    #@njit("(float32[:], float32[:], float64[:], int64)")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float32[:], float64[:], int64)", cache=True, fastmath=False)

    def sliding_pearsons_r(sample_1: np.ndarray, sample_2: np.ndarray, time_windows: np.ndarray, fps: int) -> np.ndarray:
        """
        Given two 1D arrays of size N, create sliding window of size time_windows[i] * fps and return Pearson's R
        between the values in the two 1D arrays in each window. Address "what is the correlation between Feature 1 and
        Feature 2 in the current X.X seconds of the video".

        .. image:: _static/img/sliding_pearsons.png
           :width: 600
           :align: center

        .. seealso::
           For simple two sample comparison, see :func:`simba.mixins.statistics_mixin.Statistics.pearsons_r`

        :param ndarray sample_1: First 1D array with feature values.
        :param ndarray sample_1: Second 1D array with feature values.
        :param float time_windows: The length of the sliding window in seconds.
        :param int fps: The fps of the recorded video.
        :returns: 2d array of Pearsons R of size len(sample_1) x len(time_windows). Note, if sliding window is 10 frames, the first 9 entries will be filled with 0.
        :rtype: np.ndarray

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
    # @njit(
    #     [
    #         "(float32[:], float32[:], float64[:,:], types.unicode_type)",
    #         '(float32[:], float32[:], float64[:,:], types.misc.Omitted("goodness_of_fit"))',
    #         "(float32[:], float32[:], types.misc.Omitted(None), types.unicode_type)",
    #         '(float32[:], float32[:], types.misc.Omitted(None), types.misc.Omitted("goodness_of_fit"))',
    #     ]
    # )
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes=[
            "(float32[:], float32[:], float64[:,:], types.unicode_type)",
            '(float32[:], float32[:], float64[:,:], types.misc.Omitted("goodness_of_fit"))',
            "(float32[:], float32[:], types.misc.Omitted(None), types.unicode_type)",
            '(float32[:], float32[:], types.misc.Omitted(None), types.misc.Omitted("goodness_of_fit"))',
        ], cache=True, fastmath=False)
    def chi_square(
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        critical_values: Optional[np.ndarray] = None,
        type: Optional[Literal["goodness_of_fit", "independence"]] = "goodness_of_fit") -> Tuple[float, Union[bool, None]]:
        """
        Jitted compute of chi square between two categorical distributions.

        .. note::
           Requires sample_1 and sample_2 has to be numeric. if working with strings, convert to
           numeric category values before using chi_square.

        .. warning:
           Non-overlapping values (i.e., categories exist in sample_1 that does not exist in sample2) or small values may cause inflated chi square values.
           If small contingency table small values, consider TODO Fisher's exact test

        :param ndarray sample_1: First 1d array representing feature values.
        :param ndarray sample_2: Second 1d array representing feature values.
        :param ndarray critical_values: 2D array with where indexes represent degrees of freedom and values represent critical values. Can be found in ``simba.assets.critical_values_05.pickle``
        :returns: Size-2 tuple with the chi-square value and significance threshold boolean (if critical_values is not None).
        :rtype: Tuple[float, Union[bool, None]]

        :example:
        >>> sample_1 = np.array([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5]).astype(np.float32)
        >>> sample_2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.float32)
        >>> critical_values = pickle.load(open("simba/assets/lookups/critical_values_5.pickle", "rb"))['chi_square']['one_tail'].values
        >>> Statistics.chi_square(sample_1=sample_2, sample_2=sample_1, critical_values=critical_values, type='goodness_of_fit')
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
    #njit@("(float32[:], float32, float32, float32[:,:], float32)")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float32, float32, float32[:,:], float32)", cache=True, fastmath=False)
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

        .. seealso::
           For simple two distribution commparison, see :func:`simba.mixins.statistics_mixin.Statistics.independent_samples_t`

        :param ndarray data: 1D array with feature values.
        :param float time_window: The sizes of the two feature value windows being compared in seconds.
        :param float slide_time: The slide size of the second window.
        :param ndarray critical_values: 2D array with where indexes represent degrees of freedom and values represent critical T values. Can be found in ``simba.assets.critical_values_05.pickle``.
        :parameter int fps: The fps of the recorded video.
        :returns: 1D array of size len(data) with values representing time to most recent significantly different feature distribution.
        :rtype: np.ndarray

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
    #@njit("(float32[:], float64[:], float32)")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float64[:], float32)", cache=True, fastmath=False)

    def rolling_mann_whitney(data: np.ndarray, time_windows: np.ndarray, fps: float) -> np.ndarray:
        """
        Jitted compute of rolling Mann-Whitney U comparing the current time-window of
        size N to the preceding window of size N.

        .. note::
           First time bin (where has no preceding time bin) will have fill value ``0``

           `Modified from James Webber gist <https://gist.github.com/jamestwebber/38ab26d281f97feb8196b3d93edeeb7b>`__.

        .. seealso::
           For simple two-distribution comparion, see :func:`simba.mixins.statistics_mixin.Statistics.mann_whitney`.

        :param ndarray sample_1: First 1d array representing feature values.
        :param ndarray sample_2: Second 1d array representing feature values.
        :returns: Mann-Whitney U data of size len(data) x len(time_windows).
        :rtype: np.ndarray

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
    #@njit("(int64[:, :]), bool_")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(int64[:, :]), bool_", cache=True, fastmath=False)

    def concordance_ratio(x: np.ndarray, invert: bool) -> float:
        """
        Calculate the concordance ratio of a 2D numpy array. The concordance ratio is a measure of agreement in a dataset. It is calculated as the ratio of the number of
        rows that contain only one unique value to the total number of rows.

        The equation for the concordance ratio :math:`C` is given by:

        .. math::
           C = \\frac{N_c}{N_t}

        where:
           - :math:`N_c` is the count of rows with only one unique value,
           - :math:`N_t` is the total number of rows in the array.

        If the `invert` parameter is set to `True`, the function will return the disconcordance ratio instead, defined as:

        .. math::
            D = \\frac{N_d}{N_t}

        where:

           - :math:`N_d` is the count of rows with more than one unique value.

        :param np.ndarray x: A 2D numpy array with ordinals represented as integers.
        :param bool invert: If True, the concordance ratio is inverted, and disconcordance ratio is returned
        :return: The concordance ratio, representing the count of rows with only one unique value divided by the total number of rows in the array.
        :rtype: float

        :example:
        >>> x = np.random.randint(0, 2, (5000, 4))
        >>> results = Statistics.concordance_ratio(x=x, invert=False)
        """
        conc_count = 0
        for i in prange(x.shape[0]):
            unique_cnt = np.unique((x[i])).shape[0]
            if unique_cnt == 1:
                conc_count += 1
        if invert:
            conc_count = x.shape[0] - conc_count
        return conc_count / x.shape[0]

    @staticmethod
    #@njit("(float32[:], float32[:], float64[:], int64)")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float32[:], float64[:], int64)", cache=True, fastmath=False)

    def sliding_spearman_rank_correlation(sample_1: np.ndarray, sample_2: np.ndarray, time_windows: np.ndarray, fps: int) -> np.ndarray:
        """
        Given two 1D arrays of size N, create sliding window of size time_windows[i] * fps and return Spearman's rank correlation
        between the values in the two 1D arrays in each window. Address "what is the correlation between Feature 1 and
        Feature 2 in the current X.X seconds of the video.

        .. image:: _static/img/sliding_spearman.png
           :width: 600
           :align: center

        .. seealso::
           For simple two-distribution comparion, see :func:`simba.mixins.statistics_mixin.Statistics.spearman_rank_correlation`.

        :param ndarray sample_1: First 1D array with feature values.
        :param ndarray sample_1: Second 1D array with feature values.
        :param float time_windows: The length of the sliding window in seconds.
        :param int fps: The fps of the recorded video.
        :returns: 2d array of Soearman's ranks of size len(sample_1) x len(time_windows). Note, if sliding window is 10 frames, the first 9 entries will be filled with 0. The 10th value represents the correlation in the first 10 frames.
        :rtype: np.ndarray

        :example:
        >>> sample_1 = np.array([9,10,13,22,15,18,15,19,32,11]).astype(np.float32)
        >>> sample_2 = np.array([11, 12, 15, 19, 21, 26, 19, 20, 22, 19]).astype(np.float32)
        >>> Statistics().sliding_spearman_rank_correlation(sample_1=sample_1, sample_2=sample_2, time_windows=np.array([0.5]), fps=10)
        """

        results = np.full((sample_1.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            for left, right in zip(range(0, sample_1.shape[0] + 1), range(window_size, sample_1.shape[0] + 1)):
                s1, s2 = sample_1[left:right], sample_2[left:right]
                rank_x, rank_y = np.argsort(np.argsort(s1)), np.argsort(np.argsort(s2))
                d_squared = np.sum((rank_x - rank_y) ** 2)
                n = s1.shape[0]
                s = 1 - (6 * d_squared) / (n * (n ** 2 - 1))
                results[right - 1, i] = s

        return results

    @staticmethod
    #@njit("(float32[:], float64, float64, float64)")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float64, float64, float64)", cache=True, fastmath=False)

    def sliding_autocorrelation(data: np.ndarray, max_lag: float, time_window: float, fps: float) -> np.ndarray:
        """
        Jitted computation of sliding autocorrelations, which measures the correlation of a feature with itself using lagged windows.

        :param np.ndarray data: 1D array containing feature values.
        :param float max_lag: Maximum lag in seconds for the autocorrelation window.
        :param float time_window: Length of the sliding time window in seconds.
        :param float fps: Frames per second, used to convert time-related parameters into frames.
        :return: 1D array containing the sliding autocorrelation values.
        :rtype: np.ndarray

        :example:
        >>> data = np.array([0,1,2,3,4, 5,6,7,8,1,10,11,12,13,14]).astype(np.float32)
        >>> Statistics().sliding_autocorrelation(data=data, max_lag=0.5, time_window=1.0, fps=10)
        >>> [ 0., 0., 0.,  0.,  0., 0., 0.,  0. ,  0., -3.686, -2.029, -1.323, -1.753, -3.807, -4.634]
        """

        max_frm_lag, time_window_frms = int(max_lag * fps), int(time_window * fps)
        results = np.full((data.shape[0]), -1.0)
        for right in prange(time_window_frms - 1, data.shape[0]):
            left = right - time_window_frms + 1
            w_data = data[left : right + 1]
            corrcfs = np.full((max_frm_lag), np.nan)
            corrcfs[0] = 1
            for shift in range(1, max_frm_lag):
                c = np.corrcoef(w_data[:-shift], w_data[shift:])[0][1]
                if np.isnan(c):
                    corrcfs[shift] = 1
                else:
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
    def sliding_dominant_frequencies(data: np.ndarray,
                                     fps: float,
                                     k: int,
                                     time_windows: np.ndarray,
                                     window_function: Literal["Hann", "Hamming", "Blackman"] = None) -> np.ndarray:

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
    #@njit("(float32[:], float32[:])")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float32[:])", cache=True, fastmath=False)


    def kendall_tau(sample_1: np.ndarray, sample_2: np.ndarray) -> Tuple[float, float]:
        """
        Jitted compute of Kendall Tau (rank correlation coefficient). Non-parametric method for computing correlation
        between two time-series features. Returns tau and associated z-score.

        Kendall Tau is a measure of the correspondence between two rankings. It compares the number of concordant
        pairs (pairs of elements that are in the same order in both rankings) to the number of discordant pairs
        (pairs of elements that are in different orders in the rankings).

        Kendall Tau is calculated using the following formula:

        .. math::

           \\tau = \\frac{{\\sum C - \\sum D}}{{\\sum C + \\sum D}}

        where :math:`C` is the count of concordant pairs and :math:`D` is the count of discordant pairs.

        .. seealso::
           For time-series based comparison, see :func:`simba.mixins.statistics_mixin.Statistics.sliding_kendall_tau`.

        :param ndarray sample_1: First 1D array with feature values.
        :param ndarray sample_1: Second 1D array with feature values.
        :returns: Size-2 tuple with Kendall Tau and associated z-score.
        :rtype: Tuple[float, float]

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
            cncrdnt_cnts[i] = (np.argwhere(s2_rnk[i + 1 :] > s1_rnk[i]).flatten().shape[0])
            dscrdnt_cnts[i] = (np.argwhere(s2_rnk[i + 1 :] < s1_rnk[i]).flatten().shape[0])
        t = (np.sum(cncrdnt_cnts) - np.sum(dscrdnt_cnts)) / (np.sum(cncrdnt_cnts) + np.sum(dscrdnt_cnts))
        z = (3 * t * (np.sqrt(s1_rnk.shape[0] * (s1_rnk.shape[0] - 1))) / np.sqrt(2 * ((2 * s1_rnk.shape[0]) + 5)))
        return t, z

    @staticmethod
    #@njit("(float32[:], float32[:], float64[:], int64)")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float32[:], float64[:], int64)", cache=True, fastmath=False)

    def sliding_kendall_tau(sample_1: np.ndarray, sample_2: np.ndarray, time_windows: np.ndarray, fps: float) -> np.ndarray:
        """
        Compute sliding Kendall's Tau correlation coefficient.

        Calculates Kendall's Tau correlation coefficient between two samples over sliding time windows. Kendall's Tau is a measure of correlation between two ranked datasets.

        The computation is based on the formula:

        .. math::

           \\tau = \\frac{{\\text{{concordant pairs}} - \\text{{discordant pairs}}}}{{\\text{{concordant pairs}} + \\text{{discordant pairs}}}}

        where concordant pairs are pairs of elements with the same order in both samples, and discordant pairs are pairs with different orders.


        .. seealso::
           For simple two-sample comparison, see :func:`simba.mixins.statistics_mixin.Statistics.kendall_tau`.


        References
        ----------
        .. [1] `Stephanie Glen, "Kendall’s Tau (Kendall Rank Correlation Coefficient)"  <https://www.statisticshowto.com/kendalls-tau/>`__.

        :param np.ndarray sample_1: First sample for comparison.
        :param np.ndarray sample_2: Second sample for comparison.
        :param np.ndarray time_windows: Rolling time windows in seconds.
        :param float fps: Frames per second (FPS) of the recorded video.
        :return: Array of Kendall's Tau correlation coefficients corresponding to each time window.
        :rtype: np.ndarray
        """

        results = np.full((sample_1.shape[0], time_windows.shape[0]), 0.0)
        for time_window_cnt in range(time_windows.shape[0]):
            window_size = int(time_windows[time_window_cnt] * fps)
            for left, right in zip(range(0, sample_1.shape[0] + 1), range(window_size, sample_1.shape[0] + 1)):
                sliced_sample_1, sliced_sample_2 = (sample_1[left:right], sample_2[left:right])
                rnks = np.argsort(sliced_sample_1)
                s1_rnk, s2_rnk = sliced_sample_1[rnks], sliced_sample_2[rnks]
                cncrdnt_cnts, dscrdnt_cnts = np.full((s1_rnk.shape[0] - 1), np.nan), np.full((s1_rnk.shape[0] - 1), np.nan)
                for i in range(s2_rnk.shape[0] - 1):
                    cncrdnt_cnts[i] = (np.argwhere(s2_rnk[i + 1 :] > s1_rnk[i]).flatten().shape[0])
                    dscrdnt_cnts[i] = (np.argwhere(s2_rnk[i + 1 :] < s1_rnk[i]).flatten().shape[0])
                n = np.sum(cncrdnt_cnts) - np.sum(dscrdnt_cnts)
                d = np.sum(cncrdnt_cnts) + np.sum(dscrdnt_cnts)
                if d == 0:
                    results[right][time_window_cnt] = -1
                else:
                    results[right][time_window_cnt] = n / d

        return results

    @staticmethod
    def find_collinear_features(df: pd.DataFrame,
                                threshold: float,
                                method: Optional[Literal["pearson", "spearman", "kendall"]] = "pearson",
                                verbose: Optional[bool] = False) -> List[str]:

        """
        Identify collinear features in the dataframe based on the specified correlation method and threshold.

        .. seealso::
           For multicore numba accelerated method, see :func:`simba.mixins.train_model_mixin.TrainModelMixin.find_highly_correlated_fields`.

        :param pd.DataFrame df: Input DataFrame containing features.
        :param float threshold: Threshold value to determine collinearity.
        :param Optional[Literal['pearson', 'spearman', 'kendall']] method: Method for calculating correlation. Defaults to 'pearson'.
        :return: Set of feature names identified as collinear. Returns one feature for every feature pair with correlation value above specified threshold.
        :rtype: List[str]

        :example:
        >>> x = pd.DataFrame(np.random.randint(0, 100, (100, 100)))
        >>> names = Statistics.find_collinear_features(df=x, threshold=0.2, method='pearson', verbose=True)
        """

        check_valid_dataframe(df=df, source=Statistics.find_collinear_features.__name__, valid_dtypes=Formats.NUMERIC_DTYPES.value, min_axis_1=1, min_axis_0=1)
        check_float(
            name=Statistics.find_collinear_features.__name__,
            value=threshold,
            max_value=1.0,
            min_value=0.0,
        )
        check_str(
            name=Statistics.find_collinear_features.__name__,
            value=method,
            options=("pearson", "spearman", "kendall"),
        )
        feature_names = set()
        feature_pairs = list(combinations(list(df.columns), 2))

        for cnt, i in enumerate(feature_pairs):
            if verbose:
                print(
                    f"Analyzing feature pair collinearity {cnt + 1}/{len(feature_pairs)}..."
                )
            if (i[0] not in feature_names) and (i[1] not in feature_names):
                sample_1, sample_2 = df[i[0]].values.astype(np.float32), df[
                    i[1]
                ].values.astype(np.float32)
                if method == "pearson":
                    r = Statistics.pearsons_r(sample_1=sample_1, sample_2=sample_2)
                elif method == "spearman":
                    r = Statistics.spearman_rank_correlation(
                        sample_1=sample_1, sample_2=sample_2
                    )
                else:
                    r = Statistics.kendall_tau(sample_1=sample_1, sample_2=sample_2)[0]
                if abs(r) > threshold:
                    feature_names.add(i[0])
        if verbose:
            print("Collinear analysis complete.")
        return list(feature_names)

    @staticmethod
    def local_outlier_factor(
        data: np.ndarray,
        k: Union[int, float] = 5,
        contamination: Optional[float] = 1e-10,
        normalize: Optional[bool] = False,
        groupby_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the local outlier factor of each observation.

        .. note::
           The final LOF scores are negated. Thus, higher values indicate more atypical (outlier) data points. Values
           Method calls ``sklearn.neighbors.LocalOutlierFactor`` directly. Attempted to use own jit compiled implementation,
           but runtime was 3x-ish slower than ``sklearn.neighbors.LocalOutlierFactor``.

           If groupby_idx is not None, then the index 1 of ``data`` array for which to group the data and compute LOF within each segment/cluster.
           E.g., can be field holding cluster identifier. Thus, outliers are computed within each segment/cluster, ensuring that other segments cannot affect
           outlier scores within each analyzing each cluster.

           If groupby_idx is provided, then all observations with cluster/segment variable ``-1`` will be treated as unclustered and assigned the max outlier score found withiin the clustered observations.

        .. image:: _static/img/local_outlier_factor.png
           :width: 800
           :align: center

        :param ndarray data: 2D array with feature values where rows represent frames and columns represent features.
        :param Union[int, float] k: Number of neighbors to evaluate for each observation. If the value is a float, then interpreted as the ratio of data.shape[0]. If the value is an integer, then it represent the number of neighbours to evaluate.
        :param Optional[float] contamination: Small pseudonumber to avoid DivisionByZero error.
        :param Optional[bool] normalize: Whether to normalize the distances between 0 and 1. Defaults to False.
        :param Optional[int] groupby_idx: If int, then the index 1 of ``data`` for which to group the data and compute LOF on each segment. E.g., can be field holding a cluster identifier.
        :returns: Array of size data.shape[0] with local outlier scores.
        :rtype: np.ndarray

        :example:
        >>> data, lbls = make_blobs(n_samples=2000, n_features=2, centers=10, random_state=42)
        >>> data = np.hstack((data, lbls.reshape(-1, 1)))
        >>> lof = Statistics.local_outlier_factor(data=data, groupby_idx=2, k=100, normalize=True)
        >>> results = np.hstack((data[:, 0:2], lof.reshape(lof.shape[0], 1)))
        >>> PlottingMixin.continuous_scatter(data=results, palette='seismic', bg_clr='lightgrey',size=30)
        """

        def get_lof(data, k, contamination):
            check_float(name=f"{Statistics.local_outlier_factor.__name__} k", value=k)
            if isinstance(k, int):
                k = min(k, data.shape[0])
            elif isinstance(k, float):
                k = int(data.shape[0] * k)
            lof_model = LocalOutlierFactor(n_neighbors=k, contamination=contamination)
            _ = lof_model.fit_predict(data)
            y = -lof_model.negative_outlier_factor_.astype(np.float32)
            if normalize:
                return (y - np.min(y)) / (np.max(y) - np.min(y))
            else:
                return y

        if groupby_idx is not None:
            check_int(
                name=f"{Statistics.local_outlier_factor.__name__} groupby_idx",
                value=groupby_idx,
                min_value=0,
                max_value=data.shape[1] - 1,
            )
            check_valid_array(
                source=f"{Statistics.local_outlier_factor.__name__} local_outlier_factor",
                data=data,
                accepted_sizes=[2],
                min_axis_1=3,
            )
        else:
            check_valid_array(
                source=f"{Statistics.local_outlier_factor.__name__} data",
                data=data,
                accepted_sizes=[2],
                min_axis_1=2,
            )
        check_float(
            name=f"{Statistics.local_outlier_factor.__name__} contamination",
            value=contamination,
            min_value=0.0,
        )

        if groupby_idx is None:
            return get_lof(data, k, contamination)
        else:
            results = []
            data_w_idx = np.hstack((np.arange(0, data.shape[0]).reshape(-1, 1), data))
            unique_c = np.unique(data[:, groupby_idx]).astype(np.float32)
            if -1.0 in unique_c:
                unique_c = unique_c[np.where(unique_c != -1)]
                unclustered_idx = np.argwhere(data[:, groupby_idx] == -1.0).flatten()
                unclustered = data_w_idx[unclustered_idx]
                data_w_idx = np.delete(data_w_idx, unclustered_idx, axis=0)
            else:
                unclustered = None
            for i in unique_c:
                s_data = data_w_idx[
                    np.argwhere(data_w_idx[:, groupby_idx + 1] == i)
                ].reshape(-1, data_w_idx.shape[1])
                idx = s_data[:, 0].reshape(s_data.shape[0], 1)
                s_data = np.delete(s_data, [0, groupby_idx + 1], 1)
                lof = get_lof(s_data, k, contamination).reshape(s_data.shape[0], 1)
                results.append(np.hstack((idx, lof)))
            x = np.concatenate(results, axis=0)
            if unclustered is not None:
                max_lof = np.full((unclustered.shape[0], 1), np.max(x[:, -1]))
                unclustered = np.hstack((unclustered, max_lof))[:, [0, -1]]
                x = np.vstack((x, unclustered))
            return x[np.argsort(x[:, 0])][:, -1]

    @staticmethod
    def elliptic_envelope(
        data: np.ndarray,
        contamination: Optional[float] = 1e-1,
        normalize: Optional[bool] = False,
        groupby_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Compute the Mahalanobis distances of each observation in the input array using Elliptic Envelope method.

        .. image:: _static/img/EllipticEnvelope.png
           :width: 700
           :align: center

        .. image:: _static/img/elliptic_envelope.png
           :width: 700
           :align: center

        :param data: Input data array of shape (n_samples, n_features).
        :param Optional[float] contamination: The proportion of outliers to be assumed in the data. Defaults to 0.1.
        :param Optional[bool] normalize: Whether to normalize the Mahalanobis distances between 0 and 1. Defaults to True.
        :return: The Mahalanobis distances of each observation in array. Larger values indicate outliers.
        :rtype: np.ndarray

        :example:
        >>> data, lbls = make_blobs(n_samples=2000, n_features=2, centers=1, random_state=42)
        >>> envelope_score = elliptic_envelope(data=data, normalize=True)
        >>> results = np.hstack((data[:, 0:2], envelope_score.reshape(lof.shape[0], 1)))
        >>> results = pd.DataFrame(results, columns=['X', 'Y', 'ENVELOPE SCORE'])
        >>> PlottingMixin.continuous_scatter(data=results, palette='seismic', bg_clr='lightgrey', columns=['X', 'Y', 'ENVELOPE SCORE'],size=30)

        """

        def get_envelope(data, contamination) -> np.ndarray:
            mdl = EllipticEnvelope(contamination=contamination).fit(data)
            y = -mdl.score_samples(data)
            if normalize:
                y = (y - np.min(y)) / (np.max(y) - np.min(y))
            return y

        if groupby_idx is not None:
            check_int(
                name=f"{Statistics.elliptic_envelope.__name__} groupby_idx",
                value=groupby_idx,
                min_value=0,
                max_value=data.shape[1] - 1,
            )
            check_valid_array(
                source=f"{Statistics.elliptic_envelope.__name__} local_outlier_factor",
                data=data,
                accepted_sizes=[2],
                min_axis_1=3,
            )
        else:
            check_valid_array(
                source=f"{Statistics.elliptic_envelope.__name__} data",
                data=data,
                accepted_sizes=[2],
                min_axis_1=2,
            )

        check_float(
            name=f"{Statistics.elliptic_envelope.__name__} contamination",
            value=contamination,
            min_value=0.0,
            max_value=1.0,
        )
        if groupby_idx is None:
            return get_envelope(data, contamination)
        else:
            results = []
            data_w_idx = np.hstack((np.arange(0, data.shape[0]).reshape(-1, 1), data))
            unique_c = np.unique(data[:, groupby_idx]).astype(np.float32)
            if -1.0 in unique_c:
                unique_c = unique_c[np.where(unique_c != -1)]
                unclustered_idx = np.argwhere(data[:, groupby_idx] == -1.0).flatten()
                unclustered = data_w_idx[unclustered_idx]
                data_w_idx = np.delete(data_w_idx, unclustered_idx, axis=0)
            else:
                unclustered = None
            for i in unique_c:
                s_data = data_w_idx[
                    np.argwhere(data_w_idx[:, groupby_idx + 1] == i)
                ].reshape(-1, data_w_idx.shape[1])
                idx = s_data[:, 0].reshape(s_data.shape[0], 1)
                s_data = np.delete(s_data, [0, groupby_idx + 1], 1)
                lof = get_envelope(s_data, contamination).reshape(s_data.shape[0], 1)
                results.append(np.hstack((idx, lof)))
            x = np.concatenate(results, axis=0)
            if unclustered is not None:
                max_env_score = np.full((unclustered.shape[0], 1), np.max(x[:, -1]))
                unclustered = np.hstack((unclustered, max_env_score))[:, [0, -1]]
                x = np.vstack((x, unclustered))
            return x[np.argsort(x[:, 0])][:, -1]

    @staticmethod
    def isolation_forest(
        x: np.ndarray,
        estimators: Union[int, float] = 0.2,
        groupby_idx: Optional[int] = None,
        normalize: Optional[bool] = False,
    ) -> np.ndarray:
        """
        An implementation of the Isolation Forest algorithm for outlier detection.

        .. image:: _static/img/isolation_forest.png
           :width: 700
           :align: center

        .. note::
           The isolation forest scores are negated. Thus, higher values indicate more atypical (outlier) data points.

        :param np.ndarray x: 2-D array with feature values.
        :param Union[int, float] estimators: Number of splits. If the value is a float, then interpreted as the ratio of x shape.
        :param Optional[int] groupby_idx: If int, then the index 1 of ``data`` for which to group the data and compute LOF on each segment. E.g., can be field holding a cluster identifier.
        :param Optional[bool] normalize: Whether to normalize the outlier score between 0 and 1. Defaults to False.
        :return: 2D array with the x, y and the isolation forest outlier score for each observation.
        :rtype: np.ndarray

        :example:
        >>> x, lbls = make_blobs(n_samples=10000, n_features=2, centers=10, random_state=42)
        >>> x = np.hstack((x, lbls.reshape(-1, 1)))
        >>> scores = isolation_forest(x=x, estimators=10, normalize=True)
        >>> results = np.hstack((x[:, 0:2], scores.reshape(scores.shape[0], 1)))
        >>> results = pd.DataFrame(results, columns=['X', 'Y', 'ISOLATION SCORE'])
        >>> PlottingMixin.continuous_scatter(data=results, palette='seismic', bg_clr='lightgrey', columns=['X', 'Y', 'ISOLATION SCORE'],size=30)

        :references:
        ----------
           .. [1] Liu, Fei Tony, Kai Ming Ting, and Zhi-Hua Zhou. “Isolation Forest.” In 2008 Eighth IEEE International Conference on Data Mining, 413–22. Pisa, Italy: IEEE, 2008. https://doi.org/10.1109/ICDM.2008.17.
        """

        def get_if_scores(x: np.ndarray, estimators: estimators):
            if isinstance(estimators, float):
                check_float(
                    name=f"{Statistics.isolation_forest.__name__} estimators",
                    value=estimators,
                    min_value=10e-6,
                    max_value=1.0,
                )
                estimators = x.shape[0] * estimators
                if estimators < 1:
                    estimators = 1
            else:
                check_int(
                    name=f"{Statistics.isolation_forest.__name__.__name__} estimators",
                    value=estimators,
                    min_value=1,
                )
            mdl = IsolationForest(
                n_estimators=estimators,
                n_jobs=-1,
                behaviour="new",
                contamination="auto",
            )
            r = abs(mdl.fit(x).score_samples(x))
            if normalize:
                r = (r - np.min(r)) / (np.max(r) - np.min(r))
            return r

        if groupby_idx is None:
            check_valid_array(data=x, source=Statistics.isolation_forest.__name__.__name__, accepted_ndims=(2,), min_axis_1=2, accepted_dtypes=Formats.NUMERIC_DTYPES.value)
            return get_if_scores(x=x, estimators=estimators)

        else:
            check_valid_array(data=x, source=Statistics.isolation_forest.__name__.__name__, accepted_ndims=(2,), min_axis_1=3, accepted_dtypes=Formats.NUMERIC_DTYPES.value)
            results = []
            data_w_idx = np.hstack((np.arange(0, x.shape[0]).reshape(-1, 1), x))
            unique_c = np.unique(x[:, groupby_idx]).astype(np.float32)
            if -1.0 in unique_c:
                unique_c = unique_c[np.where(unique_c != -1)]
                unclustered_idx = np.argwhere(x[:, groupby_idx] == -1.0).flatten()
                unclustered = data_w_idx[unclustered_idx]
                data_w_idx = np.delete(data_w_idx, unclustered_idx, axis=0)
            else:
                unclustered = None
            for i in unique_c:
                s_data = data_w_idx[
                    np.argwhere(data_w_idx[:, groupby_idx + 1] == i)
                ].reshape(-1, data_w_idx.shape[1])
                idx = s_data[:, 0].reshape(s_data.shape[0], 1)
                s_data = np.delete(s_data, [0, groupby_idx + 1], 1)
                i_f = get_if_scores(s_data, estimators).reshape(s_data.shape[0], 1)
                results.append(np.hstack((idx, i_f)))
            x = np.concatenate(results, axis=0)
            if unclustered is not None:
                max_if = np.full((unclustered.shape[0], 1), np.max(x[:, -1]))
                unclustered = np.hstack((unclustered, max_if))[:, [0, -1]]
                x = np.vstack((x, unclustered))
            return x[np.argsort(x[:, 0])][:, -1]

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
        :return: Array of size data.shape[0] representing outlier scores, with higher values representing greater outliers.
        :rtype: np.ndarray
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

        .. image:: _static/img/hbos.png
           :width: 1200
           :align: center

        :param np.ndarray data: 2d array with frames represented by rows and columns representing feature values.
        :param Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators.
        :return: Array of size data.shape[0] representing outlier scores, with higher values representing greater outliers.
        :rtype: np.ndarray

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
            bin_width, bin_count = bucket_data(
                data=data[:, i].flatten(), method=bucket_method
            )
            histograms[i] = self._hist_1d(
                data=data[:, i].flatten(),
                bin_count=bin_count,
                range=np.array([0, int(bin_width * bin_count)]),
            ).astype(np.int64)
            histogram_edges[i] = np.arange(0, 1 + bin_width, bin_width).astype(
                np.float64
            )
        results = self._hbos_compute(
            data=data, histograms=histograms, histogram_edges=histogram_edges
        )
        return results.astype(np.float32)

    def rolling_shapiro_wilks(self, data: np.ndarray, time_window: float, fps: int) -> np.ndarray:
        """
        Compute Shapiro-Wilks normality statistics for sequentially binned values in a time-series. E.g., compute
        the normality statistics of ``Feature N`` in each window of ``time_window`` seconds.

        :param ndarray data: 1D array of size len(frames) representing feature values.
        :param int time_window: The size of the buckets in seconds.
        :param int fps: Frame-rate of recorded video.
        :return: Array of size data.shape[0] with Shapiro-Wilks normality statistics
        :rtype: np.ndarray

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
    #@njit("(float32[:], float64[:], int64,)")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float64[:], int64,)", cache=True, fastmath=False)

    def sliding_z_scores(data: np.ndarray, time_windows: np.ndarray, fps: int) -> np.ndarray:

        """
        Calculate sliding Z-scores for a given data array over specified time windows.

        This function computes sliding Z-scores for a 1D data array over different time windows. The sliding Z-score
        is a measure of how many standard deviations a data point is from the mean of the surrounding data within
        the specified time window. This can be useful for detecting anomalies or variations in time-series data.

        :param ndarray data: 1D NumPy array containing the time-series data.
        :param ndarray time_windows: 1D NumPy array specifying the time windows in seconds over which to calculate the Z-scores.
        :param int time_windows: Frames per second, used to convert time windows from seconds to the corresponding number of data points.
        :returns: A 2D NumPy array containing the calculated Z-scores. Each row corresponds to the Z-scores calculated for a specific time window. The time windows are represented by the columns.
        :rtype: np.ndarray

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
    #@njit("(int64[:, :],)")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(int64[:, :],)", cache=True, fastmath=False)

    def phi_coefficient(data: np.ndarray) -> float:
        """
        Compute the phi coefficient for a Nx2 array of binary data.

        The phi coefficient (a.k.a Matthews Correlation Coefficient (MCC)), is a measure of association for binary data in a 2x2 contingency table. It quantifies the
        degree of association or correlation between two binary variables (e.g., binary classification targets).

        The formula for the phi coefficient is defined as:

        .. math::

           \\phi = \\frac{{(BC - AD)}}{{\sqrt{{(C\_1 + C\_2)(R\_1 + R\_2)(C\_1 + R\_1)(C\_2 + R\_2)}}}}

        where:
            - :math:`BC`: Hit rate (reponse and truth is both 1)
            - :math:`AD`: Correct rejections (response and truth are both 0)
            - :math:`C1, C2`: Counts of occurrences where the response is 1 and 0, respectively.
            - :math`R1, R2`: Counts of occurrences where the truth is 1 and 0, respectively.

        .. seealso::
           For time-series based sliding comparisons, see :func:`simba.mixins.statistics_mixin.Statistics.sliding_phi_coefficient`


        :param np.ndarray data: A NumPy array containing binary data organized in two columns. Each row represents a pair of binary values for two variables. Columns represent two features or two binary classification results.
        :returns: The calculated phi coefficient, a value between 0 and 1. A value of 0 indicates no association between the variables, while 1 indicates a perfect association.
        :rtype: float

        :example:
        >>> data = np.array([[0, 1], [1, 0], [1, 0], [1, 1]]).astype(np.int64)
        >>> Statistics().phi_coefficient(data=data)
        >>> 0.8164965809277261
        >>> data = np.random.randint(0, 2, (100, 2))
        >>> result = Statistics.phi_coefficient(data=data)
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
    def eta_squared(x: np.ndarray, y: np.ndarray) -> float:

        r"""
        Calculate eta-squared, a measure of between-subjects effect size.

        Eta-squared (\(\eta^2\)) is calculated as the ratio of the sum of squares between groups to the total sum of squares. Range from 0 to 1, where larger values indicate
        a stronger effect size.

        The equation for eta squared is defined as: :math:`\eta^2 = \frac{SS_{between}}{SS_{between} + SS_{within}}`

        where:
           - :math:`SS_{between}` is the sum of squares between groups,
           - :math:`SS_{within}` is the sum of squares within groups.

        .. seealso::
           For sliding time-windows comparisons, see :func:`simba.mixins.statistics_mixin.Statistics.sliding_eta_squared`.

        :param np.ndarray x: 1D array containing the dependent variable data.
        :param np.ndarray y: 1d array containing the grouping variable (categorical) data of same size as ``x``.
        :return: The eta-squared value representing the proportion of variance in the dependent variable that is attributable to the grouping variable.
        :rtype: float
        """

        check_valid_array(data=x, source=f'{Statistics.eta_squared.__name__} x', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, source=f'{Statistics.eta_squared.__name__} y', accepted_shapes=[x.shape])
        sum_square_within, sum_square_between = 0, 0
        for lbl in np.unique(y):
            g = x[np.argwhere(y == lbl)]
            sum_square_within += np.sum((g - np.mean(g)) ** 2)
            sum_square_between += len(g) * (np.mean(g) - np.mean(x)) ** 2
        if sum_square_between + sum_square_within == 0:
            return 0.0
        else:
            return (sum_square_between / (sum_square_between + sum_square_within)) ** .5

    @staticmethod
    @jit(nopython=True)
    def sliding_eta_squared(x: np.ndarray, y: np.ndarray, window_sizes: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Calculate sliding window eta-squared, a measure of effect size for between-subjects designs,
        over multiple window sizes.

        .. seealso::
           For two-sample comparison, see :func:`simba.mixins.statistics_mixin.Statistics.eta_squared`

        :param np.ndarray x: The array containing the dependent variable data.
        :param np.ndarray y: The array containing the grouping variable (categorical) data.
        :param np.ndarray window_sizes: 1D array of window sizes in seconds.
        :param int sample_rate: The sampling rate of the data in frames per second.
        :return: Array of size  x.shape[0] x window_sizes.shape[0] with sliding eta squared values.
        :rtype: np.ndarray

        :example:
        >>> x = np.random.randint(0, 10, (10000,))
        >>> y = np.random.randint(0, 2, (10000,))
        >>> Statistics.sliding_eta_squared(x=x, y=y, window_sizes=np.array([1.0, 2.0]), sample_rate=10)

        """
        results = np.full((x.shape[0], window_sizes.shape[0]), -1.0)
        for i in range(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(range(0, x.shape[0] + 1), range(window_size, x.shape[0] + 1)):
                sample_x = x[l:r]
                sample_y = y[l:r]
                sum_square_within, sum_square_between = 0, 0
                for lbl in np.unique(sample_y):
                    g = sample_x[np.argwhere(sample_y == lbl).flatten()]
                    sum_square_within += np.sum((g - np.mean(g)) ** 2)
                    sum_square_between += len(g) * (np.mean(g) - np.mean(sample_x)) ** 2
                if sum_square_between + sum_square_within == 0:
                    results[r - 1, i] = 0.0
                else:
                    results[r - 1, i] = (sum_square_between / (sum_square_between + sum_square_within)) ** .5
        return results


    @staticmethod
    #@njit("(int32[:, :], float64[:], int64)")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(int32[:, :], float64[:], int64)", cache=True, fastmath=False)

    def sliding_phi_coefficient(data: np.ndarray, window_sizes: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Calculate sliding phi coefficients for a 2x2 contingency table derived from binary data.

        Computes sliding phi coefficients for a 2x2 contingency table derived from binary data over different
        time windows. The phi coefficient is a measure of association between two binary variables, and sliding phi
        coefficients can reveal changes in association over time.

        .. seealso::
           For simple two distribution comparison, see :func:`simba.mixins.statistics_mixin.Statistics.phi_coefficient`.

        :param np.ndarray data: A 2D NumPy array containing binary data organized in two columns. Each row represents a pair of binary values for two variables.
        :param np.ndarray window_sizes: 1D NumPy array specifying the time windows (in seconds) over which to calculate the sliding phi coefficients.
        :param int sample_rate: The sampling rate or time interval (in samples per second, e.g., fps) at which data points were collected.
        :returns: A 2D NumPy array containing the calculated sliding phi coefficients. Each row corresponds to the phi coefficients calculated for a specific time point, the columns correspond to time-windows.
        :rtype: np.ndarray

        :example:
        >>> data = np.random.randint(0, 2, (200, 2))
        >>> Statistics().sliding_phi_coefficient(data=data, window_sizes=np.array([1.0, 4.0]), sample_rate=10)
        """

        results = np.full((data.shape[0], window_sizes.shape[0]), -1.0)
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(range(0, data.shape[0] + 1), range(window_size, data.shape[0] + 1)):
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
    def relative_risk(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the relative risk between two binary arrays.

        Relative risk (RR) is the ratio of the probability of an event occurring in one group/feature/cluster/variable (x)
        to the probability of the event occurring in another group/feature/cluster/variable (y).

        .. seealso::
           For time-series based sliding data, use :func:`simba.mixins.statistics_mixin.Statistics.sliding_relative_risk`

        :param np.ndarray x: The first 1D binary array.
        :param np.ndarray y: The second 1D binary array.
        :return: The relative risk between arrays x and y.
        :rtype: float

        :example:
        >>> Statistics.relative_risk(x=np.array([0, 1, 1]), y=np.array([0, 1, 0]))
        >>> 2.0
        """
        check_valid_array(data=x, source=f'{Statistics.relative_risk.__name__} x', accepted_ndims=(1,), accepted_values=[0, 1])
        check_valid_array(data=y, source=f'{Statistics.relative_risk.__name__} y', accepted_ndims=(1,), accepted_values=[0, 1])
        if np.sum(y) == 0:
            return -1.0
        elif np.sum(x) == 0:
            return 0.0
        else:
            return (np.sum(x) / x.shape[0]) / (np.sum(y) / y.shape[0])

    @staticmethod
    @jit(nopython=True)
    def sliding_relative_risk(x: np.ndarray, y: np.ndarray, window_sizes: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Calculate sliding relative risk values between two binary arrays using different window sizes.

        .. seealso::
           For single two distribution comparion, use :func:`simba.mixins.statistics_mixin.Statistics.relative_risk`

        :param np.ndarray x: The first 1D binary array.
        :param np.ndarray y: The second 1D binary array.
        :param np.ndarray window_sizes: One-dimensional array with the time-windows in seconds.
        :param int sample_rate: The sample rate of the input data (e.g., FPS).
        :return np.ndarray: Array of size  x.shape[0] x window_sizes.shape[0] with sliding eta squared values.

        :example:
        >>> Statistics.sliding_relative_risk(x=np.array([0, 1, 1, 0]), y=np.array([0, 1, 0, 0]), window_sizes=np.array([1.0]), sample_rate=2)
        """
        results = np.full((x.shape[0], window_sizes.shape[0]), -1.0)
        for i in range(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(range(0, x.shape[0] + 1), range(window_size, x.shape[0] + 1)):
                sample_x, sample_y = x[l:r], y[l:r]
                if np.sum(sample_y) == 0:
                    results[r - 1, i] = -1.0
                elif np.sum(sample_x) == 0:
                    results[r - 1, i] = 0.0
                else:
                    results[r - 1, i] = (np.sum(sample_x) / sample_x.shape[0]) / (np.sum(sample_y) / sample_y.shape[0])
        return results

    @staticmethod
    #@njit("int64[:], int64[:],")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="int64[:], int64[:],", cache=True, fastmath=False)

    def cohens_h(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Jitted compute Cohen's h effect size for two samples of binary [0, 1] values. Cohen's h is a measure of effect size
        for comparing two independent samples based on the differences in proportions of the two samples.

        .. note:
           Modified from `DABEST <https://github.com/ACCLAB/DABEST-python/blob/fa7df50d20ab1c9cc687c66dd8bddf55d9a9dce3/dabest/_stats_tools/effsize.py#L216>`_
           `Cohen's h wiki <https://en.wikipedia.org/wiki/Cohen%27s_h>`_

        .. math::

           \\text{Cohen's h} = 2 \\arcsin\\left(\\sqrt{\\frac{\\sum\\text{sample\_1}}{N\_1}}\\right) - 2 \\arcsin\\left(\\sqrt{\\frac{\\sum\\text{sample\_2}}{N\_2}}\\right)

        Where :math:`N_1` and :math:`N_2` are the sample sizes of sample_1 and sample_2, respectively.

        :param np.ndarray sample_1: 1D array with binary [0, 1] values (e.g., first classifier inference values).
        :param np.ndarray sample_2: 1D array with binary [0, 1] values (e.g., second classifier inference values).
        :return: Cohen's h effect size.
        :rtype: float

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
    #@jit("(float32[:], float64[:], int64,)")
    @dynamic_numba_decorator(dtypes="(float32[:], float64[:], int64,)", cache=True, fastmath=False)
    def sliding_skew(data: np.ndarray, time_windows: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Compute the skewness of a 1D array within sliding time windows.

        :param np.ndarray data: 1D array of input data.
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
                results[j - 1][i] = (1 / sample.shape[0]) * np.sum(((data - mean) / std) ** 3)
        return results

    @staticmethod
    #@jit("(float32[:], float64[:], int64,)")
    @dynamic_numba_decorator(dtypes="(float32[:], float64[:], int64,)", cache=True, fastmath=False)

    def sliding_kurtosis(data: np.ndarray, time_windows: np.ndarray, sample_rate: int) -> np.ndarray:
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
    def kmeans_1d(data: np.ndarray, k: int, max_iters: int, calc_medians: bool) -> Tuple[np.ndarray, np.ndarray, Union[None, types.DictType]]:
        """
        Perform k-means clustering on a 1-dimensional dataset.

        .. note:
           - If calc_medians is True, the function returns cluster medians in addition to centroids and labels.


        .. seealso::
           Use with brighness intensity output to detect cue lights on/off states.
           Brighness intensity in images can be obtained using :func:`simba.mixins.image_mixin.ImageMixin.brightness_intensity`, or :func:`simba.data_processors.cuda.image.img_stack_brightness` for GPU acceleration.

        :param np.ndarray data: 1d array containing feature values.
        :param int k: Number of clusters.
        :param int max_iters: Maximum number of iterations for the k-means algorithm.
        :param bool calc_medians: Flag indicating whether to calculate cluster medians.
        :returns: Tuple of three elements. Final centroids of the clusters. Labels assigned to each data point based on clusters. Cluster medians (if calc_medians is True), otherwise None.
        :rtype: Tuple[np.ndarray, np.ndarray, Union[None, types.DictType]]

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
    # @njit(
    #     [
    #         (int8[:], int8[:], types.misc.Omitted(value=False), float32[:]),
    #         (
    #             int8[:],
    #             int8[:],
    #             types.misc.Omitted(value=False),
    #             types.misc.Omitted(None),
    #         ),
    #         (int8[:], int8[:], bool_, float32[:]),
    #         (int8[:], int8[:], bool_, types.misc.Omitted(None)),
    #     ]
    # )
    # @jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes=[
            (int8[:], int8[:], types.misc.Omitted(value=False), float32[:]),
            (
                int8[:],
                int8[:],
                types.misc.Omitted(value=False),
                types.misc.Omitted(None),
            ),
            (int8[:], int8[:], bool_, float32[:]),
            (int8[:], int8[:], bool_, types.misc.Omitted(None)),
        ], cache=True, fastmath=False)

    def hamming_distance(x: np.ndarray,
                         y: np.ndarray,
                         sort: Optional[bool] = False,
                         w: Optional[np.ndarray] = None) -> float:
        """
        Jitted compute of the Hamming similarity between two vectors.

        The Hamming similarity measures the similarity between two binary vectors by counting the number of positions at which the corresponding elements are different.

        .. note::
           If w is not provided, equal weights are assumed. Adapted from `pynndescent <https://pynndescent.readthedocs.io/en/latest/>`_.

        .. math::

           \\text{Hamming distance}(x, y) = \\frac{{\\sum_{i=1}^{n} w_i}}{{n}}

        where:
           - :math:`n` is the length of the vectors,
           - :math:`w_i` is the weight associated with the math:`i`th element of the vectors.

        .. seealso::
           For GPU method, see :func:`simba.data_processors.cuda.statistics.hamming_distance_gpu`.

        :param np.ndarray x: First binary vector.
        :param np.ndarray x: Second binary vector.
        :param Optional[np.ndarray] w: Optional weights for each element. Can be classification probabilities. If not provided, equal weights are assumed.
        :param Optional[bool] sort: If True, sorts x and y prior to hamming distance calculation. Default, False.
        :return: Hamming similarity
        :rtype: float

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
    # @njit(
    #     [(int8[:], int8[:], float32[:]), (int8[:], int8[:], types.misc.Omitted(None))]
    # )
    # @jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes=[(int8[:], int8[:], float32[:]), (int8[:], int8[:], types.misc.Omitted(None))], cache=True, fastmath=False)
    def yule_coef(
        x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None
    ) -> float64:
        """
        Jitted calculate of the yule coefficient between two binary vectors (e.g., to classified behaviors). 0 represent independence, 2 represents
        complete interdependence.

        .. math::
           Yule Coefficient = \\frac{{2 \cdot t_f \cdot f_t}}{{t_t \cdot f_f + t_f \cdot f_t}}

        .. note::
           Adapted from `pynndescent <https://pynndescent.readthedocs.io/en/latest/>`_.

        :param np.ndarray x: First binary vector.
        :param np.ndarray x: Second binary vector.
        :param Optional[np.ndarray] w: Optional weights for each element. Can be classification probabilities. If not provided, equal weights are assumed.
        :returns: yule coefficient
        :rtype: float

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
    # @njit(
    #     [(int8[:], int8[:], types.misc.Omitted(None)), (int8[:], int8[:], float32[:])]
    # )
    # @jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes=[(int8[:], int8[:], types.misc.Omitted(None)), (int8[:], int8[:], float32[:])], cache=True, fastmath=False)

    def sokal_sneath(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> float64:
        """
        Jitted calculate of the sokal sneath coefficient between two binary vectors (e.g., to classified behaviors). 0 represent independence, 1 represents complete interdependence.

        .. math::
           Sokal-Sneath = \\frac{{f_t + t_f}}{{2 \cdot (t_{{cnt}} + f_{{cnt}}) + f_t + t_f}}

        .. note::
           Adapted from `pynndescent <https://pynndescent.readthedocs.io/en/latest/>`_.

        .. seealso::
           For GPU method, see :func:`simba.data_processors.cuda.statistics.sokal_sneath_gpu`

        :param np.ndarray x: First binary vector.
        :param np.ndarray x: Second binary vector.
        :param Optional[np.ndarray] w: Optional weights for each element. Can be classification probabilities. If not provided, equal weights are assumed.
        :returns: sokal sneath coefficient
        :rtype: float

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
    # # @njit([(float32[:, :], float32[:, :]), (float32[:, :], types.misc.Omitted(None))])
    # @jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes=[(float32[:, :], float32[:, :]), (float32[:, :], types.misc.Omitted(None))], cache=True, fastmath=False)
    def bray_curtis_dissimilarity(x: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Jitted compute of the Bray-Curtis dissimilarity matrix between samples based on feature values.

        The Bray-Curtis dissimilarity measures the dissimilarity between two samples based on their feature values.
        It is useful for finding similar frames based on behavior.

        Useful for finding similar frames based on behavior.

        .. note::
           Adapted from `pynndescent <https://pynndescent.readthedocs.io/en/latest/>`_.

        :param np.ndarray x: 2d array with likely normalized feature values.
        :param Optional[np.ndarray] w: Optional 2d array with weights of same size as x. Default None and all observations will have the same weight.
        :returns: 2d array with same size as x representing dissimilarity values. 0 and the observations are identical and at 1 the observations are completly disimilar.
        :rtype: np.ndarray

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
    #@njit((float32[:], float32[:]))
    # @jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes=(float32[:], float32[:]), cache=True, fastmath=False)

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

    def hellinger_distance(self, x: np.ndarray, y: np.ndarray, bucket_method: Optional[Literal["fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"]] = "auto") -> float:

        r"""
        Compute the Hellinger distance between two vector distributions.

        .. note::
           The Hellinger distance is bounded and ranges from 0 to √2. Distance of √2 indicates that the two distributions are maximally dissimilar

        .. math::

           H(P, Q) = \frac{1}{\sqrt{2}} \sqrt{ \sum_{i=1}^{n} (\sqrt{P(i)} - \sqrt{Q(i)})^2 }

        where:
        - :math:`P(i)` is the probability of the :math:`i`-th event in distribution :math:`P`,
        - :math:`Q(i)` is the probability of the :math:`i`-th event in distribution :math:`Q`,
        - :math:`n` is the number of events.

        :param np.ndarray x: First 1D array representing a probability distribution.
        :param np.ndarray y: Second 1D array representing a probability distribution.
        :param Optional[Literal['fd', 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', 'sqrt']] bucket_method: Method for computing histogram bins. Default is 'auto'.
        :returns: Hellinger distance between the two input probability distributions.
        :rtype: float

        :example:
        >>> x = np.random.randint(0, 9000, (500000,))
        >>> y = np.random.randint(0, 9000, (500000,))
        >>> Statistics().hellinger_distance(x=x, y=y, bucket_method='auto')
        """

        check_valid_array(data=x, source=f'{Statistics.hellinger_distance.__name__} x', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, source=f'{Statistics.hellinger_distance.__name__} y', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_str(name=f"{Statistics.hellinger_distance.__name__} method", value=bucket_method, options=Options.BUCKET_METHODS.value)
        bin_width, bin_count = bucket_data(data=x, method=bucket_method)
        s1_h = self._hist_1d(data=x, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
        s2_h = self._hist_1d(data=y, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
        return self._hellinger_helper(x=s1_h.astype(np.float32), y=s2_h.astype(np.float32))

    @staticmethod
    def youden_j(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Calculate Youden's J statistic from two binary arrays.

        Youden's J statistic is a measure of the overall performance of a binary classification test, taking into account both sensitivity (true positive rate) and specificity (true negative rate).

        :param sample_1: The first binary array.
        :param sample_2: The second binary array.
        :return: Youden's J statistic.
        :rtype: float
        """

        check_valid_array(data=sample_1, source=f'{Statistics.youden_j.__name__} sample_1', accepted_ndims=(1,), accepted_values=[0, 1])
        check_valid_array(data=sample_2, source=f'{Statistics.youden_j.__name__} sample_2', accepted_ndims=(1,), accepted_shapes=[(sample_1.shape)], accepted_values=[0, 1])
        tp = np.sum((sample_1 == 1) & (sample_2 == 1))
        tn = np.sum((sample_1 == 0) & (sample_2 == 0))
        fp = np.sum((sample_1 == 0) & (sample_2 == 1))
        fn = np.sum((sample_1 == 1) & (sample_2 == 0))
        if tp + fn == 0 or tn + fp == 0:
            return np.nan
        else:
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            return sensitivity + specificity - 1

    @staticmethod
    def geometric_mean(x: np.ndarray) -> float:
        """
        Computes the geometric mean of a 1D NumPy array.

        :param x: A 1D NumPy array of numeric type containing non-negative values. Must have at least two elements.
        :return: The geometric mean of the values in `x`.
        :rtype: float
        """

        check_valid_array(data=x,
                          source=f'{Statistics.geometric_mean.__name__} x',
                          accepted_ndims=(1,),
                          accepted_dtypes=Formats.NUMERIC_DTYPES.value,
                          min_value=0,
                          min_axis_0=2)

        return np.prod(x) ** (1 / x.shape[0])

    @staticmethod
    def jaccard_distance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the Jaccard distance between two 1D NumPy arrays.

        The Jaccard distance is a measure of dissimilarity between two sets. It is defined as the size of the
        intersection of the sets divided by the size of the union of the sets.

        :param np.ndarray x: The first 1D NumPy array.
        :param np.ndarray y: The second 1D NumPy array.
        :return: The Jaccard distance between arrays x and y.
        :rtype: float

        :example:
        >>> x = np.random.randint(0, 5, (100))
        >>> y = np.random.randint(0, 7, (100))
        >>> Statistics.jaccard_distance(x=x, y=y)
        >>> 0.2857143
        """
        check_valid_array(data=x, source=f'{Statistics.jaccard_distance.__name__} x', accepted_ndims=(1,))
        check_valid_array(data=y, source=f'{Statistics.jaccard_distance.__name__} y', accepted_ndims=(1,), accepted_dtypes=[x.dtype.type])
        u_x, u_y = np.unique(x), np.unique(y)
        return np.float32(1 -(len(np.intersect1d(u_x, u_y)) / len(np.unique(np.hstack((u_x, u_y))))))

    @staticmethod
    def manhattan_distance_cdist(data: np.ndarray) -> np.ndarray:
        """
        Compute the pairwise Manhattan distance matrix between points in a 2D array.

        Can be preferred over Euclidean distance in scenarios where the movement is restricted
        to grid-based paths and/or the data is high dimensional.

        .. math::
           D_{\text{Manhattan}} = |x_2 - x_1| + |y_2 - y_1|

        :param data: 2D array where each row represents a featurized observation (e.g., frame)
        :return np.ndarray: Pairwise Manhattan distance matrix where element (i, j) represents the distance between points i and j.

        :example:
        >>> data = np.random.randint(0, 50, (10000, 2))
        >>> Statistics.manhattan_distance_cdist(data=data)
        """
        check_valid_array(data=data, source=f'{Statistics.manhattan_distance_cdist} data', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        differences = np.abs(data[:, np.newaxis, :] - data)
        results = np.sum(differences, axis=-1)
        return results

    @staticmethod
    def get_clustering_purity(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute clustering quality using purity score.

        An external evaluation metric for clustering quality. It measures the extent to which clusters contain a single class.
        The score ranges from 0 to 1, where 1 indicates perfect purity.

        .. note::
           Adapted from Ugurite's Stack Overflow answer: https://stackoverflow.com/a/51672699


        :param np.ndarray x: Predicted cluster labels (1D array of integers).
        :param np.ndarray y: Ground truth class labels (1D array of integers, same length as `x`).
        :returns: Purity score in the range [0, 1].
        :rtype: float

        :example:
        >>> x = np.random.randint(0, 5, (100000,))
        >>> y = np.random.randint(0, 4, (100000,))
        >>> p = Statistics.get_clustering_purity(x=x, y=y)

        :references:
           .. [1] Evaluation of clustering. *Introduction to Information Retrieval*. Available at: https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
        """

        check_valid_array(data=x, source=f'{Statistics.get_clustering_purity.__name__} x', accepted_ndims=(1,), accepted_dtypes=Formats.INTEGER_DTYPES.value)
        check_valid_array(data=y, source=f'{Statistics.get_clustering_purity.__name__} y', accepted_ndims=(1,), accepted_axis_0_shape=[x.shape[0]], accepted_dtypes=Formats.INTEGER_DTYPES.value)
        c = get_confusion_matrix(x=x, y=y)
        return np.sum(np.amax(c, axis=0)) / np.sum(c)

    @staticmethod
    #@jit('(float32[:,:],)')
    @dynamic_numba_decorator(dtypes='(float32[:,:],)', cache=True, fastmath=False)

    def mahalanobis_distance_cdist(data: np.ndarray) -> np.ndarray:
        """
        Compute the Mahalanobis distance between every pair of observations in a 2D array using numba.

        The Mahalanobis distance is a measure of the distance between a point and a distribution. It accounts for correlations between variables and the scales of the variables, making it suitable for datasets where features are not independent and have different variances.

        .. note::
           Significantly reduced runtime versus Mahalanobis scipy.cdist only with larger feature sets ( > 10-50).

        However, Mahalanobis distance may not be suitable in certain scenarios, such as:
        - When the dataset is small and the covariance matrix is not accurately estimated.
        - When the dataset contains outliers that significantly affect the estimation of the covariance matrix.
        - When the assumptions of multivariate normality are violated.

        :param np.ndarray data: 2D array with feature observations. Frames on axis 0 and feature values on axis 1
        :return: Pairwise Mahalanobis distance matrix where element (i, j) represents the Mahalanobis distance between  observations i and j.
        :rtype: np.ndarray

        :example:
        >>> data = np.random.randint(0, 50, (1000, 200)).astype(np.float32)
        >>> x = Statistics.mahalanobis_distance_cdist(data=data)
        """

        covariance_matrix = np.cov(data, rowvar=False)
        inv_covariance_matrix = np.linalg.inv(covariance_matrix).astype(np.float32)
        n = data.shape[0]
        distances = np.zeros((n, n))
        for i in prange(n):
            for j in range(n):
                diff = data[i] - data[j]
                diff = diff.astype(np.float32)
                distances[i, j] = np.sqrt(np.dot(np.dot(diff, inv_covariance_matrix), diff.T))
        return distances

    @staticmethod
    # @njit("(int64[:], int64[:])")
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(int64[:], int64[:])", cache=True, fastmath=False)

    def cohens_kappa(sample_1: np.ndarray, sample_2: np.ndarray):
        """
        Jitted compute Cohen's Kappa coefficient for two binary samples.

        Cohen's Kappa coefficient measures the agreement between two sets of binary ratings, taking into account agreement occurring by chance.
        It ranges from -1 to 1, where 1 indicates perfect agreement, 0 indicates agreement by chance, and -1 indicates complete disagreement.

        .. math::

           \\kappa = 1 - \\frac{\sum{w_{ij} \\cdot D_{ij}}}{\\sum{w_{ij} \\cdot E_{ij}}}

        where:
           - :math:`\kappa` is Cohen's Kappa coefficient,
           - :math:`w_{ij}` are the weights,
           - :math:`D_{ij}` are the observed frequencies,
           - :math:`E_{ij}` are the expected frequencies.

        :param np.ndarray sample_1: The first binary sample, a 1D NumPy array of integers.
        :param np.ndarray sample_2: The second binary sample, a 1D NumPy array of integers.
        :return: Cohen's Kappa coefficient between the two samples.
        :rtype: float

        :example:
        >>> sample_1 = np.random.randint(0, 2, size=(10000,))
        >>> sample_2 = np.random.randint(0, 2, size=(10000,))
        >>> Statistics.cohens_kappa(sample_1=sample_1, sample_2=sample_2))
        """

        sample_1 = np.ascontiguousarray(sample_1)
        sample_2 = np.ascontiguousarray(sample_2)
        data = np.hstack((sample_1.reshape(-1, 1), sample_2.reshape(-1, 1)))
        tp = len(np.argwhere((data[:, 0] == 1) & (data[:, 1] == 1)).flatten())
        tn = len(np.argwhere((data[:, 0] == 0) & (data[:, 1] == 0)).flatten())
        fp = len(np.argwhere((data[:, 0] == 1) & (data[:, 1] == 0)).flatten())
        fn = len(np.argwhere((data[:, 0] == 0) & (data[:, 1] == 1)).flatten())
        data = np.array(([tp, fp], [fn, tn]))
        sum0 = data.sum(axis=0)
        sum1 = data.sum(axis=1)
        expected = np.outer(sum0, sum1) / np.sum(sum0)
        w_mat = np.full(shape=(2, 2), fill_value=1)
        w_mat[0, 0] = 0
        w_mat[1, 1] = 0
        return 1 - np.sum(w_mat * data) / np.sum(w_mat * expected)

    @staticmethod
    def d_prime(
        x: np.ndarray,
        y: np.ndarray,
        lower_limit: Optional[float] = 0.0001,
        upper_limit: Optional[float] = 0.9999,
    ) -> float:
        """
        Computes d-prime from two Boolean 1d arrays, e.g., between classifications and ground truth.

        D-prime (d') is a measure of signal detection performance, indicating the ability to discriminate between signal and noise.
        It is computed as the difference between the inverse cumulative distribution function (CDF) of the hit rate and the false alarm rate.

        .. math::

           d' = \\Phi^{-1}(hit\\_rate) - \\Phi^{-1}(false\\_alarm\\_rate)

        where:
        - :math:`\\Phi^{-1}` is the inverse of the cumulative distribution function (CDF) of the normal distribution,
        - :math:`hit\\_rate` is the proportion of true positives correctly identified,
        - :math:`false\\_alarm\\_rate` is the proportion of false positives incorrectly identified.

        :param np.ndarray x: Boolean 1D array of response values, where 1 represents presence, and 0 representing absence.
        :param np.ndarray y: Boolean 1D array of ground truth, where 1 represents presence, and 0 representing absence.
        :param Optional[float] lower_limit: Lower limit to bound hit and false alarm rates. Defaults to 0.0001.
        :param Optional[float] upper_limit: Upper limit to bound hit and false alarm rates. Defaults to 0.9999.
        :return: The calculated d' (d-prime) value.
        :rtype: float

        :example:
        >>> x = np.random.randint(0, 2, (1000,))
        >>> y = np.random.randint(0, 2, (1000,))
        >>> Statistics.d_prime(x=x, y=y)
        """

        check_valid_array(
            data=x,
            source=Statistics.d_prime.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.int64, np.int32, np.int8),
        )
        check_valid_array(
            data=y,
            source=Statistics.d_prime.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.int64, np.int32, np.int8),
        )
        if len(list({x.shape[0], y.shape[0]})) != 1:
            raise CountError(
                msg=f"The two arrays has to be equal lengths but got: {x.shape[0], y.shape[0]}",
                source=Statistics.d_prime.__name__,
            )
        for i in [x, y]:
            additional = list(set(list(np.sort(np.unique(i)))) - {0, 1})
            if len(additional) > 0:
                raise InvalidInputError(
                    msg=f"D-prime requires binary input data but found {additional}",
                    source=Statistics.d_prime().__name__,
                )
        target_idx = np.argwhere(y == 1).flatten()
        hit_rate = np.sum(x[np.argwhere(y == 1)]) / target_idx.shape[0]
        false_alarm_rate = np.sum(x[np.argwhere(y == 0)]) / target_idx.shape[0]
        if hit_rate < lower_limit:
            hit_rate = lower_limit
        elif hit_rate > upper_limit:
            hit_rate = upper_limit
        if false_alarm_rate < lower_limit:
            false_alarm_rate = lower_limit
        elif false_alarm_rate > upper_limit:
            false_alarm_rate = upper_limit
        return stats.norm.ppf(hit_rate) - stats.norm.ppf(false_alarm_rate)

    @staticmethod
    def mcnemar(
        x: np.ndarray,
        y: np.ndarray,
        ground_truth: np.ndarray,
        continuity_corrected: Optional[bool] = True,
    ) -> Tuple[float, float]:
        """
        Perform McNemar's test to compare the predictive accuracy of two models. This test is used
        to evaluate if the accuracies of two classifiers are significantly different when tested on the same data.

        The chi-squared statistic (with continuity correction if `continuity_corrected=True`) is calculated as:

        .. math::
          X^2 = \\frac{(|b - c| - 1)^2}{b + c} \\,\\text{ if corrected, or }\\, X^2 = \\frac{(b - c)^2}{b + c}

        where:
           - `b` is the number of instances misclassified by the first model but correctly classified by the second model.
           - `c` is the number of instances correctly classified by the first model but misclassified by the second model.
        .. note::
           Adapted from `mlextend <https://github.com/rasbt/mlxtend/blob/master/mlxtend/evaluate/mcnemar.py>`__.

        :param np.ndarray x: 1-dimensional Boolean array with predictions of the first model.
        :param np.ndarray y: 1-dimensional Boolean array with predictions of the second model.
        :param np.ndarray ground_truth: 1-dimensional Boolean array with ground truth labels.
        :param Optional[bool] continuity_corrected: Whether to apply continuity correction. Default is True.
        :returns: McNemar score are significance level.
        :rtype: Tuple[float, float]

        :example:
        >>> x = np.random.randint(0, 2, (100000, ))
        >>> y = np.random.randint(0, 2, (100000, ))
        >>> ground_truth = np.random.randint(0, 2, (100000, ))
        >>> Statistics.mcnemar(x=x, y=y, ground_truth=ground_truth)
        """

        check_valid_array(
            data=x,
            source=Statistics.mcnemar.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.int64, np.int32, np.int8),
        )
        check_valid_array(
            data=y,
            source=Statistics.mcnemar.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.int64, np.int32, np.int8),
        )
        check_valid_array(
            data=ground_truth,
            source=Statistics.mcnemar.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.int64, np.int32, np.int8),
        )
        if len(list({x.shape[0], y.shape[0], ground_truth.shape[0]})) != 1:
            raise CountError(
                msg=f"The three arrays has to be equal lengths but got: {x.shape[0], y.shape[0], ground_truth.shape[0]}",
                source=Statistics.mcnemar.__name__,
            )
        for i in [x, y, ground_truth]:
            additional = list(set(list(np.sort(np.unique(i)))) - {0, 1})
            if len(additional) > 0:
                raise InvalidInputError(
                    msg=f"Mcnemar requires binary input data but found {additional}",
                    source=Statistics.mcnemar.__name__,
                )
        data = np.hstack(
            (x.reshape(-1, 1), y.reshape(-1, 1), ground_truth.reshape(-1, 1))
        )
        b = (
            np.where((data == (0, 1, 0)).all(axis=1))[0].shape[0]
            + np.where((data == (1, 0, 1)).all(axis=1))[0].shape[0]
        )
        c = (
            np.where((data == (1, 0, 0)).all(axis=1))[0].shape[0]
            + np.where((data == (0, 1, 1)).all(axis=1))[0].shape[0]
        )
        if not continuity_corrected:
            x = (np.square(b - c)) / (b + c)
        else:
            x = (np.square(np.abs(b - c) - 1)) / (b + c)
        p = chi2.sf(x, 1)
        return x, p

    @staticmethod
    def cochrans_q(data: np.ndarray) -> Tuple[float, float]:

        r"""

        Compute Cochrans Q for 2-dimensional boolean array.

        Cochran's Q statistic is used to test for significant differences between more than two proportions.
        It can be used to evaluate if the performance of multiple (>=2) classifiers on the same data is the same or significantly different.

        .. note::
           If two classifiers, consider :func:`simba.mixins.statistics.Statistics.mcnemar()`.

           Useful background: https://psych.unl.edu/psycrs/handcomp/hccochran.PDF

        :math:`Q = \frac{(k - 1) \left( kG^2 - \left( \sum_{j=1}^{k} C_j \right)^2 \right)}{kR - S}`

        where:

        - :math:`k` is the number of classifiers,
        - :math:`G = \sum_{j=1}^{k} C_j^2` (the sum of the squares of the column sums),
        - :math:`C_j` is the sum of the :math:`j`-th column (number of successes for the :math:`j`-th classifier),
        - :math:`R = \sum_{i=1}^{n} R_i` (the total number of successes across all classifiers),
        - :math:`S = \sum_{i=1}^{n} R_i^2` (the sum of the squares of the row sums),
        - :math:`R_i` is the sum of the :math:`i`-th row (number of successes for the :math:`i`-th observation).

        :param np.ndarray data: Two-dimensional array of boolean values where axis 1 represents classifiers or features and rows represent frames.
        :return: Cochran's Q statistic signidicance value.
        :rtype: Tuple[float, float]

        :example:
        >>> data = np.random.randint(0, 2, (100000, 4))
        >>> Statistics.cochrans_q(data=data)
        """
        check_valid_array(
            data=data, source=Statistics.cochrans_q.__name__, accepted_ndims=(2,)
        )
        additional = list(set(list(np.sort(np.unique(data)))) - {0, 1})
        if len(additional) > 0:
            raise InvalidInputError(
                msg=f"Cochrans Q requires binary input data but found {additional}",
                source=Statistics.cochrans_q.__name__,
            )
        col_sums = np.sum(data, axis=0)
        row_sum_sum = np.sum(np.sum(data, axis=1))
        row_sum_square_sum = np.sum(np.square(np.sum(data, axis=1)))
        k = data.shape[1]
        g2 = np.sum(sum(np.square(col_sums)))
        nominator = (k - 1) * ((k * g2) - np.square(np.sum(col_sums)))
        denominator = (k * row_sum_sum) - row_sum_square_sum
        if nominator == 0 or denominator == 0:
            return -1.0, -1.0
        else:
            q = (nominator / denominator,)
            return q, stats.chi2.sf(q, k - 1)

    @staticmethod
    def hartley_fmax(x: np.ndarray, y: np.ndarray) -> float:

        r"""
        Compute Hartley's Fmax statistic to test for equality of variances between two features or groups.

        Hartley's Fmax statistic is used to test whether two samples have equal variances.
        It is calculated as the ratio of the largest sample variance to the smallest sample variance.
        Values close to one represent closer to equal variance.

        .. math::
            \text{Hartley's } F_{max} = \frac{\max(\text{Var}(x), \text{Var}(y))}{\min(\text{Var}(x), \text{Var}(y))}

        where:
        - :math:`\text{Var}(x)` is the variance of sample :math:`x`,
        - :math:`\text{Var}(y)` is the variance of sample :math:`y`.

        :param np.ndarray x: 1D array representing numeric data of the first group/feature.
        :param np.ndarray x: 1D array representing numeric data of the second group/feature.
        :return: Hartley's Fmax statistic.
        :rtype: float

        :example:
        >>> x = np.random.random((100,))
        >>> y = np.random.random((100,))
        >>> Statistics.hartley_fmax(x=x, y=y)
        """
        check_valid_array(
            data=x,
            source=Statistics.hartley_fmax.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.float32, np.float64, np.int64, np.float32),
        )
        check_valid_array(
            data=y,
            source=Statistics.hartley_fmax.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.float32, np.float64, np.int64, np.float32),
        )
        max_var = np.max((np.var(x), np.var(y)))
        min_var = np.min((np.var(x), np.var(y)))
        if (max_var == 0) or (min_var == 0):
            return -1.0
        return max_var / min_var

    @staticmethod
    def grubbs_test(x: np.ndarray, left_tail: Optional[bool] = False) -> float:

        r"""
        Perform Grubbs' test to detect outliers if the minimum or maximum value in a feature series is an outlier.

        Grubbs' test is a statistical test used to detect outliers in a univariate data set.
        It calculates the Grubbs' test statistic as the absolute difference between the
        extreme value (either the minimum or maximum) and the sample mean, divided by the sample standard deviation.

        .. math::
           \text{Grubbs' Test Statistic} = \frac{|\bar{x} - x_{\text{min/max}}|}{s}

        where:
           - :math:`\bar{x}` is the sample mean,
           - :math:`x_{\text{min/max}}` is the minimum or maximum value of the sample (depending on the tail being tested),
           - :math:`s` is the sample standard deviation.

        :param np.ndarray x: 1D array representing numeric data.
        :param Optional[bool] left_tail: If True, the test calculates the Grubbs' test statistic for the left tail (minimum value). If False (default), it calculates the statistic for the right tail (maximum value).
        :return: The computed Grubbs' test statistic.
        :rtype: float

        :example:
        >>> x = np.random.random((100,))
        >>> Statistics.grubbs_test(x=x)
        """
        check_valid_array(
            data=x,
            source=Statistics.grubbs_test.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.float32, np.float64, np.int64, np.float32),
        )
        x = np.sort(x)
        if left_tail:
            return (np.mean(x) - np.min(x)) / np.std(x)
        else:
            return (np.max(x) - np.mean(x)) / np.std(x)

    @staticmethod
    def wilcoxon(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """
        Perform the Wilcoxon signed-rank test for paired samples.

        Wilcoxon signed-rank test is a non-parametric statistical hypothesis test used
        to compare two related samples, matched samples, or repeated measurements on a single sample
        to assess whether their population mean ranks differ.

        :param np.ndarray x: 1D array representing the observations for the first sample.
        :param np.ndarray y: 1D array representing the observations for the second sample.
        :return: A tuple containing the test statistic (z-score) and the effect size (r).
        - The test statistic (z-score) measures the deviation of the observed ranks sum from the expected sum.
        - The effect size (r) measures the strength of association between the variables.
        :rtype: Tuple[float, float]
        """

        data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
        n = data.shape[0]
        diff = np.diff(data).flatten()
        diff_abs = np.abs(diff)
        rank_w_ties = fast_mean_rank(data=diff_abs, descending=False)
        signed_rank_w_ties = np.full((rank_w_ties.shape[0]), np.nan)
        t_plus, t_minus = 0, 0

        for i in range(diff.shape[0]):
            if diff[i] < 0:
                signed_rank_w_ties[i] = -rank_w_ties[i]
                t_minus += np.abs(rank_w_ties[i])
            else:
                signed_rank_w_ties[i] = rank_w_ties[i]
                t_plus += np.abs(rank_w_ties[i])
        u_w = (n * (n + 1)) / 4
        std_correction = 0
        for i in range(signed_rank_w_ties.shape[0]):
            same_rank_n = (
                np.argwhere(signed_rank_w_ties == signed_rank_w_ties[i])
                .flatten()
                .shape[0]
            )
            if same_rank_n > 1:
                std_correction += ((same_rank_n**3) - same_rank_n) / 2

        std = np.sqrt(((n * (n + 1)) * ((2 * n) + 1) - std_correction) / 24)
        W = np.min((t_plus, t_minus))
        z = (W - u_w) / std
        r = z / np.sqrt(n)
        return z, r

    @staticmethod
    # @njit(
    #     "(float32[:,:], )",
    # )
    # @jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:,:], )", cache=True, fastmath=False)
    def cov_matrix(data: np.ndarray):
        """
        Jitted helper to compute the covariance matrix of the input data. Helper for computing cronbach alpha,
        multivariate analysis, and distance computations.

        :param np.ndarray data: 2-dimensional numpy array representing the input data with shape (n, m), where n is the number of observations and m is the number of features.
        :return: Covariance matrix of the input data with shape (m, m). The (i, j)-th element of the matrix represents the covariance between the i-th and j-th features in the data.

        :example:
        >>> data = np.random.randint(0,2, (200, 40)).astype(np.float32)
        >>> covariance_matrix = Statistics.cov_matrix(data=data)
        """
        n, m = data.shape
        cov = np.full((m, m), 0.0)
        for i in prange(m):
            mean_i = np.sum(data[:, i]) / n
            for j in range(m):
                mean_j = np.sum(data[:, j]) / n
                cov[i, j] = np.sum((data[:, i] - mean_i) * (data[:, j] - mean_j)) / (
                    n - 1
                )
        return cov

    @staticmethod
    # @njit("(float32[:], int64,)")
    # @jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], int64,)", cache=True, fastmath=False)
    def mad_median_rule(data: np.ndarray, k: int) -> np.ndarray:
        """
        Detects outliers in the given data using the Median Absolute Deviation (MAD) rule.
        Returns a 1D array of size `data.shape[0]`, where `1` represents an outlier and `0`
        represents an inlier.

        .. seealso::
           :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_descriptive_statistics`
           :func:`simba.mixins.statistics_mixin.Statistics.sliding_mad_median_rule`

        :param np.ndarray data: A 1-dimensional array of numerical values to check for outliers.
        :param int k: The multiplier for the MAD threshold. Higher values make the rule less sensitive to deviations from the median.
        :returns: A 1D binary array of the same length as `data`, where each element is `1` if the corresponding element in `data` is classified as an outlier, and `0` otherwise.
        :rtype: np.ndarray

        :example:
        >>> data = np.random.randint(0, 600, (9000000,)).astype(np.float32)
        >>> Statistics.mad_median_rule(data=data, k=1)
        """

        median = np.median(data)
        mad = np.median(np.abs(data - median))
        threshold = k * mad
        outliers = np.abs(data - median) > threshold
        return outliers * 1

    @staticmethod
    # @njit("(float32[:], int64, float64[:], float64)")
    # @jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], int64, float64[:], float64)", cache=True, fastmath=False)
    def sliding_mad_median_rule(data: np.ndarray, k: int, time_windows: np.ndarray, fps: float) -> np.ndarray:

        """
        Count the number of outliers in a sliding time-window using the MAD-Median Rule.

        The MAD-Median Rule is a robust method for outlier detection. It calculates the median absolute deviation (MAD)
        and uses it to identify outliers based on a threshold defined as k times the MAD.

        .. seealso::
           For alternative method, see :func:`simba.mixins.timeseries_features_mixin.TimeseriesFeatureMixin.sliding_descriptive_statistics`
           For single dataset, use :func:`simba.mixins.statistics_mixin.Statistics.mad_median_rule`

        :param np.ndarray data: 1D numerical array representing feature.
        :param int k: The outlier threshold defined as k * median absolute deviation in each time window.
        :param np.ndarray time_windows: 1D array of time window sizes in seconds.
        :param float fps: The frequency of the signal.
        :return: Array of size (data.shape[0], time_windows.shape[0]) with counts if outliers detected.
        :rtype: np.ndarray

        :example:
        >>> data = np.random.randint(0, 50, (50000,)).astype(np.float32)
        >>> Statistics.sliding_mad_median_rule(data=data, k=2, time_windows=np.array([20.0]), fps=1.0)
        """
        results = np.full((data.shape[0], time_windows.shape[0]), -1)
        for cnt, time_window in enumerate(time_windows):
            w = int(fps * time_window)
            for i in range(w, data.shape[0] + 1, 1):
                w_data = data[i - w : i]
                median = np.median(w_data)
                mad = np.median(np.abs(w_data - median))
                threshold = k * mad
                outliers = np.abs(w_data - median) > threshold
                results[i - 1][cnt] = np.sum(outliers * 1)
        return results

    @staticmethod
    def dunn_index(x: np.ndarray, y: np.ndarray, sample: Optional[float] = None) -> float:

        r"""
        Calculate the Dunn index to evaluate the quality of clustered labels.

        This function calculates the Dunn Index, which is a measure of clustering quality.
        The index considers the ratio of the minimum inter-cluster distance to the maximum
        intra-cluster distance. The Dunn Index range from zero to infinity and larger values indicate better clustering.
        The Dunn Index uses Euclidean distances.

        The Dunn Index is calculated using the following steps:

        1. **Inter-cluster distance**: Compute the distances between each pair of clusters and find the minimum distance.
        2. **Intra-cluster distance**: Determine the distances within each cluster and find the maximum distance.

        The Dunn Index is given by:

        .. math::
           D = \frac{\min_{i \neq j} \{ \delta(C_i, C_j) \}}{\max_k \{ \Delta(C_k) \}}

        where :math:`\delta(C_i, C_j)` is the distance between clusters :math:`C_i` and :math:`C_j`, and
        :math:`\Delta(C_k)` is the diameter of cluster :math:`C_k`.

        .. note::
           Modified from `jqmviegas <https://github.com/jqmviegas/jqm_cvi/>`_

           Wiki `https://en.wikipedia.org/wiki/Dunn_index <https://en.wikipedia.org/wiki/Dunn_index>`_

           If Dunn Index can not be calculated, `-1` is returned.

        .. note::
           For GPU accelerated method, use :func:`simba.data_processors.cuda.statistics.dunn_index`.

        :param np.ndarray x: 2D array representing the data points. Shape (n_samples, n_features).
        :param np.ndarray y: 1D array representing cluster labels for each data point. Shape (n_samples,).
        :return: The Dunn index value
        :rtype: float

        :example:
        >>> x = np.random.randint(0, 100, (100, 2))
        >>> y = np.random.randint(0, 3, (100,))
        >>> Statistics.dunn_index(x=x, y=y)
        """

        check_valid_array(data=x, source=Statistics.dunn_index.__name__, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, source=Statistics.dunn_index.__name__, accepted_ndims=(1,), accepted_shapes=[(x.shape[0],)], accepted_dtypes=Formats.NUMERIC_DTYPES.value,)
        if sample is not None:
            check_float(name=Statistics.dunn_index.__name__, value=sample, min_value=10e-6, max_value=1.0)
            sample_idx = np.random.choice(np.arange(0, x.shape[0]+1), 10)
            x, y = x[sample_idx, :], y[sample_idx]
        y = np.nan_to_num(y, nan=-1, posinf=-1, neginf=-1)
        x = np.nan_to_num(x, nan=-1, posinf=-1, neginf=-1)
        distances = FeatureExtractionMixin.cdist(array_1=x.astype(np.float32), array_2=x.astype(np.float32))
        ks = np.sort(np.unique(y)).astype(np.int64)
        deltas = np.full((ks.shape[0], ks.shape[0]), np.inf)
        big_deltas = np.zeros([ks.shape[0], 1])
        for i, j in list(permutations(np.arange(0, ks.shape[0]), 2)):
            k, l = ks[i], ks[j]
            values = distances[np.where((y == k))][:, np.where((y == l))]
            deltas[i, j] = np.min(values[np.nonzero(values)])
        for m in np.arange(0, ks.shape[0]):
            values = distances[np.where((y == ks[m]))][:, np.where((y == ks[m]))]
            big_deltas[m] = np.max(values)
        v = np.min(deltas) / np.max(big_deltas)
        if v == np.inf: return -1
        else: return v

    def davis_bouldin(self, x: np.ndarray, y: np.ndarray) -> float:

        r"""
        Calculate the Davis-Bouldin index for evaluating clustering performance.

        Davis-Bouldin index measures the clustering quality based on the within-cluster
        similarity and between-cluster dissimilarity. Lower values indicate better clustering.

        .. seealso::
           For GPU acceleration, use :func:`simba.data_processors.cuda.statistics.davis_bouldin`

        .. note::
           Modified from `scikit-learn <https://github.com/scikit-learn/scikit-learn/blob/f07e0138bfee41cd2c0a5d0251dc3fe03e6e1084/sklearn/metrics/cluster/_unsupervised.py#L390>`_

        .. math::
           DB = \\frac{1}{N} \sum_{i=1}^{N} \max_{j \neq i} \left( \frac{\sigma_i + \sigma_j}{d_{ij}} \right)

        where:
        - :math:`N` is the number of clusters,
        - :math:`\sigma_i` is the average distance between each point in cluster :math:`i` and the centroid of cluster :math:`i`,
        - :math:`d_{ij}` is the distance between the centroids of clusters :math:`i` and :math:`j`.

        :param np.ndarray x: 2D array representing the data points. Shape (n_samples, n_features/n_dimension).
        :param np.ndarray y: 2D array representing cluster labels for each data point. Shape (n_samples,).
        :return: Davis-Bouldin score.
        :rtype: float

        :example:
        >>> x = np.random.randint(0, 100, (100, 2))
        >>> y = np.random.randint(0, 3, (100,))
        >>> Statistics.davis_bouldin(x=x, y=y)
        """

        check_valid_array(
            data=x,
            source=Statistics.davis_bouldin.__name__,
            accepted_ndims=[(2,)],
            accepted_dtypes=Formats.NUMERIC_DTYPES.value,
        )
        check_valid_array(
            data=y,
            source=Statistics.davis_bouldin.__name__,
            accepted_ndims=[(1,)],
            accepted_shapes=[(x.shape[0],)],
            accepted_dtypes=Formats.NUMERIC_DTYPES.value,
        )
        n_labels = np.unique(y).shape[0]
        labels = np.unique(y)
        intra_dists = np.full((n_labels), 0.0)
        centroids = np.full((n_labels, x.shape[1]), 0.0)
        for k in range(n_labels):
            cluster_k = x[np.argwhere(y == labels[k])].reshape(-1, 2)
            cluster_mean = np.full((x.shape[1]), np.nan)
            for i in range(cluster_mean.shape[0]):
                cluster_mean[i] = np.mean(cluster_k[:, i].flatten())
            centroids[k] = cluster_mean
            intra_dists[k] = np.average(
                FeatureExtractionMixin.framewise_euclidean_distance(
                    location_1=cluster_k.astype(np.float64),
                    location_2=np.full(cluster_k.shape, cluster_mean).astype(np.float64),
                    px_per_mm=np.float64(1), centimeter=False
                )
            )
        centroid_distances = FeatureExtractionMixin.cdist(
            array_1=centroids.astype(np.float32), array_2=centroids.astype(np.float32)
        )
        if np.allclose(intra_dists, 0) or np.allclose(centroid_distances, 0):
            return 0.0
        centroid_distances[centroid_distances == 0] = np.inf
        combined_intra_dists = intra_dists[:, None] + intra_dists
        return np.mean(np.max(combined_intra_dists / centroid_distances, axis=1))

    @staticmethod
    # @njit("(float32[:,:], int64[:])", cache=True)
    # @jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:,:], int64[:])", cache=True, fastmath=False)

    def calinski_harabasz(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Calinski-Harabasz score to evaluate clustering quality.

        The Calinski-Harabasz score is a measure of cluster separation and compactness.
        It is calculated as the ratio of the between-cluster dispersion to the
        within-cluster dispersion. A higher score indicates better clustering.

        .. note::
           Modified from `scikit-learn <https://github.com/scikit-learn/scikit-learn/blob/8721245511de2f225ff5f9aa5f5fadce663cd4a3/sklearn/metrics/cluster/_unsupervised.py#L326>`_

        The Calinski-Harabasz score (CH) is calculated as:

        .. math::

            CH = \\frac{B}{W} \\times \\frac{N - k}{k - 1}

        where:
        - :math:`B` is the sum of squared distances between cluster centroids,
        - :math:`W` is the sum of squared distances from each point to its assigned cluster centroid,
        - :math:`N` is the total number of data points,
        - :math:`k` is the number of clusters.

        :param x: 2D array representing the data points. Shape (n_samples, n_features/n_dimension).
        :param y: 2D array representing cluster labels for each data point. Shape (n_samples,).
        :return: Calinski-Harabasz score.
        :float: float

        :example:
        :example:
        >>> x = np.random.random((100, 2)).astype(np.float32)
        >>> y = np.random.randint(0, 100, (100,)).astype(np.int64)
        >>> Statistics.calinski_harabasz(x=x, y=y)
        """

        n_labels = np.unique(y).shape[0]
        labels = np.unique(y)
        extra_dispersion, intra_dispersion = 0.0, 0.0
        global_mean = np.full((x.shape[1]), np.nan)
        for i in range(x.shape[1]):
            global_mean[i] = np.mean(x[:, i].flatten())
        for k in range(n_labels):
            cluster_k = x[np.argwhere(y == labels[k]).flatten(), :]
            mean_k = np.full((cluster_k.shape[1]), np.nan)
            for i in range(cluster_k.shape[1]):
                mean_k[i] = np.mean(cluster_k[:, i].flatten())
            extra_dispersion += len(cluster_k) * np.sum((mean_k - global_mean) ** 2)
            intra_dispersion += np.sum((cluster_k - mean_k) ** 2)

        denominator = intra_dispersion * (n_labels - 1.0)
        if denominator == 0.0:
            return 0.0
        else:
            return extra_dispersion * (x.shape[0] - n_labels) / denominator

    def silhouette_score(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the silhouette score for the given dataset and labels.


        .. seealso::
           For GPU implementation, see :func:`simba.data_processors.cuda.statistics.silhouette_score_gpu`


        :param np.ndarray x: The dataset as a 2D NumPy array of shape (n_samples, n_features).
        :param np.ndarray y: Cluster labels for each data point as a 1D NumPy array of shape (n_samples,).
        :returns: The average silhouette score for the dataset.
        :rtype: float

        :example:
        >>> x, y = make_blobs(n_samples=10000, n_features=400, centers=5, cluster_std=10, center_box=(-1, 1))
        >>> score = silhouette_score(x=x, y=y)

        >>> from sklearn.metrics import silhouette_score as sklearn_silhouette # SKLEARN ALTERNATIVE
        >>> score_sklearn = sklearn_silhouette(x, y)

        """
        dists = cdist(x, x)
        results = np.full(x.shape[0], fill_value=-1.0, dtype=np.float32)
        cluster_ids = np.unique(y)
        cluster_indices = {cluster_id: np.argwhere(y == cluster_id).flatten() for cluster_id in cluster_ids}

        for i in range(x.shape[0]):
            intra_idx = cluster_indices[y[i]]
            if len(intra_idx) <= 1:
                a_i = 0.0
            else:
                intra_distances = dists[i, intra_idx]
                a_i = np.sum(intra_distances) / (intra_distances.shape[0] - 1)
            b_i = np.inf
            for cluster_id in cluster_ids:
                if cluster_id != y[i]:
                    inter_idx = cluster_indices[cluster_id]
                    inter_distances = dists[i, inter_idx]
                    b_i = min(b_i, np.mean(inter_distances))
            results[i] = (b_i - a_i) / max(a_i, b_i)

        return np.mean(results)

    @staticmethod
    def xie_beni(x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the Xie-Beni index for clustering evaluation.

        The score is calculated as the ratio between the average intra-cluster variance and the squared minimum distance between cluster centroids. This ensures that the index penalizes both loosely packed clusters and clusters that are too close to each other.

        A lower Xie-Beni index indicates better clustering quality, signifying well-separated and compact clusters.

        .. seealso::
           To compute Xie-Beni on the GPU, use :func:`~simba.mixins.statistics_mixin.Statistics.xie_beni`.
           Significant GPU savings detected at about 1m features, 25 clusters

        .. math::
           \\text{XB} = \\frac{\\frac{1}{n} \\sum_{i=1}^{n} \\| x_i - c_{y_i} \\|^2}{\\min_{i \\neq j} \\| c_i - c_j \\|^2}

        where :math:`n` is the total number of points in the dataset, :math:`x_i` is the :math:`i`-th data point, :math:`c_{y_i}` is the centroid of the cluster to which :math:`x_i` belongs, and :math:`\\| \\cdot \\|` denotes the Euclidean norm.

        :param np.ndarray x: The dataset as a 2D NumPy array of shape (n_samples, n_features).
        :param np.ndarray y: Cluster labels for each data point as a 1D NumPy array of shape (n_samples,).
        :returns: The Xie-Beni score for the dataset.
        :rtype: float

        :example:
        >>> from sklearn.datasets import make_blobs
        >>> X, y = make_blobs(n_samples=100000, centers=40, n_features=600, random_state=0, cluster_std=0.3)
        >>> Statistics.xie_beni(x=X, y=y)

        :references:
        .. [1] X. L. Xie, G. Beni (1991). A validity measure for fuzzy clustering.
               In: IEEE Transactions on Pattern Analysis and Machine Intelligence 13(8), 841 - 847. DOI: 10.1109/34.85677
        """
        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0], ])
        cluster_ids = np.unique(y)
        centroids = np.full(shape=(cluster_ids.shape[0], x.shape[1]), fill_value=-1.0, dtype=np.float32)
        intra_centroid_distances = np.full(shape=(y.shape[0]), fill_value=-1.0, dtype=np.float32)
        obs_cnt = 0
        for cnt, cluster_id in enumerate(cluster_ids):
            cluster_obs = x[np.argwhere(y == cluster_id).flatten()]
            centroids[cnt] = np.mean(cluster_obs, axis=0)
            intra_dist = np.linalg.norm(cluster_obs - centroids[cnt], axis=1)
            intra_centroid_distances[obs_cnt: cluster_obs.shape[0] + obs_cnt] = intra_dist
            obs_cnt += cluster_obs.shape[0]
        compactness = np.mean(np.square(intra_centroid_distances))
        cluster_dists = cdist(centroids, centroids).flatten()
        d = np.sqrt(cluster_dists[np.argwhere(cluster_dists > 0).flatten()])
        separation = np.min(d)

        return compactness / separation

    @staticmethod
    def bouguessa_wang_sun_v2(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Bouguessa-Wang-Sun (BWS) index using covariance matrices and means.

        :param np.ndarray x: A 2D array of shape (n_samples, n_features) representing the feature vectors of the data points.
        :param np.ndarray y: A 1D array of shape (n_samples,) containing the cluster labels for each data point.
        :returns: The BWS index value. Lower values indicate better clustering.
        :rtype: float

        :example:
        >>> from sklearn.datasets import make_blobs
        >>> X, y = make_blobs(n_samples=500, centers=3, random_state=42)
        >>> Statistics.bouguessa_wang_sun_v2(X, y)

        :references:
        .. [1] Bouguessa, Wang & Sun (2006).Bouguessa M, Wang S, Sun H. An objective approach to cluster validation.
               Pattern Recognition Letters. 2006;27:1419–1430. doi: 10.1016/j.patrec.2006.01.015.
        """

        unique_y = np.unique(y)
        global_mean = np.mean(x, axis=0)
        compactness, separation = 0, 0
        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0], ])

        for cluster_id in unique_y:
            cluster_data = x[y == cluster_id]
            cluster_mean = np.mean(cluster_data, axis=0)
            cov_matrix = np.cov(cluster_data, rowvar=False)
            compactness += np.trace(cov_matrix)
            diff_mean = cluster_mean - global_mean
            separation += len(cluster_data) * np.outer(diff_mean, diff_mean)

        separation_trace = np.trace(separation)
        return separation_trace / compactness

    @staticmethod
    def i_index(x: np.ndarray, y: np.ndarray):

        """
        Calculate the I-Index for evaluating clustering quality.

        The I-Index is a metric that measures the compactness and separation of clusters.
        A higher I-Index indicates better clustering with compact and well-separated clusters.

        .. seealso::
           To compute I-index on GPU, use :func:`~simba.data_processors.cuda.statistics.i_index`

        :param np.ndarray x: The dataset as a 2D NumPy array of shape (n_samples, n_features).
        :param np.ndarray y: Cluster labels for each data point as a 1D NumPy array of shape (n_samples,).
        :returns: The I-index score for the dataset.
        :rtype: float

        :references:
            .. [1] Zhao, Q., Xu, M., Fränti, P. (2009). Sum-of-Squares Based Cluster Validity Index and Significance Analysis.
                   In: Kolehmainen, M., Toivanen, P., Beliczynski, B. (eds) Adaptive and Natural Computing Algorithms. ICANNGA 2009.
                    Lecture Notes in Computer Science, vol 5495. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-642-04921-7_32

        :example:
        >>> X, y = make_blobs(n_samples=5000, centers=20, n_features=3, random_state=0, cluster_std=0.1)
        >>> Statistics.i_index(x=X, y=y)
        """
        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0], ])
        _ = get_unique_values_in_iterable(data=y, name=Statistics.i_index.__name__, min=2)
        unique_y = np.unique(y)
        n_y = unique_y.shape[0]
        global_centroid = np.mean(x, axis=0)
        sst = np.sum(np.linalg.norm(x - global_centroid, axis=1) ** 2)

        swc = 0
        for cluster_cnt, cluster_id in enumerate(unique_y):
            cluster_obs = x[np.argwhere(y == cluster_id).flatten()]
            cluster_centroid = np.mean(cluster_obs, axis=0)
            swc += np.sum(np.linalg.norm(cluster_obs - cluster_centroid, axis=1) ** 2)

        return sst / (n_y * swc)

    @staticmethod
    def sd_index(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the SD (Scatter and Discriminant) Index for evaluating the quality of a clustering solution.

        The SD Index combines two components to measure clustering quality:
        1. **Scatter (SCAT)**: Evaluates the compactness of clusters by measuring the ratio of intra-cluster variance to the global standard deviation.
        2. **Discriminant (DIS)**: Measures the separation between clusters relative to their distance from the global mean.

        A lower SD Index indicates better clustering quality, reflecting compact and well-separated clusters.

        :param np.ndarray x: A 2D array of shape (n_samples, n_features) representing the feature vectors of the data points.
        :param np.ndarray y: A 1D array of shape (n_samples,) containing the cluster labels for each data point.
        :returns: The SD Index value. Lower values indicate better clustering quality with more compact and well-separated clusters.
        :rtype: float

        :example:
        >>> X, y = make_blobs(n_samples=800, centers=2, n_features=3, random_state=0, cluster_std=0.1)
        >>> Statistics.sd_index(x=X, y=y)

        :references:
        .. [1] Halkidi, M., Vazirgiannis, M., Batistakis, Y. (2000). Quality Scheme Assessment in the Clustering Process. In: Zighed, D.A., Komorowski, J., Żytkow, J. (eds) Principles of Data Mining and Knowledge Discovery. PKDD 2000.
               Lecture Notes in Computer Science(), vol 1910. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-45372-5_26
        """
        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=(int,), accepted_axis_0_shape=[x.shape[0], ])
        _ = get_unique_values_in_iterable(data=y, name=Statistics.sd_index.__name__, min=2)
        global_std = np.std(x)
        global_m = np.mean(x, axis=0)
        unique_clusters = np.unique(y)
        cnt_y = unique_clusters.shape[0]
        scat, dis = 0, 0

        centroids = np.full(shape=(cnt_y, x.shape[1]), fill_value=-1.0, dtype=np.float32)
        for cnt, cluster in enumerate(unique_clusters):
            cluster_data = x[y == cluster]
            centroids[cnt] = np.mean(cluster_data, axis=0)
            scat += np.mean(np.std(cluster_data, axis=0)) / global_std

        for i in range(cnt_y):
            for j in range(i + 1, cnt_y):
                dist_between_clusters = np.linalg.norm(centroids[i] - centroids[j])
                dist_to_global = (np.linalg.norm(centroids[i] - global_m) + np.linalg.norm(centroids[j] - global_m)) / 2
                dis += dist_between_clusters / dist_to_global

        scat /= cnt_y
        dis /= (cnt_y * (cnt_y - 1) / 2)

        return scat + dis

    @staticmethod
    def c_index(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the C Index for clustering evaluation.

        :param np.ndarray x: A 2D array of shape (n_samples, n_features) containing the data points.
        :param np.ndarray x: A 1D array of shape (n_samples,) containing cluster labels for the data points.
        :return: The C Index value, ranging from 0 to 1.
        :rtype: float

        The C Index ranges from 0 to 1:
           - 0 indicates perfect clustering (clusters are as compact as possible).
           - 1 indicates worst clustering (clusters are highly spread out).

        :references:
        .. [1] Ubert, L. J., & Levin, J. R. (1976). A general statistical framework for assessing categorical clustering in free recall. Psychological Bulletin, 83(5), 1072–1080.


        :example:
        >>> X, y = make_blobs(n_samples=800, centers=2, n_features=3, random_state=0, cluster_std=0.1)
        >>> Statistics.c_index(x=X, y=y)
        """

        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=(int,), accepted_axis_0_shape=[x.shape[0], ])
        unique_y = np.unique(y)
        S_w = 0
        N_w = 0
        for cluster_id in unique_y:
            cluster_obs = x[np.argwhere(y == cluster_id).flatten()]
            dists = cdist(cluster_obs, cluster_obs)
            triu_indices = np.triu_indices_from(dists, k=1)
            S_w += np.sum(dists[triu_indices])
            N_w += len(triu_indices[0])

        all_dists = cdist(x, x)
        triu_indices = np.triu_indices_from(all_dists, k=1)
        sorted_dists = np.sort(all_dists[triu_indices])
        S_min = np.sum(sorted_dists[:N_w])
        S_max = np.sum(sorted_dists[-N_w:])

        return (S_w - S_min) / (S_max - S_min)

    @staticmethod
    def adjusted_rand(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the Adjusted Rand Index (ARI) between two clusterings.

        The Adjusted Rand Index (ARI) is a measure of the similarity between two clusterings. It considers all pairs of samples and counts pairs that are assigned to the same or different clusters in both the true and predicted clusterings.

        The ARI is defined as:

        .. math::
           ARI = \\frac{TP + TN}{TP + FP + FN + TN}

        where:
            - :math:`TP` (True Positive) is the number of pairs of elements that are in the same cluster in both x and y,
            - :math:`FP` (False Positive) is the number of pairs of elements that are in the same cluster in y but not in x,
            - :math:`FN` (False Negative) is the number of pairs of elements that are in the same cluster in x but not in y,
            - :math:`TN` (True Negative) is the number of pairs of elements that are in different clusters in both x and y.

        The ARI value ranges from -1 to 1. A value of 1 indicates perfect clustering agreement, 0 indicates random clustering, and negative values indicate disagreement between the clusterings.

        .. note::
           Modified from `scikit-learn <https://github.com/scikit-learn/scikit-learn/blob/8721245511de2f225ff5f9aa5f5fadce663cd4a3/sklearn/metrics/cluster/_supervised.py#L353>`_

        .. seealso::
           For GPU call, see :func:`simba.data_processors.cuda.statistics.adjusted_rand_gpu`

        :param np.ndarray x: 1D array representing the labels of the first model.
        :param np.ndarray y: 1D array representing the labels of the second model.
        :return: A value of 1 indicates perfect clustering agreement, a value of 0 indicates random clustering, and negative values indicate disagreement between the clusterings.
        :rtype: float

        :example:
        >>> x = np.array([0, 0, 0, 0, 0])
        >>> y = np.array([1, 1, 1, 1, 1])
        >>> Statistics.adjusted_rand(x=x, y=y)
        >>> 1.0
        """

        check_valid_array(
            data=x,
            source=Statistics.adjusted_rand.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.int64, np.int32, int),
            min_axis_0=1,
        )
        check_valid_array(
            data=y,
            source=Statistics.adjusted_rand.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.int64, np.int32, np.integer),
            accepted_shapes=[(x.shape[0],)],
        )
        return adjusted_rand_score(labels_true=x, labels_pred=y)

    @staticmethod
    def rmsstd(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Root-Mean-Square Standard Deviation (RMSSTD) for a clustering result.

        :param np.ndarray x: A 2D array of shape (n_samples, n_features) representing the feature vectors of the data points.
        :param np.ndarray y: A 1D array of shape (n_samples,) containing the cluster labels for each data point.
        :returns: The RMSSTD index value. Lower values indicate better clustering.
        :rtype: float

        :references:
        .. [1] Milligan, G. W., & Cooper, M. C. (1985). An examination of procedures for determining the number of clusters in a data set.
               Psychometrika, 50(2), 159–179. https://doi.org/10.1007/BF02294245

        :example:
        >>> X, y = make_blobs(n_samples=100, centers=10, n_features=3, random_state=0, cluster_std=0.1)
        >>> d = Statistics.rmsstd(x=X, y=y)
        """

        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value,
                          accepted_axis_0_shape=[x.shape[0], ])
        _ = get_unique_values_in_iterable(data=y, name=Statistics.rmsstd.__name__, min=2)
        unique_clusters = np.unique(y)
        p = x.shape[1]
        numerator, denominator = 0, 0
        for cluster_id in unique_clusters:
            cluster_points = x[y == cluster_id]
            cluster_mean = np.mean(cluster_points, axis=0)
            squared_diff = np.sum((cluster_points - cluster_mean) ** 2)
            numerator += squared_diff
            denominator += (cluster_points.shape[0] - 1) * p

        return np.sqrt(numerator / denominator)

    @staticmethod
    def krzanowski_lai_index(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-16) -> float:
        """
        Computes the Krzanowski-Lai (KL) Index for a given clustering result.

        :param np.ndarray x: A 2D array of shape (n_samples, n_features) representing the feature vectors of the data points.
        :param np.ndarray y: A 1D array of shape (n_samples,) containing the cluster labels for each data point.
        :param float epsilon: Small correction factor to avoid division by zero. Default 1e-16.
        :returns: The KL index value. Higher values indicate better clustering.
        :rtype: float

        :references:
        .. [1] Krzanowski, W. J., & Lai, Y. T. (1988). A criterion for determining the number of groups in a data set using sum-of-squares clustering.
               Biometrics, 44(1), 23–34. https://doi.org/10.2307/2531893

        :example:
        >>> X, y = make_blobs(n_samples=100, centers=10, n_features=3, random_state=0, cluster_std=100)
        >>> Statistics.krzanowski_lai_index(x=X, y=y)
        """

        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value,
                          accepted_axis_0_shape=[x.shape[0], ])
        unique_clusters = np.unique(y)
        _ = get_unique_values_in_iterable(data=y, name=Statistics.krzanowski_lai_index.__name__, min=2)
        x_center = np.mean(x, axis=0)
        BSS, WSS = 0.0, 0.0
        for cluster_id in unique_clusters:
            cluster_x = x[y == cluster_id]
            cluster_center = np.mean(cluster_x, axis=0)
            BSS += cluster_x.shape[0] * np.linalg.norm(cluster_center - x_center) ** 2
            WSS += np.sum(np.linalg.norm(cluster_x - cluster_center, axis=1) ** 2)
        return BSS / (WSS + epsilon)

    @staticmethod
    def cop_index(x: np.ndarray, y: np.ndarray, epsilon: float = 1e-16) -> float:
        """
        Computes the Clustering Overall Performance (COP) Index for evaluating clustering quality.

        The COP Index is defined as the ratio of the average intra-cluster compactness (C)
        to the average inter-cluster separation (S). A lower COP index indicates better clustering,
        as it implies tight clusters and greater separation between them.

        :param np.ndarray x: A 2D array of shape (n_samples, n_features) representing the feature vectors of the data points.
        :param np.ndarray y: A 1D array of shape (n_samples,) containing the cluster labels for each data point.
        :returns: The COP index value. Lower values indicate better clustering.
        :rtype: float

        :references:
        .. [1] Gurrutxaga, I., Albisua, I., Arbelaitz, O., Martín, J. I., Muguerza, J., Pérez, J. M., & Perona, I. (2011).
               SEP/COP: An efficient method to find the best partition in hierarchical clustering based on a new cluster validity index.
               Pattern Recognition, 44(4), 810-820. https://doi.org/10.1016/j.patcog.2010.10.002

        :example:
        >>> X, y = make_blobs(n_samples=50000, centers=10, n_features=3, random_state=0, cluster_std=1)
        >>> Statistics.cop_index(x=X, y=y)
        """

        unique_clusters = np.unique(y)
        cluster_combinations = list(combinations(unique_clusters, 2))
        intra_cluster_dists = np.full(shape=(len(unique_clusters)), fill_value=np.nan, dtype=np.float64)
        inter_cluster_dists = np.full(shape=(len(cluster_combinations)), fill_value=np.nan, dtype=np.float32)

        for cluster_cnt, cluster_id in enumerate(unique_clusters):
            cluster_x = x[np.argwhere(y == cluster_id).flatten()]
            intra_cluster_dist = np.sum(cdist(cluster_x, cluster_x))
            intra_cluster_dists[cluster_cnt] = intra_cluster_dist / (len(cluster_x) ** 2)

        C = np.mean(intra_cluster_dists)
        for cnt, (k, j) in enumerate(cluster_combinations):
            cluster_k = x[np.argwhere(y == k).flatten()]
            cluster_j = x[np.argwhere(y == j).flatten()]
            inter_cluster_dists[cnt] = np.min(cdist(cluster_k, cluster_j))

        S = np.mean(inter_cluster_dists)

        return C / (S + epsilon)

    @staticmethod
    def pbm_index(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the PBM (Performance of the Best Matching) Index, a measure of clustering quality that combines the compactness
        of the clusters and the separation between them. The PBM index evaluates how well-defined the clusters are in terms
        of their intra-cluster distance and the distance between their centroids.

        Higher values indicates better clustering.

        :param np.ndarray x: A 2D array of shape (n_samples, n_features) containing the data points.
        :param np.ndarray x: A 1D array of shape (n_samples,) containing cluster labels for the data points.
        :return: The PBM Index value.
        :rtype: float

        :references:
        .. [1] Pakhira, M. K., Bandyopadhyay, S., & Maulik, U. (2004). Validity index for crisp and fuzzy clusters.
               Pattern Recognition, 37(4), 487–501. https://doi.org/10.1016/j.patcog.2003.09.021
        .. [2] Bernard Desgraupes, University Paris Ouest Lab Modal’X, https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf

        :example:
        >>> X, y = make_blobs(n_samples=5, centers=2, n_features=3, random_state=0, cluster_std=5)
        >>> Statistics.pbm_index(x=X, y=y)

        """

        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0], ])
        unique_clusters, X_cnt = np.unique(y), x.shape[1]
        N_clusters = get_unique_values_in_iterable(data=y, name=Statistics.pbm_index.__name__, min=2)
        x_center = np.mean(x, axis=0)
        center_dists = np.linalg.norm(x - x_center, axis=1)
        E1 = np.sum(center_dists)
        intra_cluster_dists = np.full(shape=(len(unique_clusters)), fill_value=np.nan, dtype=np.float64)
        cluster_centers = np.full(shape=(len(unique_clusters), X_cnt), fill_value=np.nan, dtype=np.float64)
        for cnt, cluster_id in enumerate(unique_clusters):
            cluster_x = x[np.argwhere(y == cluster_id).flatten()]
            cluster_centers[cnt] = np.mean(cluster_x, axis=0)
            center_center_dists = np.linalg.norm(cluster_x - cluster_centers[cnt], axis=1)
            intra_cluster_dists[cnt] = np.sum(center_center_dists)

        EK = np.sum(intra_cluster_dists)

        cluster_dists = cdist(cluster_centers, cluster_centers)
        cluster_dists[cluster_dists == 0] = np.inf
        Dmin = np.min(cluster_dists)

        return (((1 / N_clusters) * E1) ** 2) / (EK * Dmin)

    @staticmethod
    def banfeld_raftery_index(x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the Banfeld-Raftery index for clustering evaluation.

        Smaller values represent better clustering. Values can be negative.

        :param x: 2D NumPy array of shape (n_samples, n_features) representing the dataset.
        :param y: 1D NumPy array of shape (n_samples,) containing cluster labels for each data point.
        :return: The Banfeld-Raftery index.
        :rtype: float

        :references:
           .. [1] Banfield, J. D., & Raftery, A. E. (1993). Model-based Gaussian and non-Gaussian clustering. Biometrics, 49(3), 803-821. https://doi.org/10.2307/2532201

        """
        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0], ])
        _ = get_unique_values_in_iterable(data=y, name=Statistics.banfeld_raftery_index.__name__, min=2)
        unique_labels = np.unique(y)
        val = 0.0
        for cluster_label in unique_labels:
            cluster_data = x[y == cluster_label]
            n_k = cluster_data.shape[0]
            covariance_matrix = np.cov(cluster_data, rowvar=False)
            determinant = np.linalg.det(covariance_matrix)
            determinant = max(determinant, 1e-10)
            val += n_k * np.log(determinant)

        return val

    @staticmethod
    def scott_symons_index(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Scott-Symons index for clustering evaluation.

        Smaller values represent better clustering. Values can be negative.

        :param np.ndarray x: The dataset as a 2D NumPy array of shape (n_samples, n_features).
        :param np.ndarray y: Cluster labels for each data point as a 1D NumPy array of shape (n_samples,).
        :returns: The Scott-Symons index score.
        :rtype: float


        :references:
           .. [1] . J. Scott and M. J. Symons. Clustering methods based on likelihood ratio criteria. Biometrics, 27:387–397, 1971.
        """

        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0], ])
        _ = get_unique_values_in_iterable(data=y, name=Statistics.scott_symons_index.__name__, min=2)
        unique_labels = np.unique(y)
        val = 0.0

        for label in unique_labels:
            cluster_points = x[y == label]
            n_k = cluster_points.shape[0]
            cov_matrix = np.cov(cluster_points, rowvar=False)
            det_cov = np.linalg.det(cov_matrix)
            val += n_k * np.log(det_cov / n_k)
        return val

    @staticmethod
    def wemmert_gancarski_index(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Wemmert-Gançarski index for clustering evaluation.

        The best case is when the index approaches 1, indicating good clustering. The worst case is when the index approaches 0, indicating poor clustering.

        :param np.ndarray x: The dataset as a 2D NumPy array of shape (n_samples, n_features).
        :param np.ndarray y: Cluster labels for each data point as a 1D NumPy array of shape (n_samples,).
        :returns: The Wemmert-Gançarski index score.
        :rtype: float

        :references:
           .. [1] Bernard Desgraupes, University Paris Ouest Lab Modal’X, https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf
        """

        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0], ])
        _ = get_unique_values_in_iterable(data=y, name=Statistics.wemmert_gancarski_index.__name__, min=2)
        unique_labels = np.unique(y)
        total_score = 0.0

        for label in unique_labels:
            cluster_points = x[y == label]
            n_k = cluster_points.shape[0]
            G_k = np.mean(cluster_points, axis=0)

            R_values = []
            for point in cluster_points:
                dist_to_G_k = np.linalg.norm(point - G_k)
                distances_to_other_centroids = [np.linalg.norm(point - np.mean(x[y == other_label], axis=0)) for other_label in unique_labels if other_label != label]
                min_dist_to_other_centroids = min(distances_to_other_centroids)
                R_values.append(dist_to_G_k / min_dist_to_other_centroids)

            J_k = max(0, 1 - (1 / n_k) * np.sum(R_values))
            total_score += n_k * J_k

        return total_score / x.shape[0]

    @staticmethod
    def mclain_rao_index(x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the McClain-Rao Index, which measures the quality of clustering by evaluating the ratio of
        the mean within-cluster distances to the mean between-cluster distances.

        The McClain-Rao Index is computed by calculating the mean ratio of intra-cluster distances (distances
        between points within the same cluster) to inter-cluster distances (distances between points from
        different clusters). A lower value indicates a better clustering result, with clusters being compact and well-separated.

        :param np.ndarray x: The dataset as a 2D NumPy array of shape (n_samples, n_features).
        :param np.ndarray y: Cluster labels for each data point as a 1D NumPy array of shape (n_samples,).
        :returns: The McClain-Rao Index score, a lower value indicates better clustering quality.
        :rtype: float

        :references:
           .. [1] McClain, J. O., & Rao, V. R. (1975). CLUSTISZ: A program to test for the quality of clustering of a set of objects.  *Journal of Marketing Research, 12*(4), 456-460. https://doi.org/10.1177/002224377501200410
        """

        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0], ])
        _ = get_unique_values_in_iterable(data=y, name=Statistics.mclain_rao_index.__name__, min=2)
        unique_labels = np.unique(y)
        ratios = np.full(shape=(len(unique_labels)), fill_value=np.nan, dtype=np.float64)
        for cluster_cnt, cluster_id in enumerate(unique_labels):
            cluster_obs = x[np.argwhere(y == cluster_id).flatten()]
            noncluster_obs = x[np.argwhere(y != cluster_id).flatten()]
            intra_dists = cdist(cluster_obs, cluster_obs)
            np.fill_diagonal(intra_dists, np.nan)
            intra_dist_mean = np.nanmean(intra_dists)
            inter_dist_mean = np.mean(cdist(cluster_obs, noncluster_obs))
            ratios[cluster_cnt] = intra_dist_mean / inter_dist_mean

        return np.mean(ratios)

    @staticmethod
    def s_dbw_index(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the S_Dbw index for evaluating the clustering quality.

        A lower value indicates a better clustering result.

        :param np.ndarray x: The dataset as a 2D NumPy array of shape (n_samples, n_features).
        :param np.ndarray y: Cluster labels for each data point as a 1D NumPy array of shape (n_samples,).
        :returns: The S_Dbw index score.
        :rtype: float

        .. note::
           Behaves weird as the number of dimensions increase (> 20).

        :example:
        >>> from sklearn.datasets import make_blobs
        >>> X, labels = make_blobs(n_samples=5000, centers=5, random_state=42, n_features=3, cluster_std=2)
        >>> score = Statistics.s_dbw_index(X, labels)

        :references:
           .. [1]  M. Halkidi and M. Vazirgiannis. Clustering validity assessment: Finding the optimal partitioning of a data set. Proceedings IEEE International Conference on Data Mining, pages 187–194, 2001.
        """

        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value,
                          accepted_axis_0_shape=[x.shape[0], ])
        _ = get_unique_values_in_iterable(data=y, name=Statistics.s_dbw_index.__name__, min=2)
        unique_labels = np.unique(y)
        K = len(unique_labels)
        centroids = np.array([x[y == label].mean(axis=0) for label in unique_labels])
        variances = np.array([np.var(x[y == label], axis=0) for label in unique_labels])
        sigma = np.sqrt(np.sum(np.linalg.norm(variances, axis=1)) / K)
        s_dbw = 0.0
        for k in range(K):
            for k_prime in range(k + 1, K):
                cluster_k = x[y == unique_labels[k]]
                cluster_k_prime = x[y == unique_labels[k_prime]]
                G_k = centroids[k]
                G_k_prime = centroids[k_prime]
                H_kk_prime = (G_k + G_k_prime) / 2
                density_at_Gk = np.sum(np.linalg.norm(cluster_k - G_k, axis=1) < sigma) + np.sum(np.linalg.norm(cluster_k_prime - G_k, axis=1) < sigma)
                density_at_Gk_prime = np.sum(np.linalg.norm(cluster_k - G_k_prime, axis=1) < sigma) + np.sum(np.linalg.norm(cluster_k_prime - G_k_prime, axis=1) < sigma)
                density_at_Hkk_prime = np.sum(np.linalg.norm(cluster_k - H_kk_prime, axis=1) < sigma) + np.sum(np.linalg.norm(cluster_k_prime - H_kk_prime, axis=1) < sigma)
                if max(density_at_Gk, density_at_Gk_prime) == 0:
                    pass
                else:
                    Rkk_prime = density_at_Hkk_prime / max(density_at_Gk, density_at_Gk_prime)
                    s_dbw += Rkk_prime

        s_dbw /= (K * (K - 1)) / 2
        return s_dbw

    @staticmethod
    def dunn_symmetry_idx(x: np.ndarray, y: np.ndarray) -> float:
        """
        DunnSym index output range positive real numbers 0 -> ∞ where 0 is extremely poor clustering and higher values represent better cluster separation.

        :param x: 2D array representing the data points. Shape (n_samples, n_features/n_dimension).
        :param y: 2D array representing cluster labels for each data point. Shape (n_samples,).
        :return float: Dynn-Symmetry index.

        :references:
           .. [1]  Ikotun, A. M., Habyarimana, F., & Ezugwu, A. E. (2025). Cluster validity indices for automatic clustering: A comprehensive review. Heliyon, 11(2), e41953. https://doi.org/10.1016/j.heliyon.2025.e41953
           .. [2]  Hassan, B. A., Tayfor, N. B., Hassan, A. A., Ahmed, A. M., Rashid, T. A., & Abdalla, N. N. (2024). From A-to-Z review of clustering validation indices. arXiv. https://doi.org/10.48550/arXiv.2407.20246


        :example:
        >>> x, y = make_blobs(n_samples=1000, n_features=2, centers=5, random_state=42, cluster_std=0.1)
        >>> Statistics.dunn_symmetry_idx(x=x, y=y)
        """

        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0], ])
        _ = get_unique_values_in_iterable(data=y, name=Statistics.dunn_symmetry_idx.__name__, min=2)
        unique_clusters = np.unique(y)
        min_inter_distance, max_intra_diameter = np.inf, -np.inf
        for i in range(len(unique_clusters)):
            for j in range(i + 1, len(unique_clusters)):
                i_points = x[y == i]
                j_points = x[y == j]
                min_inter_distance = min(min_inter_distance, np.min(cdist(i_points, j_points)))

        for i in range(len(unique_clusters)):
            i_points = x[y == i]
            max_intra_diameter = max(max_intra_diameter, np.max(cdist(i_points, i_points)))

        return min_inter_distance / max_intra_diameter

    @staticmethod
    def ray_turi_index(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Ray-Turi index for evaluating the clustering quality.

        A lower value indicates a better clustering result.

        :param np.ndarray x: The dataset as a 2D NumPy array of shape (n_samples, n_features).
        :param np.ndarray y: Cluster labels for each data point as a 1D NumPy array of shape (n_samples,).
        :returns: The Ray-Turi  index score.
        :rtype: float

        :example:
        >>> from sklearn.datasets import make_blobs
        >>> X, labels = make_blobs(n_samples=5000, centers=5, random_state=42, n_features=3, cluster_std=2)
        >>> score = Statistics.s_dbw_index(X, labels)

        :references:
           .. [1] Ray, S., & Turi, R. H. (1999). Determination of number of clusters in k-means clustering and application in colour image segmentation. Proceedings of the 4th International Conference on Advances in Pattern Recognition and Digital Techniques, 137–143.
        """

        check_valid_array(data=x, accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[x.shape[0], ])
        n_clusters = get_unique_values_in_iterable(data=y, name=Statistics.ray_turi_index.__name__, min=2)
        unique_labels = np.unique(y)

        centroids = np.array([x[y == label].mean(axis=0) for label in unique_labels])
        intra_dists = np.full(shape=(x.shape[0]), fill_value=np.nan, dtype=np.float32)
        min_cluster_distance = np.inf
        obs_cnt = 0
        for cnt, cluster_id in enumerate(unique_labels):
            cluster_obs = x[np.argwhere(y == cluster_id).flatten()]
            centroids[cnt] = np.mean(cluster_obs, axis=0)
            dists = np.linalg.norm(cluster_obs - centroids[cnt], axis=1) ** 2
            intra_dists[obs_cnt: obs_cnt + dists.shape[0]] = dists
            obs_cnt += dists.shape[0]

        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                distance = np.sum((centroids[i] - centroids[j]) ** 2)
                min_cluster_distance = min(min_cluster_distance, distance)

        return np.mean(intra_dists) / min_cluster_distance

    @staticmethod
    def fowlkes_mallows(x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the Fowlkes-Mallows Index (FMI) between two clusterings.

        The Fowlkes-Mallows index (FMI) is a measure of similarity between two clusterings. It compares the similarity of the clusters obtained by two different clustering algorithms or procedures.

        The index is defined as the geometric mean of the pairwise precision and recall:

        .. math::
           FMI = \\sqrt{\\frac{TP}{TP + FP} \\times \\frac{TP}{TP + FN}}

        where:
        - :math:`TP` (True Positive) is the number of pairs of elements that are in the same cluster in both x and y,
        - :math:`FP` (False Positive) is the number of pairs of elements that are in the same cluster in y but not in x,
        - :math:`FN` (False Negative) is the number of pairs of elements that are in the same cluster in x but not in y.

        .. note::
           Modified from `scikit-learn <https://github.com/scikit-learn/scikit-learn/blob/8721245511de2f225ff5f9aa5f5fadce663cd4a3/sklearn/metrics/cluster/_supervised.py#L1184>`_


        :param np.ndarray x: 1D array representing the labels of the first model.
        :param np.ndarray y: 1D array representing the labels of the second model.
        :return float: Score between 0 and 1. 1 indicates perfect clustering agreement, 0 indicates random clustering.
        """

        check_valid_array(
            data=x,
            source=Statistics.fowlkes_mallows.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.int64, np.int32, int),
            min_axis_0=1,
        )
        check_valid_array(
            data=y,
            source=Statistics.fowlkes_mallows.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.int64, np.int32, int),
            accepted_shapes=[(x.shape[0],)],
        )
        return fowlkes_mallows_score(labels_true=x, labels_pred=y)

    @staticmethod
    def adjusted_mutual_info(x: np.ndarray, y: np.ndarray) -> float:

        r"""
        Calculate the Adjusted Mutual Information (AMI) between two clusterings as a measure of similarity.

        Calculates the Adjusted Mutual Information (AMI) between two sets of cluster labels.
        AMI measures the agreement between two clustering results, accounting for chance agreement.
        The value of AMI ranges from 0 (indicating no agreement) to 1 (perfect agreement).

        .. math::

           \mathrm{AMI}(x, y) = \frac{\mathrm{MI}(x, y) - E(\mathrm{MI}(x, y))}{\max(H(x), H(y)) - E(\mathrm{MI}(x, y))}

        where:

           - :math:`\text{MI}(x, y)` is the mutual information between :math:`x` and :math:`y`.
           - :math:`E(\text{MI}(x, y))` is the expected mutual information.
           - :math:`H(x)` and :math:`H(y)` are the entropies of :math:`x` and :math:`y`, respectively.

        :param np.ndarray x: 1D array representing the labels of the first model.
        :param np.ndarray y: 1D array representing the labels of the second model.
        :return: Score between 0 and 1, where 1 indicates perfect clustering agreement.
        :rtype: float

        """
        check_valid_array(
            data=x,
            source=Statistics.adjusted_mutual_info.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.int64, np.int32, int),
            min_axis_0=1,
        )
        check_valid_array(
            data=y,
            source=Statistics.adjusted_mutual_info.__name__,
            accepted_ndims=(1,),
            accepted_dtypes=(np.int64, np.int32, int),
            accepted_shapes=[(x.shape[0],)],
        )
        return adjusted_mutual_info_score(labels_true=x, labels_pred=y)

    @staticmethod
    @jit(nopython=True)
    def czebyshev_distance(sample_1: np.ndarray, sample_2: np.ndarray) -> float:

        r"""
        Calculate the Czebyshev distance between two N-dimensional samples.

        The Czebyshev distance is defined as the maximum absolute difference
        between the corresponding elements of the two arrays.

        .. note::
           Normalize arrays sample_1 and sample_2 before passing it to ensure accurate results.

        The equation for the Czebyshev distance is given by :math:`D_\infty(p, q) = \max_i \left| p_i - q_i \right|`.

        .. seealso:
           :func:`simba.mixins.statistics_mixin.Statistics.sliding_czebyshev_distance`

        :param np.ndarray sample_1: The first sample, an N-dimensional NumPy array.
        :param np.ndarray sample_2: The second sample, an N-dimensional NumPy array.
        :return float: The Czebyshev distance between the two samples.

        :example:
        >>> sample_1 = np.random.randint(0, 10, (10000,100))
        >>> sample_2 = np.random.randint(0, 10, (10000,100))
        >>> Statistics.czebyshev_distance(sample_1=sample_1, sample_2=sample_2)
        """

        c = 0.0
        for idx in np.ndindex(sample_1.shape):
            c = max((c, np.abs(sample_1[idx] - sample_2[idx])))
        return c

    @staticmethod
    #@njit(["(float32[:, :], float64[:], int64)", ])
    #@jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes=["(float32[:, :], float64[:], int64)", ], cache=True, fastmath=False)

    def sliding_czebyshev_distance(x: np.ndarray, window_sizes: np.ndarray, sample_rate: float) -> np.ndarray:
        """
        Calculate the sliding Chebyshev distance for a given signal with different window sizes.

        This function computes the sliding Chebyshev distance for a signal `x` using
        different window sizes specified by `window_sizes`. The Chebyshev distance measures
        the maximum absolute difference between the corresponding elements of two signals.

        .. note::
           Normalize array x before passing it to ensure accurate results.

        .. seealso:
           For simple 2-sample comparison, use :func:`simba.mixins.statistics_mixin.Statistics.czebyshev_distance`

        :param np.ndarray x: Input signal, a 2D array with shape (n_samples, n_features).
        :param np.ndarray window_sizes: Array containing window sizes for sliding computation.
        :param float sample_rate: Sampling rate of the signal.
        :return: 2D array of Chebyshev distances for each window size and position.
        :rtype: np.ndarray

        :example:
        >>> sample_1 = np.random.randint(0, 10, (200,5)).astype(np.float32)
        >>> sample_2 = np.random.randint(0, 10, (10000,100))
        >>> Statistics.sliding_czebyshev_distance(x=sample_1, window_sizes=np.array([1.0, 2.0]), sample_rate=10.0)
        """

        result = np.full((x.shape[0], window_sizes.shape[0]), 0.0)
        for i in range(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * sample_rate)
            for l, r in zip(range(0, x.shape[0] + 1), range(window_size, x.shape[0] + 1)):
                sample, c = x[l:r, :], 0.0
                for j in range(sample.shape[1]):
                    c = max(c, (np.abs(np.min(sample[:, j]) - np.max(sample[:, j]))))
                result[r - 1, i] = c
        return result

    @staticmethod
    # @njit(["(int64[:], int64[:], float64[:])", "(int64[:], int64[:], types.misc.Omitted(None))",
    #        "(int64[:, :], int64[:, :], float64[:])", "(int64[:, :], int64[:, :], types.misc.Omitted(None))"])
    # @jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes=["(int64[:], int64[:], float64[:])", "(int64[:], int64[:], types.misc.Omitted(None))", "(int64[:, :], int64[:, :], float64[:])", "(int64[:, :], int64[:, :], types.misc.Omitted(None))"], cache=True, fastmath=False)
    def sokal_michener(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> float:

        r"""
        Jitted compute of the Sokal-Michener dissimilarity between two binary vectors or matrices.

        Higher values indicate more dissimilar vectors or matrices, while lower values indicate more similar vectors or matrices.

        The Sokal-Michener dissimilarity is a measure of dissimilarity between two sets
        based on the presence or absence of similar attributes. This implementation supports weighted dissimilarity.

        .. note::
           Adapted from `umap <https://github.com/lmcinnes/umap/blob/e7f2fb9e5e772edd5c8f38612365ec6a35a54373/umap/distances.py#L468>`_.

        .. math::
           D(x, y) = \\frac{2 \cdot \sum_{i} w_i \cdot \mathbb{1}(x_i \neq y_i)}{N + \sum_{i} w_i \cdot \mathbb{1}(x_i \neq y_i)}

        where:
        - :math:`x` and :math:`y` are the binary vectors or matrices.
        - :math:`w_i` is the weight for the i-th element.
        - :math:`\mathbb{1}(x_i \neq y_i)` is an indicator function that is 1 if :math:`x_i \neq y_i` and 0 otherwise.
        - :math:`N` is the total number of elements in :math:`x` or :math:`y`.

        :param np.ndarray x: First binary vector or matrix.
        :param np.ndarray y: Second binary vector or matrix.
        :param Optional[np.ndarray] w: Optional weight vector. If None, all weights are considered as 1.
        :return: Sokal-Michener dissimilarity between `x` and `y`.
        :rtype: float

        :example:
        >>> x = np.random.randint(0, 2, (200,))
        >>> y = np.random.randint(0, 2, (200,))
        >>> sokal_michener = Statistics.sokal_michener(x=x, y=y)
        """

        if w is None:
            w = np.ones(x.shape[0]).astype(np.float64)
        unequal_cnt = 0.0
        for i in np.ndindex(x.shape):
            x_i, y_i = x[i], y[i]
            if x_i != y_i:
                unequal_cnt += 1 * w[i[0]]
        return (2.0 * unequal_cnt) / (x.size + unequal_cnt)

    def kumar_hassebrook_similarity(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Kumar-Hassebrook similarity is a measure used to quantify the similarity between two vectors.

        .. note::
            Kumar-Hassebrook similarity score of 1 indicates identical vectors and 0 indicating no similarity

        :param np.ndarray x: 1D array representing the first feature values.
        :param np.ndarray y: 1D array representing the second feature values.
        :return: Kumar-Hassebrook similarity between vectors x and y.
        :rtype: float

        :example:
        >>> x, y = np.random.randint(0, 500, (1000,)), np.random.randint(0, 500, (1000,))
        >>> Statistics.kumar_hassebrook_similarity(x=x, y=y)
        """
        check_valid_array(data=x, source=f'{Statistics.kumar_hassebrook_similarity.__name__} x', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, source=f'{Statistics.kumar_hassebrook_similarity.__name__} y', accepted_ndims=(1,), accepted_shapes=(x.shape,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        dot_product = np.dot(x, y)
        norm_x = np.linalg.norm(x)
        norm_y = np.linalg.norm(y)
        return dot_product / (norm_x ** 2 + norm_y ** 2 - dot_product)

    def wave_hedges_distance(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the Wave-Hedges distance between two 1-dimensional arrays `x` and `y`. The Wave-Hedges distance is a measure of dissimilarity between arrays.

        .. note::
            Wave-Hedges distance score of 0 indicate identical arrays. There is no upper bound.

        :param np.ndarray x: 1D array representing the first feature values.
        :param np.ndarray y: 1D array representing the second feature values.
        :returns: Wave-Hedges distance
        :rtype: float

        :example:
        >>> x = np.random.randint(0, 500, (1000,))
        >>> y = np.random.randint(0, 500, (1000,))
        >>> Statistics().wave_hedges_distance(x=x, y=y)

        :references:
           .. [1] Hedges, T. S. (1976). An empirical modification to linear wave theory. Proceedings of the Institution of Civil Engineers, Part 2, 61(3), 575–579. https://doi.org/10.1680/iicep.1976.3408
        """

        check_valid_array(data=x, source=f'{Statistics.wave_hedges_distance.__name__} x', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, source=f'{Statistics.wave_hedges_distance.__name__} y', accepted_ndims=(1,), accepted_shapes=(x.shape,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        x_y = abs(x - y)
        xy_max = np.maximum(x, y)
        return np.sum(np.where(((x_y != 0) & (xy_max != 0)), x_y / xy_max, 0))

    @staticmethod
    def gower_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute Gower-like distance vector between corresponding rows of two numerical matrices.
        Gower distance is a measure of dissimilarity between two vectors (or rows in this case).

        .. note::
           This function assumes x and y have the same shape and only considers numerical attributes.
            Each observation in x is compared to the corresponding observation in y based on normalized
            absolute differences across numerical columns.

        :param np.ndarray x: First numerical matrix with shape (m, n).
        :param np.ndarray y: Second numerical matrix with shape (m, n).
        :return: Gower-like distance vector with shape (m,).
        :rtype: np.ndarray

        :example:
        >>> x, y = np.random.randint(0, 500, (1000, 6000)), np.random.randint(0, 500, (1000, 6000))
        >>> Statistics.gower_distance(x=x, y=y)


        :references:
           .. [1] Gower, J. C. (1971). A general coefficient of similarity and some of its properties. Biometrics, 27(4), 857–874. https://doi.org/10.2307/2528823


        """
        check_valid_array(data=x, source=f'{Statistics.gower_distance.__name__} x', accepted_ndims=(1, 2), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, source=f'{Statistics.gower_distance.__name__} y', accepted_ndims=(x.ndim,), accepted_shapes=(x.shape,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        field_ranges = np.max(x, axis=0) - np.min(x, axis=0)
        results = np.full((x.shape[0]), np.nan)
        for i in range(x.shape[0]):
            u, v = x[i], y[i]
            dist = 0.0
            for j in range(u.shape[0]):
                if field_ranges[j] != 0:
                    dist += np.abs(u[j] - v[j]) / field_ranges[j]
            results[i] = dist / u.shape[0]
        return results

    @staticmethod
    def normalized_google_distance(x: np.ndarray, y: np.ndarray) -> float:

        r"""
        Compute the Normalized Google Distance (NGD) between two vectors or matrices.

        Normalized Google Distance is a measure of similarity between two sets based on
        the relationship between the sums and minimum values of their elements.

        The NGD is calculated as:

        .. math::
          NGD(x, y) = \frac{\max(\sum x, \sum y) - \sum \min(x, y)}{(\sum x + \sum y) - \min(\sum x, \sum y)}

        where:
        - :math:`\sum x` is the sum of elements in `x`
        - :math:`\sum y` is the sum of elements in `y`
        - :math:`\sum \min(x, y)` is the sum of element-wise minimums of `x` and `y`

        .. note::
           This function assumes x and y have the same shape. It computes NGD based on the sum of elements and the minimum values between corresponding elements of x and y.

        :param np.ndarray x: First numerical matrix with shape (m, n).
        :param np.ndarray y: Second array or matrix with shape (m, n).
        :return: Normalized Google Distance between x and y.
        :rtype: float

        :example:
        >>> x, y = np.random.randint(0, 500, (1000,200)), np.random.randint(0, 500, (1000,200))
        >>> Statistics.normalized_google_distance(x=y, y=x)

        :references:
           .. [1] Cilibrasi, R., & Vitányi, P. (2007). Clustering by compression. IEEE Transactions on Information Theory, 51(4), 1523-1545. https://doi.org/10.1109/TIT.2005.862080
        """
        check_valid_array(data=x, source=f'{Statistics.normalized_google_distance.__name__} x', accepted_ndims=(1, 2), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, source=f'{Statistics.normalized_google_distance.__name__} y', accepted_ndims=(x.ndim,), accepted_shapes=(x.shape,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)

        sum_x, sum_y = np.sum(x), np.sum(y)
        sum_min = np.sum(np.minimum(x, y))
        D = (sum_x + sum_y) - np.min([sum_x, sum_y])
        N = np.max([sum_x, sum_y]) - sum_min
        if D == 0:
            return -1.0
        else:
            return N / D

    @staticmethod
    def symmetry_index(x: np.ndarray, y: np.ndarray, agg_type: Literal['mean', 'median'] = 'mean') -> float:

        """
        Calculate the Symmetry Index (SI) between two arrays of measurements, `x` and `y`, over a given time series.
        The Symmetry Index quantifies the relative difference between two measurements at each time point, expressed as a percentage.
        The function returns either the mean or median Symmetry Index over the entire series, based on the specified aggregation type.

        Zero indicates perfect symmetry. Positive values pepresent increasing asymmetry between the two measurements.

        The Symmetry Index (SI) is calculated as:

        .. math::
           SI = \frac{|x_i - y_i|}{0.5 \times (x_i + y_i)} \times 100

        where :math:`x_i` and :math:`y_i` are the values of the two measurements at each time point.

        :param np.ndarray x: A 1-dimensional array of measurements from one side (e.g., left side), representing a time series or sequence of measurements.
        :param np.ndarray y: A 1-dimensional array of measurements from the other side (e.g., right side), of the same length as `x`.
        :param Literal['mean', 'median'] agg_type: The aggregation method used to summarize the Symmetry Index across all time points.
        :return: The aggregated Symmetry Index over the series, either as the mean or median SI.
        :rtype: float

        :example:
        >>> x = np.random.randint(0, 155, (100,))
        >>>y = np.random.randint(0, 155, (100,))
        >>> Statistics.symmetry_index(x=x, y=y)
        """

        check_valid_array(data=x, source=f'{Statistics.symmetry_index.__name__} x', accepted_ndims=(1,), min_axis_0=1, accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=x, source=f'{Statistics.symmetry_index.__name__} y', accepted_ndims=(1,), min_axis_0=1, accepted_axis_0_shape=[x.shape[0]], accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_str(name=f'{Statistics.symmetry_index.__name__} agg_type', value=agg_type, options=('mean', 'median'))
        si_values = np.abs(x - y) / (0.5 * (x + y)) * 100
        if agg_type == 'mean':
            return np.float32(np.nanmean(si_values))
        else:
            return np.float32(np.nanmedian(si_values))

    @staticmethod
    # @njit("(float32[:], float64, float64)")
    # @jit(nopython=True, cache=True)
    @dynamic_numba_decorator(dtypes="(float32[:], float64, float64)", cache=True, fastmath=False)

    def sliding_iqr(x: np.ndarray, window_size: float, sample_rate: float) -> np.ndarray:
        """
        Compute the sliding interquartile range (IQR) for a 1D array of feature values.

        :param ndarray x: 1D array representing the feature values for which the IQR will be calculated.
        :param float window_size: Size of the sliding window, in seconds.  This value determines how many samples are included in each window.
        :param float sample_rate: The sampling rate in samples per second, e.g., fps.
        :returns : Sliding IQR values
        :rtype: np.ndarray

        :references:
            .. [1] Hession, Leinani E., Gautam S. Sabnis, Gary A. Churchill, and Vivek Kumar. “A Machine-Vision-Based Frailty Index for Mice.” Nature Aging 2, no. 8 (August 16, 2022): 756–66. https://doi.org/10.1038/s43587-022-00266-0.

        :example:
        >>> data = np.random.randint(0, 50, (90,)).astype(np.float32)
        >>> window_size = 0.5
        >>> Statistics.sliding_iqr(x=data, window_size=0.5, sample_rate=10.0)
        """

        frm_win = max(1, int(window_size * sample_rate))
        results = np.full(shape=(x.shape[0],), dtype=np.float32, fill_value=-1.0)
        for r in range(frm_win, x.shape[0] + 1):
            sorted_sample = np.sort(x[r - frm_win:r])
            lower_idx = sorted_sample.shape[0] // 4
            upper_idx = (3 * sorted_sample.shape[0]) // 4
            lower_val = sorted_sample[lower_idx]
            upper_val = sorted_sample[upper_idx]
            results[r - 1] = upper_val - lower_val
        return results

    @staticmethod
    def one_way_anova_scipy(x: np.ndarray,
                            y: np.ndarray,
                            variable_names: List[str],
                            x_name: str = '',
                            y_name: str = '') -> pd.DataFrame:
        """
        Compute one-way ANOVAs comparing each column (axis 1) on two arrays.

        .. notes::
           Use for computing and presenting aggregate statistics. Not suitable for featurization.

        .. seealso::
           For featurization instead use :func:`simba.mixins.statistics_mixin.Statistics.rolling_one_way_anova` or
           :func:`simba.mixins.statistics_mixin.Statistics.one_way_anova`

        :param np.ndarray x: First 2d array with observations rowwise and variables columnwise.
        :param np.ndarray y: Second 2d array with observations rowwise and variables columnwise. Must be same number of columns as x.
        :param List[str, ...] variable_names: Names of columnwise variable names. Same length as number of data columns.
        :param str x_name: Name of the first group (x).
        :param str y_name: Name of the second group (y).
        :return: Dataframe with one row per column representing the ANOVA F-statistic and P-values comparing the variables between x and y.
        :rtype: pd.DataFrame
        """

        check_valid_array(data=x, source=f'{Statistics.one_way_anova_scipy.__name__} x', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, source=f'{Statistics.one_way_anova_scipy.__name__} y', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_1_shape=(x.shape[1],))
        check_str(name=f'{Statistics.one_way_anova_scipy.__name__} x_name', value=x_name, allow_blank=True)
        check_str(name=f'{Statistics.one_way_anova_scipy.__name__} y_name', value=y_name, allow_blank=True)
        check_valid_lst(source=f'{Statistics.one_way_anova_scipy.__name__} variable_names', data=variable_names, valid_dtypes=(str,), exact_len=x.shape[1])
        results = pd.DataFrame(variable_names, columns=['FEATURE'])
        results[['GROUP_1', 'GROUP_2']] = x_name, y_name
        results['F-STATISTIC'], results['P-VALUE'] = stats.f_oneway(x, y)

        results['P-VALUE'] = results['P-VALUE'].round(8)

        return results

    @staticmethod
    def kruskal_scipy(x: np.ndarray,
                      y: np.ndarray,
                      variable_names: List[str],
                      x_name: str = '',
                      y_name: str = '') -> pd.DataFrame:
        """
        Compute Kruskal-Wallis comparing each column (axis 1) on two arrays.

        .. notes::
           Use for computing and presenting aggregate statistics. Not suitable for featurization.

        .. seealso::
           For featurization instead use :func:`simba.mixins.statistics_mixin.Statistics.kruskal_wallis`

        :param np.ndarray x: First 2d array with observations rowwise and variables columnwise.
        :param np.ndarray y: Second 2d array with observations rowwise and variables columnwise. Must be same number of columns as x.
        :param List[str, ...] variable_names: Names of columnwise variable names. Same length as number of data columns.
        :param str x_name: Name of the first group (x).
        :param str y_name: Name of the second group (y).
        :return: Dataframe with one row per column representing the Kruskal-Wallis statistic and P-values comparing the variables between x and y.
        :rtype: pd.DataFrame
        """

        check_valid_array(data=x, source=f'{Statistics.kruskal_scipy.__name__} x', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=y, source=f'{Statistics.kruskal_scipy.__name__} y', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_1_shape=(x.shape[1],))
        check_str(name=f'{Statistics.kruskal_scipy.__name__} x_name', value=x_name, allow_blank=True)
        check_str(name=f'{Statistics.kruskal_scipy.__name__} y_name', value=y_name, allow_blank=True)
        check_valid_lst(source=f'{Statistics.kruskal_scipy.__name__} variable_names', data=variable_names, valid_dtypes=(str,), exact_len=x.shape[1])
        results = pd.DataFrame(variable_names, columns=['FEATURE'])
        results[['GROUP_1', 'GROUP_2']] = x_name, y_name
        results['STATISTIC'], results['P-VALUE'] = stats.kruskal(x, y)

        results['P-VALUE'] = results['P-VALUE'].round(8)

        return results

    @staticmethod
    def pairwise_tukeyhsd_scipy(data: np.ndarray,
                                group: np.ndarray,
                                variable_names: List[str],
                                verbose: bool = False) -> pd.DataFrame:

        """
        Compute pairwise grouped Tukey-HSD tests.

        .. notes::
           Use for computing and presenting aggregate statistics. Not suitable for featurization.

        :param np.ndarray data: 2D array  with observations rowwise (axis 0) and features columnwise (axis 1)
        :param np.ndarray group: 1D array with the same number of observations as rows in ``data`` containing the group for each sample.
        :param List[str, ...] variable_names: Names of columnwise variable names. Same length as number of data columns.
        :return: Dataframe comparing each group for each variable.
        :rtype: pd.DataFrame
        """

        check_valid_array(data=data, source=f'{Statistics.pairwise_tukeyhsd_scipy.__name__} data', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=group, source=f'{Statistics.pairwise_tukeyhsd_scipy.__name__} group', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=(data.shape[0],))
        check_valid_lst(source=f'{Statistics.pairwise_tukeyhsd_scipy.__name__} variable_names', data=variable_names, valid_dtypes=(str,), exact_len=data.shape[1])
        results = []
        for var in range(data.shape[1]):
            if verbose:
                print(f'Computing Tukey HSD for variable {var+1}/{data.shape[1]}...')
            tukey_data = pairwise_tukeyhsd(data[:, var], group)
            df = pd.DataFrame(data=tukey_data._results_table.data[1:], columns=tukey_data._results_table.data[0])
            df['P-VALUE'] = psturng(np.abs(tukey_data.meandiffs / tukey_data.std_pairs), len(tukey_data.groupsunique), tukey_data.df_total)
            df['FEATURE'] = variable_names[var]
            results.append(df)

        return pd.concat(results, axis=0)


    @staticmethod
    @dynamic_numba_decorator(dtypes="(float32[:], float32[:])", cache=True, fastmath=False)
    def circular_euclidean_kantorovich(x: np.ndarray, y: np.ndarray):
        """
        Compute the circular Euclidean Kantorovich (Wasserstein) distance between two discrete distributions.

        Suitable for comparing distributions of circular data such as angles, time-of-day, phase etc.

        :param: np.ndarray x: 1D array representing the first discrete distribution or histogram.
        :param: np.ndarray x: 1D array representing the second discrete distribution or histogram.

        .. note::
           Distance metric: smaller values represent similar distributions.
           Adapted from `pynndescent <https://pynndescent.readthedocs.io/en/latest/>`_.

        :example:
        >>> x, y = np.random.normal(loc=65, scale=10, size=10000000), np.random.normal(loc=90, scale=1, size=10000000)
        >>> b =Statistics.circular_euclidean_kantorovich(x, y)
        """
        x_sum, y_sum = np.sum(x), np.sum(y)
        x_cdf, y_cdf = x / x_sum, y / y_sum
        x_cdf, y_cdf = np.cumsum(x_cdf), np.cumsum(y_cdf)
        mu = np.median((x_cdf - y_cdf) ** 2)
        result = 0.0
        for i in prange(x_cdf.shape[0]):
            val = x_cdf[i] - y_cdf[i] - mu
            result += val * val
        return np.sqrt(result)
