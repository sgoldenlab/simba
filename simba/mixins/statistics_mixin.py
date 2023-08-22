__author__ = "Simon Nilsson"

import time

try:
    from typing import Literal
except:
    from typing_extensions import Literal
import numpy as np
from numba import njit, jit, prange
from scipy import stats

from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.data import bucket_data, fast_mean_rank, fast_minimum_rank

class Statistics(FeatureExtractionMixin):

    """
    Primarily frequentist statistics methods used for feature extraction of drift assessment.

    """

    def __init__(self):
        FeatureExtractionMixin.__init__(self)


    @staticmethod
    @jit(nopython=True)
    def _hist_1d(data: np.ndarray,
                bin_count: int,
                range: np.ndarray):
        """
        Jitted helper to compute 1D histograms with counts.

        :parameter np.ndarray data: 1d array containing feature values.
        :parameter int bins: The number of bins.
        :parameter: np.ndarray range: 1d array with two values representing minimum and maximum value to bin.
        """

        hist = np.histogram(data, bin_count, (range[0], range[1]))[0]
        return hist

    @staticmethod
    @jit(nopython=True)
    def rolling_independent_sample_t(data: np.ndarray,
                                     time_window: float,
                                     fps: float) -> np.ndarray:
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
        >>> Statistics().rolling_independent_sample_t(data, group_size_s=1, fps=10)
        >>> [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389])

        """

        results = np.full((data.shape[0]), -1.0)
        window_size = int(time_window * fps)
        data = np.split(data, list(range(window_size, data.shape[0], window_size)))
        for cnt, i in enumerate(prange(1, len(data))):
            start, end = int((cnt + 1) * window_size), int(((cnt + 1) * window_size) + window_size)
            mean_1, mean_2 = np.mean(data[i-1]), np.mean(data[i])
            stdev_1, stdev_2 = np.std(data[i-1]), np.std(data[i])
            results[start:end] = (mean_1 - mean_2) / np.sqrt((stdev_1 / data[i-1].shape[0]) + (stdev_2 / data[i].shape[0]))
        return results

    def independent_samples_t(self,
                              sample_1: np.ndarray,
                              sample_2: np.ndarray) -> (float, float):
        """
        Compute independent-samples t-test statistic and p-values between two distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns (float float) t_statistic, p_value: Representing t-statistic and associated probability value.

        :example:
        >>> sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
        >>> sample_2 = np.array([8, 7, 5, 3, 8, 9, 6, 10, 7, 10, 10])
        >>> Statistics().independent_samples_t(sample_1=sample_1, sample_2=sample_2)
        >>> (-4.877830567527609, 9.106532603464572e-05)
        """
        t_statistic = (np.mean(sample_1) - np.mean(sample_2)) / np.sqrt((np.std(sample_1) / sample_1.shape[0]) + (np.std(sample_2) / sample_1.shape[0]))
        dof = (sample_1.shape[0] + sample_2.shape[0]) - 2
        p_value = 2 * (stats.t.cdf(-abs(t_statistic), dof))
        return t_statistic, p_value


    def cohens_d(self,
                 sample_1: np.ndarray,
                 sample_2: np.ndarray) -> float:
        """
        Jitted compute of Cohen's d between two distributions

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns float: Cohens D statistic.

        :example:
        >>> sample_1 = np.array([8, 1, 5, 1, 8, 1, 1, 10, 7, 10, 10])
        >>> sample_2 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
        >>> Statistics().cohens_d(sample_1=sample_1, sample_2=sample_2)
        >>> 0.41197143155579974
        """

        dof = (sample_1.shape[0] + sample_2.shape[0]) - 2
        return (np.mean(sample_1) - np.mean(sample_2)) / np.sqrt(((sample_1.shape[0] - 1) * np.std(sample_1, ddof=1) ** 2 + (sample_2.shape[0] - 1) * np.std(sample_2, ddof=1) ** 2) / dof)

    @staticmethod
    @jit(nopython=True)
    def rolling_cohens_d(data: np.ndarray,
                         time_windows: np.ndarray,
                         fps: float) -> np.ndarray:

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
            data_split = np.split(data, list(range(window_size, data.shape[0], window_size)))
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j-1].astype(np.float32), data_split[j].astype(np.float32)
                dof = (sample_1.shape[0] + sample_2.shape[0]) - 2
                mean_sample_1, mean_sample_2 = np.mean(sample_1), np.mean(sample_2)
                std_sample_1 = np.sum((sample_1 - mean_sample_1)**2) / sample_1.shape[0]
                std_sample_2 = np.sum((sample_2 - mean_sample_2)**2) / sample_2.shape[0]
                d = (mean_sample_1 - mean_sample_2) / np.sqrt(((sample_1.shape[0] - 1) * std_sample_1 ** 2 + (sample_2.shape[0] - 1) * std_sample_2 ** 2) / dof)
                results[window_start: window_end, i] = d
        return results

    def rolling_two_sample_ks(self,
                              data: np.ndarray,
                              time_window: float,
                              fps: float) -> np.ndarray:
        """
        Compute Kolmogorov two-sample statistics for sequentially binned values in a time-series.
        E.g., compute KS statistics when comparing ``Feature N`` in the current 1s time-window, versus ``Feature N`` in the previous 1s time-window.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter int time_window: The size of the buckets in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :return np.ndarray: Array of size data.shape[0] with KS statistics

        :example:
        >>> data = np.random.randint(low=0, high=100, size=(200)).astype('float32')
        >>> results = self.rolling_two_sample_ks(data=data, group_size_s=1, fps=30)
        """

        window_size, results = int(time_window * fps), np.full((data.shape[0]), -1.0)
        data = np.split(data, list(range(window_size, data.shape[0], window_size)))
        for cnt, i in enumerate(prange(1, len(data))):
            start, end = int((cnt + 1) * window_size), int(((cnt + 1) * window_size) + window_size)
            results[start:end] = stats.ks_2samp(data1=data[i-1], data2=data[i]).statistic
        return results

    def two_sample_ks(self,
                      sample_1: np.ndarray,
                      sample_2: np.ndarray) -> (float, float):

        """
        Compute Kolmogorov-Smirnov statistics and p-value for two distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns (float float): Representing Kolmogorov-Smirnov statistic and associated probability value.

        :example:
        >>> sample_1 = np.array([8, 1, 5, 1, 8, 1, 1, 10, 7, 10, 10])
        >>> sample_2 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
        >>> Statistics().two_sample_ks(sample_1=sample_1, sample_2=sample_2)
        >>>
        """

        ks = stats.ks_2samp(data1=sample_1, data2=sample_2)
        return ks.statistic, ks.pvalue,


    def one_way_anova(self,
                      sample_1: np.ndarray,
                      sample_2: np.ndarray) -> (float, float):
        """
        Compute One-way ANOVA F statistics and associated p-value for two distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns (float float): Representing ANOVA F statistic and associated probability value.

        :example:
        >>> sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
        >>> sample_2 = np.array([8, 5, 5, 8, 8, 9, 10, 1, 7, 10, 10])
        >>> Statistics().one_way_anova(sample_1=sample_2, sample_2=sample_1)
        >>> (5.848375451263537, 0.025253351261409433)
        """

        f, p = stats.f_oneway(sample_1, sample_2)
        return f, p

    @staticmethod
    @jit(nopython=True)
    def rolling_one_way_anova(data: np.ndarray,
                              time_windows: np.ndarray,
                              fps: int) -> np.ndarray:

        """
        Jitted compute of rolling one-way ANOVA F-statistic comparing the current time-window of
        size N to the preceding window of size N.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter np.ndarray[ints] time_windows: Time windows to compute ANOVAs for in seconds.
        :parameter int fps: Frame-rate of recorded video.

        :example:
        >>> sample = np.random.normal(loc=10, scale=1, size=10)
        >>> Statistics().rolling_one_way_anova(data=sample, window_sizes=np.array([1]), fps=2)
        >>> [[0.00000000e+00][0.00000000e+00][2.26221263e-06][2.26221263e-06][5.39119950e-03][5.39119950e-03][1.46725486e-03][1.46725486e-03][1.16392111e-02][1.16392111e-02]]
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(data, list(range(window_size, data.shape[0], window_size)))
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j-1].astype(np.float32), data_split[j].astype(np.float32)
                overall_sample = np.concatenate((sample_1, sample_2))
                sample_1_mean, sample_2_mean, overall_mean = np.mean(sample_1), np.mean(sample_2), np.mean(overall_sample)
                sample_1_ssq, sample_2_ssq, overall_ssq = np.sum(sample_1**2), np.sum(sample_2**2), np.sum(overall_sample**2)
                within_group_ssq = sample_1_ssq + sample_2_ssq
                between_groups_ssq = (sample_1_mean - overall_mean)**2 +(sample_2_mean-overall_mean)**2
                total_dfg, between_groups_dfg = (sample_1.shape[0] + sample_2.shape[0]) - 1, 1
                within_group_dfg = total_dfg - between_groups_dfg
                mean_squares_between, mean_squares_within = between_groups_ssq /between_groups_dfg, within_group_ssq / within_group_dfg
                f = mean_squares_between / mean_squares_within
                results[window_start: window_end, i] = f

        return results

    def kullback_leibler_divergence(self,
                                    sample_1: np.ndarray,
                                    sample_2: np.ndarray,
                                    fill_value: int = 1,
                                    bucket_method: Literal['fd', 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] = 'auto') -> float:

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
        sample_1_hist = self._hist_1d(data=sample_1, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
        sample_2_hist = self._hist_1d(data=sample_2, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
        sample_1_hist[sample_1_hist == 0] = fill_value; sample_2_hist[sample_2_hist == 0] = fill_value
        sample_1_hist, sample_2_hist = sample_1_hist / np.sum(sample_1_hist), sample_2_hist / np.sum(sample_2_hist)
        return stats.entropy(pk=sample_1_hist, qk=sample_2_hist)

    def rolling_kullback_leibler_divergence(self,
                                            data: np.ndarray,
                                            time_windows: np.ndarray,
                                            fps: int,
                                            fill_value: int = 1,
                                            bucket_method: Literal['fd', 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] = 'auto') -> np.ndarray:

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
        >>> Statistics().rolling_kullback_leibler_divergence(data=data, window_sizes=np.array([1]), fps=2)
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(data, list(range(window_size, data.shape[0], window_size)))
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j-1].astype(np.float32), data_split[j].astype(np.float32)
                bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
                sample_1_hist = self._hist_1d(data=sample_1, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
                sample_2_hist = self._hist_1d(data=sample_2, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
                sample_1_hist[sample_1_hist == 0] = fill_value; sample_2_hist[sample_2_hist == 0] = fill_value
                sample_1_hist, sample_2_hist = sample_1_hist / np.sum(sample_1_hist), sample_2_hist / np.sum(sample_2_hist)
                kl = stats.entropy(pk=sample_1_hist, qk=sample_2_hist)
                results[window_start: window_end, i] = kl
        return results


    def jensen_shannon_divergence(self,
                                  sample_1: np.ndarray,
                                  sample_2: np.ndarray,
                                  bucket_method: Literal['fd', 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] = 'auto') -> float:

        """
        Compute Jensen-Shannon divergence between two distributions.
        Useful for (i) measure drift in datasets, and (ii) featurization of distribution shifts across
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
        sample_1_hist = self._hist_1d(data=sample_1, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
        sample_2_hist = self._hist_1d(data=sample_2, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
        mean_hist = np.mean([sample_1_hist, sample_2_hist], axis=0)
        kl_sample_1, kl_sample_2 = stats.entropy(pk=sample_1_hist, qk=mean_hist), stats.entropy(pk=sample_2_hist, qk=mean_hist)
        return (kl_sample_1 + kl_sample_2) / 2


    def rolling_jensen_shannon_divergence(self,
                                          data: np.ndarray,
                                          time_windows: np.ndarray,
                                          fps = int,
                                          bucket_method: Literal['fd', 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] = 'auto') -> np.ndarray:
        """
        Compute rolling Jensen-Shannon divergence comparing the current time-window of
        size N to the preceding window of size N.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter np.ndarray[ints] time_windows: Time windows to compute JS for in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(data, list(range(window_size, data.shape[0], window_size)))
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j-1].astype(np.float32), data_split[j].astype(np.float32)
                bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
                sample_1_hist = self._hist_1d(data=sample_1, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
                sample_2_hist = self._hist_1d(data=sample_2, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
                sample_1_hist, sample_2_hist = sample_1_hist / np.sum(sample_1_hist), sample_2_hist / np.sum(sample_2_hist)
                mean_hist = np.mean([sample_1_hist, sample_2_hist], axis=0)
                kl_sample_1, kl_sample_2 = stats.entropy(pk=sample_1_hist, qk=mean_hist), stats.entropy(pk=sample_2_hist, qk=mean_hist)
                js = (kl_sample_1 + kl_sample_2) / 2
                results[window_start: window_end, i] = js
        return results


    def wasserstein_distance(self,
                            sample_1: np.ndarray,
                            sample_2: np.ndarray,
                            bucket_method: Literal['fd', 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] = 'auto') -> float:

        """
        Compute Wasserstein distance between two distributions.

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
        sample_1_hist = self._hist_1d(data=sample_1, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
        sample_2_hist = self._hist_1d(data=sample_2, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
        sample_1_hist, sample_2_hist = sample_1_hist / np.sum(sample_1_hist), sample_2_hist / np.sum(sample_2_hist)
        return stats.wasserstein_distance(u_values=sample_1_hist, v_values=sample_2_hist)


    def rolling_wasserstein_distance(self,
                                     data: np.ndarray,
                                     time_windows: np.ndarray,
                                     fps: int,
                                     bucket_method: Literal['fd', 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] = 'auto') -> np.ndarray:

        """
        Compute rolling Wasserstein distance comparing the current time-window of
        size N to the preceding window of size N.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter np.ndarray[ints] time_windows: Time windows to compute JS for in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns np.ndarray: Size data.shape[0] x window_sizes.shape with Wasserstein distance. Columns represent different time windows.
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(data, list(range(window_size, data.shape[0], window_size)))
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j-1].astype(np.float32), data_split[j].astype(np.float32)
                bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
                sample_1_hist = self._hist_1d(data=sample_1, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
                sample_2_hist = self._hist_1d(data=sample_2, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
                sample_1_hist, sample_2_hist = sample_1_hist / np.sum(sample_1_hist), sample_2_hist / np.sum(sample_2_hist)
                w = stats.wasserstein_distance(u_values=sample_1_hist, v_values=sample_2_hist)
                results[window_start: window_end, i] = w

        return results

    def population_stability_index(self,
                                   sample_1: np.ndarray,
                                   sample_2: np.ndarray,
                                   fill_value: int = 1,
                                   bucket_method: Literal['fd', 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] = 'auto') -> float:
        """
        Compute Population Stability Index (PSI) comparing two distributions.

        .. note::
           Empty bins (0 observations in bin) in is replaced with ``fill_value``.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter int fill_value: Empty bins (0 observations in bin) in is replaced with ``fill_value``.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns float: PSI distance between ``sample_1`` and ``sample_2``
        """

        bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
        sample_1_hist = self._hist_1d(data=sample_1, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
        sample_2_hist = self._hist_1d(data=sample_2, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
        sample_1_hist[sample_1_hist == 0] = fill_value; sample_2_hist[sample_2_hist == 0] = fill_value
        sample_1_hist, sample_2_hist = sample_1_hist / np.sum(sample_1_hist), sample_2_hist / np.sum(sample_2_hist)

        samples_diff = sample_2_hist - sample_1_hist
        log = np.log(sample_2_hist / sample_1_hist)
        return np.sum(samples_diff * log)


    def rolling_population_stability_index(self,
                                           data: np.ndarray,
                                           time_windows: np.ndarray,
                                           fps: int,
                                           fill_value: int = 1,
                                           bucket_method: Literal['fd', 'doane', 'auto', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] = 'auto') -> np.ndarray:
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
            data_split = np.split(data, list(range(window_size, data.shape[0], window_size)))
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[j].astype(np.float32)
                bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
                sample_1_hist = self._hist_1d(data=sample_1, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
                sample_2_hist = self._hist_1d(data=sample_2, bin_count=bin_count, range=np.array([0, int(bin_width * bin_count)]))
                sample_1_hist[sample_1_hist == 0] = fill_value; sample_2_hist[sample_2_hist == 0] = fill_value
                sample_1_hist, sample_2_hist = sample_1_hist / np.sum(sample_1_hist), sample_2_hist / np.sum(sample_2_hist)
                samples_diff = sample_2_hist - sample_1_hist
                log = np.log(sample_2_hist / sample_1_hist)
                psi = np.sum(samples_diff * log)
                results[window_start: window_end, i] = psi

        return results

    def shapiro_wilks(self,
                      data: np.ndarray,
                      time_window: float,
                      fps: int) -> np.ndarray:
        """
        Compute Shapiro-Wilks normality statistics for sequentially binned values in a time-series. E.g., compute
        the normality statistics of ``Feature N`` in each window of ``time_window`` seconds.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter int time_window: The size of the buckets in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :return np.ndarray: Array of size data.shape[0] with Shapiro-Wilks normality statistics

        :example:
        >>> data = np.random.randint(low=0, high=100, size=(200)).astype('float32')
        >>> results = self.shapiro_wilks(data=data, time_window=1, fps=30)
        """

        window_size, results = int(time_window * fps), np.full((data.shape[0]), -1.0)
        data = np.split(data, list(range(window_size, data.shape[0], window_size)))
        for cnt, i in enumerate(prange(1, len(data))):
            start, end = int((cnt + 1) * window_size), int(((cnt + 1) * window_size) + window_size)
            results[start:end] = stats.shapiro(data[i])[0]
        return results


    @staticmethod
    @jit(nopython=True, cache=True)
    def kruskal_wallis(sample_1: np.ndarray,
                       sample_2: np.ndarray) -> float:

        """
        Jitted compute of Kruskal-Wallis H between two distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns float: Kruskal-Wallis H.

        :example:
        >>> sample_1 = np.array([1, 1, 3, 4, 5])
        >>> sample_2 = np.array([6, 7, 8, 9, 10])
        >>> results = Statistics().kruskal_wallis(sample_1=sample_1, sample_2=sample_2)
        """
        sample_1 = np.concatenate((np.zeros((sample_1.shape[0], 1)), sample_1.reshape(-1, 1)), axis=1)
        sample_2 = np.concatenate((np.ones((sample_2.shape[0], 1)), sample_2.reshape(-1, 1)), axis=1)
        data = np.vstack((sample_1, sample_2))
        ranks = fast_mean_rank(data=data[:, 1], descending=False)
        data = np.hstack((data, ranks.reshape(-1, 1)))
        sample_1_summed_rank = np.sum(data[0: sample_1.shape[0], 2].flatten())
        sample_2_summed_rank = np.sum(data[sample_1.shape[0]:, 2].flatten())
        h1 = 12 / (data.shape[0] * (data.shape[0] + 1))
        h2 = (np.square(sample_1_summed_rank) / sample_1.shape[0]) + (np.square(sample_2_summed_rank) / sample_2.shape[0])
        h3 = 3 * (data.shape[0] + 1)
        return  h1 * h2 - h3

    @staticmethod
    @jit(nopython=True, cache=True)
    def mann_whitney(sample_1: np.ndarray,
                     sample_2: np.ndarray) -> float:

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
        >>> results = Statistics().kruskal_wallis(sample_1=sample_1, sample_2=sample_2)
        """

        n1, n2 = sample_1.shape[0], sample_2.shape[0]
        ranked = fast_mean_rank(np.concatenate((sample_1, sample_2)))
        u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(ranked[:n1], axis=0)
        u2 = n1 * n2 - u1
        return min(u1, u2)

    @staticmethod
    @jit(nopython=True, cache=True)
    def rolling_mann_whitney(data: np.ndarray,
                             time_windows: np.ndarray,
                             fps: float):

        """
        Jitted compute of rolling Mann-Whitney U comparing the current time-window of
        size N to the preceding window of size N.

        .. note::
           First time bin (where has no preceding time bin) will have fill value ``0``

           `Modified from James Webber gist <https://gist.github.com/jamestwebber/38ab26d281f97feb8196b3d93edeeb7b>`__.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns np.ndarray: Mann-Whitney U data of size len(data) x len(time_windows).
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            data_split = np.split(data, list(range(window_size, data.shape[0], window_size)))
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[j].astype(np.float32)
                n1, n2 = sample_1.shape[0], sample_2.shape[0]
                ranked = fast_mean_rank(np.concatenate((sample_1, sample_2)))
                u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(ranked[:n1], axis=0)
                u2 = n1 * n2 - u1
                u = min(u1, u2)
                results[window_start: window_end, i] = u

        return results


    @staticmethod
    @jit(nopython=True, cache=True)
    def levenes(sample_1: np.ndarray,
                sample_2: np.ndarray) -> float:

        """
        Jitted compute of two-sample Leven's W.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns float: Leven's W.

        :examples:
        >>> sample_1 = np.array(list(range(0, 50)))
        >>> sample_2 = np.array(list(range(25, 100)))
        >>> Statistics().levenes(sample_1=sample_1, sample_2=sample_2)
        >>> 12.63909108903254
        """
        Ni_x, Ni_y = len(sample_1), len(sample_2)
        Yci_x, Yci_y = np.median(sample_1), np.median(sample_2)
        Ntot = Ni_x + Ni_y

        Zij_x, Zij_y = np.abs(sample_1 - Yci_x).astype(np.float32),  np.abs(sample_2 - Yci_y).astype(np.float32)
        Zbari_x, Zbari_y = np.mean(Zij_x), np.mean(Zij_y)
        Zbar = ((Zbari_x * Ni_x) + (Zbari_y * Ni_y)) / Ntot
        numer = (Ntot - 2) * np.sum(np.array([Ni_x, Ni_y]) * (np.array([Zbari_x, Zbari_y]) - Zbar) ** 2)
        dvar = np.sum((Zij_x - Zbari_x) ** 2) + np.sum((Zij_y - Zbari_y) ** 2)
        denom = (2 - 1.0) * dvar
        return numer / denom

    @staticmethod
    @njit('(float64[:], float64[:], float64)', cache=True)
    def rolling_levenes(data: np.ndarray,
                        time_windows: np.ndarray,
                        fps: float):

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
            data_split = np.split(data, list(range(window_size, data.shape[0], window_size)))
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[j].astype(np.float32)
                Ni_x, Ni_y = len(sample_1), len(sample_2)
                Yci_x, Yci_y = np.median(sample_1), np.median(sample_2)
                Ntot = Ni_x + Ni_y
                Zij_x, Zij_y = np.abs(sample_1 - Yci_x).astype(np.float32), np.abs(sample_2 - Yci_y).astype(np.float32)
                Zbari_x, Zbari_y = np.mean(Zij_x), np.mean(Zij_y)
                Zbar = ((Zbari_x * Ni_x) + (Zbari_y * Ni_y)) / Ntot
                numer = (Ntot - 2) * np.sum(np.array([Ni_x, Ni_y]) * (np.array([Zbari_x, Zbari_y]) - Zbar) ** 2)
                dvar = np.sum((Zij_x - Zbari_x) ** 2) + np.sum((Zij_y - Zbari_y) ** 2)
                denom = (2 - 1.0) * dvar
                w = numer / denom
                results[window_start: window_end, i] = w
        return results


    @staticmethod
    @jit(nopython=True, cache=True)
    def brunner_munzel(sample_1: np.ndarray,
                       sample_2: np.ndarray) -> float:
        """
        Jitted compute of Brunner-Munzel W between two distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns float: Brunner-Munzel W.

        .. note::
           Modified from ``scipy.stats.brunnermunzel``.

        :example:
        >>> sample_1, sample_2 = np.random.normal(loc=10, scale=2, size=10), np.random.normal(loc=20, scale=2, size=10)
        >>> Statistics().kruskal_wallis(sample_1=sample_1, sample_2=sample_2)
        >>> 0.5751408161437165
        """
        nx, ny = len(sample_1), len(sample_2)
        rankc = fast_mean_rank(np.concatenate((sample_1, sample_2)))
        rankcx, rankcy = rankc[0:nx], rankc[nx:nx + ny]
        rankcx_mean, rankcy_mean = np.mean(rankcx), np.mean(rankcy)
        rankx, ranky = fast_mean_rank(sample_1), fast_mean_rank(sample_2)
        rankx_mean, ranky_mean = np.mean(rankx), np.mean(ranky)
        Sx = np.sum(np.power(rankcx - rankx - rankcx_mean + rankx_mean, 2.0)) / nx - 1
        Sy = np.sum(np.power(rankcy - ranky - rankcy_mean + ranky_mean, 2.0)) / ny - 1
        wbfn = nx * ny * (rankcy_mean - rankcx_mean)
        wbfn /= (nx + ny) * np.sqrt(nx * Sx + ny * Sy)
        return -wbfn


    @staticmethod
    @jit(nopython=True)
    def sliding_independent_samples_t(data: np.ndarray,
                                      time_window: float,
                                      slide_time: float,
                                      critical_values: np.ndarray,
                                      fps: float) -> np.ndarray:
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
        :parameter ndarray critical_values: 1D array with where indexes represent degrees of freedom and values
                                            represent critical T values. Can be found in ``simba.assets.critical_values_05.pickle``.
        :parameter int fps: The fps of the recorded video.
        :returns np.ndarray: 1D array of size len(data) with values representing time to most recent significantly different feature distribution.

        :example:
        >>> data = np.random.randint(0, 50, (10)).astype(np.float64)
        >>> critical_values = pickle.load(open( "simba/assets/lookups/critical_values_05.pickle","rb"))['independent_t_test']['one_tail'].values
        >>> results = Statistics().sliding_independent_samples_t(data=data, time_window=1, fps=10, critical_values=critical_values, slide_time=0.30)
        """

        results = np.full((data.shape[0]), 0.0)
        window_size, slide_size = int(time_window * fps), int(slide_time * fps)
        for i in range(1, data.shape[0]):
            sample_1_left, sample_1_right = i, i+window_size
            sample_2_left, sample_2_right = sample_1_left-slide_size, sample_1_right-slide_size
            sample_1 = data[sample_1_left:sample_1_right]
            dof, steps_taken = (sample_1.shape[0] + sample_1.shape[0]) - 2, 1
            while sample_2_left >= 0:
                sample_2 = data[sample_2_left:sample_2_right]
                t_statistic = (np.mean(sample_1) - np.mean(sample_2)) / np.sqrt((np.std(sample_1) / sample_1.shape[0]) + (np.std(sample_2) / sample_1.shape[0]))
                critical_val = critical_values[dof-1][1]
                if t_statistic >= critical_val:
                    break
                else:
                    sample_2_left -= 1; sample_2_right -= 1; steps_taken += 1
                if sample_2_left < 0:
                    steps_taken = -1
            if steps_taken == -1:
                results[i + window_size] = -1
            else:
                results[i+window_size] = steps_taken * slide_time

        return results

# data = np.random.randint(0, 50, (20)).astype(np.float64)
# import pickle
# critical_values = pickle.load(open( "/Users/simon/Desktop/envs/simba_dev/simba/assets/lookups/critical_values_05.pickle","rb"))['independent_t_test']['one_tail'].values
# test = Statistics().sliding_independent_samples_t(data=data, time_window=1, fps=2, critical_values=critical_values, slide_time=1)





# sample_2 = np.random.randint(25, 50, (100)).astype(np.float64)
# start = time.time()
# for i in range(1000000):
#     test = Statistics().levenes(sample_1=sample_1, sample_2=sample_2)
# print(time.time() - start)


