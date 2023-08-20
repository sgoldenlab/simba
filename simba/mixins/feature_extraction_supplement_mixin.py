__author__ = "Simon Nilsson"

import time

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import numpy as np
from numba import jit, prange
from scipy import stats

from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.data import bucket_data, freedman_diaconis


class FeatureExtractionSupplemental(FeatureExtractionMixin):

    """
    Additional feature extraction method not called by default feature extraction classes.
    Primarily frequentist statistics methods.
    """

    def __init__(self):
        FeatureExtractionMixin.__init__(self)

    @staticmethod
    @jit(nopython=True)
    def _helper_euclidean_distance_timeseries_change(
        distances: np.ndarray, time_windows: np.ndarray, fps: int
    ):
        """
        Private jitted helper called by ``simba.mixins.feature_extraction_supplemental_mixin.FeatureExtractionSupplemental.euclidean_distance_timeseries_change``
        """
        results = np.full((distances.shape[0], time_windows.shape[0]), np.nan)
        for window_cnt in prange(time_windows.shape[0]):
            frms = int(time_windows[window_cnt] * fps)
            shifted_distances = np.copy(distances)
            shifted_distances[0:frms] = np.nan
            shifted_distances[frms:] = distances[:-frms]
            shifted_distances[np.isnan(shifted_distances)] = distances[
                np.isnan(shifted_distances)
            ]
            results[:, window_cnt] = distances - shifted_distances

        return results

    @staticmethod
    @jit(nopython=True)
    def hist_1d(data: np.ndarray, bin_count: int, range: np.ndarray):
        """
        Jitted helper to compute 1D histograms with counts.

        :parameter np.ndarray data: 1d array containing feature values.
        :parameter int bins: The number of bins.
        :parameter: np.ndarray range: 1d array with two values representing minimum and maximum value to bin.
        """

        hist = np.histogram(data, bin_count, (range[0], range[1]))[0]
        return hist

    def euclidean_distance_timeseries_change(
        self,
        location_1: np.ndarray,
        location_2: np.ndarray,
        fps: int,
        px_per_mm: float,
        time_windows: np.ndarray = np.array([0.2, 0.4, 0.8, 1.6]),
    ) -> np.ndarray:
        """
        Compute the difference in distance between two points in the current frame versus N.N seconds ago. E.g.,
        computes if two points are traveling away from each other (positive output values) or towards each other
        (negative output values) relative to reference time-point(s)

        .. image:: _static/img/euclid_distance_change.png
           :width: 700
           :align: center

        :parameter ndarray location_1: 2D array of size len(frames) x 2 representing pose-estimated locations of body-part one
        :parameter ndarray location_2: 2D array of size len(frames) x 2 representing pose-estimated locations of body-part two
        :parameter int fps: Fps of the recorded video.
        :parameter float px_per_mm: The pixels per millimeter in the video.
        :parameter np.ndarray time_windows: Time windows to compare.
        :return np.array: Array of size location_1.shape[0] x time_windows.shape[0]

        :example:
        >>> location_1 = np.random.randint(low=0, high=100, size=(2000, 2)).astype('float32')
        >>> location_2 = np.random.randint(low=0, high=100, size=(2000, 2)).astype('float32')
        >>> distances = self.euclidean_distance_timeseries_change(location_1=location_1, location_2=location_2, fps=10, px_per_mm=4.33, time_windows=np.array([0.2, 0.4, 0.8, 1.6]))
        """
        distances = self.framewise_euclidean_distance(
            location_1=location_1, location_2=location_2, px_per_mm=px_per_mm
        )
        return self._helper_euclidean_distance_timeseries_change(
            distances=distances, fps=fps, time_windows=time_windows
        ).astype(int)

    @staticmethod
    @jit(nopython=True)
    def rolling_independent_sample_t(
        data: np.ndarray, group_size_s: int, fps: int
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
        >>> FeatureExtractionSupplemental().rolling_independent_sample_t(data, group_size_s=1, fps=10)
        >>> [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389, -6.88741389])

        """

        results = np.full((data.shape[0]), -1.0)
        window_size = int(group_size_s * fps)
        data = np.split(data, list(range(window_size, data.shape[0], window_size)))
        for cnt, i in enumerate(prange(1, len(data))):
            start, end = int((cnt + 1) * window_size), int(
                ((cnt + 1) * window_size) + window_size
            )
            mean_1, mean_2 = np.mean(data[i - 1]), np.mean(data[i])
            stdev_1, stdev_2 = np.std(data[i - 1]), np.std(data[i])
            results[start:end] = (mean_1 - mean_2) / np.sqrt(
                (stdev_1 / data[i - 1].shape[0]) + (stdev_2 / data[i].shape[0])
            )
        return results

    def independent_samples_t(
        self, sample_1: np.ndarray, sample_2: np.ndarray
    ) -> (float, float):
        """
        Compute independent-samples t-test statistic and p-values between two distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns (float float) t_statistic, p_value: Representing t-statistic and associated probability value.

        :example:
        >>> sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
        >>> sample_2 = np.array([8, 7, 5, 3, 8, 9, 6, 10, 7, 10, 10])
        >>> FeatureExtractionSupplemental().independent_samples_t(sample_1=sample_1, sample_2=sample_2)
        >>> (-4.877830567527609, 9.106532603464572e-05)
        """
        t_statistic = (np.mean(sample_1) - np.mean(sample_2)) / np.sqrt(
            (np.std(sample_1) / sample_1.shape[0])
            + (np.std(sample_2) / sample_1.shape[0])
        )
        dof = (sample_1.shape[0] + sample_2.shape[0]) - 2
        p_value = 2 * (stats.t.cdf(-abs(t_statistic), dof))
        return t_statistic, p_value

    def cohens_d(self, sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Compute Cohen's d effect size between two distributions

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns float: Cohens D statistic.

        :example:
        >>> sample_1 = np.array([8, 1, 5, 1, 8, 1, 1, 10, 7, 10, 10])
        >>> sample_2 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
        >>> FeatureExtractionSupplemental().cohens_d(sample_1=sample_1, sample_2=sample_2)
        >>> 0.41197143155579974
        """

        dof = (sample_1.shape[0] + sample_2.shape[0]) - 2
        return (np.mean(sample_1) - np.mean(sample_2)) / np.sqrt(
            (
                (sample_1.shape[0] - 1) * np.std(sample_1, ddof=1) ** 2
                + (sample_2.shape[0] - 1) * np.std(sample_2, ddof=1) ** 2
            )
            / dof
        )

    @staticmethod
    @jit(nopython=True)
    def rolling_cohens_d(
        data: np.ndarray, window_sizes: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Jitted compute of rolling Cohen's D statistic comparing the current time-window of
        size N to the preceding window of size N.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter np.ndarray[ints] window_sizes: Time windows to compute ANOVAs for in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :returns np.ndarray: Array of size data.shape[0] x window_sizes.shape[1] with Cohens D.

        :example:
        >>> sample_1, sample_2 = np.random.normal(loc=10, scale=1, size=4), np.random.normal(loc=11, scale=2, size=4)
        >>> sample = np.hstack((sample_1, sample_2))
        >>> FeatureExtractionSupplemental().rolling_cohens_d(data=sample, window_sizes=np.array([1]), fps=4)
        >>> [[0.],[0.],[0.],[0.],[0.14718302],[0.14718302],[0.14718302],[0.14718302]])
        """

        results = np.full((data.shape[0], window_sizes.shape[0]), 0.0)
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * fps)
            data_split = np.split(
                data, list(range(window_size, data.shape[0], window_size))
            )
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[
                    j
                ].astype(np.float32)
                dof = (sample_1.shape[0] + sample_2.shape[0]) - 2
                mean_sample_1, mean_sample_2 = np.mean(sample_1), np.mean(sample_2)
                std_sample_1 = (
                    np.sum((sample_1 - mean_sample_1) ** 2) / sample_1.shape[0]
                )
                std_sample_2 = (
                    np.sum((sample_2 - mean_sample_2) ** 2) / sample_2.shape[0]
                )
                d = (mean_sample_1 - mean_sample_2) / np.sqrt(
                    (
                        (sample_1.shape[0] - 1) * std_sample_1**2
                        + (sample_2.shape[0] - 1) * std_sample_2**2
                    )
                    / dof
                )
                results[window_start:window_end, i] = d
        return results

    def rolling_two_sample_ks(
        self, data: np.ndarray, group_size_s: int, fps: int
    ) -> np.ndarray:
        """
        Compute Kolmogorov two-sample statistics for sequentially binned values in a time-series.
        E.g., compute KS statistics when comparing ``Feature N`` in the current 1s time-window, versus ``Feature N`` in the previous 1s time-window.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter int group_size_s: The size of the buckets in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :return np.ndarray: Array of size data.shape[0] with KS statistics

        :example:
        >>> data = np.random.randint(low=0, high=100, size=(200)).astype('float32')
        >>> results = self.rolling_two_sample_ks(data=data, group_size_s=1, fps=30)
        """

        window_size, results = int(group_size_s * fps), np.full((data.shape[0]), -1.0)
        data = np.split(data, list(range(window_size, data.shape[0], window_size)))
        for cnt, i in enumerate(prange(1, len(data))):
            start, end = int((cnt + 1) * window_size), int(
                ((cnt + 1) * window_size) + window_size
            )
            results[start:end] = stats.ks_2samp(
                data1=data[i - 1], data2=data[i]
            ).statistic
        return results

    def two_sample_ks(
        self, sample_1: np.ndarray, sample_2: np.ndarray
    ) -> (float, float):
        """
        Compute Kolmogorov-Smirnov statistics and p-value for two distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns (float float): Representing Kolmogorov-Smirnov statistic and associated probability value.

        :example:
        >>> sample_1 = np.array([8, 1, 5, 1, 8, 1, 1, 10, 7, 10, 10])
        >>> sample_2 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
        >>> FeatureExtractionSupplemental().two_sample_ks(sample_1=sample_1, sample_2=sample_2)
        >>>
        """

        ks = stats.ks_2samp(data1=sample_1, data2=sample_2)
        return (
            ks.statistic,
            ks.pvalue,
        )

    def one_way_anova(
        self, sample_1: np.ndarray, sample_2: np.ndarray
    ) -> (float, float):
        """
        Compute One-way ANOVA F statistics and associated p-value for two distributions.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :returns (float float): Representing ANOVA F statistic and associated probability value.

        :example:
        >>> sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
        >>> sample_2 = np.array([8, 5, 5, 8, 8, 9, 10, 1, 7, 10, 10])
        >>> FeatureExtractionSupplemental().one_way_anova(sample_1=sample_2, sample_2=sample_1)
        >>> (5.848375451263537, 0.025253351261409433)
        """

        f, p = stats.f_oneway(sample_1, sample_2)
        return f, p

    @staticmethod
    @jit(nopython=True)
    def rolling_one_way_anova(
        data: np.ndarray, window_sizes: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Jitted compute of rolling one-way ANOVA F-statistic comparing the current time-window of
        size N to the preceding window of size N.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter np.ndarray[ints] window_sizes: Time windows to compute ANOVAs for in seconds.
        :parameter int fps: Frame-rate of recorded video.

        :example:
        >>> sample = np.random.normal(loc=10, scale=1, size=10)
        >>> FeatureExtractionSupplemental().rolling_one_way_anova(data=sample, window_sizes=np.array([1]), fps=2)
        >>> [[0.00000000e+00][0.00000000e+00][2.26221263e-06][2.26221263e-06][5.39119950e-03][5.39119950e-03][1.46725486e-03][1.46725486e-03][1.16392111e-02][1.16392111e-02]]
        """

        results = np.full((data.shape[0], window_sizes.shape[0]), 0.0)
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * fps)
            data_split = np.split(
                data, list(range(window_size, data.shape[0], window_size))
            )
            for j in prange(1, len(data_split)):
                window_start = int(window_size * j)
                window_end = int(window_start + window_size)
                sample_1, sample_2 = data_split[j - 1].astype(np.float32), data_split[
                    j
                ].astype(np.float32)
                overall_sample = np.concatenate((sample_1, sample_2))
                sample_1_mean, sample_2_mean, overall_mean = (
                    np.mean(sample_1),
                    np.mean(sample_2),
                    np.mean(overall_sample),
                )
                sample_1_ssq, sample_2_ssq, overall_ssq = (
                    np.sum(sample_1**2),
                    np.sum(sample_2**2),
                    np.sum(overall_sample**2),
                )
                within_group_ssq = sample_1_ssq + sample_2_ssq
                between_groups_ssq = (sample_1_mean - overall_mean) ** 2 + (
                    sample_2_mean - overall_mean
                ) ** 2
                total_dfg, between_groups_dfg = (
                    sample_1.shape[0] + sample_2.shape[0]
                ) - 1, 1
                within_group_dfg = total_dfg - between_groups_dfg
                mean_squares_between, mean_squares_within = (
                    between_groups_ssq / between_groups_dfg,
                    within_group_ssq / within_group_dfg,
                )
                f = mean_squares_between / mean_squares_within
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
        sample_1_hist = self.hist_1d(
            data=sample_1,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
        )
        sample_2_hist = self.hist_1d(
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
        window_sizes: np.ndarray,
        fps: int,
        fill_value: int = 1,
        bucket_method: Literal[
            "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"
        ] = "auto",
    ) -> np.ndarray:
        """
        Jitted compute of rolling Kullback-Leibler divergence comparing the current time-window of
        size N to the preceding window of size N.

        .. note::
           Empty bins (0 observations in bin) in is replaced with ``fill_value``.

        :parameter ndarray sample_1: 1d array representing feature values.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns np.ndarray: Size data.shape[0] x window_sizes.shape with Kullback-Leibler divergence. Columns represents different tiem windows.

        :example:
        >>> sample_1, sample_2 = np.random.normal(loc=10, scale=700, size=5), np.random.normal(loc=50, scale=700, size=5)
        >>> data = np.hstack((sample_1, sample_2))
        >>> FeatureExtractionSupplemental().rolling_kullback_leibler_divergence(data=data, window_sizes=np.array([1]), fps=2)
        """

        results = np.full((data.shape[0], window_sizes.shape[0]), 0.0)
        for i in prange(window_sizes.shape[0]):
            window_size = int(window_sizes[i] * fps)
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
                sample_1_hist = self.hist_1d(
                    data=sample_1,
                    bin_count=bin_count,
                    range=np.array([0, int(bin_width * bin_count)]),
                )
                sample_2_hist = self.hist_1d(
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
        Compute Jensen-Shannon divergence between two distributions.
        Useful for (i) measure drift in datasets, and (ii) featurization of distribution shifts across
        sequential time-bins.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators.
        :returns float: Jensen-Shannon divergence between ``sample_1`` and ``sample_2``

        :example:
        >>> sample_1, sample_2 = np.array([1, 2, 3, 4, 5, 10, 1, 2, 3]), np.array([1, 5, 10, 9, 10, 1, 10, 6, 7])
        >>> FeatureExtractionSupplemental().jensen_shannon_divergence(sample_1=sample_1, sample_2=sample_2, bucket_method='fd')
        >>> 0.30806541358219786
        """

        bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
        sample_1_hist = self.hist_1d(
            data=sample_1,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
        )
        sample_2_hist = self.hist_1d(
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
        Jitted compute of rolling Jensen-Shannon divergence comparing the current time-window of
        size N to the preceding window of size N.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter np.ndarray[ints] window_sizes: Time windows to compute JS for in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
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
                sample_1_hist = self.hist_1d(
                    data=sample_1,
                    bin_count=bin_count,
                    range=np.array([0, int(bin_width * bin_count)]),
                )
                sample_2_hist = self.hist_1d(
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

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns float: Wasserstein distance between ``sample_1`` and ``sample_2``
        :example:

        >>> sample_1 = np.random.normal(loc=10, scale=2, size=10)
        >>> sample_2 = np.random.normal(loc=10, scale=3, size=10)
        >>> FeatureExtractionSupplemental().wasserstein_distance(sample_1=sample_1, sample_2=sample_2)
        >>> 0.020833333333333332
        """

        bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
        sample_1_hist = self.hist_1d(
            data=sample_1,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
        )
        sample_2_hist = self.hist_1d(
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

    def population_stability_index(
        self,
        sample_1: np.ndarray,
        sample_2: np.ndarray,
        fill_value: int = 1,
        bucket_method: Literal[
            "fd", "doane", "auto", "scott", "stone", "rice", "sturges", "sqrt"
        ] = "auto",
    ) -> float:
        """
        Compute population stability index comparing two distributions.

        .. note::
           Empty bins (0 observations in bin) in is replaced with ``fill_value``.

        :parameter ndarray sample_1: First 1d array representing feature values.
        :parameter ndarray sample_2: Second 1d array representing feature values.
        :parameter int fill_value: Empty bins (0 observations in bin) in is replaced with ``fill_value``.
        :parameter Literal bucket_method: Estimator determining optimal bucket count and bucket width. Default: The maximum of the Sturges and Freedman-Diaconis estimators
        :returns float: PSI distance between ``sample_1`` and ``sample_2``
        """

        bin_width, bin_count = bucket_data(data=sample_1, method=bucket_method)
        sample_1_hist = self.hist_1d(
            data=sample_1,
            bin_count=bin_count,
            range=np.array([0, int(bin_width * bin_count)]),
        )
        sample_2_hist = self.hist_1d(
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

    def shapiro_wilks(self, data: np.ndarray, bin_size_s: int, fps: int) -> np.ndarray:
        """
        Compute Shapiro-Wilks normality statistics for sequentially binned values in a time-series. E.g., compute
        the normality statistics of ``Feature N`` in each window of ``group_size_s`` seconds.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter int group_size_s: The size of the buckets in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :return np.ndarray: Array of size data.shape[0] with Shapiro-Wilks normality statistics

        :example:
        >>> data = np.random.randint(low=0, high=100, size=(200)).astype('float32')
        >>> results = self.two_sample_ks(data=data, bin_size_s=1, fps=30)
        """

        window_size, results = int(bin_size_s * fps), np.full((data.shape[0]), -1.0)
        data = np.split(data, list(range(window_size, data.shape[0], window_size)))
        for cnt, i in enumerate(prange(1, len(data))):
            start, end = int((cnt + 1) * window_size), int(
                ((cnt + 1) * window_size) + window_size
            )
            results[start:end] = stats.shapiro(data[i])[0]
        return results

    @staticmethod
    @jit(nopython=True)
    def peak_ratio(data: np.ndarray, bin_size_s: int, fps: int):
        """
        Compute the ratio of peak values relative to number of values within each seqential
        time-period represented of ``bin_size_s`` seconds. Peak is defined as value is higher than
        in the prior observation (i.e., no future data is involved in comparison).

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter int bin_size_s: The size of the buckets in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :return np.ndarray: Array of size data.shape[0] with peak counts as ratio of len(frames).

        .. image:: _static/img/peak_cnt.png
           :width: 700
           :align: center

        :example:
        >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> FeatureExtractionSupplemental().peak_ratio(data=data, bin_size_s=1, fps=10)
        >>> [0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9]
        >>> data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        >>> FeatureExtractionSupplemental().peak_ratio(data=data, bin_size_s=1, fps=10)
        >>> [0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.9 0.  0.  0.  0.  0.  0.  0.  0. 0.  0. ]
        """

        window_size, results = int(bin_size_s * fps), np.full((data.shape[0]), -1.0)
        data = np.split(data, list(range(window_size, data.shape[0], window_size)))
        start, end = 0, data[0].shape[0]
        for i in prange(len(data)):
            peak_cnt = 0
            if data[i][0] > data[i][1]:
                peak_cnt += 1
            if data[i][-1] > data[i][-2]:
                peak_cnt += 1
            for j in prange(1, len(data[i]) - 1):
                if data[i][j] > data[i][j - 1]:
                    peak_cnt += 1
            peak_ratio = peak_cnt / data[i].shape[0]
            results[start:end] = peak_ratio
            start, end = start + len(data[i]), end + len(data[i])
        return results

    @staticmethod
    @jit(nopython=True)
    def rolling_peak_count_ratio(data: np.ndarray, time_windows: np.ndarray, fps: int):
        results = np.full((data.shape[0], time_windows.shape[0]), -1.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            for j in prange(window_size, data.shape[0]):
                window_data = data[j - window_size : j]
                peak_cnt = 0
                if window_data[0] > window_data[1]:
                    peak_cnt += 1
                if window_data[-1] > window_data[-2]:
                    peak_cnt += 1
                for k in prange(1, len(window_data) - 1):
                    if window_data[j] > window_data[j - 1]:
                        peak_cnt += 1
                peak_ratio = peak_cnt / window_data.shape[0]
                results[j, i] = peak_ratio
        print(results)

    @staticmethod
    @jit(nopython=True)
    def rolling_categorical_switches_ratio(
        data: np.ndarray, time_windows: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Compute the ratio of in categorical feature switches within rolling windows.

        :parameter np.ndarray data: 1d array of feature values
        :parameter np.ndarray time_windows: Rolling time-windows as floats in seconds. E.g., [0.2, 0.4, 0.6]
        :parameter int fps: fps of the recorded video
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] array

        .. attention::
           Output for initial frames where [current_frm - window_size] < 0, are populated with ``0``.

        .. image:: _static/img/feature_switches.png
           :width: 700
           :align: center

        :example:
        >>> data = np.array([0, 1, 1, 1, 4, 5, 6, 7, 8, 9])
        >>> FeatureExtractionSupplemental().rolling_categorical_switches_ratio(data=data, time_windows=np.array([1.0]), fps=10)
        >>> [[-1][-1][-1][-1][-1][-1][-1][-1][-1][ 0.7]]
        >>> data = np.array(['A', 'B', 'B', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        >>> FeatureExtractionSupplemental().rolling_categorical_switches_ratio(data=data, time_windows=np.array([1.0]), fps=10)
        >>> [[-1][-1][-1][-1][-1][-1][-1][-1][-1][ 0.7]]
        """

        results = np.full((data.shape[0], time_windows.shape[0]), -1.0)
        for time_window in prange(time_windows.shape[0]):
            jump_frms = int(time_windows[time_window] * fps)
            for current_frm in prange(jump_frms, data.shape[0] + 1):
                time_slice = data[current_frm - jump_frms : current_frm]
                current_value, unique_cnt = time_slice[0], 0
                for i in prange(1, time_slice.shape[0]):
                    if time_slice[i] != current_value:
                        unique_cnt += 1
                    current_value = time_slice[i]
                print(unique_cnt, time_slice.shape[0])
                results[current_frm - 1][time_window] = unique_cnt / time_slice.shape[0]
        return results

    @staticmethod
    @jit(nopython=True)
    def consecutive_time_series_categories_count(data: np.ndarray, fps: int):
        """
        Compute the count of consecutive milliseconds the feature value has remained static. For example,
        compute for how long in milleseconds the animal has remained in the current cardinal direction or the
        within an ROI.

        .. image:: _static/img/categorical_consecitive_time.png
           :width: 700
           :align: center

        :parameter np.ndarray data: 1d array of feature values
        :parameter int fps: Frame-rate of video.
        :returns np.ndarray: Array of size data.shape[0]

        :example:
        >>> data = np.array([0, 1, 1, 1, 4, 5, 6, 7, 8, 9])
        >>> FeatureExtractionSupplemental().consecutive_time_series_categories_count(data=data, fps=10)
        >>> [0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        >>> data = np.array(['A', 'B', 'B', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
        >>> [0.1, 0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        """

        results = np.full((data.shape[0]), 0.0)
        results[0] = 1
        for i in prange(1, data.shape[0]):
            if data[i] == data[i - 1]:
                results[i] = results[i - 1] + 1
            else:
                results[i] = 1

        return results / fps

    @staticmethod
    @jit(nopython=True)
    def rolling_horizontal_vs_vertical_movement(
        data: np.ndarray, pixels_per_mm: float, time_windows: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Compute the movement along the x-axis relative to the y-axis in rolling time bins.

        .. attention::
           Output for initial frames where [current_frm - window_size] < 0, are populated with ``0``.

        .. image:: _static/img/x_vs_y_movement.png
           :width: 700
           :align: center

        :parameter np.ndarray data: 2d array of size len(frames)x2 with body-part coordinates.
        :parameter int fps: FPS of the recorded video
        :parameter float pixels_per_mm: Pixels per millimeter of recorded video.
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] array
        :parameter np.ndarray time_windows: Rolling time-windows as floats in seconds. E.g., [0.2, 0.4, 0.6]
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0]. Greater values denote greater movement on x-axis relative to y-axis.

        :example:
        >>> data = np.array([[250, 250], [250, 250], [250, 250], [250, 500], [500, 500], 500, 500]]).astype(float)
        >>> FeatureExtractionSupplemental().rolling_horizontal_vs_vertical_movement(data=data, time_windows=np.array([1.0]), fps=2, pixels_per_mm=1)
        >>> [[  -1.][   0.][   0.][-250.][ 250.][   0.]]
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0)
        for time_window in prange(time_windows.shape[0]):
            jump_frms = int(time_windows[time_window] * fps)
            for current_frm in prange(jump_frms, results.shape[0] + 1):
                x_movement = (
                    np.sum(
                        np.abs(
                            np.ediff1d(data[current_frm - jump_frms : current_frm, 0])
                        )
                    )
                    / pixels_per_mm
                )
                y_movement = (
                    np.sum(
                        np.abs(
                            np.ediff1d(data[current_frm - jump_frms : current_frm, 1])
                        )
                    )
                    / pixels_per_mm
                )
                results[current_frm - 1][time_window] = x_movement - y_movement

        return results

    @staticmethod
    @jit(nopython=True)
    def border_distances(
        data: np.ndarray,
        pixels_per_mm: float,
        img_resolution: np.ndarray,
        time_window: float,
        fps: int,
    ):
        """
        Compute the mean distance of key-point to the left, right, top, and bottom sides of the image in
        rolling time-windows. Uses a straight line.

        .. image:: _static/img/border_distance.png
           :width: 700
           :align: center

        .. attention::
           Output for initial frames where [current_frm - window_size] < 0 will be populated with ``-1``.

        :parameter np.ndarray data: 2d array of size len(frames)x2 with body-part coordinates.
        :parameter np.ndarray img_resolution: Resolution of video in WxH format.
        :parameter float pixels_per_mm: Pixels per millimeter of recorded video.
        :parameter int fps: FPS of the recorded video
        :parameter float time_windows: Rolling time-window as floats in seconds. E.g., ``0.2``
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] array with millimeter distances from LEFT, RIGH, TOP, BOTTOM,

        :example:
        >>> data = np.array([[250, 250], [250, 250], [250, 250], [500, 500],[500, 500], [500, 500]]).astype(float)
        >>> img_resolution = np.array([500, 500])
        >>> FeatureExtractionSupplemental().border_distances(data=data, img_resolution=img_resolution, time_window=1, fps=2, pixels_per_mm=1)
        >>> [[-1, -1, -1, -1][250, 250, 250, 250][250, 250, 250, 250][375, 125, 375, 125][500, 0, 500, 0][500, 0, 500, 0]]
        """

        results = np.full((data.shape[0], 4), -1.0)
        window_size = int(time_window * fps)
        for current_frm in prange(window_size, results.shape[0] + 1):
            distances = np.full((4, window_size, 1), np.nan)
            windowed_locs = data[current_frm - window_size : current_frm]
            for bp_cnt, bp_loc in enumerate(windowed_locs):
                distances[0, bp_cnt] = np.linalg.norm(
                    np.array([0, bp_loc[1]]) - bp_loc
                )  # left
                distances[1, bp_cnt] = np.linalg.norm(
                    np.array([img_resolution[0], bp_loc[1]]) - bp_loc
                )  # right
                distances[2, bp_cnt] = np.linalg.norm(
                    np.array([bp_loc[0], 0]) - bp_loc
                )  # top
                distances[3, bp_cnt] = np.linalg.norm(
                    np.array([bp_loc[0], img_resolution[1]]) - bp_loc
                )  # bottpm
            for i in prange(4):
                results[current_frm - 1][i] = np.mean(distances[i]) / pixels_per_mm

        return results.astype(np.int32)

    @staticmethod
    @jit(nopython=True)
    def acceleration(data: np.ndarray, pixels_per_mm: float, fps: int):
        """
        Compute acceleration.

        :parameter np.ndarray data: 2d array of size len(frames)x2 with body-part coordinates.
        :parameter np.ndarray img_resolution: Resolution of video in HxW format.
        :parameter float pixels_per_mm: Pixels per millimeter of recorded video.
        :parameter int fps: FPS of the recorded video
        :parameter float time_windows: Rolling time-window as floats in seconds. E.g., ``0.2``
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] array with millimeter distances from TOP, BOTTOM, LEFT RIGHT

        :example:
        >>> data = np.random.randint(low=0, high=500, size=(231, 2)).astype('float32')
        >>> results = FeatureExtractionSupplemental().acceleration(data=data, pixels_per_mm=4.33, fps=10)
        """

        velocity, results = np.full((data.shape[0]), -1), np.full((data.shape[0]), -1)
        shifted_loc = np.copy(data)
        shifted_loc[0:fps] = np.nan
        shifted_loc[fps:] = data[:-fps]
        for i in prange(fps, shifted_loc.shape[0]):
            velocity[i] = np.linalg.norm(shifted_loc[i] - data[i]) / pixels_per_mm
        for current_frm in prange(fps, velocity.shape[0], fps):
            print(current_frm - fps, current_frm, current_frm, current_frm + fps)
            prior_window = np.mean(velocity[current_frm - fps : current_frm])
            current_window = np.mean(velocity[current_frm : current_frm + fps])
            results[current_frm : current_frm + fps] = current_window - prior_window
        return results


# sample_1 = np.random.normal(loc=10, scale=700, size=5)
# sample_2 = np.random.normal(loc=50, scale=700, size=5)
# data = np.hstack((sample_1, sample_2))
# results = FeatureExtractionSupplemental().rolling_kullback_leibler_divergence(data=data, window_sizes=np.array([1]), fps=2)


#
# sample_1 = np.array([1, 2, 3, 4, 5, 10, 1, 2, 3, 8, 9, 7, 1, 10, 1, 10, 1])
# results = FeatureExtractionSupplemental().rolling_peak_count_ratio(data=sample_1, time_windows=np.array([1]), fps=10)


# def rolling_peak_count_ratio_(data: np.ndarray,
#                               time_window: int,
#                               fps: int):


# sample_2 = np.array([1, 5, 10, 9, 10, 1, 10, 6, 7])
#
# start = time.time()
# for i in range(10000):
#     results = FeatureExtractionSupplemental().kullback_leibler_divergence(sample_1=sample_1, sample_2=sample_2, fill_value=1)
# print(results)
# print(time.time() - start)


#
# for i in range(0, 10000):
#     print(i)
#     results = FeatureExtractionSupplemental().kullback_leibler_divergence(sample_1=sample_1, sample_2=sample_2)
#


# sample_1 = np.random.normal(loc=10, scale=1, size=10)
# sample_2 = np.random.normal(loc=10, scale=2, size=10)
# results = FeatureExtractionSupplemental().population_stability_index(sample_1=sample_1, sample_2=sample_2)


# sample_1 = np.array([1, 2, 3, 1, 3, 2, 1, 10, 8, 4, 10])
# sample_2 = np.array([8, 5, 5, 8, 8, 9, 10, 1, 7, 10, 10])
# FeatureExtractionSupplemental().one_way_anova(sample_1=sample_2, sample_2=sample_1)

# data_1 = np.random.normal(loc=10, scale=2, size=10)
# data_2 = np.random.normal(loc=20, scale=2, size=10)
# data = np.hstack([data_1, data_2])
# results = FeatureExtractionSupplemental().timeseries_independent_sample_t(data, group_size_s=1, fps=10).astype()


# data = np.array([0, 1, 1, 1, 4, 5, 6, 7, 8, 9])
# #data = np.array(['A', 'B', 'B', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
# results = FeatureExtractionSupplemental().consecutive_time_series_categories_count(data=data, fps=10)
#

# data = np.array([0, 1, 1, 1, 4, 5, 6, 7, 8, 9, 10, 11, 11, 12, 13, 13, 1, 1, 1, 1])
# results = FeatureExtractionSupplemental().rolling_categorical_switches_ratio(data=data, time_windows=np.array([1.0]), fps=10)
# print(results)
# data = np.array(['A', 'B', 'B', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])
# results = FeatureExtractionSupplemental().rolling_categorical_switches_ratio(data=data, time_windows=np.array([1.0]), fps=10)
# print(results)


#
# data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# results = FeatureExtractionSupplemental().peak_ratio(data=data, bin_size_s=1, fps=10)
# print(results)
#
# data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
# results = FeatureExtractionSupplemental().peak_ratio(data=data, bin_size_s=1, fps=10)
# print(results)


# data = np.random.randint(low=0, high=100, size=(200)).astype('float32')
# results = FeatureExtractionSupplemental().peak_ratio(data=data, group_size_s=1, fps=30)


# data = np.array([[250, 250],
#                  [250, 250],
#                  [250, 250],
#                  [250, 500],
#                  [500, 500],
#                  [500, 500]]).astype(float)
#
# results = FeatureExtractionSupplemental().rolling_horizontal_vs_vertical_movement(data=data, time_windows=np.array([1.0]), fps=2, pixels_per_mm=1)
# print(results)


# data = np.array([[250, 250],
#                  [250, 250],
#                  [250, 250],
#                  [500, 500],
#                  [500, 500],
#                  [500, 500]]).astype(float)
# img_resolution = np.array([500, 500])
# results = FeatureExtractionSupplemental().border_distances(data=data, img_resolution=img_resolution, time_window=1, fps=2, pixels_per_mm=1)
# print(results)
# def border_distances(data: np.ndarray,
#                      pixels_per_mm: float,
#                      img_resolution: np.ndarray,
#                      time_window: float,
#                      fps: int):

#
#
#
#
#
# start = time.time()
# nose_loc = np.random.randint(low=0, high=500, size=(231, 2)).astype('float32')
# results = FeatureExtractionSupplemental().horizontal_vs_vertical_movement(data=nose_loc, pixels_per_mm=4.33, fps=10, time_windows=np.array([0.4]))


# results = FeatureExtractionSupplemental().border_distances(data=nose_loc, pixels_per_mm=4.33, fps=10, time_window=0.2, img_resolution=np.array([600, 400]))

# results = FeatureExtractionSupplemental().acceleration(data=nose_loc, pixels_per_mm=4.33, fps=10)


# left_ear_loc = np.random.randint(low=0, high=500, size=(10000, 2)).astype('float32')
# right_ear_loc = np.random.randint(low=0, high=500, size=(10000, 2)).astype('float32')
# angle_data = FeatureExtractionSupplemental().head_direction(nose_loc=nose_loc, left_ear_loc=left_ear_loc, right_ear_loc=right_ear_loc)

#
# degree_angles = np.random.randint(low=0, high=50, size=(1000)).astype('int')
# rotation = pd.DataFrame(list(FeatureExtractionSupplemental().degrees_to_compass_cardinal(degree_angles=degree_angles)))
# rotation = rotation[0].map(get_cardinality_lookup())
#
# #switches = FeatureExtractionSupplemental().rolling_categorical_switches(data=rotation.values, time_windows=np.array([0.4]), fps=10)
#
# static_count = FeatureExtractionSupplemental().consecutive_time_series_categories_count(data=rotation.values, fps=10)


# rolling_angular_dispersion = FeatureExtractionSupplemental().rolling_angular_dispersion(data=angle_data, time_windows=np.array([0.4]), fps=10)


# print(time.time() - start)


# # data = np.random.randint(low=0, high=100, size=(223)).astype('float32')
# # results = FeatureExtractionSupplemental().two_sample_ks(data=data, group_size_s=1, fps=30)
#
# # data = np.random.randint(low=0, high=100, size=(223)).astype('float32')
# # results = FeatureExtractionSupplemental().shapiro_wilks(data=data, group_size_s=1, fps=30)
#
# start = time.time()
# data = np.random.randint(low=0, high=100, size=(50000000)).astype('float32')
# results = FeatureExtractionSupplemental().peak_ratio(data=data, group_size_s=1, fps=10)
# print(time.time() - start)


# 100
#
# 2 * np.sqrt(100)
