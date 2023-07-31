import multiprocessing
import time
from typing import List

import numpy as np
import pandas as pd
from numba import jit, prange, typed
from scipy import stats

from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.lookups import get_cardinality_lookup


class FeatureExtractionSupplemental(FeatureExtractionMixin):
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

        :parameter ndarray location_1: 2D array of size len(frames) x 2 representing pose-estimated locations of body-part one
        :parameter ndarray location_2: 2D array of size len(frames) x 2 representing pose-estimated locations of body-part two
        :parameter int fps: Fps of the recorded video.
        :parameter float px_per_mm: The pixels per millimeter in the video.
        :parameter bool time_windows np.ndarray: Time windows to compare.
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
    def timeseries_independent_sample_t(
        data: np.ndarray, group_size_s: int, fps: int
    ) -> np.ndarray:
        """
        Compute independent-sample t-statistics for sequentially binned values in a time-series.
        E.g., compute t-test statistics when comparing ``Feature N`` in the current 1s
        time-window, versus ``Feature N`` in the previous 1s time-window.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter int group_size_s: The size of the buckets in seconds.
        :parameter int fps: Frame-rate of recorded video.

        :example:
        >>> data = np.random.randint(low=0, high=100, size=(200)).astype('float32')
        >>> results = FeatureExtractionSupplemental().timeseries_independent_sample_t(data=data, group_size_s=1, fps=30)
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

    def two_sample_ks(
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
        >>> results = self.two_sample_ks(data=data, group_size_s=1, fps=30)
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
        bin of frames of ``bin_size_s`` seconds.

        :parameter ndarray data: 1D array of size len(frames) representing feature values.
        :parameter int bin_size_s: The size of the buckets in seconds.
        :parameter int fps: Frame-rate of recorded video.
        :return np.ndarray: Array of size data.shape[0] with peak counts as ratio of len(frames).

        :example:
        >>> data = np.random.randint(low=0, high=100, size=(200)).astype('float32')
        >>> results = FeatureExtractionSupplemental().peak_ratio(data=data, group_size_s=1, fps=30)
        """

        window_size, results = int(bin_size_s * fps), np.full((data.shape[0]), -1.0)
        data = np.split(data, list(range(window_size, data.shape[0], window_size)))
        for cnt, i in enumerate(prange(len(data))):
            start, end = int((cnt + 1) * window_size), int(
                ((cnt + 1) * window_size) + window_size
            )
            results[start:end] = (
                np.sum(
                    (data[i] > np.roll(data[i], 1)) & (data[i] > np.roll(data[i], -1))
                )
                / data[i].shape[0]
            )
        return results

    @staticmethod
    @jit(nopython=True)
    def head_direction(
        nose_loc: np.ndarray, left_ear_loc: np.ndarray, right_ear_loc: np.ndarray
    ) -> np.ndarray:
        """
        Jitted helper to compute the degree angle of nose direction. Computes the angle in degrees left_ear <-> nose
        and right_ear_nose and returns the midpoint.

        :parameter ndarray nose_loc: 2D array of size len(frames)x2 representing nose coordinates
        :parameter ndarray left_ear_loc: 2D array of size len(frames)x2 representing left ear coordinates
        :parameter ndarray right_ear_loc: 2D array of size len(frames)x2 representing right ear coordinates
        :return np.ndarray: Array of size nose_loc.shape[0] with direction in degrees.

        :example:
        >>> nose_loc = np.random.randint(low=0, high=500, size=(50, 2)).astype('float32')
        >>> left_ear_loc = np.random.randint(low=0, high=500, size=(50, 2)).astype('float32')
        >>> right_ear_loc = np.random.randint(low=0, high=500, size=(50, 2)).astype('float32')
        >>> results = FeatureExtractionSupplemental().head_direction(nose_loc=nose_loc, left_ear_loc=left_ear_loc, right_ear_loc=right_ear_loc)
        """

        results = np.full((nose_loc.shape[0]), np.nan)
        for i in prange(nose_loc.shape[0]):
            left_ear_to_nose = np.degrees(
                np.arctan2(
                    left_ear_loc[i][0] - nose_loc[i][1],
                    left_ear_loc[i][1] - nose_loc[i][0],
                )
            )
            right_ear_nose = np.degrees(
                np.arctan2(
                    right_ear_loc[i][0] - nose_loc[i][1],
                    right_ear_loc[i][1] - nose_loc[i][0],
                )
            )
            results[i] = ((left_ear_to_nose + right_ear_nose) % 360) / 2
        return results

    @staticmethod
    @jit(nopython=True)
    def degrees_to_compass_cardinal(degree_angles: np.ndarray) -> List[str]:
        """
        Convert degree angles to cardinal direction bucket e.g., 0 -> "N", 180 -> "S"

        .. note::
           To convert the cardinal directionality literals to integers, map using ``simba.utils.enums.lookups.get_cardinality_lookup``.

        :parameter degree_angles nose_loc: 1d array of degrees. Note: return by ``self.head_direction``.
        :return List[str]: List of strings representing frame-wise cardinality

        """
        results = typed.List(["str"])
        DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        for i in prange(degree_angles.shape[0]):
            ix = round(degree_angles[i] / (360.0 / len(DIRECTIONS)))
            direction = DIRECTIONS[ix % len(DIRECTIONS)]
            results.append(direction)
        return results[1:]

    @staticmethod
    @jit(nopython=True)
    def rolling_categorical_switches(
        data: np.ndarray, time_windows: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Compute the count of in categorical feature switches within rolling windows.

        :parameter np.ndarray data: 1d array of feature values
        :parameter np.ndarray time_windows: Rolling time-windows as floats in seconds. E.g., [0.2, 0.4, 0.6]
        :parameter int fps: fps of the recorded video
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] array

        :example:
        >>> data = np.random.randint(low=1, high=10, size=(10)).astype('int')
        >>> switches = FeatureExtractionSupplemental().rolling_categorical_switches(data=data, time_windows=np.array([0.4]), fps=10)
        """

        results = np.full((data.shape[0], time_windows.shape[0]), -1)
        for time_window in prange(time_windows.shape[0]):
            jump_frms = int(time_windows[time_window] * fps)
            for current_frm in prange(jump_frms, data.shape[0]):
                time_slice = data[current_frm - jump_frms : current_frm]
                current_value, unique_cnt = time_slice[0], 0
                for i in prange(1, time_slice.shape[0]):
                    if time_slice[i] != current_value:
                        unique_cnt += 1
                    current_value = time_slice[i]
                results[current_frm][time_window] = unique_cnt
        return results

    @staticmethod
    @jit(nopython=True)
    def consecutive_time_series_categories_count(data: np.ndarray, fps: int):
        """
        Compute the count of consecutive milliseconds the feature value has remained static. For example,
        compute for how long in milleseconds the animal has remained in the current cardinal direction.

        :parameter np.ndarray data: 1d array of feature values
        :parameter int fps: Frame-rate of video.
        :returns np.ndarray: Array of size data.shape[0]

        :example:
        >>> data = np.random.randint(low=1, high=8, size=(10)).astype('int')
        >>> results = FeatureExtractionSupplemental().consecutive_time_series_categories_count(data=data)
        """

        results = np.full((data.shape[0]), 0.0)
        for i in prange(1, data.shape[0]):
            if data[i] == data[i - 1]:
                results[i] = results[i - 1] + 1
            else:
                results[i] = 0

        return results / fps

    @staticmethod
    def rolling_mean_dispersion(
        data: np.ndarray, time_windows: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Compute the angular mean dispersion in degrees within rolling temporal windows.

        :parameter np.ndarray data: 1d array with feature values in degrees.
        :parameter np.ndarray time_windows: Rolling time-windows as floats in seconds. E.g., [0.2, 0.4, 0.6]
        :parameter int fps: fps of the recorded video
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] array

        :example:
        >>> data = np.random.randint(low=0, high=8, size=(10)).astype('int')
        >>> results = FeatureExtractionSupplemental().rolling_mean_dispersion(data=angle_data, time_windows=np.array([0.4]), fps=10)
        """

        results = np.full((data.shape[0], time_windows.shape[0]), -1)
        for time_window in prange(time_windows.shape[0]):
            jump_frms = int(time_windows[time_window] * fps)
            for current_frm in prange(jump_frms, results.shape[0]):
                data_window = np.deg2rad(data[current_frm - jump_frms : current_frm])
                results[current_frm][time_window] = np.rad2deg(
                    stats.circmean(data_window)
                ).astype(int)
        return results

    @staticmethod
    @jit(nopython=True)
    def horizontal_vs_vertical_movement(
        data: np.ndarray, pixels_per_mm: float, time_windows: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Compute the movement along the x-axis relative to the y-axis in rolling time bins.

        :parameter np.ndarray data: 2d array of size len(frames)x2 with body-part coordinates.
        :parameter int fps: FPS of the recorded video
        :parameter float pixels_per_mm: Pixels per millimeter of recorded video.
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] array
        :parameter np.ndarray time_windows: Rolling time-windows as floats in seconds. E.g., [0.2, 0.4, 0.6]
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0]. Greater values denote greater movement on x-axis relative to y-axis.
        """

        results = np.full((data.shape[0], time_windows.shape[0]), -1.0)
        for time_window in prange(time_windows.shape[0]):
            jump_frms = int(time_windows[time_window] * fps)
            for current_frm in prange(jump_frms, results.shape[0]):
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
                results[current_frm][time_window] = x_movement - y_movement

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
        Compute the mean distance of key-point to the top, bottom, left, and right sides of the imaged in
        rolling time-windows. Uses a straight line.

        :parameter np.ndarray data: 2d array of size len(frames)x2 with body-part coordinates.
        :parameter np.ndarray img_resolution: Resolution of video in HxW format.
        :parameter float pixels_per_mm: Pixels per millimeter of recorded video.
        :parameter int fps: FPS of the recorded video
        :parameter float time_windows: Rolling time-window as floats in seconds. E.g., ``0.2``
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] array with millimeter distances from TOP, BOTTOM, LEFT RIGHT
        """

        results = np.full((data.shape[0], 4), -1.0)
        window_size = int(time_window * fps)
        for current_frm in prange(window_size, results.shape[0]):
            distances = np.full((4, window_size, 1), np.nan)
            windowed_locs = data[current_frm - window_size : current_frm, :]
            for bp_cnt, bp_loc in enumerate(windowed_locs):
                distances[0, bp_cnt] = np.linalg.norm(np.array([bp_loc[0], 0]) - bp_loc)
                distances[1, bp_cnt] = np.linalg.norm(
                    np.array([bp_loc[0], img_resolution[0]]) - bp_loc
                )
                distances[2, bp_cnt] = np.linalg.norm(np.array([0, bp_loc[1]]) - bp_loc)
                distances[3, bp_cnt] = np.linalg.norm(
                    np.array([0, img_resolution[1]]) - bp_loc
                )
            for i in prange(4):
                results[current_frm][i] = np.mean(distances[i]) / pixels_per_mm
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


#
#
#
#
#
# start = time.time()
# nose_loc = np.random.randint(low=0, high=500, size=(231, 2)).astype('float32')
# results = FeatureExtractionSupplemental().horizontal_vs_vertical_movement(data=nose_loc, pixels_per_mm=4.33, fps=10, time_windows=np.array([0.4]))


# results = FeatureExtractionSupplemental().border_distances(data=nose_loc, pixels_per_mm=4.33, fps=10, time_window=0.2, img_resolution=np.array([600, 400]))

results = FeatureExtractionSupplemental().acceleration(
    data=nose_loc, pixels_per_mm=4.33, fps=10
)


# left_ear_loc = np.random.randint(low=0, high=500, size=(10000, 2)).astype('float32')
# right_ear_loc = np.random.randint(low=0, high=500, size=(10000, 2)).astype('float32')
# angle_data = FeatureExtractionSupplemental().head_direction(nose_loc=nose_loc, left_ear_loc=left_ear_loc, right_ear_loc=right_ear_loc)
#
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
