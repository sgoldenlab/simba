__author__ = "Simon Nilsson"

try:
    from typing import Literal
except:
    from typing_extensions import Literal
import numpy as np
from numba import jit, prange

from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin

class FeatureExtractionSupplemental(FeatureExtractionMixin):

    """
    Additional feature extraction method not called by default feature extraction classes from ``simba.feature_extractors``.

    """

    def __init__(self):
        FeatureExtractionMixin.__init__(self)

    @staticmethod
    @jit(nopython=True)
    def _helper_euclidean_distance_timeseries_change(distances: np.ndarray,
                                                     time_windows: np.ndarray,
                                                     fps: int):
        """
        Private jitted helper called by ``simba.mixins.feature_extraction_supplemental_mixin.FeatureExtractionSupplemental.euclidean_distance_timeseries_change``
        """
        results = np.full((distances.shape[0], time_windows.shape[0]), np.nan)
        for window_cnt in prange(time_windows.shape[0]):
            frms = int(time_windows[window_cnt] * fps)
            shifted_distances = np.copy(distances)
            shifted_distances[0:frms] = np.nan
            shifted_distances[frms:] = distances[:-frms]
            shifted_distances[np.isnan(shifted_distances)] = distances[np.isnan(shifted_distances)]
            results[:, window_cnt] = distances - shifted_distances

        return results


    def euclidean_distance_timeseries_change(self,
                                             location_1: np.ndarray,
                                             location_2: np.ndarray,
                                             fps: int,
                                             px_per_mm: float,
                                             time_windows: np.ndarray = np.array([0.2, 0.4, 0.8, 1.6])) -> np.ndarray:
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
        distances = self.framewise_euclidean_distance(location_1=location_1, location_2=location_2, px_per_mm=px_per_mm)
        return self._helper_euclidean_distance_timeseries_change(distances=distances, fps=fps, time_windows=time_windows).astype(int)

    @staticmethod
    @jit(nopython=True)
    def peak_ratio(data: np.ndarray,
                   bin_size_s: int,
                   fps: int):
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
            if data[i][0] > data[i][1]: peak_cnt +=1
            if data[i][-1] > data[i][-2]: peak_cnt += 1
            for j in prange(1, len(data[i])-1):
                if data[i][j] > data[i][j-1]: peak_cnt += 1
            peak_ratio = peak_cnt / data[i].shape[0]
            results[start:end] = peak_ratio
            start, end = start + len(data[i]), end + len(data[i])
        return results


    @staticmethod
    @jit(nopython=True)
    def rolling_peak_count_ratio(data: np.ndarray,
                                  time_windows: np.ndarray,
                                  fps: int):

        results = np.full((data.shape[0], time_windows.shape[0]), -1.0)
        for i in prange(time_windows.shape[0]):
            window_size = int(time_windows[i] * fps)
            for j in prange(window_size, data.shape[0]):
                window_data = data[j-window_size:j]
                peak_cnt = 0
                if window_data[0] > window_data[1]: peak_cnt += 1
                if window_data[-1] > window_data[-2]: peak_cnt += 1
                for k in prange(1, len(window_data) - 1):
                    if window_data[j] > window_data[j - 1]: peak_cnt += 1
                peak_ratio = peak_cnt / window_data.shape[0]
                results[j, i] = peak_ratio
        print(results)


    @staticmethod
    @jit(nopython=True)
    def rolling_categorical_switches_ratio(data: np.ndarray,
                                           time_windows: np.ndarray,
                                           fps: int) -> np.ndarray:

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
            for current_frm in prange(jump_frms, data.shape[0]+1):
                time_slice = data[current_frm-jump_frms: current_frm]
                current_value, unique_cnt = time_slice[0], 0
                for i in prange(1, time_slice.shape[0]):
                    if time_slice[i] != current_value:
                        unique_cnt += 1
                    current_value = time_slice[i]
                print(unique_cnt, time_slice.shape[0])
                results[current_frm-1][time_window] = unique_cnt / time_slice.shape[0]
        return results

    @staticmethod
    @jit(nopython=True)
    def consecutive_time_series_categories_count(data: np.ndarray,
                                                 fps: int):
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
            if data[i] == data[i-1]:
                results[i] = results[i-1]+1
            else:
                results[i] = 1

        return results / fps


    @staticmethod
    @jit(nopython=True)
    def rolling_horizontal_vs_vertical_movement(data: np.ndarray,
                                                pixels_per_mm: float,
                                                time_windows: np.ndarray,
                                                fps: int) -> np.ndarray:

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
            for current_frm in prange(jump_frms, results.shape[0]+1):
                x_movement = np.sum(np.abs(np.ediff1d(data[current_frm-jump_frms: current_frm, 0]))) / pixels_per_mm
                y_movement = np.sum(np.abs(np.ediff1d(data[current_frm-jump_frms: current_frm, 1]))) / pixels_per_mm
                results[current_frm-1][time_window] = x_movement - y_movement

        return results

    @staticmethod
    @jit(nopython=True)
    def border_distances(data: np.ndarray,
                         pixels_per_mm: float,
                         img_resolution: np.ndarray,
                         time_window: float,
                         fps: int):

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
        for current_frm in prange(window_size, results.shape[0]+1):
            distances = np.full((4, window_size, 1), np.nan)
            windowed_locs = data[current_frm - window_size: current_frm]
            for bp_cnt, bp_loc in enumerate(windowed_locs):
                distances[0, bp_cnt] = np.linalg.norm(np.array([0, bp_loc[1]]) - bp_loc)  #left
                distances[1, bp_cnt] = np.linalg.norm(np.array([img_resolution[0], bp_loc[1]]) - bp_loc) #right
                distances[2, bp_cnt] = np.linalg.norm(np.array([bp_loc[0], 0]) - bp_loc) #top
                distances[3, bp_cnt] = np.linalg.norm(np.array([bp_loc[0], img_resolution[1]]) - bp_loc) #bottpm
            for i in prange(4):
                results[current_frm-1][i] = np.mean(distances[i]) / pixels_per_mm

        return results.astype(np.int32)

    @staticmethod
    @jit(nopython=True)
    def acceleration(data: np.ndarray,
                     pixels_per_mm: float,
                     fps: int):

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
            print(current_frm-fps, current_frm, current_frm, current_frm+fps)
            prior_window = np.mean(velocity[current_frm-fps: current_frm])
            current_window = np.mean(velocity[current_frm: current_frm+fps])
            results[current_frm:current_frm+fps] = current_window - prior_window
        return results

# import time
#
# start = time.time()
# for i in range(10000):
#     results = FeatureExtractionSupplemental().brunner_munzel(sample_1=sample_1, sample_2=sample_2)
# print(time.time() - start)
# #
# start = time.time()
# for i in range(10000):
#     bm = stats.brunnermunzel(x=sample_1, y=sample_2)
# print(time.time() - start)
#
#
#
# # print(results)
# # print(bm)