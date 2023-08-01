import numpy as np
from scipy import stats
from numba import jit, prange, typed
from typing import List

from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin

class FeatureExtractionCircularMixin(FeatureExtractionMixin):

    def __init__(self):
        FeatureExtractionMixin.__init__(self)

    @staticmethod
    def rolling_mean_dispersion(data: np.ndarray,
                                time_windows: np.ndarray,
                                fps: int) -> np.ndarray:

        """
        Compute the angular mean dispersion (circular mean) in degrees within rolling temporal windows.

        :parameter np.ndarray data: 1d array with feature values in degrees.
        :parameter np.ndarray time_windows: Rolling time-windows as floats in seconds. E.g., [0.2, 0.4, 0.6]
        :parameter int fps: fps of the recorded video
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] array

        :example:
        >>> data = np.random.randint(low=0, high=8, size=(10)).astype('int')
        >>> results = FeatureExtractionCircularMixin().rolling_mean_dispersion(data=data, time_windows=np.array([0.4]), fps=10)
        """

        results = np.full((data.shape[0], time_windows.shape[0]), -1)
        for time_window in prange(time_windows.shape[0]):
            jump_frms = int(time_windows[time_window] * fps)
            for current_frm in prange(jump_frms, results.shape[0]):
                data_window = np.deg2rad(data[current_frm-jump_frms:current_frm])
                results[current_frm][time_window] = np.rad2deg(stats.circmean(data_window)).astype(int)
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
        results = typed.List(['str'])
        DIRECTIONS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        for i in prange(degree_angles.shape[0]):
            ix = round(degree_angles[i] / (360. / len(DIRECTIONS)))
            direction = DIRECTIONS[ix % len(DIRECTIONS)]
            results.append(direction)
        return results[1:]

    @staticmethod
    @jit(nopython=True)
    def direction_three_bps(nose_loc: np.ndarray,
                            left_ear_loc: np.ndarray,
                             right_ear_loc: np.ndarray) -> np.ndarray:

        """
        Jitted helper to compute the degree angle from three body-parts. Computes the angle in degrees left_ear <-> nose
        and right_ear_nose and returns the midpoint.

        :parameter ndarray nose_loc: 2D array of size len(frames)x2 representing nose coordinates
        :parameter ndarray left_ear_loc: 2D array of size len(frames)x2 representing left ear coordinates
        :parameter ndarray right_ear_loc: 2D array of size len(frames)x2 representing right ear coordinates
        :return np.ndarray: Array of size nose_loc.shape[0] with direction in degrees.

        :example:
        >>> nose_loc = np.random.randint(low=0, high=500, size=(50, 2)).astype('float32')
        >>> left_ear_loc = np.random.randint(low=0, high=500, size=(50, 2)).astype('float32')
        >>> right_ear_loc = np.random.randint(low=0, high=500, size=(50, 2)).astype('float32')
        >>> results = FeatureExtractionCircularMixin().direction_three_bps(nose_loc=nose_loc, left_ear_loc=left_ear_loc, right_ear_loc=right_ear_loc)
        """

        results = np.full((nose_loc.shape[0]), np.nan)
        for i in prange(nose_loc.shape[0]):
            left_ear_to_nose = np.degrees(np.arctan2(left_ear_loc[i][0] - nose_loc[i][1], left_ear_loc[i][1] - nose_loc[i][0]))
            right_ear_nose = np.degrees(np.arctan2(right_ear_loc[i][0] - nose_loc[i][1], right_ear_loc[i][1] - nose_loc[i][0]))
            results[i] = ((left_ear_to_nose + right_ear_nose) % 360) / 2
        return results

    @staticmethod
    @jit(nopython=True)
    def direction_two_bps(bp_x: np.ndarray,
                          bp_y: np.ndarray) -> np.ndarray:

        results = np.full((bp_x.shape[0]), np.nan)
        for i in prange(bp_x.shape[0]):
            angle_degrees = np.degrees(np.arctan2(bp_x[i][0] - bp_y[i][0], bp_y[i][1] - bp_x[i][1]))
            angle_degrees = angle_degrees + 360 if angle_degrees < 0 else angle_degrees
            results[i] = angle_degrees
        return results



    @staticmethod
    @jit(nopython=True)
    def rolling_resultant_vector_length(data: np.ndarray,
                                        fps: int,
                                        time_window: float = 1.0) -> np.array:

        """
        Jitted helper computing the mean resultant vector within rolling time window.

        .. note:
           Adapted from ``pingouin.circular.circ_r``.

        :parameter ndarray data: 1D array of size len(frames) representing degrees.
        :parameter np.ndarray time_window: Rolling time-window as float in seconds. Default: 1s rolling time-window.
        :parameter int fps: fps of the recorded video
        :returns np.ndarray: Size len(frames) representing resultant vector length in the prior ``time_window``.
        """

        data = np.deg2rad(data)
        results = np.full((data.shape[0]), np.nan)
        window_size = int(time_window * fps)
        for window_end in prange(window_size, data.shape[0], 1):
            window_data = data[window_end - fps:window_end]
            w = np.ones(window_data.shape[0])
            r = np.nansum(np.multiply(w, np.exp(1j * window_data)))
            results[window_end] = np.abs(r) / np.nansum(w)
        return results


    @staticmethod
    @jit(nopython=True)
    def _helper_rayleigh_z(data: np.ndarray, window_size: int):
        results = np.full((data.shape[0], 2), np.nan)
        for i in range(data.shape[0]):
            r = window_size * data[i]
            results[i][0] = (r ** 2) / window_size
            results[i][1] = np.exp(np.sqrt(1 + 4 * window_size + 4 * (window_size**2 - r**2)) - (1 + 2 * window_size))
        return results

    def rolling_rayleigh_z(self,
                           data: np.ndarray,
                           fps: int,
                           time_window: float = 1.0) -> np.array:
        """
        Compute Rayleigh Z (test of non-uniformity) of circular data within rolling time-window.

        .. note:
           Adapted from ``pingouin.circular.circ_rayleigh``.

        :parameter ndarray data: 1D array of size len(frames) representing degrees.
        :parameter np.ndarray time_window: Rolling time-window as float in seconds. Default: 1s rolling time-window.
        :parameter int fps: fps of the recorded video
        :returns np.ndarray: Size data.shape[0] x 2 with Rayleigh Z statistics in first column and associated p_values in second column
        """

        results, window_size = np.full((data.shape[0], 2), np.nan), int(time_window * fps)
        resultant_vector_lengths = FeatureExtractionCircularMixin().rolling_resultant_vector_length(data=data, fps=fps, time_window=time_window)
        return np.nan_to_num(self._helper_rayleigh_z(data=resultant_vector_lengths, window_size=window_size), nan=-1.0)

    @staticmethod
    @jit(nopython=True)
    def rolling_circular_correlation(data_x: np.ndarray,
                                     data_y: np.ndarray,
                                     fps: int,
                                     time_window: float = 1.0):

        data_x, data_y = np.deg2rad(data_x), np.deg2rad(data_y)
        results = np.full((data_x.shape[0]), np.nan)
        window_size = int(time_window * fps)
        for window_start in prange(0, data_x.shape[0]-window_size+1):
            data_x_window = data_x[window_start:window_start+window_size]
            data_y_window = data_y[window_start:window_start+window_size]
            x_sin = np.sin(data_x_window - np.angle(np.nansum(np.multiply(1, np.exp(1j * data_x_window)))))
            y_sin = np.sin(data_y_window - np.angle(np.nansum(np.multiply(1, np.exp(1j * data_y_window)))))
            r = np.sum(x_sin * y_sin) / np.sqrt(np.sum(x_sin ** 2) * np.sum(y_sin ** 2))
            results[window_start+window_size] = (np.sqrt((data_x_window.shape[0] * (x_sin ** 2).mean() * (y_sin ** 2).mean()) / np.mean(x_sin ** 2 * y_sin ** 2)) * r)

        return results

# nose_loc = np.random.randint(low=0, high=500, size=(200, 2)).astype('float32')
# left_ear_loc = np.random.randint(low=0, high=500, size=(200, 2)).astype('float32')
#
# angle_data = FeatureExtractionCircularMixin().direction_two_bps(bp_x=nose_loc, bp_y=left_ear_loc)
#

#
# def direction_two_bps(bp_x: np.ndarray,
#                       bp_y: np.ndarray) -> np.ndarray:




# right_ear_loc = np.random.randint(low=0, high=500, size=(200, 2)).astype('float32')
# angle_data = FeatureExtractionCircularMixin().head_direction(nose_loc=nose_loc, left_ear_loc=left_ear_loc, right_ear_loc=right_ear_loc)
#
# resultant_length = FeatureExtractionCircularMixin().rolling_resultant_vector_length(data=angle_data.astype(np.int8), time_window=1.0, fps=25)
#
# #resultant_length = FeatureExtractionCircularMixin().rolling_rayleigh_z(data=angle_data.astype(np.int8), time_window=2.0, fps=5)

# start = time.time()
# correlation = FeatureExtractionCircularMixin().rolling_circular_correlation(data_x=angle_data.astype(np.int8), data_y=angle_data.astype(np.int8), time_window=2.0, fps=5)
# print(time.time() - start)