from typing import List

import numpy as np
from numba import jit, prange, typed
from scipy import stats


class FeatureExtractionCircularMixin(object):
    """
    Mixin for circular statistics. Support for multiple animals and base
    radial directions derived from two or three body-parts.

    .. important::
        See references below for  mature packages computing extensive circular measurements.

    .. image:: _static/img/circular_statistics.png
       :width: 800
       :align: center

    References
    ----------
    .. [1] `pycircstat <https://github.com/circstat/pycircstat>`_.
    .. [2] `circstat <https://www.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-toolbox-directional-statistics>`_.
    .. [3] `pingouin.circular <https://github.com/raphaelvallat/pingouin/blob/master/pingouin/circular.py>`_.
    .. [4] `pycircular <https://github.com/albahnsen/pycircular>`_.
    .. [5] `scipy.stats.directional_stats <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.directional_stats.html>`_.
    .. [6] `astropy.stats.circstats <https://docs.astropy.org/en/stable/_modules/astropy/stats/circstats.html>`_.
    """

    def __init__(self):
        pass

    @staticmethod
    def rolling_mean_dispersion(
        data: np.ndarray, time_windows: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Compute the angular mean dispersion (circular mean) in degrees within rolling temporal windows.

        :parameter np.ndarray data: 1d array with feature values in degrees.
        :parameter np.ndarray time_windows: Rolling time-windows as floats in seconds. E.g., [0.2, 0.4, 0.6]
        :parameter int fps: fps of the recorded video
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] array

        .. image:: _static/img/mean_rolling_timeseries_angle.png
           :width: 1000
           :align: center

        .. attention::
           The returned values represents the angular mean dispersion in the time-window ``[current_frame-time_window->current_frame]``.
           `-1` is returned when ``current_frame-time_window`` is less than 0.

        :example:
        >>> data = np.random.normal(loc=45, scale=1, size=20)
        >>> FeatureExtractionCircularMixin().rolling_mean_dispersion(data=data,time_windows=np.array([0.5]), fps=10)
        >>> [ [-1],[-1],[-1],[-1], [-1],[44],[44],[43],[44],[44],[44],[44],[44],[44],[44],[45],[45],[45],[45],[45]])
        """

        results = np.full((data.shape[0], time_windows.shape[0]), -1)
        for time_window in prange(time_windows.shape[0]):
            jump_frms = int(time_windows[time_window] * fps)
            for current_frm in prange(jump_frms, results.shape[0] + 1):
                data_window = np.deg2rad(data[current_frm - jump_frms : current_frm])
                results[current_frm - 1][time_window] = np.rad2deg(
                    stats.circmean(data_window)
                ).astype(int)
        return results

    @staticmethod
    @jit(nopython=True)
    def degrees_to_compass_cardinal(degree_angles: np.ndarray) -> List[str]:
        """
        Convert degree angles to cardinal direction bucket e.g., 0 -> "N", 180 -> "S"

        .. note::
           To convert cardinal literals to integers, map using ``simba.utils.enums.lookups.cardinality_to_integer_lookup``.
           To convert integers to cardinal literals, map using ``simba.utils.enums.lookups.integer_to_cardinality_lookup``.

        :parameter degree_angles nose_loc: 1d array of degrees. Note: return by ``self.head_direction``.
        :return List[str]: List of strings representing frame-wise cardinality

        :example:
        >>> data = np.array(list(range(0, 405, 45)))
        >>> FeatureExtractionCircularMixin().degrees_to_compass_cardinal(degree_angles=data)
        >>> ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
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
    def direction_three_bps(
        nose_loc: np.ndarray, left_ear_loc: np.ndarray, right_ear_loc: np.ndarray
    ) -> np.ndarray:
        """
        Jitted helper to compute the degree angle from three body-parts. Computes the angle in degrees left_ear <-> nose
        and right_ear_nose and returns the midpoint.

        .. image:: _static/img/angle_from_3_bps.png
          :width: 600
          :align: center

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
    def direction_two_bps(bp_x: np.ndarray, bp_y: np.ndarray) -> np.ndarray:
        """
        Jitted method computing degree directionality from two body-parts. E.g., ``nape`` and ``nose``,
        or ``swim_bladder`` and ``tail``.

        .. image:: _static/img/angle_from_2_bps.png
           :width: 1200
           :align: center

        :parameter np.ndarray bp_x: Size len(frames) x 2 representing x and y coordinates for first body-part.
        :parameter np.ndarray bp_y: Size len(frames) x 2 representing x and y coordinates for second body-part.
        :return np.ndarray: Frame-wise directionality in degrees.

        :example:
        >>> swim_bladder_loc = np.random.randint(low=0, high=500, size=(50, 2)).astype('float32')
        >>> tail_loc = np.random.randint(low=0, high=500, size=(50, 2)).astype('float32')
        >>> results = FeatureExtractionCircularMixin().direction_three_bps(bp_x=swim_bladder_loc, bp_y=tail_loc)
        """

        results = np.full((bp_x.shape[0]), np.nan)
        for i in prange(bp_x.shape[0]):
            angle_degrees = np.degrees(
                np.arctan2(bp_x[i][0] - bp_y[i][0], bp_y[i][1] - bp_x[i][1])
            )
            angle_degrees = angle_degrees + 360 if angle_degrees < 0 else angle_degrees
            results[i] = angle_degrees
        return results

    @staticmethod
    @jit(nopython=True)
    def rolling_resultant_vector_length(
        data: np.ndarray, fps: int, time_windows: np.ndarray = np.array([1.0])
    ) -> np.ndarray:
        """
        Jitted helper computing the mean resultant vector within rolling time window.

        .. note:
           Adapted from ``pingouin.circular.circ_r``.

        .. attention::
           The returned values represents resultant vector length in the time-window ``[current_frame-time_window->current_frame]``.
           `-1` is returned where ``current_frame-time_window`` is less than 0.

        :parameter np.ndarray data: 1D array of size len(frames) representing degrees.
        :parameter np.ndarray time_window: Rolling time-window as float in seconds. Default: 1s rolling time-window.
        :parameter int fps: fps of the recorded video
        :returns np.ndarray: Size len(frames) representing resultant vector length in the prior ``time_window``.

        :example:
        >>> data_1, data_2 = np.random.normal(loc=45, scale=3, size=20), np.random.normal(loc=45, scale=150, size=30)
        >>> data = np.hstack([data_1, data_2])
        >>> FeatureExtractionCircularMixin().rolling_resultant_vector_length(data=data,time_windows=np.array([1]), fps=20)
        >>> [[-1],[-1],[-1],[-1],[-1],[ 0.998],[ 0.999],[ 0.999],[ 0.999],[ 0.997],[ 0.997],[ 0.996],[ 0.996],[ 0.996],[ 0.998],[ 0.998],[ 0.999],[ 0.998],[ 0.998],[ 0.999],[ 0.998],[ 0.655],[ 0.644],[ 0.367],[ 0.277],[ 0.193],[ 0.582],[ 0.200],[ 0.254],[ 0.235],[ 0.126],[ 0.145],[ 0.336],[ 0.719],[ 0.682],[ 0.780],[ 0.576],[ 0.314],[ 0.333],[ 0.291],[ 0.304],[ 0.095],[ 0.410],[ 0.061],[ 0.052],[ 0.262],[ 0.217],[ 0.485],[ 0.411],[ 0.736]])
        """

        data = np.deg2rad(data)
        results = np.full((data.shape[0], time_windows.shape[0]), -1.0)
        for time_window_cnt in prange(time_windows.shape[0]):
            window_size = int(time_windows[time_window_cnt] * fps)
            for window_end in prange(window_size, data.shape[0] + 1, 1):
                window_data = data[window_end - window_size : window_end]
                w = np.ones(window_data.shape[0])
                r = np.nansum(np.multiply(w, np.exp(1j * window_data)))
                results[window_end - 1][time_window_cnt] = np.abs(r) / np.nansum(w)
        return results

    @staticmethod
    @jit(nopython=True)
    def _helper_rayleigh_z(data: np.ndarray, window_size: int):
        results = np.full((data.shape[0], 2), np.nan)
        for i in range(data.shape[0]):
            r = window_size * data[i]
            results[i][0] = (r**2) / window_size
            results[i][1] = np.exp(
                np.sqrt(1 + 4 * window_size + 4 * (window_size**2 - r**2))
                - (1 + 2 * window_size)
            )
        return results

    def rolling_rayleigh_z(
        self, data: np.ndarray, fps: int, time_window: float = 1.0
    ) -> np.array:
        """
        Compute Rayleigh Z (test of non-uniformity) of circular data within rolling time-window.

        .. note:
           Adapted from ``pingouin.circular.circ_rayleigh``.

        :parameter ndarray data: 1D array of size len(frames) representing degrees.
        :parameter np.ndarray time_window: Rolling time-window as float in seconds. Default: 1s rolling time-window.
        :parameter int fps: fps of the recorded video
        :returns np.ndarray: Size data.shape[0] x 2 with Rayleigh Z statistics in first column and associated p_values in second column
        """

        results, window_size = np.full((data.shape[0], 2), np.nan), int(
            time_window * fps
        )
        resultant_vector_lengths = (
            FeatureExtractionCircularMixin().rolling_resultant_vector_length(
                data=data, fps=fps, time_window=time_window
            )
        )
        return np.nan_to_num(
            self._helper_rayleigh_z(
                data=resultant_vector_lengths, window_size=window_size
            ),
            nan=-1.0,
        )

    @staticmethod
    @jit(nopython=True)
    def rolling_circular_correlation(
        data_x: np.ndarray, data_y: np.ndarray, fps: int, time_window: float = 1.0
    ) -> np.ndarray:
        """
        Compute correlations between two angular distributions in rolling time-windows.

        .. image:: _static/img/cicle_correlation.png
           :width: 800
           :align: center

        :parameter np.ndarray data_x: Angular data for e.g., Animal 1
        :parameter np.ndarray data_y: Angular data for e.g., Animal 2
        :parameter int fps: Frame-rate of video.
        :parameter float time_window: Time window in seconds.
        :return np.ndarray: Correlation coefficients for ``data_x`` and ``data_y``.
        """

        data_x, data_y = np.deg2rad(data_x), np.deg2rad(data_y)
        results = np.full((data_x.shape[0]), np.nan)
        window_size = int(time_window * fps)
        for window_start in prange(0, data_x.shape[0] - window_size + 1):
            data_x_window = data_x[window_start : window_start + window_size]
            data_y_window = data_y[window_start : window_start + window_size]
            x_sin = np.sin(
                data_x_window
                - np.angle(np.nansum(np.multiply(1, np.exp(1j * data_x_window))))
            )
            y_sin = np.sin(
                data_y_window
                - np.angle(np.nansum(np.multiply(1, np.exp(1j * data_y_window))))
            )
            r = np.sum(x_sin * y_sin) / np.sqrt(np.sum(x_sin**2) * np.sum(y_sin**2))
            results[window_start + window_size] = (
                np.sqrt(
                    (data_x_window.shape[0] * (x_sin**2).mean() * (y_sin**2).mean())
                    / np.mean(x_sin**2 * y_sin**2)
                )
                * r
            )

        return results

    @staticmethod
    def rolling_circular_stdev(
        data: np.ndarray, fps: int, time_windows: np.ndarray
    ) -> np.ndarray:
        """
        Compute standard deviation of angular data in rolling time windows.

        .. image:: _static/img/angle_stdev.png
           :width: 800
           :align: center

        :parameter ndarray data: 1D array of size len(frames) representing degrees.
        :parameter np.ndarray time_window: Rolling time-window as float in seconds.
        :parameter int fps: fps of the recorded video
        :returns np.ndarray: Size data.shape[0] x time_windows.shape[0] with angular standard deviations in rolling time windows.

        :example:
        >>> data_1, data_2 = np.random.normal(loc=180, scale=3, size=10), np.random.normal(loc=90, scale=700, size=10)
        >>> data = np.hstack([data_1, data_2])
        >>> results = FeatureExtractionCircularMixin().rolling_circular_stdev(data=data, time_windows=np.array([1]), fps=10)
        """

        data = np.deg2rad(data)
        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for time_window_cnt in prange(time_windows.shape[0]):
            window_size = int(time_windows[time_window_cnt] * fps)
            for window_end in prange(window_size, data.shape[0] + 1, 1):
                window_data = data[window_end - window_size : window_end]
                results[window_end - 1][time_window_cnt] = stats.circvar(window_data)
        return np.round(results, 4)

    @staticmethod
    @jit(nopython=True)
    def rolling_angular_difference(
        data: np.ndarray, fps: int, time_windows: np.ndarray
    ):
        """
        Computes the angular difference in the current frame versus N seconds previously.
        For example, if the current angle is 45 degrees, and the angle N seconds previously was 350 degrees, then the difference
        is 55 degrees.

        .. note::
           Frames where current frame - N seconds prior equal a negative value is populated with 0.

        :parameter ndarray data: 1D array of size len(frames) representing degrees.
        :parameter np.ndarray time_window: Rolling time-window as float in seconds.
        :parameter int fps: fps of the recorded video

        :example:
        >>> data = np.array([350, 350, 1, 1])
        >>> FeatureExtractionCircularMixin().rolling_angular_difference(data=data, fps=1, time_windows=np.array([1]))
        >>> [[ 0.], [ 0.], [11.], [ 0.]]
        """

        data = np.deg2rad(data)
        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for time_window_cnt in prange(time_windows.shape[0]):
            window_size = int(time_windows[time_window_cnt] * fps)
            for window_end in prange(window_size, data.shape[0], 1):
                point_one, point_two = data[window_end - window_size], data[window_end]
                print(point_one, point_two)
                distance = np.pi - np.abs(np.pi - np.abs(point_one - point_two))
                results[window_end][time_window_cnt] = np.rad2deg(distance)

        return results

    @staticmethod
    @jit(nopython=True)
    def agg_angular_difference_timebins(
        data: np.ndarray, fps: int, time_windows: np.ndarray
    ):
        """
        Compute the difference between the median angle in the current time-window versus the previous time window.
        For example, computes the difference between the mean angle in the first 1s of the video versus
        the second 1s of the video, the second 1s of the video versus the third 1s of the video, ... etc.

        .. note::
           The first time-bin of the video can't be compared against the prior time-bin of the video and the results
           for this first time-bin will be populated with `0`.

        :parameter ndarray data: 1D array of size len(frames) representing degrees.
        :parameter np.ndarray time_window: Rolling time-window as float in seconds.
        :parameter int fps: fps of the recorded video

        :example:
        >>> data = np.random.normal(loc=45, scale=3, size=20)
        >>> FeatureExtractionCircularMixin().agg_angular_difference_timebins(data=data,time_windows=np.array([1]), fps=5)
        """

        data = np.deg2rad(data)
        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for time_window_cnt in prange(time_windows.shape[0]):
            window_size = int(time_windows[time_window_cnt] * fps)
            prior_window = [0, window_size]
            for win_cnt, window_end in enumerate(
                prange(int(window_size * 2), data.shape[0] + 1, window_size)
            ):
                window_start = (window_end - window_size) - 1
                current_data = data[window_start:window_end]
                prior_data = data[prior_window[0] : prior_window[1]]
                prior_median = np.arctan2(
                    np.median(np.cos(prior_data)), np.median(np.sin(prior_data))
                )
                current_median = np.arctan2(
                    np.median(np.cos(current_data)), np.median(np.sin(current_data))
                )
                distance = np.pi - np.abs(np.pi - np.abs(prior_median - current_median))
                results[window_start:window_end, win_cnt] = np.rad2deg(distance)
                prior_window = [window_start, window_end]

        return results

    @staticmethod
    @jit(nopython=True)
    def instantaneous_angular_velocity(data: np.ndarray, bin_size: int = 1):
        """
        Jitted compute of absolute angular change in the smallest possible time bin.

        .. note::
            If the smallest possible frame-to-frame time-bin in Video 1 is 33ms (recorded at 30fps), and the
            smallest possible frame-to-frame time-bin in Video 2 is 66ms (recorded at 15fps) we have to correct for
            this across recordings using the ``bin_size`` argument. E.g., when passing angular data from Video 1
            we would set bin_size to ``2``, and when passing angular data for Video 2 we would set bin_size to ``1`` to
            allow comparisons of instantaneous angular velocity between Video 1 and Video 2.

            When current frame minus bin_size results in a negative index, 0 is returned.

        :parameter ndarray data: 1D array of size len(frames) representing degrees.
        :parameter int bin_size: The number of frames prior to compare the current angular velocity against.

        :example:
        >>> data = np.array([350, 355, 356, 357])
        >>> FeatureExtractionCircularMixin().instantaneous_angular_velocity(data=data, bin_size=1)
        >>> [0., 5., 1., 1.]
        >>> FeatureExtractionCircularMixin().instantaneous_angular_velocity(data=data, bin_size=2)
        >>> [0., 0., 6., 2.]
        """
        data = np.deg2rad(data)
        results = np.full((data.shape[0]), 0.0)
        left_idx, right_idx = 0, bin_size
        for end_idx in prange(right_idx, data.shape[0] + 1, 1):
            results[end_idx] = np.rad2deg(
                np.pi - np.abs(np.pi - np.abs(data[left_idx] - data[end_idx]))
            )
            left_idx += 1
        return results

    @staticmethod
    def sliding_rao_spacing(
        data: np.ndarray, time_window: float, fps: int
    ) -> np.ndarray:
        """
        Compute the uniformity of a circular dataset in sliding window of size ``time_window``.

        :parameter ndarray data: 1D array of size len(frames) representing degrees.
        :parameter np.ndarray time_window: Rolling time-window as float in seconds.
        :parameter int fps: fps of the recorded video
        :return np.ndarray: representing rao-spacing U in every sliding windows [-window:n]

        .. image:: _static/img/raospacing.png
           :width: 800
           :align: center

        :references:
        .. [1] `UCSB <https://jammalam.faculty.pstat.ucsb.edu/html/favorite/test.htm>`__.

        :example:
        >>> data = np.random.randint(low=0, high=360, size=(500,))
        >>> result = FeatureExtractionCircularMixin().sliding_rao_spacing(data=data, time_window=0.5, fps=10)
        """

        results = np.full((data.shape[0]), -1.0)
        window_size = int(time_window * fps)
        for i in range(window_size, data.shape[0]):
            w_data = np.sort(data[i - window_size : i])
            Ti, TiL = np.full((w_data.shape[0]), np.nan), np.full(
                (w_data.shape[0]), np.nan
            )
            l = 360 / len(w_data)
            Ti[-1] = np.rad2deg(
                np.pi
                - np.abs(np.pi - np.abs(np.deg2rad(w_data[0]) - np.deg2rad(w_data[-1])))
            )
            for j in prange(w_data.shape[0] - 1, -1, -1):
                Ti[j] = np.rad2deg(
                    np.pi
                    - np.abs(
                        np.pi
                        - np.abs(np.deg2rad(w_data[j]) - np.deg2rad(w_data[j - 1]))
                    )
                )
            for k in prange(Ti.shape[0]):
                TiL[k] = np.max((l, Ti[k])) - np.min((l, Ti[k]))
            S = np.sum(TiL)
            U = int(S / 2)
            results[i] = U
        return results


# data = np.random.normal(loc=90, scale=360, size=500)
# data = np.random.randint(low=0, high=360, size=(500,))
# result = FeatureExtractionCircularMixin().sliding_rao_spacing(data=data, time_window=0.5, fps=10)
# print(result)
# raospacing(np.deg2rad(data[-6:-1]), axis=0)

# #data = np.random.normal(loc=90, scale=2, size=20)
# data = np.array([350, 355, 356, 357])
# angle_data = FeatureExtractionCircularMixin().instantaneous_angular_velocity(data=data, bin_size=2)
#
#


# data = np.random.normal(loc=45, scale=3, size=20)
# FeatureExtractionCircularMixin().agg_angular_difference_timebins(data=data,time_windows=np.array([1]), fps=5)


# nose_loc = np.random.randint(low=0, high=500, size=(200, 2)).astype('float32')
# left_ear_loc = np.random.randint(low=0, high=500, size=(200, 2)).astype('float32')
#
# angle_data = FeatureExtractionCircularMixin().direction_two_bps(bp_x=nose_loc, bp_y=left_ear_loc)
#


# data_1 = np.random.normal(loc=45, scale=3, size=20)
# data_2 = np.random.normal(loc=45, scale=150, size=30)
# data = np.hstack([data_1, data_2])
# FeatureExtractionCircularMixin().rolling_resultant_vector_length(data=data,time_windows=np.array([1]), fps=5)
#
#


# data = np.random.normal(loc=45, scale=1, size=20)
# FeatureExtractionCircularMixin().rolling_mean_dispersion(data=data,time_windows=np.array([0.5]), fps=10)


# data = np.array(list(range(0, 405, 45)))
# results = FeatureExtractionCircularMixin().degrees_to_compass_cardinal(degree_angles=data)


data = np.array(list(range(0, 405, 45)))
results = FeatureExtractionCircularMixin().degrees_to_compass_cardinal(
    degree_angles=data
)


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
