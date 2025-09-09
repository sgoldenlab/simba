__author__ = "Simon Nilsson"

from typing import List, Optional, Tuple

import numpy as np
from numba import float32, float64, int64, jit, njit, prange, typed, types

from simba.utils.checks import check_float, check_valid_array
from simba.utils.data import get_mode
from simba.utils.enums import Formats
from simba.utils.errors import InvalidInputError


class CircularStatisticsMixin(object):
    """
    Mixin for circular statistics. Unlike linear data, circular data wrap around in a circular or periodic
    manner such as two measurements of e.g., 360 vs. 1 are more similar than two measurements of 1 vs. 3. Thus, the
    minimum and maximum values are connected, forming a closed loop, and we therefore need specialized
    statistical methods.

    These methods have support for multiple animals and base radial directions derived from two or three body-parts.

    Methods are adopted from the referenced packages below which are **far** more reliable. However,
    runtime on standard hardware (multicore CPU) is prioritized and typically orders of magnitude faster than referenced libraries.

    See image below for example of expected run-times for a small set of method examples included in this class.

    .. note::
        Many method has numba typed `signatures <https://numba.pydata.org/numba-doc/latest/reference/types.html>`_ to decrease
        compilation time through reduced type inference. Make sure to pass the correct dtypes as indicated by signature decorators.

    .. important::
        See references below for mature packages computing more extensive circular measurements.

    .. image:: _static/img/circular_statistics.png
       :width: 800
       :align: center

    .. image:: _static/img/circular_stats_runtimes.png
       :width: 1200
       :align: center

    References
    ----------
    .. [1] `pycircstat <https://github.com/circstat/pycircstat>`_.
    .. [2] `circstat <https://www.mathworks.com/matlabcentral/fileexchange/10676-circular-statistics-toolbox-directional-statistics>`_.
    .. [3] `pingouin.circular <https://github.com/raphaelvallat/pingouin/blob/master/pingouin/circular.py>`_.
    .. [4] `pycircular <https://github.com/albahnsen/pycircular>`_.
    .. [5] `scipy.stats.directional_stats <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.directional_stats.html>`_.
    .. [6] `astropy.stats.circstats <https://docs.astropy.org/en/stable/_modules/astropy/stats/circstats.html>`_.
    .. [7] `pycircstat2 <https://github.com/circstat/pycircstat2/>`_.

    """

    def __init__(self):
        pass

    @staticmethod
    @njit("(float32[:],)")
    def mean_resultant_vector_length(data: np.ndarray) -> float:
        """
        Jitted compute of the mean resultant vector length of a single sample. Captures the overall "pull" or "tendency" of the
        data points towards a central direction on the circle with a range between 0 and 1.

        .. image:: _static/img/mean_resultant_vector.png
           :width: 400
           :align: center

        .. math::

            R = \\frac{{\\sqrt{{\\sum_{{i=1}}^N \\cos(\\theta_i - \\bar{\theta})^2 + \\sum_{{i=1}}^N \\sin(\theta_i - \\bar{\\theta})^2}}}}{{N}}

        where :math:`N` is the number of data points, :math:`\\theta_i` is the angle of the ith data point, and \(\bar{\theta}\) is the mean angle.

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.sliding_resultant_vector_length`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_mean_resultant_vector_length`


        :parameter np.ndarray data: 1D array of size len(frames) representing angles in degrees.
        :returns: The mean resultant vector of the angles. 1 represents tendency towards a single point. 0 represents no central point.
        :rtype: float

        :example:
        >>> data = np.array([50, 90, 70, 60, 20, 90]).astype(np.float32)
        >>> CircularStatisticsMixin().mean_resultant_vector_length(data=data)
        >>> 0.9132277170817057
        """

        data = np.deg2rad(data)
        mean_angles = np.arctan2(np.mean(np.sin(data)), np.mean(np.cos(data)))
        return np.sqrt(np.sum(np.cos(data - mean_angles)) ** 2 + np.sum(np.sin(data - mean_angles)) ** 2) / len(data)

    @staticmethod
    @njit("(float32[:], float64, float64[:])")
    def sliding_mean_resultant_vector_length(data: np.ndarray, fps: float, time_windows: np.ndarray) -> np.ndarray:

        """
        Jitted compute of the mean resultant vector within sliding time window. Captures the overall "pull" or "tendency" of the
        data points towards a central direction on the circle with a range between 0 and 1.

        .. attention::
           The returned values represents resultant vector length in the time-window ``[(current_frame-time_window)->current_frame]``.
           `-1` is returned where ``current_frame-time_window`` is less than 0.

        .. image:: _static/img/sliding_mean_resultant_length.png
           :width: 600
           :align: center

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.sliding_resultant_vector_length`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.mean_resultant_vector_length`

        :parameter np.ndarray data: 1D array of size len(data) representing degrees.
        :parameter np.ndarray time_window: Rolling time-window as float in seconds.
        :parameter int fps: fps of the recorded video
        :returns: Size len(data) x len(time_windows) representing resultant vector length in the prior ``time_window``.
        :rtype: np.ndarray

        :example:
        >>> data_1, data_2 = np.random.normal(loc=45, scale=1, size=100), np.random.normal(loc=90, scale=45, size=100)
        >>> data = np.hstack([data_1, data_2])
        >>> CircularStatisticsMixin().sliding_mean_resultant_vector_length(data=data.astype(np.float32),time_windows=np.array([1.0]), fps=10)

        """
        data = np.deg2rad(data)
        results = np.full((data.shape[0], time_windows.shape[0]), -1.0)
        for time_window_cnt in prange(time_windows.shape[0]):
            window_size = int(time_windows[time_window_cnt] * fps)
            for window_end in prange(window_size, data.shape[0] + 1, 1):
                window_data = data[window_end - window_size: window_end]
                cos_sum = np.sum(np.cos(window_data))
                sin_sum = np.sum(np.sin(window_data))
                r = np.sqrt(cos_sum ** 2 + sin_sum ** 2) / len(window_data)
                results[window_end - 1, time_window_cnt] = r
        return results

    @staticmethod
    @njit("(float32[:],)")
    def circular_mean(data: np.ndarray) -> float:

        r"""
        Jitted compute of the circular mean of single sample.


        The circular mean is calculated as:

        .. math::

           \mu = \text{atan2}\left(\frac{1}{N} \sum_{i=1}^{N} \sin(\theta_i), \frac{1}{N} \sum_{i=1}^{N} \cos(\theta_i)\right)

        Where:

        - :math:`\theta_i` are the angles in radians within the sample
        - :math:`N` is the number of samples
        - :math:`\mu` is the circular mean angle


        :param np.ndarray data: 1D array of size len(frames) representing angles in degrees.
        :returns: The circular mean of the angles in degrees.
        :rtype: float

        .. image:: _static/img/mean_angle.png
           :width: 400
           :align: center

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.sliding_circular_mean`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_mean`

        :example:
        >>> data = np.array([50, 90, 70, 60, 20, 90]).astype(np.float32)
        >>> CircularStatisticsMixin().circular_mean(data=data)
        >>> 63.737892150878906
        """

        m = np.rad2deg(
            np.arctan2(
                np.mean(np.sin(np.deg2rad(data))), np.mean(np.cos(np.deg2rad(data)))
            )
        )
        return np.abs(np.round(m, 4))

    @staticmethod
    @njit("(float32[:], float64[:], float64)")
    def sliding_circular_mean(data: np.ndarray, time_windows: np.ndarray, fps: int) -> np.ndarray:
        """
        Compute the circular mean in degrees within sliding temporal windows.

        .. image:: _static/img/mean_rolling_timeseries_angle.png
           :width: 600
           :align: center

        .. attention::
           The returned values represents the angular mean dispersion in the time-window ``[current_frame-time_window->current_frame]``.
           `-1` is returned when ``current_frame-time_window`` is less than 0.

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.sliding_circular_mean`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.circular_mean`


        :parameter np.ndarray data: 1d array with feature values in degrees.
        :parameter np.ndarray time_windows: Rolling time-windows as floats in seconds. E.g., [0.2, 0.4, 0.6]
        :parameter int fps: fps of the recorded video
        :returns: Size data.shape[0] x time_windows.shape[0] array
        :rtype: np.ndarray

        :example:
        >>> data = np.random.normal(loc=45, scale=1, size=20).astype(np.float32)
        >>> CircularStatisticsMixin().sliding_circular_mean(data=data,time_windows=np.array([0.5, 1.0]), fps=10)
        """

        data = np.deg2rad(data)
        results = np.full((data.shape[0], time_windows.shape[0]), -1.0)
        for time_window in range(time_windows.shape[0]):
            window_size = int(time_windows[time_window] * fps)
            for current_frm in range(window_size - 1, results.shape[0]):
                data_window = data[(current_frm - window_size) + 1 : current_frm + 1]
                m = np.rad2deg(np.arctan2(np.mean(np.sin(data_window)), np.mean(np.cos(data_window))))
                m = (m + 360) % 360
                results[current_frm, time_window] = np.round(m, 4)
        return results

    @staticmethod
    @njit("(float32[:],)")
    def circular_std(data: np.ndarray) -> float:
        """
        Jitted compute of the circular standard deviation from a single distribution of angles in degrees.

        .. image:: _static/img/circular_std.png
           :width: 600
           :align: center

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.sliding_circular_std`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_std`

        .. math::

           \\sigma_{\\text{circular}} = \\text{rad2deg}\\left(\\sqrt{-2 \\cdot \\log\\left(|\text{mean}(\\exp(j \\cdot \\theta))|\\right)}\\right)

        where :math:`\\theta` represents the angles in radians


        :param ndarray data: 1D array of size len(frames) with angles in degrees
        :returns: The standard deviation of the data sample in degrees.
        :rtype: float

        :example:
        >>> data = np.array([180, 221, 32, 42, 212, 101, 139, 41, 69, 171, 149, 200]).astype(np.float32)
        >>> CircularStatisticsMixin().circular_std(data=data)
        >>> 75.03725024504664
        """

        data = np.deg2rad(data)
        return np.abs(
            np.rad2deg(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * data))))))
        )

    @staticmethod
    @njit("(float32[:], int64, float64[:])", parallel=True)
    def sliding_circular_std(
        data: np.ndarray, fps: int, time_windows: np.ndarray
    ) -> np.ndarray:

        r"""
        Compute standard deviation of angular data in sliding time windows.

        .. image:: _static/img/angle_stdev.png
           :width: 600
           :align: center

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.sliding_circular_std`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.circular_std`


        :param ndarray data: 1D array of size len(frames) representing degrees.
        :param np.ndarray time_window: Sliding time-window as float in seconds.
        :param int fps: fps of the recorded video
        :returns: Size data.shape[0] x time_windows.shape[0] with angular standard deviations in rolling time windows in degrees.
        :rtype: np.ndarray

        :example:
        >>> data = np.array([180, 221, 32, 42, 212, 101, 139, 41, 69, 171, 149, 200]).astype(np.float32)
        >>> CircularStatisticsMixin().sliding_circular_std(data=data.astype(np.float32), time_windows=np.array([0.5]), fps=10)
        """

        data = np.deg2rad(data)
        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for time_window_cnt in prange(time_windows.shape[0]):
            window_size = int(time_windows[time_window_cnt] * fps)
            for window_end in prange(window_size, data.shape[0] + 1, 1):
                window_data = data[window_end - window_size : window_end]
                m = np.rad2deg(np.sqrt(-2 * np.log(np.abs(np.mean(np.exp(1j * window_data))))))
                results[window_end - 1][time_window_cnt] = np.abs(np.round(m, 4))

        return results.astype(np.float32)

    @staticmethod
    @njit("(float32[:], int64)")
    def instantaneous_angular_velocity(data: np.ndarray, bin_size: int) -> np.ndarray:
        """
        Jitted compute of absolute angular change in the smallest possible time bin.

        .. note::
            If the smallest possible frame-to-frame time-bin in Video 1 is 33ms (recorded at 30fps), and the
            smallest possible frame-to-frame time-bin in Video 2 is 66ms (recorded at 15fps), we correct for
            this across recordings using the ``bin_size`` argument. E.g., when passing angular data from Video 1
            we pass bin_size as ``2``, and when passing angular data for Video 2 we set bin_size to ``1`` to
            allow comparisons of instantaneous angular velocity between Video 1 and Video 2.

            When current frame minus bin_size results in a negative index, -1 is returned.

        .. image:: _static/img/instantaneous_angular_velocity.png
           :width: 600
           :align: center

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.instantaneous_angular_velocity`


        :parameter ndarray data: 1D array of size len(frames) representing degrees.
        :parameter int bin_size: The number of frames prior to compare the current angular velocity against.
        :returns: 1D array with instantanous angular velocities according to bin size.
        :rtype: np.ndarray

        :example:
        >>> data = np.array([350, 360, 365, 360]).astype(np.float32)
        >>> CircularStatisticsMixin().instantaneous_angular_velocity(data=data, bin_size=1.0)
        >>> [-1., 10.00002532, 4.999999, 4.999999]
        >>> CircularStatisticsMixin().instantaneous_angular_velocity(data=data, bin_size=2)
        >>> [-1., -1., 15.00002432, 0.]
        """
        data = np.deg2rad(data)
        results = np.full((data.shape[0]), -1)
        left_idx, right_idx = 0, bin_size
        for end_idx in prange(right_idx, data.shape[0] + 1, 1):
            v = np.rad2deg(
                np.pi - np.abs(np.pi - np.abs(data[left_idx] - data[end_idx]))
            )
            results[end_idx] = int(np.round(v, 4))
            left_idx += 1
        return results

    @staticmethod
    @njit("(float32[:],)")
    def degrees_to_cardinal(data: np.ndarray) -> List[str]:
        """
        Convert degree angles to cardinal direction bucket e.g., 0 -> "N", 180 -> "S"

        .. note::
           To convert cardinal literals to integers, map using :func:`simba.utils.enums.lookups.cardinality_to_integer_lookup`.
           To convert integers to cardinal literals, map using :func:`simba.utils.enums.lookups.integer_to_cardinality_lookup`.

        .. image:: _static/img/degrees_to_cardinal.png
           :width: 600
           :align: center

        :parameter np.ndarray degree_angles: 1d array of degrees. Note: return by ``self.head_direction``.
        :return: List of strings representing frame-wise cardinality.
        :rtype: List[str]

        :example:
        >>> data = np.array(list(range(0, 405, 45))).astype(np.float32)
        >>> CircularStatisticsMixin().degrees_to_cardinal(degree_angles=data)
        >>> ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']
        """

        DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        results = typed.List(["str"])
        for i in prange(data.shape[0]):
            ix = round(data[i] / (360.0 / len(DIRECTIONS)))
            direction = DIRECTIONS[ix % len(DIRECTIONS)]
            results.append(direction)
        return results[1:]

    @staticmethod
    @njit("(int32[:,:], float64, float64)")
    def sliding_bearing(x: np.ndarray, lag: float, fps: float) -> np.ndarray:
        """
        Calculates the sliding bearing (direction) of movement in degrees for a sequence of 2D points representing a single body-part.

        .. note::
           To calculate frame-by-frame bearing, pass fps == 1 and lag == 1.

        .. image:: _static/img/sliding_bearing.png
           :width: 600
           :align: center

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.sliding_bearing`

        :param np.ndarray x: An array of shape (n, 2) representing the time-series sequence of 2D points.
        :param float lag: The lag time (in seconds) used for calculating the sliding bearing. E.g., if 1, then bearing will be calculated using coordinates in the current frame vs the frame 1s previously.
        :param float fps: The sample rate (frames per second) of the sequence.
        :return: An array containing the sliding bearings (in degrees) for each point in the sequence.
        :rtype: np.ndarray

        :example:
        >>> x = np.array([[10, 10], [20, 10]])
        >>> CircularStatisticsMixin.sliding_bearing(x=x, lag=1, fps=1)
        >>> [-1. 90.]
        """

        results = np.full((x.shape[0]), -1.0)
        lag = int(lag * fps)
        for i in range(lag, x.shape[0]):
            x1, y1 = x[i - lag, 0], x[i - lag, 1]
            x2, y2 = x[i, 1], x[i, 1]
            bearing = np.degrees(np.arctan2(x2 - x1, y2 - y1))
            results[i] = (bearing + 360) % 360
        return results

    @staticmethod
    @njit("(float32[:,:], float32[:, :], float32[:, :])")
    def direction_three_bps(nose_loc: np.ndarray, left_ear_loc: np.ndarray, right_ear_loc: np.ndarray) -> np.ndarray:
        """
        Jitted helper to compute the degree angle from three body-parts. Computes the angle in degrees left_ear <-> nose
        and right_ear_nose and returns the midpoint.

        .. image:: _static/img/angle_from_3_bps.png
          :width: 600
          :align: center

        .. seealso:
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.direction_two_bps`
           :func:`simba.data_processors.cuda.circular_statistics.direction_from_two_bps`
           :func:`simba.data_processors.cuda.circular_statistics.direction_from_three_bps`

        :param ndarray nose_loc: 2D array of size len(frames)x2 representing nose coordinates
        :param ndarray left_ear_loc: 2D array of size len(frames)x2 representing left ear coordinates
        :param ndarray right_ear_loc: 2D array of size len(frames)x2 representing right ear coordinates
        :return: Array of size nose_loc.shape[0] with direction in degrees.
        :rtype: np.ndarray

        :example:
        >>> nose_loc = np.random.randint(low=0, high=500, size=(50, 2)).astype(np.float32)
        >>> left_ear_loc = np.random.randint(low=0, high=500, size=(50, 2)).astype(np.float32)
        >>> right_ear_loc = np.random.randint(low=0, high=500, size=(50, 2)).astype(np.float32)
        >>> results = CircularStatisticsMixin().direction_three_bps(nose_loc=nose_loc, left_ear_loc=left_ear_loc, right_ear_loc=right_ear_loc)
        """

        results = np.full((nose_loc.shape[0]), np.nan)
        for i in prange(nose_loc.shape[0]):
            left_ear_to_nose = np.arctan2(
                nose_loc[i][0] - left_ear_loc[i][0], left_ear_loc[i][1] - nose_loc[i][1]
            )
            right_ear_nose = np.arctan2(
                nose_loc[i][0] - right_ear_loc[i][0],
                right_ear_loc[i][1] - nose_loc[i][1],
            )
            mean_angle_rad = np.arctan2(
                np.sin(left_ear_to_nose) + np.sin(right_ear_nose),
                np.cos(left_ear_to_nose) + np.cos(right_ear_nose),
            )
            results[i] = (np.degrees(mean_angle_rad) + 360) % 360
        return results

    @staticmethod
    def three_point_direction(nose_loc: np.ndarray,
                              left_ear_loc: np.ndarray,
                              right_ear_loc: np.ndarray) -> np.ndarray:
        """
        Calculate animal heading direction using three anatomical landmarks with input validation.

        Computes the mean directional angle of an animal based on nose and ear coordinates
        using circular statistics. Provides a robust estimate of the animal's facing direction by calculating individual directional vectors from each ear to the nose, then computing their
        circular mean to handle angular discontinuities properly.

        The function serves as a validated wrapper around the underlying numba-accelerated implementation ensuring input data meet requirements before computation.

        .. seealso::
           For the underlying numba-accelerated implementation, see :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.direction_three_bps`.
           For two-point direction calculation, see :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.direction_two_bps`.

        .. image:: _static/img/angle_from_3_bps.png
           :width: 600
           :align: center

        .. csv-table::
           :header: EXPECTED RUNTIMES
           :file: ../../docs/tables/three_point_direction.csv
           :widths: 10, 45, 45
           :align: center
           :header-rows: 1

        :param np.ndarray nose_loc: 2D array with shape (n_frames, 2) containing [x, y] pixel coordinates  of the nose for each frame. Must contain non-negative numeric values.
        :param np.ndarray left_ear_loc: 2D array with shape (n_frames, 2) containing [x, y] pixel coordinates  of the left ear for each frame. Must have the same number of frames as nose_loc.
        :param np.ndarray right_ear_loc: 2D array with shape (n_frames, 2) containing [x, y] pixel coordinates of the right ear for each frame. Must have the same number of frames as nose_loc.
        :return: 1D array with shape (n_frames,) containing directional angles in degrees [0, 360)  for each frame. Contains NaN values for frames where computation fails.
        :rtype: np.ndarray

        :example:
        >>> nose_loc = np.array([[100, 150], [102, 148], [105, 145]], dtype=np.float32)
        >>> left_ear_loc = np.array([[95, 160], [97, 158], [100, 155]], dtype=np.float32)
        >>> right_ear_loc = np.array([[105, 160], [107, 158], [110, 155]], dtype=np.float32)
        >>> directions = CircularStatisticsMixin.direction_three_bps( nose_loc=nose_loc, left_ear_loc=left_ear_loc, right_ear_loc=right_ear_loc)
        """

        check_valid_array(data=nose_loc, source=f'{CircularStatisticsMixin.direction_three_bps.__name__} nose_loc', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_value=0.0, raise_error=True, accepted_axis_1_shape=[2,])
        check_valid_array(data=left_ear_loc, source=f'{CircularStatisticsMixin.three_point_direction.__name__} left_ear_loc', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_value=0.0, raise_error=True, accepted_axis_0_shape=(nose_loc.shape[0],), accepted_axis_1_shape=[2,])
        check_valid_array(data=right_ear_loc, source=f'{CircularStatisticsMixin.three_point_direction.__name__} right_ear_loc', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_value=0.0, raise_error=True, accepted_axis_0_shape=(right_ear_loc.shape[0],), accepted_axis_1_shape=[2,])

        results = CircularStatisticsMixin().direction_three_bps(nose_loc=nose_loc.astype(np.float32),
                                                                left_ear_loc=left_ear_loc.astype(np.float32),
                                                                right_ear_loc=right_ear_loc.astype(np.float32))
        return results

    @staticmethod
    @njit("float32[:](float32[:, :], float32[:, :])", fastmath=True, parallel=True)
    def direction_two_bps(anterior_loc: np.ndarray, posterior_loc: np.ndarray) -> np.ndarray:
        """
        Compute directional angle from two body parts using numba acceleration.

        Calculates frame-wise directionality between two anatomical landmarks, such as nape to nose
        or swim bladder to tail. Uses arctangent to determine the heading direction in degrees.

        .. image:: _static/img/angle_from_2_bps.png
           :width: 1200
           :align: center

        .. seealso::
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.direction_three_bps`
           :func:`simba.data_processors.cuda.circular_statistics.direction_from_two_bps`
           :func:`simba.data_processors.cuda.circular_statistics.direction_from_three_bps`


        :parameter np.ndarray anterior_loc: Size len(frames) x 2 representing x and y coordinates for first body-part.
        :parameter np.ndarray posterior_loc: Size len(frames) x 2 representing x and y coordinates for second body-part.
        :return np.ndarray: Frame-wise directionality in degrees.

        :example:
        >>> swim_bladder_loc = np.random.randint(low=0, high=500, size=(50, 2)).astype(np.float32)
        >>> tail_loc = np.random.randint(low=0, high=500, size=(50, 2)).astype(np.float32)
        >>> CircularStatisticsMixin().direction_two_bps(anterior_loc=swim_bladder_loc, posterior_loc=tail_loc)
        """

        results = np.full((anterior_loc.shape[0]), np.nan)
        for i in prange(anterior_loc.shape[0]):
            angle_degrees = np.degrees(np.arctan2(anterior_loc[i][0] - posterior_loc[i][0], posterior_loc[i][1] - anterior_loc[i][1]))
            results[i] = angle_degrees + 360 if angle_degrees < 0 else angle_degrees
        return results

    @staticmethod
    def two_point_direction(anterior_loc: np.ndarray, posterior_loc: np.ndarray) -> np.ndarray:

        """
        Calculate directional angles between two body parts.

        Computes frame-wise directional angles from posterior to anterior body parts (e.g., tail to nose, nape to head) using arctangent calculations.

        It is a validated wrapper around the optimized numba implementation.

        .. image:: _static/img/angle_from_2_bps.png
           :width: 1200
           :align: center

        .. seealso::
           For the underlying numba-accelerated implementation, see :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.direction_two_bps`
           For three-point direction calculation, see :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.three_point_direction` or :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.direction_three_bps`

        .. csv-table::
           :header: EXPECTED RUNTIMES
           :file: ../../docs/tables/two_point_direction.csv
           :widths: 10, 45, 45
           :align: center
           :header-rows: 1

        :param np.ndarray anterior_loc: 2D array with shape (n_frames, 2) containing [x, y] coordinates for the anterior body part (e.g., nose, head). Must contain non-negative numeric values.
        :param np.ndarray posterior_loc : np.ndarray 2D array with shape (n_frames, 2) containing [x, y] coordinates for the posterior body part (e.g., tail base, nape). Must contain non-negative numeric values.
        :return: 1D array with shape (n_frames,) containing directional angles in degrees [0, 360)  for each frame at type float32. Contains NaN values for frames where computation fails.
        :rtype: np.ndarray
        """

        check_valid_array(data=anterior_loc, source=f'{CircularStatisticsMixin.two_point_direction.__name__} anterior_loc', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_value=0.0, raise_error=True, accepted_axis_1_shape=[2, ])
        check_valid_array(data=posterior_loc, source=f'{CircularStatisticsMixin.two_point_direction.__name__} posterior_loc', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_value=0.0, raise_error=True, accepted_axis_0_shape=(anterior_loc.shape[0],), accepted_axis_1_shape=[2, ])
        results = CircularStatisticsMixin().direction_two_bps(anterior_loc=anterior_loc.astype(np.float32), posterior_loc=posterior_loc.astype(np.float32))

        return results


    @staticmethod
    @njit("(float32[:],)")
    def rayleigh(data: np.ndarray) -> Tuple[float, float]:

        r"""
        Jitted compute of Rayleigh Z (test of non-uniformity) of single sample of circular data in degrees.

        .. note::
           Adapted from ``pingouin.circular.circ_rayleigh`` and ``pycircstat.tests.rayleigh``.


        The Rayleigh Z score is calculated as follows:

        .. math::
           Z = nR^2

        where :math:`n` is the sample size and :math:`R` is the mean resultant length.

        The associated p-value is calculated as follows:

        .. math::
           p = e^{\\sqrt{1 + 4n + 4(n^2 - R^2)} - (1 + 2n)}

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.sliding_rayleigh_z`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_rayleigh_z`

        :parameter ndarray data: 1D array of size len(frames) representing degrees.
        :returns: Tuple with Rayleigh Z score and associated probability value.
        :rtype: Tuple[float, float]

        >>> data = np.array([350, 360, 365, 360, 100, 109, 232, 123, 42, 3,4, 145]).astype(np.float32)
        >>> CircularStatisticsMixin().rayleigh(data=data)
        >>> (2.3845645695246467, 0.9842236169985417)
        """

        data = np.deg2rad(data)
        R = np.sqrt(np.sum(np.cos(data)) ** 2 + np.sum(np.sin(data)) ** 2) / len(data)
        p = np.exp(
            np.sqrt(1 + 4 * len(data) + 4 * (len(data) ** 2 - R**2))
            - (1 + 2 * len(data))
        )
        return len(data) * R**2, p

    @staticmethod
    @njit("(float32[:], float64[:], float64)", parallel=True)
    def sliding_rayleigh_z(data: np.ndarray, time_windows: np.ndarray, fps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Jitted compute of Rayleigh Z (test of non-uniformity) of circular data within sliding time-window.

        .. note::
           Adapted from ``pingouin.circular.circ_rayleigh`` and ``pycircstat.tests.rayleigh``.

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.sliding_rayleigh_z`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.rayleigh`

        :parameter ndarray data: 1D array of size len(frames) representing degrees.
        :parameter np.ndarray time_window: Rolling time-window as float in seconds. Two windows of 0.5s and 1s would be represented as np.array([0.5, 1.0])
        :parameter int fps: fps of the recorded video
        :returns: Two 2d arrays with the first representing Rayleigh Z scores and second representing associated p values.
        :rtype: Tuple[np.ndarray, np.ndarray]

        :example:
        >>> data = np.random.randint(low=0, high=361, size=(100,)).astype(np.float32)
        >>> CircularStatisticsMixin().sliding_rayleigh_z(data=data, time_windows=np.array([0.5, 1.0]), fps=10)
        """

        data = np.deg2rad(data)
        Z_results, P_results = np.full((data.shape[0], time_windows.shape[0]), 0.0), np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for i in range(time_windows.shape[0]):
            win_size = int(time_windows[i] * fps)
            for j in prange(win_size, len(data) + 1):
                data_win = data[j - win_size : j]
                R = np.sqrt(np.sum(np.cos(data_win)) ** 2 + np.sum(np.sin(data_win)) ** 2) / len(data_win)
                Z_results[j - 1][i] = len(data_win) * R**2
                P_results[j - 1][i] = np.exp(np.sqrt(1 + 4 * len(data_win) + 4 * (len(data_win) ** 2 - R**2)) - (1 + 2 * len(data_win)))
        return Z_results, P_results

    @staticmethod
    @njit("(float32[:], float32[:],)")
    def circular_correlation(sample_1: np.ndarray, sample_2: np.ndarray) -> float:

        r"""
        Jitted compute of circular correlation coefficient of two samples using the cross-correlation coefficient.
        Ranges from -1 to 1: 1 indicates perfect positive correlation, -1 indicates perfect negative correlation, 0
        indicates no correlation.

        The circular correlation coefficient is calculated as:

        .. math::
           R = \\frac{\\sum \\sin(\\theta_1 - \\bar{\\theta}_1) \\sin(\\theta_2 - \\bar{\\theta}_2)}
           {\\sqrt{\\sum \\sin^2(\\theta_1 - \\bar{\\theta}_1) \\sum \\sin^2(\\theta_2 - \\bar{\\theta}_2)}}

           R = \frac{\sum \sin(\theta_1 - \bar{\theta}_1) \sin(\theta_2 - \bar{\theta}_2)}{\sqrt{\sum \sin^2(\theta_1 - \bar{\theta}_1) \sum \sin^2(\theta_2 - \bar{\theta}_2)}}

        Where:

        - :math:`\theta_1` and :math:`\theta_2` are the angles (in radians) from `sample_1` and `sample_2`, respectively
        - :math:`\bar{\theta}_1` and :math:`\bar{\theta}_2` are the mean directions of `sample_1` and `sample_2`, respectively
        - :math:`R` is the circular correlation coefficient ranging from -1 to 1

        .. note::
           Adapted from ``astropy.stats.circstats.circcorrcoef``.

        .. image:: _static/img/circular_correlation.png
           :width: 700
           :align: center

        .. seealso:
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_correlation`

        :param np.ndarray sample_1: Angular data for e.g., Animal 1
        :param np.ndarray sample_1: Angular data for e.g., Animal 2
        :return: The correlation between the two distributions.
        :rtype: float

        :example:
        >>> sample_1 = np.array([50, 90, 20, 60, 20, 90]).astype(np.float32)
        >>> sample_2 = np.array([50, 90, 70, 60, 20, 90]).astype(np.float32)
        >>> CircularStatisticsMixin().circular_correlation(sample_1=sample_1, sample_2=sample_2)
        >>> 0.7649115920066833

        :references:
           .. [1] Mardia, K. V. (1976). Linear-circular correlation coefficients and rhythmometry. Biometrika, 63(2), 403–405
        """

        sample_1, sample_2 = np.deg2rad(sample_1), np.deg2rad(sample_2)
        m1 = np.arctan2(np.mean(np.sin(sample_1)), np.mean(np.cos(sample_1)))
        m2 = np.arctan2(np.mean(np.sin(sample_2)), np.mean(np.cos(sample_2)))
        sin_1, sin_2 = np.sin(sample_1 - m1), np.sin(sample_2 - m2)
        return np.abs(np.sum(sin_1 * sin_2) / np.sqrt(np.sum(sin_1 * sin_1) * np.sum(sin_2 * sin_2)))

    @staticmethod
    @njit("(float32[:], float32[:], float64[:], int64)")
    def sliding_circular_correlation(sample_1: np.ndarray, sample_2: np.ndarray, time_windows: np.ndarray, fps: float) -> np.ndarray:

        r"""
        Jitted compute of correlations between two angular distributions in sliding time-windows
        using the cross-correlation coefficient.

        .. image:: _static/img/cicle_correlation.png
           :width: 800
           :align: center

        .. note::
           Values prior to the ending of the first time window will be filles with ``0``.

        .. math::
            r = \frac{\sum \sin(\theta_1 - \bar{\theta_1}) \cdot \sin(\theta_2 - \bar{\theta_2})}{\sqrt{\sum \sin^2(\theta_1 - \bar{\theta_1}) \cdot \sum \sin^2(\theta_2 - \bar{\theta_2})}}

        Where:
        - :math:`r` is the circular correlation coefficient.
        - :math:`\theta_1` and :math:`\theta_2` are the angular data points from the two samples.
        - :math:`\bar{\theta_1}` and :math:`\bar{\theta_2}` are the mean angles of the two samples.

        .. seealso:
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.circular_correlation`

        :param np.ndarray sample_1: Angular data for e.g., Animal 1
        :param np.ndarray sample_1: Angular data for e.g., Animal 2
        :param float time_windows: Size of sliding time window in seconds. E.g., two windows of 0.5s and 1s would be represented as np.array([0.5, 1.0])
        :param int fps: Frame-rate of recorded video.
        :return: Array of size len(sample_1) x len(time_window) with correlation coefficients.
        :rtype: np.ndarray

        :example:
        >>> sample_1 = np.random.randint(0, 361, (200,)).astype(np.float32)
        >>> sample_2 = np.random.randint(0, 361, (200,)).astype(np.float32)
        >>> CircularStatisticsMixin().sliding_circular_correlation(sample_1=sample_1, sample_2=sample_2, time_windows=np.array([0.5, 1.0]), fps=10.0)
        """

        sample_1, sample_2 = np.deg2rad(sample_1), np.deg2rad(sample_2)
        results = np.full((sample_1.shape[0], time_windows.shape[0]), 0.0)
        for i in prange(time_windows.shape[0]):
            win_size = int(time_windows[i] * fps)
            for j in prange(win_size, sample_1.shape[0] + 1):
                data_1_window = sample_1[j - win_size : j]
                data_2_window = sample_2[j - win_size : j]
                m1 = np.arctan2(np.mean(np.sin(data_1_window)), np.mean(np.cos(data_1_window)))
                m2 = np.arctan2(np.mean(np.sin(data_2_window)), np.mean(np.cos(data_2_window)))
                sin_1, sin_2 = np.sin(data_1_window - m1), np.sin(data_2_window - m2)
                denominator = np.sqrt(np.sum(sin_1 * sin_1) * np.sum(sin_2 * sin_2))
                numerator = np.sum(sin_1 * sin_2)
                if denominator == 0:
                    results[j - 1][i] = 0.0
                else:
                    results[j - 1][i] = np.abs(numerator / denominator)

        return results.astype(np.float32)

    @staticmethod
    @njit("(float32[:], float64[:], int64)")
    def sliding_angular_diff(data: np.ndarray, time_windows: np.ndarray, fps: float) -> np.ndarray:

        """
        Computes the angular difference in the current frame versus N seconds previously.
        For example, if the current angle is 45 degrees, and the angle N seconds previously was 350 degrees, then the difference
        is 55 degrees.

        .. note::
           Frames where current frame - N seconds prior equal a negative value is populated with ``0``.

           Results are returned in rounded nearest integer.

        .. image:: _static/img/sliding_angular_difference.png
           :width: 600
           :align: center

        :parameter ndarray data: 1D array of size len(frames) representing degrees. Can be computed by `CircularStatisticsMixin().direction_three_bps` or `CircularStatisticsMixin().direction_two_bps`.
        :parameter np.ndarray time_window: Rolling time-window as float in seconds.
        :parameter int fps: fps of the recorded video

        :example:
        >>> data = np.array([350, 350, 1, 1]).astype(np.float32)
        >>> CircularStatisticsMixin().sliding_angular_diff(data=data, fps=1.0, time_windows=np.array([1.0]))
        """

        data = np.deg2rad(data)
        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for time_window_cnt in prange(time_windows.shape[0]):
            window_size = int(time_windows[time_window_cnt] * fps)
            for left, right in zip(
                range(0, (data.shape[0] - window_size)),
                range(window_size, data.shape[0] + 1),
            ):
                distance = np.pi - np.abs(np.pi - np.abs(data[left] - data[right]))
                results[right][time_window_cnt] = np.abs(np.rint(np.rad2deg(distance)))

        return results.astype(np.int64)

    @staticmethod
    @njit("(float32[:], float64[:], int64)")
    def agg_angular_diff_timebins(
        data: np.ndarray, time_windows: np.ndarray, fps: int
    ) -> np.ndarray:
        """
        Compute the difference between the median angle in the current time-window versus the previous time window.
        For example, computes the difference between the mean angle in the first 1s of the video versus
        the second 1s of the video, the second 1s of the video versus the third 1s of the video, ... etc.

        .. note::
           The first time-bin of the video cannot be compared against the prior time-bin and is populated with `0`.

        .. image:: _static/img/circular_difference_time_bins.png
           :width: 800
           :align: center

        :parameter ndarray data: 1D array of size len(frames) representing degrees.
        :parameter np.ndarray time_window: Rolling time-window as float in seconds.
        :parameter int fps: fps of the recorded video

        :example:
        >>> data = np.random.normal(loc=45, scale=3, size=20).astype(np.float32)
        >>> CircularStatisticsMixin().agg_angular_diff_timebins(data=data,time_windows=np.array([1.0]), fps=5.0))
        """

        data = np.deg2rad(data)
        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for win_cnt, time_window_cnt in enumerate(prange(time_windows.shape[0])):
            window_size = int(time_windows[time_window_cnt] * fps)
            w1_start, w1_end = 0, window_size
            w2_start, w2_end = window_size, window_size * 2
            while w2_start < data.shape[0]:
                current_data = data[w2_start:w2_end]
                prior_data = data[w1_start:w1_end]
                prior_median = np.arctan2(
                    np.sum(np.sin(prior_data)), np.sum(np.cos(prior_data))
                )
                if prior_median < 0:
                    prior_median += 2 * np.pi
                current_median = np.arctan2(
                    np.sum(np.sin(current_data)), np.sum(np.cos(current_data))
                )
                if current_median < 0:
                    current_median += 2 * np.pi
                distance = np.pi - np.abs(np.pi - np.abs(prior_median - current_median))
                results[w2_start:w2_end, win_cnt] = np.rad2deg(distance)
                w1_start, w1_end = w2_start, w2_end
                w2_start, w2_end = w2_start + window_size, w2_end + window_size

        return np.rint(results)

    @staticmethod
    @njit("(float32[:],)", parallel=True)
    def rao_spacing(data: np.array):
        """
        Jitted compute of Rao's spacing for angular data.

        Computes the uniformity of a circular dataset in degrees. Low output values represent concentrated angularity,
        while high values represent dispersed angularity.

        The Rao's Spacing (:math:`U`) is calculated as follows:

        .. math::

           U = \\frac{1}{2} \\sum_{i=1}^{N} |l - T_i|

        where :math:`N` is the number of data points in the sliding window, :math:`T_i` is the spacing between adjacent data points, and :math:`l` is the equal angular spacing.


        :parameter ndarray data: 1D array of size len(frames) with data in degrees.
        :return: Rao's spacing measure, indicating the dispersion or concentration of angular data points.
        :rtype: int

        .. seealso::
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_rao_spacing`

        :references:
        .. [1] `UCSB <https://jammalam.faculty.pstat.ucsb.edu/html/favorite/test.htm>`__.

        :example:
        >>> data = np.random.randint(0, 360, (5000,)).astype(np.float32)
        >>> CircularStatisticsMixin.rao_spacing(data=data)
        """

        data = np.sort(data)
        Ti, TiL = np.full((data.shape[0]), np.nan), np.full((data.shape[0]), np.nan)
        l = np.int8(360 / len(data))
        Ti[-1] = np.rad2deg(np.pi - np.abs(np.pi - np.abs(np.deg2rad(data[0]) - np.deg2rad(data[-1])))
        )
        for j in range(data.shape[0] - 1, -1, -1):
            Ti[j] = np.rad2deg(
                np.pi
                - np.abs(np.pi - np.abs(np.deg2rad(data[j]) - np.deg2rad(data[j - 1])))
            )
        for k in prange(Ti.shape[0]):
            TiL[int(k)] = max((l, Ti[k])) - min((l, Ti[k]))
        S = np.sum(TiL)
        U = int(S / 2)
        return U

    @staticmethod
    @njit("(float32[:], float64[:], int64)", parallel=True)
    def sliding_rao_spacing(data: np.ndarray, time_windows: np.ndarray, fps: int) -> np.ndarray:
        """
        Jitted compute of the uniformity of a circular dataset in sliding windows.

        :param ndarray data: 1D array of size len(frames) representing degrees.
        :param np.ndarray time_window: Rolling time-window as float in seconds.
        :param int fps: fps of the recorded video
        :return np.ndarray: representing rao-spacing U in every sliding windows [-window:n]

        .. image:: _static/img/raospacing.png
           :width: 800
           :align: center

        The Rao's Spacing (:math:`U`) is calculated as follows:

        .. math::

           U = \\frac{1}{2} \\sum_{i=1}^{N} |l - T_i|

        where :math:`N` is the number of data points in the sliding window, :math:`T_i` is the spacing between adjacent data points, and :math:`l` is the equal angular spacing.

        .. note::
           For frames occuring before a complete time window, 0.0 is returned.

        .. seealso::
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.rao_spacing`

        :references:
        .. [1] `UCSB <https://jammalam.faculty.pstat.ucsb.edu/html/favorite/test.htm>`__.

        :example:
        >>> data = np.random.randint(low=0, high=360, size=(500,)).astype(np.float32)
        >>> result = CircularStatisticsMixin().sliding_rao_spacing(data=data, time_windows=np.array([0.5, 1.0]), fps=10)
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for win_cnt in prange(time_windows.shape[0]):
            window_size = int(time_windows[win_cnt] * fps)
            for i in range(window_size, data.shape[0]):
                w_data = np.sort(data[i - window_size : i])
                Ti, TiL = np.full((w_data.shape[0]), np.nan), np.full((w_data.shape[0]), np.nan)
                l = np.int16(360 / len(w_data))
                Ti[-1] = np.rad2deg(np.pi - np.abs(np.pi - np.abs(np.deg2rad(w_data[0]) - np.deg2rad(w_data[-1]))))
                for j in range(w_data.shape[0] - 1, -1, -1):
                    Ti[j] = np.rad2deg(np.pi - np.abs(np.pi - np.abs(np.deg2rad(w_data[j]) - np.deg2rad(w_data[j - 1]))))
                for k in prange(Ti.shape[0]):
                    TiL[int(k)] = max((l, Ti[k])) - min((l, Ti[k]))
                S = np.sum(TiL)
                U = int(S / 2)
                results[i][win_cnt] = U
        return results

    @staticmethod
    @njit("(float32[:], float32[:])")
    def kuipers_two_sample_test(sample_1: np.ndarray, sample_2: np.ndarray) -> float:
        """
        Compute the Kuiper's two-sample test statistic for circular distributions.

        Kuiper's two-sample test is a non-parametric test used to determine if two samples are drawn from the same circular distribution. It is particularly useful for circular data, such as angles or directions.

        The Kuiper test statistic is calculated as the sum of the maximum positive and negative deviations between the cumulative distribution functions of the two samples:

        .. math::

           V = \max(F_1(\theta) - F_2(\theta)) + \max(F_2(\theta) - F_1(\theta))

        Where:

        - :math:`F_1(\theta)` and :math:`F_2(\theta)` are the empirical cumulative distribution functions (CDFs) of the two circular samples.
        - :math:`\\theta` are the sorted angles in the two samples.

        .. note::
           Adapted from `Kuiper <https://github.com/aarchiba/kuiper/tree/master>`__ by `Anne Archibald <https://github.com/aarchiba>`_.

        .. seealso::
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_kuipers_two_sample_test`


        :param ndarray data: The first circular sample array in degrees.
        :param ndarray data: The second circular sample array in degrees.
        :return: Kuiper's test statistic.
        :rtype: float

        :example:
        >>> sample_1, sample_2 = np.random.normal(loc=45, scale=1, size=100).astype(np.float32), np.random.normal(loc=180, scale=20, size=100).astype(np.float32)
        >>> CircularStatisticsMixin().kuipers_two_sample_test(sample_1=sample_1, sample_2=sample_2)
        """

        sample_1, sample_2 = np.deg2rad(np.sort(sample_1)), np.deg2rad(np.sort(sample_2))
        cdfv1 = np.searchsorted(sample_2, sample_1) / float(len(sample_2))
        cdfv2 = np.searchsorted(sample_1, sample_2) / float(len(sample_1))
        return np.amax(cdfv1 - np.arange(len(sample_1)) / float(len(sample_1))) + np.amax(cdfv2 - np.arange(len(sample_2)) / float(len(sample_2)))

    @staticmethod
    @njit("(float32[:], float32[:], float64[:], int64)")
    def sliding_kuipers_two_sample_test(sample_1: np.ndarray, sample_2: np.ndarray, time_windows: np.ndarray, fps: int) -> np.ndarray:
        """
        Jitted compute of Kuipers two-sample test comparing two distributions with sliding time window.

        This function calculates the Kuipers two-sample test statistic for each time window, sliding through the given circular data sequences.

        .. seealso::
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.kuipers_two_sample_test`

        :param np.ndarray data: The first circular sample array in degrees.
        :param np.ndarray data: The second circular sample array in degrees.
        :param np.ndarray time_windows: An array containing the time window sizes (in seconds) for which the Kuipers two-sample test will be computed.
        :param int fps: The frames per second, representing the sampling rate of the data.
        :returns: A 2D array containing the Kuipers two-sample test statistics for each time window and each time step.
        :rtype: np.ndarray

        :examples:
        >>> data = np.random.randint(low=0, high=360, size=(100,)).astype(np.float64)
        >>> D = CircularStatisticsMixin().sliding_kuipers_two_sample_test(data=data, time_windows=np.array([0.5, 5]), fps=2)
        """
        sample_1, sample_2 = np.deg2rad(sample_1), np.deg2rad(sample_2)
        results = np.full((sample_1.shape[0], time_windows.shape[0]), -1.0)
        for time_window_cnt in prange(time_windows.shape[0]):
            win_size = int(time_windows[time_window_cnt] * fps)
            for i in range(win_size, sample_1.shape[0]):
                sample_1_win, sample_2_win = (
                    sample_1[i - win_size : i],
                    sample_2[i - win_size : i],
                )
                cdfv1 = np.searchsorted(sample_2, sample_1_win) / float(
                    len(sample_2_win)
                )
                cdfv2 = np.searchsorted(sample_1_win, sample_2_win) / float(
                    len(sample_1_win)
                )
                D = np.amax(
                    cdfv1 - np.arange(len(sample_1_win)) / float(len(sample_1_win))
                ) + np.amax(
                    cdfv2 - np.arange(len(sample_2_win)) / float(len(sample_2_win))
                )
                results[i][time_window_cnt] = D

        return results

    @staticmethod
    def sliding_hodges_ajne(data: np.ndarray, time_window: float, fps: int) -> np.ndarray:

        data = np.deg2rad(data)
        results, window_size = np.full((data.shape[0]), -1.0), int(time_window * fps)
        for i in range(window_size, data.shape[0]):
            w_data = data[i - window_size : i]
            v = 1 - np.abs(np.mean(np.exp(1j * w_data)))
            n = len(w_data)
            H = n * (1 - v)
            results[i] = H
        return results

    @staticmethod
    def hodges_ajne(sample: np.ndarray):
        v = 1 - np.abs(np.mean(np.exp(1j * sample)))
        n = len(sample)
        H = n * (1 - v)
        return H

    @staticmethod
    @jit(nopython=True)
    def watson_williams_test(sample_1: np.ndarray, sample_2: np.ndarray):

        variance1 = 1 - np.abs(np.mean(np.exp(1j * sample_1)))
        variance2 = 1 - np.abs(np.mean(np.exp(1j * sample_2)))
        numerator = (variance1 + variance2) / 2
        denominator = (variance1**2 / len(sample_1)) + (variance2**2 / len(sample_2))
        F = numerator / denominator
        return F

    @staticmethod
    def watsons_u(data: np.ndarray):
        data = np.deg2rad(data)
        mean_vector = np.exp(1j * data).mean()
        n = len(data)
        return n * (1 - np.abs(mean_vector))

    @staticmethod
    @njit("(float32[:],)")
    def circular_range(data: np.ndarray) -> float:
        r"""
        Jitted compute of circular range in degrees. The range is defined as the angular span of the
        shortest arc that can contain all the data points. A smaller range indicates a more concentrated
        distribution, while a larger range suggests a more dispersed distribution.

        .. math::
           \text{Range} = \min \left( 2\pi - \max(\delta \theta_i), \theta_{\text{max}} - \theta_{\text{min}} \right)

        where:

        - :math:`\delta \theta_i` is the difference between consecutive angular data points.
        - :math:`\theta_{\text{max}}` and :math:`\theta_{\text{min}}` are the maximum and minimum angles in the data.


        .. image:: _static/img/circular_range.png
           :width: 400
           :align: center

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.sliding_circular_range`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_range`

        :param ndarray data: 1D array of circular data measured in degrees
        :return: The circular range in degrees.
        :rtype: np.ndarray

        :example:
        >>> CircularStatisticsMixin().circular_range(np.ndarray([350, 20, 60, 100]))
        >>> 110.0
        >>> CircularStatisticsMixin().circular_range(np.ndarray([110, 20, 60, 100]))
        >>> 90.0
        """

        data = np.sort(np.deg2rad(data))
        angular_diffs = np.diff(data)
        circular_range = np.max(angular_diffs)
        return np.ceil(np.rad2deg(min(2 * np.pi - circular_range, data[-1] - data[0])))

    @staticmethod
    @njit("(float32[:], float64[:], int64)")
    def sliding_circular_range(data: np.ndarray, time_windows: np.ndarray, fps: int ) -> np.ndarray:
        """
        Jitted compute of sliding circular range for a time series of circular data. The range is defined as the angular span of the
        shortest arc that can contain all the data points. Measures the circular spread of data within sliding time windows of specified duration.

        .. image:: _static/img/sliding_circular_range.png
           :width: 600
           :align: center

        .. note::
           Output data in the beginning of the series where a full time-window is not satisfied (e.g., first 9 observations when
           fps equals 10 and time_windows = [1.0], will be populated by ``0``.

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.sliding_circular_range`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.circular_range`

        :param np.ndarray data: 1D array of circular data measured in degrees
        :param np.ndarray time_windows: Size of sliding time window in seconds. E.g., two windows of 0.5s and 1s would be represented as np.array([0.5, 1.0])
        :param int fps: Frame-rate of recorded video.
        :return: Array of size len(sample_1) x len(time_window) with angular ranges in degrees.
        :rtype: np.ndarray

        :examples:
        >>> data = np.array([260, 280, 300, 340, 360, 0, 10, 350, 0, 15]).astype(np.float32)
        >>> CircularStatisticsMixin().sliding_circular_range(data=data, time_windows=np.array([0.5]), fps=10)
        >>> [[ -1.],[ -1.],[ -1.],[ -1.],[100.],[80],[70],[30],[20],[25]]
        """

        results = np.full((data.shape[0], time_windows.shape[0]), 0.0)
        for time_window_cnt in range(time_windows.shape[0]):
            win_size = int(time_windows[time_window_cnt] * fps)
            for left, right in zip(range(0, (data.shape[0] - win_size) + 1), range(win_size-1, data.shape[0])):
                sample = np.sort(np.deg2rad(data[left : right + 1]))
                angular_diffs = np.diff(sample)
                circular_range = np.max(angular_diffs)
                results[right][time_window_cnt] = np.abs(np.rint(np.rad2deg(min(2 * np.pi - circular_range, sample[-1] - sample[0]))))
        return results

    @staticmethod
    @njit("(float32[:], int64[:, :],)")
    def circular_hotspots(data: np.ndarray, bins: np.ndarray) -> np.ndarray:
        """
        Calculate the proportion of data points falling within circular bins.

        .. image:: _static/img/circular_hotspots.png
           :width: 700
           :align: center

        .. warning:
           Make sure the ``bins`` argument do not contain overlapping bin edge definitions. E.g.,
           bins = np.array([[270, 0], [1, 90], [91, 180], [181, 269]]) is accepted but
           bins = np.array([[270, 0], [0, 90], [90, 180], [180, 270]]) is not.

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.sliding_circular_hotspots`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_hotspots`

        :parameter ndarray data: 1D array of circular data measured in degrees.
        :parameter ndarray bins: 2D array of shape representing circular bins defining [start_degree, end_degree] inclusive.
        :return: 1D array containing the proportion of data points that fall within each specified circular bin.
        :rtype: np.ndarray

        :example:
        >>> data = np.array([270, 360, 10, 90, 91, 180, 185, 260]).astype(np.float32)
        >>> bins = np.array([[270, 90], [91, 269]])
        >>> CircularStatisticsMixin().circular_hotspots(data=data, bins=bins)
        >>> [0.5, 0.5]
        >>> bins = np.array([[270, 0], [1, 90], [91, 180], [181, 269]])
        >>> CircularStatisticsMixin().circular_hotspots(data=data, bins=bins)
        >>> [0.25, 0.25, 0.25, 0.25]
        """
        results = np.full((bins.shape[0]), -1.0)
        for bin_cnt in range(bins.shape[0]):
            if bins[bin_cnt][0] > bins[bin_cnt][1]:
                mask = ((data >= bins[bin_cnt][0]) & (data <= 360)) | (
                    (data >= 0) & (data <= bins[bin_cnt][1])
                )
            else:
                mask = (data >= bins[bin_cnt][0]) & (data <= bins[bin_cnt][1])
            results[bin_cnt] = data[mask].shape[0] / data.shape[0]
        return results

    @staticmethod
    @njit("(float32[:], int64[:, :], float64, float64)")
    def sliding_circular_hotspots(data: np.ndarray, bins: np.ndarray, time_window: float, fps: float) -> np.ndarray:
        """
        Jitted compute of sliding circular hotspots in a dataset. Calculates circular hotspots in a time-series dataset by sliding a time window
        across the data and computing hotspot statistics for specified circular bins.

        :parameter ndarray data: 1D array of circular data measured in degrees.
        :parameter ndarray bins: 2D array of shape representing circular bins defining [start_degree, end_degree] inclusive.
        :parameter float time_window: The size of the sliding window in seconds.
        :parameter float fps: The frame-rate of the video.

        :return np.ndarray: A 2D numpy array where each row corresponds to a time point in `data`, and each column represents a circular bin. The values in the array represent the proportion of data points within each bin at each time point.

        .. note::
           - The function utilizes the Numba JIT compiler for improved performance.
           - Circular bin definitions should follow the convention where angles are specified in degrees
             within the range [0, 360], and the bins are defined using start and end angles inclusive.
             For example, (0, 90) represents the first quadrant in a circular space.

             For example, bins = np.array([[270, 90], [91, 269]]) divides the space into a top and bottom circular space.

              bins = np.array([[0, 179], [180, 364]]) divides the space into a left and right circular space.


             Output data in the beginning of the series where a full time-window is not satisfied (e.g., first 9 observations when
             fps equals 10 and time_windows = [1.0], will be populated by ``0``.

        .. warning::
          Note that ``0`` is noted as a bin-edge, ``360`` should not be a bin-edge. Instead, use ``0`` and ``359`` or ``1`` and ``360``.

        .. image:: _static/img/sliding_circular_hotspot.png
           :width: 600
           :align: center

        .. seealso::
           :func:`simba.data_processors.cuda.circular_statistics.circular_hotspots`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_circular_hotspots`

        :example:
        >>> data = np.array([270, 360, 10, 20, 90, 91, 180, 185, 260, 265]).astype(np.float32)
        >>> bins = np.array([[270, 90], [91, 269]])
        >>> CircularStatisticsMixin().sliding_circular_hotspots(data=data, bins=bins, time_window=0.5, fps=10)
        >>> [[-1. , -1. ],
        >>>  [-1. , -1. ],
        >>>  [-1. , -1. ],
        >>>  [-1. , -1. ],
        >>>  [ 0.5,  0. ],
        >>>  [ 0.4,  0.1],
        >>>  [ 0.3,  0.2],
        >>>  [ 0.2,  0.3],
        >>>  [ 0.1,  0.4],
        >>>  [ 0. ,  0.5]]

        """
        results = np.full((data.shape[0], bins.shape[0]), 0.0)
        win_size = int(time_window * fps)
        for left, right in zip(prange(0, (data.shape[0] - win_size) + 1), prange(win_size - 1, data.shape[0] + 1)):
            sample = data[left : right + 1]
            for bin_cnt in range(bins.shape[0]):
                if bins[bin_cnt][0] > bins[bin_cnt][1]:
                    mask = ((sample >= bins[bin_cnt][0]) & (sample <= 360)) | ((sample >= 0) & (sample <= bins[bin_cnt][1]))
                else:
                    mask = (sample >= bins[bin_cnt][0]) & (sample <= bins[bin_cnt][1])
                results[right][bin_cnt] = np.float32(data[mask].shape[0] / sample.shape[0])
        return results

    @staticmethod
    @njit([(float32[:], int64), (float32[:], types.misc.Omitted(value=1))])
    def rotational_direction(data: np.ndarray, stride: int = 1) -> np.ndarray:
        """
        Jitted compute of frame-by-frame rotational direction within a 1D timeseries array of angular data.

        .. note::
           * For the first frame, no rotation is possible so is populated with -1.
           * Frame-by-frame rotations of 180° degrees are denoted as clockwise rotations.

        .. seealso::
           See :func:`~simba.data_processors.cuda.circular_statistics.rotational_direction` for GPU acceleration.

        .. image:: _static/img/rotational_direction.png
           :width: 600
           :align: center

        The result array contains values:
        - `-1`: Indicates no rotation is possible for the first frame. This serves as a placeholder since there is no prior frame to compare to.
        - `0`: Represents no change in the angular value between consecutive frames
        - `1`: Indicates an increase in the angular value (rotation in the positive direction, counterclockwise)
        - `2`: Indicates a decrease in the angular value (rotation in the negative direction, clockwise)

        :param np.ndarray data: 1D array of size len(frames) representing degrees.
        :return: An array of directional indicators.
        :rtype: numpy.ndarray

        :example:
        >>> data = np.array([45, 50, 35, 50, 80, 350, 350, 0 , 180]).astype(np.float32)
        >>> CircularStatisticsMixin().rotational_direction(data)
        >>> [-1.,  1.,  2.,  1.,  1.,  2.,  0.,  1.,  1.]
        """

        data = data % 360
        data = np.deg2rad(data)
        result, prior_idx = np.full((data.shape[0]), 0.0), 0
        for i in prange(int(stride), data.shape[0]):
            prior_angle = data[prior_idx]
            current_angle = data[i]
            angle_diff = current_angle - prior_angle

            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            if angle_diff == 0:
                result[i] = 0
            elif angle_diff > 0:
                result[i] = 1
            else:
                result[i] = 2
            prior_idx += 1
        return result.astype(np.int8)

    @staticmethod
    @njit(
        [(float64[:, :, :], int64), (float64[:, :, :], types.misc.Omitted(value=400))]
    )
    def fit_circle(data: np.ndarray, max_iterations: Optional[int] = 400) -> np.ndarray:
        """
        Fit a circle to a dataset using the least squares method.

        This function fits a circle to a dataset using the least squares method. The circle is defined
        by the equation:

        .. math::
           X^2 + Y^2 = R^2

        .. note::
           Adapted to numba JIT from `circle-fit <https://github.com/AlliedToasters/circle-fit>`_ ``hyperLSQ`` method.

        .. image:: _static/img/fit_circle.png
           :width: 600
           :align: center

        References
        ----------
        .. [1] Kanatani, Rangarajan, Hyper least squares fitting of circles and ellipses, `Computational Statistics & Data Analysis`,
               vol. 55, pp. 2197-2208, 2011.
        .. [2] Lapp, Salazar, Champagne. Automated maternal behavior during early life in rodents (AMBER) pipeline, `Scientific Reports`,
               13:18277, 2023.

        :parameter np.ndarray data: A 3D NumPy array with shape (N, M, 2). N represent frames, M represents the number of body-parts, and 2 represents x and y coordinates.
        :parameter int max_iterations: The maximum number of iterations for fitting the circle.
        :return: Array with shape (N, 3) with N representing frame and 3 representing (i) X-coordinate of the circle center, (ii) Y-coordinate of the circle center, and (iii) Radius of the circle
        :rtype: np.ndarray

        :example:
        >>> data = np.array([[[5, 10], [10, 5], [15, 10], [10, 15]]])
        >>> CircularStatisticsMixin().fit_circle(data=data, iter_max=88)
        >>> [[10, 10, 5]]
        """

        results = np.full((data.shape[0], 3), np.nan)
        for i in range(data.shape[0]):
            frm_data = data[i]
            x, y, n = frm_data[:, 0], frm_data[:, 1], frm_data.shape[0]

            Xi, Yi = x - x.mean(), y - y.mean()
            Zi = Xi * Xi + Yi * Yi
            Mxx, Mxy, Mxz = (
                (Xi * Xi).sum() / n,
                (Xi * Yi).sum() / n,
                (Xi * Zi).sum() / n,
            )
            Myy, Myz = (Yi * Yi).sum() / n, (Yi * Zi).sum() / n
            Mzz = (Zi * Zi).sum() / n

            Mz = Mxx + Myy
            Var_z = Mzz - Mz * Mz
            Cov_xy = Mxx * Myy - Mxy * Mxy

            A2 = 4 * Cov_xy - 3 * Mz * Mz - Mzz
            A1 = Var_z * Mz + 4.0 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz
            A0 = (Mxz * (Mxz * Myy - Myz * Mxy) + Myz * (Myz * Mxx - Mxz * Mxy) - Var_z * Cov_xy)
            A22 = A2 + A2

            Y, X = A0, 0.0
            if A1 == 0:
                continue

            for j in range(max_iterations):
                Dy = A1 + X * (A22 + 16.0 * (X**2))
                xnew = X - Y / Dy
                if xnew == X or not np.isfinite(xnew):
                    break
                ynew = A0 + xnew * (A1 + xnew * (A2 + 4.0 * xnew * xnew))
                if abs(ynew) >= abs(Y):
                    break
                X, Y = xnew, ynew

            det = X**2 - X * Mz + Cov_xy
            Xcenter = (Mxz * (Myy - X) - Myz * Mxy) / det / 2.0
            Ycenter = (Myz * (Mxx - X) - Mxz * Mxy) / det / 2.0

            results[i, :] = np.array(
                [
                    Xcenter + x.mean(),
                    Ycenter + y.mean(),
                    np.sqrt(abs(Xcenter**2 + Ycenter**2 + Mz)),
                ]
            )

        return results

    @staticmethod
    def preferred_turning_direction(x: np.ndarray) -> int:
        """
        Determines the preferred turning direction from a 1D array of circular directional data.

        .. note::
           The input ``x`` can be created using any of the following methods:
           - :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.direction_two_bps`
           - :func:`simba.data_processors.cuda.circular_statistics.direction_from_two_bps`
           - :func:`simba.data_processors.cuda.circular_statistics.direction_from_three_bps`
           - :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.direction_three_bps`

        .. seealso::
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.rotational_direction`, :func:`~simba.data_processors.cuda.circular_statistics.rotational_direction`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.sliding_preferred_turning_direction`

        :param np.ndarray x: 1D array of circular directional data (values between 0 and 360, inclusive). The array represents angular directions measured in degrees.
        :return:
            The most frequent turning direction from the input data:
            - `0`: No change in the angular value between consecutive frames.
            - `1`: An increase in the angular value (rotation in the positive direction, counterclockwise).
            - `2`: A decrease in the angular value (rotation in the negative direction, clockwise).
        :rtype: int

        :example:
        >>> x = np.random.randint(0, 361, (200,))
        >>> CircularStatisticsMixin.preferred_turning_direction(x=x)
        """

        check_valid_array(data=x, source=CircularStatisticsMixin.preferred_turning_direction.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        if np.max(x) > 360 or np.min(x) < 0:
            raise InvalidInputError(msg='x has to be values between 0 and 360 inclusive', source=CircularStatisticsMixin.preferred_turning_direction.__name__)
        rotational_direction = CircularStatisticsMixin.rotational_direction(data=x.astype(np.float32))
        return get_mode(x=rotational_direction)

    @staticmethod
    def sliding_preferred_turning_direction(x: np.ndarray,
                                            time_window: float,
                                            sample_rate: float) -> np.ndarray:
        """
        Computes the sliding preferred turning direction over a given time window from a 1D array of circular directional data.

        Calculates the most frequent turning direction (mode) within a sliding window  of a specified duration.

        .. seealso::
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.rotational_direction`, :func:`~simba.data_processors.cuda.circular_statistics.rotational_direction`,
           :func:`simba.mixins.circular_statistics.CircularStatisticsMixin.preferred_turning_direction`


        :param np.ndarray x: A 1D array of circular directional data (values between 0 and 360, inclusive).  Each value represents an angular direction in degrees.
        :param float time_window:  The duration of the sliding window in seconds.
        :param float sample_rate: The sampling rate of the data in Hz (samples per second) or FPS (frames per seconds)
        :return:
            A 1D array of integers indicating the preferred turning direction for each window:
            - `0`: No change in angular values within the window.
            - `1`: An increase in angular values (counterclockwise rotation).
            - `2`: A decrease in angular values (clockwise rotation).
            For indices before the first full window, the value is `-1`.
        :rtype: np.ndarray

        :example:
        >>> x = np.random.randint(0, 361, (213,))
        >>> CircularStatisticsMixin.sliding_preferred_turning_direction(x=x, time_window=1, sample_rate=10)
        """
        check_valid_array(data=x, source=CircularStatisticsMixin.sliding_preferred_turning_direction.__name__, accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        if np.max(x) > 360 or np.min(x) < 0:
            raise InvalidInputError(msg='x has to be values between 0 and 360 inclusive', source=CircularStatisticsMixin.sliding_preferred_turning_direction.__name__)
        check_float(name=f'{CircularStatisticsMixin.sliding_preferred_turning_direction.__name__} time_window', value=time_window)
        check_float(name=f'{CircularStatisticsMixin.sliding_preferred_turning_direction.__name__} sample_rate', value=sample_rate)
        rotational_directions = CircularStatisticsMixin.rotational_direction(data=x.astype(np.float32))
        window_size = np.int64(np.max((1.0, (time_window * sample_rate))))
        results = np.full(shape=(x.shape[0]), fill_value=-1, dtype=np.int32)
        for r in range(window_size, x.shape[0] + 1):
            l = r - window_size
            sample = rotational_directions[l:r]
            results[r - 1] = get_mode(x=sample)
        return results.astype(np.int32)



# data = np.array([260, 280, 300, 340, 360, 0, 10, 350, 0, 15]).astype(np.float32)
# CircularStatisticsMixin().sliding_circular_range(data=data, time_windows=np.array([0.5]), fps=10)

# data_sizes = [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]
# # #
# # for data_size in data_sizes:
# #     data = np.random.randint(low=0, high=360, size=(data_size,)).astype(np.float32)
# #     start = time.time()
# #     CircularStatisticsMixin().rayleigh(data=data)
# #     print(time.time() - start)
#
#
# for data_size in data_sizes:
#     data = np.random.randint(low=0, high=360, size=(data_size,)).astype(np.float32)
#     start = time.time()
#     CircularStatisticsMixin().rolling_rayleigh_z(data=data, fps=10, time_windows=np.array([0.5, 1.0]))
#     print(time.time() - start)


# for data_size in data_sizes:
#     data = np.random.randint(low=0, high=360, size=(data_size,)).astype(np.float32)
#     start = time.time()
#     CircularStatisticsMixin().degrees_to_cardinal(degree_angles=data)
#     print(time.time() - start)


# for data_size in data_sizes:
#     nose_loc = np.random.randint(low=0, high=500, size=(data_size, 2)).astype(np.float32)
#     left_ear_loc = np.random.randint(low=0, high=500, size=(data_size, 2)).astype(np.float32)
#     right_ear_loc = np.random.randint(low=0, high=500, size=(data_size, 2)).astype(np.float32)
#     start = time.time()
#     results = CircularStatisticsMixin().direction_two_bps(swim_bladder_loc=nose_loc,tail_loc=left_ear_loc)
#     print(time.time() - start)
