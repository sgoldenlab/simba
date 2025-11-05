__author__ = "Simon Nilsson"

import warnings

warnings.filterwarnings("ignore")
import glob
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from numba import jit, njit, prange, types
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

import simba
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_filepath_list_is_empty,
                                check_minimum_roll_windows, check_valid_array,
                                check_valid_boolean)
from simba.utils.enums import Formats, Options, Paths
from simba.utils.errors import CountError
from simba.utils.read_write import (get_bp_headers, read_config_file,
                                    read_project_path_and_file_type,
                                    read_video_info_csv)


class FeatureExtractionMixin(object):
    """
    Methods for featurizing pose-estimation data.

    :param Optional[configparser.Configparser] config_path: Optional path to SimBA project_config.ini
    """

    def __init__(self, config_path: Optional[str] = None):

        if config_path:
            self.config_path = config_path
            self.config = read_config_file(config_path=config_path)
            self.project_path, self.file_type = read_project_path_and_file_type(
                config=self.config
            )
            self.video_info_path = os.path.join(
                self.project_path, Paths.VIDEO_INFO.value
            )
            self.video_info_df = read_video_info_csv(file_path=self.video_info_path)
            self.data_in_dir = os.path.join(
                self.project_path, Paths.OUTLIER_CORRECTED.value
            )
            self.save_dir = os.path.join(
                self.project_path, Paths.FEATURES_EXTRACTED_DIR.value
            )
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            bp_path = os.path.join(self.project_path, Paths.BP_NAMES.value)
            check_file_exist_and_readable(file_path=bp_path)
            self.body_parts_lst = list(pd.read_csv(bp_path, header=None)[0])
            self.roll_windows_values = check_minimum_roll_windows(
                Options.ROLLING_WINDOW_DIVISORS.value, self.video_info_df["fps"].min()
            )
            self.files_found = glob.glob(self.data_in_dir + "/*." + self.file_type)
            check_if_filepath_list_is_empty(
                filepaths=self.files_found,
                error_msg=f"No files of type {self.file_type} found in {self.data_in_dir}",
            )
            self.col_headers = get_bp_headers(body_parts_lst=self.body_parts_lst)
            self.col_headers_shifted = [bp + "_shifted" for bp in self.col_headers]

    @staticmethod
    @jit(nopython=True)
    def euclidean_distance(bp_1_x: np.ndarray,
                           bp_2_x: np.ndarray,
                           bp_1_y: np.ndarray,
                           bp_2_y: np.ndarray,
                           px_per_mm: float) -> np.ndarray:

        """
        Compute Euclidean distance in millimeters between two body-parts.

        .. seealso::
            For improved runtime on CPU, use :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_euclidean_distance` or :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.keypoint_distances`.
            For GPU acceleration through CuPy, use :func:`simba.data_processors.cuda.statistics.get_euclidean_distance_cupy`.
            For GPU acceleration through numba CUDA, use :func:`simba.data_processors.cuda.statistics.get_euclidean_distance_cuda`.

        :param np.ndarray bp_1_x: 2D array of size len(frames) x 1 with bodypart 1 x-coordinates.
        :param np.ndarray bp_2_x: 2D array of size len(frames) x 1 with bodypart 2 x-coordinates.
        :param np.ndarray bp_1_y: 2D array of size len(frames) x 1 with bodypart 1 y-coordinates.
        :param np.ndarray bp_2_y: 2D array of size len(frames) x 1 with bodypart 2 y-coordinates.
        :return: 2D array of size len(frames) x 1 with distances between body-part 1 and body-part 2 in millimeters
        :rtype: np.ndarray

        :example:
        >>> x1, x2 = np.random.randint(1, 10, size=(10, 1)), np.random.randint(1, 10, size=(10, 1))
        >>> y1, y2 = np.random.randint(1, 10, size=(10, 1)), np.random.randint(1, 10, size=(10, 1))
        >>> FeatureExtractionMixin.euclidean_distance(bp_1_x=x1, bp_2_x=x2, bp_1_y=y1, bp_2_y=y2, px_per_mm=4.56)
        """
        series = (np.sqrt((bp_1_x - bp_2_x) ** 2 + (bp_1_y - bp_2_y) ** 2)) / px_per_mm
        return series

    @staticmethod
    @jit(nopython=True, fastmath=True, cache=True)
    def angle3pt(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
        """
        Compute 3-point angle using thre body-parts.

        .. seealso::
           For multicore numba based method across multiple observations, see :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.angle3pt_vectorized`.
           For GPU acceleration, use :func:`simba.data_processors.cuda.statistics.get_3pt_angle`.


        .. image:: _static/img/get_3pt_angle.webp
           :width: 300
           :align: center

         :param float ax: x coordinate of the first body-part (e.g., nape).
         :param float ax: y coordinate of the first body-part (e.g., nape).
         :param float bx: x coordinate of the second body-part (e.g., center).
         :param float bx: y coordinate of the second body-part (e.g., center).
         :param float cx: x coordinate of the second body-part (e.g., tail-base).
         :param float cx: y coordinate of the second body-part (e.g., tail-base).
         :return: Angle between 0-360.
         :rtype: float

        :example:
        >>> FeatureExtractionMixin.angle3pt(ax=122.0, ay=198.0, bx=237.0, by=138.0, cx=191.0, cy=109)
        >>> 59.78156901181637
        """
        ang = math.degrees(math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx))
        return ang + 360 if ang < 0 else ang

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def angle3pt_vectorized(data: np.ndarray) -> np.ndarray:
        """
        Numba accelerated compute of frame-wise 3-point angles.

        .. image:: _static/img/get_3pt_angle.webp
           :width: 300
           :align: center

        .. seealso::
           For GPU acceleration, use :func:`simba.data_processors.cuda.statistics.get_3pt_angle` for single frame alternative,
           see :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.angle3pt`

        :param ndarray data: 2D numerical array with frame number on x and [ax, ay, bx, by, cx, cy] on y.
        :return: 1d float numerical array of size data.shape[0] with angles.
        :rtype: ndarray

        :examples:
        >>> coordinates = np.random.randint(1, 10, size=(6, 6))
        >>> FeatureExtractionMixin.angle3pt_vectorized(data=coordinates)
        >>> [ 67.16634582,   1.84761027, 334.23067238, 258.69006753, 11.30993247, 288.43494882]
        """

        results = np.full((data.shape[0]), 0.0)
        for i in prange(data.shape[0]):
            angle = math.degrees(
                math.atan2(data[i][5] - data[i][3], data[i][4] - data[i][2])
                - math.atan2(data[i][1] - data[i][3], data[i][0] - data[i][2])
            )
            if angle < 0:
                angle += 360
            results[i] = angle

        return results

    @staticmethod
    def convex_hull_calculator_mp(arr: np.ndarray, px_per_mm: float) -> float:
        """
        Calculate single frame convex hull perimeter length in millimeters.

        .. image:: _static/img/framewise_perim_length.png
           :width: 300
           :align: center

        .. note::
           For acceptable run-time, call using parallel processing.

        .. seealso::
           For numba CPU based acceleration, use :func:`simba.feature_extractors.perimeter_jit.jitted_hull`.
           For multicore based acceleration, use :func:`simba.mixins.geometry_mixin.GeometryMixin.bodyparts_to_polygon`.
           For numba CUDA based acceleration, use :func:`simba.data_processors.cuda.geometry.get_convex_hull`,

        :param np.ndarray arr: 2D array of size len(body-parts) x 2.
        :param float px_per_mm: Video pixels per millimeter.
        :return: The length of the animal perimeter in millimeters.
        :rtype: float

        :example:
        >>> coordinates = np.random.randint(1, 200, size=(6, 2)).astype(np.float32)
        >>> FeatureExtractionMixin.convex_hull_calculator_mp(arr=coordinates, px_per_mm=4.56)
        >>> 98.6676814218373
        """
        arr = np.unique(arr, axis=0).astype(int)
        if arr.shape[0] < 3:
            return 0
        for i in range(1, arr.shape[0]):
            if (arr[i] != arr[0]).all():
                try:
                    return ConvexHull(arr).area / px_per_mm
                except QhullError:
                    return 0
            else:
                pass
        return 0

    @staticmethod
    @jit(nopython=True)
    def count_values_in_range(data: np.ndarray, ranges: np.ndarray) -> np.ndarray:
        """
        Jitted helper finding count of values that falls within ranges. E.g., count number of pose-estimated
        body-parts that fall within defined bracket of probabilities per frame.

        .. image:: _static/img/count_ranges.png
           :width: 300
           :align: center

        .. image:: _static/img/count_ranges_2.png
           :width: 600
           :align: center

        .. seealso::
           For GPU acceleration, use :func:`simba.data_processors.cuda.statistics.count_values_in_ranges`

        :param np.ndarray data: 2D numpy array with frames on X.
        :param np.ndarray ranges: 2D numpy array representing the brackets. E.g., [[0, 0.1], [0.1, 0.5]]
        :return: 2D numpy array of size data.shape[0], ranges.shape[1]
        :rtype: np.ndarray

        :example:
        >>> FeatureExtractionMixin.count_values_in_range(data=np.random.random((3,10)), ranges=np.array([[0.0, 0.25], [0.25, 0.5]]))
        >>> [[6, 1], [3, 2],[2, 1]]
        """

        results = np.full((data.shape[0], ranges.shape[0]), 0)
        for i in prange(data.shape[0]):
            for j in prange(ranges.shape[0]):
                lower_bound, upper_bound = ranges[j][0], ranges[j][1]
                results[i][j] = data[i][
                    np.logical_and(data[i] >= lower_bound, data[i] <= upper_bound)
                ].shape[0]
        return results

    @staticmethod
    @jit(nopython=True)
    def framewise_euclidean_distance_roi(location_1: np.ndarray,
                                         location_2: np.ndarray,
                                         px_per_mm: float,
                                         centimeter: bool = False) -> np.ndarray:
        """
        Find frame-wise distances between a moving location (location_1) and
        static location (location_2) in millimeter or centimeter.

        .. image:: _static/img/distance_to_static.png
           :width: 600
           :align: center

        .. seealso::
           For distances between two moving targets, use :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_euclidean_distance`,
           For GPU implementation, see :func:`simba.data_processors.cuda.statistics.get_euclidean_distance_cupy` or :func:`simba.data_processors.cuda.statistics.get_euclidean_distance_cuda`
           For numpy method (which appears faster than numba) use :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.keypoint_distances`.

        :param ndarray location_1: 2D numpy array of size len(frames) x 2.
        :param ndarray location_1: 1D numpy array holding the X and Y of the static location.
        :param float px_per_mm: The pixels per millimeter in the video.
        :param bool centimeter: If true, the value in centimeters is returned. Else the value in millimeters.
        :return: 1D array of size location_1.shape[0] with distances in millimeter or centimeter.
        :rtype: np.ndarray

        :example:
        >>> loc_1 = np.random.randint(1, 200, size=(6, 2)).astype(np.float32)
        >>> loc_2 = np.random.randint(1, 200, size=(1, 2)).astype(np.float32)
        >>> FeatureExtractionMixin.framewise_euclidean_distance_roi(location_1=loc_1, location_2=loc_2, px_per_mm=4.56, centimeter=False)
        >>> [11.31884926, 13.84534585,  6.09712224, 17.12773976, 19.32066031, 12.18043378]
        >>> FeatureExtractionMixin.framewise_euclidean_distance_roi(location_1=loc_1, location_2=loc_2, px_per_mm=4.56, centimeter=True)
        >>> [1.13188493, 1.38453458, 0.60971222, 1.71277398, 1.93206603, 1.21804338]
        """

        results = np.full((location_1.shape[0]), np.nan)
        for i in prange(location_1.shape[0]):
            results[i] = np.linalg.norm(location_1[i] - location_2) / px_per_mm
        if centimeter:
            results = results / 10
        return results

    @staticmethod
    @jit(nopython=True)
    def framewise_inside_rectangle_roi(bp_location: np.ndarray, roi_coords: np.ndarray) -> np.ndarray:
        """
        Frame-wise analysis if animal is inside static rectangular ROI.

        .. image:: _static/img/inside_rectangle.png
           :width: 300
           :align: center

        .. seealso::
           For GPU acceleration, use :func:`simba.data_processors.cuda.geometry.is_inside_rectangle`.

        :param np.ndarray bp_location:  2d numeric np.ndarray size len(frames) x 2
        :param np.ndarray roi_coords: 2d numeric np.ndarray size 2x2 (top left[x, y], bottom right[x, y])
        :return: 2d numeric boolean np.ndarray size len(frames) x 1, with 0 representing outside the rectangle and 1 representing inside the rectangle.
        :rtype: ndarray

        :example:
        >>> bp_loc = np.random.randint(1, 10, size=(6, 2)).astype(np.float32)
        >>> roi_coords = np.random.randint(1, 10, size=(2, 2)).astype(np.float32)
        >>> FeatureExtractionMixin.framewise_inside_rectangle_roi(bp_location=bp_loc, roi_coords=roi_coords)
        >>> [0, 0, 0, 0, 0, 0]
        """
        results = np.full((bp_location.shape[0]), 0)
        within_x_idx = np.argwhere(
            (bp_location[:, 0] <= roi_coords[1][0])
            & (bp_location[:, 0] >= roi_coords[0][0])
        ).flatten()
        within_y_idx = np.argwhere(
            (bp_location[:, 1] <= roi_coords[1][1])
            & (bp_location[:, 1] >= roi_coords[0][1])
        ).flatten()
        for i in prange(within_x_idx.shape[0]):
            match = np.argwhere(within_y_idx == within_x_idx[i])
            if match.shape[0] > 0:
                results[within_x_idx[i]] = 1
        return results

    @staticmethod
    @jit(nopython=True)
    def is_inside_circle(bp: np.ndarray, roi_center: np.ndarray, roi_radius: int) -> np.ndarray:
        """
        Determines whether each body part in `bp` is inside or outside a given circular region.

        This function calculates the Euclidean distance between each body part's (x, y) coordinates
        and the center of the region of interest (ROI). If the distance is less than or equal to
        the specified radius, the body part is considered inside the circle (marked as 1); otherwise,
        it is considered outside (marked as 0).

        .. seealso::
           For GPU acceleration, see :func:`simba.data_processors.cuda.geometry.is_inside_circle`

        :param np.ndarray bp: A (N, 2) array containing the (x, y) coordinates of N body parts.
        :param np.ndarray roi_center: A (2,) array representing the (x, y) coordinates of the circle center.
        :param int roi_radius: The radius of the circular region of interest.
        :return: A 1D numpy array of size `len(bp)`, where 1 represents a body part inside the circle and 0 represents a body part outside the circle.
        :rtype: np.ndarray
        """

        results = np.full(shape=(bp.shape[0]), fill_value=-1, dtype=np.int32)
        for i in prange(bp.shape[0]):
            px_dist = int(np.sqrt((bp[i][0] - roi_center[0]) ** 2 + (bp[i][1] - roi_center[1]) ** 2))
            if px_dist <= roi_radius:
                results[i] = 1
            else:
                results[i] = 0
        return results

    @staticmethod
    @jit(nopython=True)
    def framewise_inside_polygon_roi(bp_location: np.ndarray, roi_coords: np.ndarray) -> np.ndarray:
        """
        Jitted helper for frame-wise detection if animal is inside static polygon ROI.

        .. note::
           Modified from `epifanio <https://stackoverflow.com/a/48760556>`_

        .. image:: _static/img/inside_polygon.png
           :width: 300
           :align: center

        .. seealso::
           For GPU acceleration, use :func:`simba.data_processors.cuda.geometry.is_inside_polygon`

        :param np.ndarray bp_location:  2d numeric np.ndarray size len(frames) x 2
        :param np.ndarray roi_coords: 2d numeric np.ndarray size len(polygon points) x 2
        :return: 2d numeric boolean np.ndarray size len(frames) x 1, with 0 representing outside the polygon and 1 representing inside the polygon.
        :rtype: np.ndarray

        :example:
        >>> bp_loc = np.random.randint(1, 10, size=(6, 2)).astype(np.float32)
        >>> roi_coords = np.random.randint(1, 10, size=(10, 2)).astype(np.float32)
        >>> FeatureExtractionMixin.framewise_inside_polygon_roi(bp_location=bp_loc, roi_coords=roi_coords)
        >>> [0, 0, 0, 1]
        """

        results = np.full((bp_location.shape[0]), 0)
        for i in prange(0, results.shape[0]):
            x, y, n = bp_location[i][0], bp_location[i][1], len(roi_coords)
            p2x, p2y, xints, inside = 0.0, 0.0, 0.0, False
            p1x, p1y = roi_coords[0]
            for j in prange(n + 1):
                p2x, p2y = roi_coords[j % n]
                if (
                    (y > min(p1y, p2y))
                    and (y <= max(p1y, p2y))
                    and (x <= max(p1x, p2x))
                ):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
                p1x, p1y = p2x, p2y
            if inside:
                results[i] = 1

        return results

    @staticmethod
    def windowed_frequentist_distribution_tests(data: np.ndarray, feature_name: str, fps: int) -> pd.DataFrame:
        """
        Calculates feature value distributions and feature peak counts in 1-s sequential time-bins.

        Computes (i) feature value distributions in 1-s sequential time-bins: Kolmogorov-Smirnov and T-tests.
        Computes (ii)  feature values against a normal distribution: Shapiro-Wilks.
        Computes (iii) peak count in *rolling* 1s long feature window: scipy.find_peaks.

        .. warning::
           This is a **legacy** method. For KS test, use :func:`simba.mixins.statistics_mixin.Statistics.two_sample_ks`.
           For t-tests, use :func:`simba.mixins.statistics_mixin.Statistics.independent_samples_t.
           For Shapiro-Wilks, use :func:`simba.mixins.statistics_mixin.Statistics.rolling_shapiro_wilks`.
           For peaks, use :func:`simba.mixins.feature_extraction_supplement_mixin.FeatureExtractionSupplemental.peak_ratio`.

        :param np.ndarray data: Single feature 1D array
        :param np.ndarray feature_name: The name of the input feature.
        :param int fps: The framerate of the video representing the data.
        :return: Of size len(data) x 4 with columns representing KS, T, Shapiro-Wilks, and peak count statistics.
        :rtype: pd.DataFrame

        :example:
        >>> feature_data = np.random.randint(1, 10, size=(100))
        >>> FeatureExtractionMixin.windowed_frequentist_distribution_tests(data=feature_data, fps=25, feature_name='Anima_1_velocity')
        """

        (ks_results,) = (np.full((data.shape[0]), -1.0),)
        t_test_results = np.full((data.shape[0]), -1.0)
        shapiro_results = np.full((data.shape[0]), -1.0)
        peak_cnt_results = np.full((data.shape[0]), -1.0)

        for i in range(fps, data.shape[0] - fps, fps):
            bin_1_idx, bin_2_idx = [i - fps, i], [i, i + fps]
            bin_1_data, bin_2_data = (
                data[bin_1_idx[0] : bin_1_idx[1]],
                data[bin_2_idx[0] : bin_2_idx[1]],
            )
            ks_results[i : i + fps + 1] = stats.ks_2samp(
                data1=bin_1_data, data2=bin_2_data
            ).statistic
            t_test_results[i : i + fps + 1] = stats.ttest_ind(
                bin_1_data, bin_2_data
            ).statistic

        for i in range(0, data.shape[0] - fps, fps):
            shapiro_results[i : i + fps + 1] = stats.shapiro(data[i : i + fps])[0]

        rolling_idx = np.arange(fps)[None, :] + 1 * np.arange(data.shape[0])[:, None]
        for i in range(rolling_idx.shape[0]):
            bin_start_idx, bin_end_idx = rolling_idx[i][0], rolling_idx[i][-1]
            peaks, _ = find_peaks(data[bin_start_idx:bin_end_idx], height=0)
            peak_cnt_results[i] = len(peaks)

        columns = [
            f"{feature_name}_KS",
            f"{feature_name}_TTEST",
            f"{feature_name}_SHAPIRO",
            f"{feature_name}_PEAK_CNT",
        ]

        return (
            pd.DataFrame(
                np.column_stack(
                    (ks_results, t_test_results, shapiro_results, peak_cnt_results)
                ),
                columns=columns,
            )
            .round(4)
            .fillna(0)
        )

    @staticmethod
    @jit(nopython=True, cache=True)
    def cdist(array_1: np.ndarray, array_2: np.ndarray) -> np.ndarray:
        """
        Analogue of meth:`scipy.cdist` for two 2D arrays. Use to calculate Euclidean distances between
        all coordinates in one array and all coordinates in a second array. E.g., computes the distances between
        all body-parts of one animal and all body-parts of a second animal. Acceleration though numba.

        .. image:: _static/img/cdist.png
           :width: 600
           :align: center

        .. seealso::
           For GPU acceleration, use `cupyx.scipy.spatial.distance.cdist`

        :param np.ndarray array_1: 2D array of body-part coordinates
        :param np.ndarray array_2: 2D array of body-part coordinates
        :return: 2D array of Euclidean distances between body-parts in ``array_1`` and ``array_2``
        :rtype: np.ndarray

        :example:
        >>> array_1 = np.random.randint(1, 10, size=(3, 2)).astype(np.float32)
        >>> array_2 = np.random.randint(1, 10, size=(3, 2)).astype(np.float32)
        >>> FeatureExtractionMixin.cdist(array_1=array_1, array_2=array_2)
        >>> [[7.07106781, 1.        , 3.60555124],
        >>> [3.60555124, 6.3245554 , 2.        ],
        >>>  [3.1622777 , 5.38516474, 4.12310553]])
        """
        results = np.full((array_1.shape[0], array_2.shape[0]), np.nan)
        for i in prange(array_1.shape[0]):
            for j in prange(array_2.shape[0]):
                results[i][j] = np.linalg.norm(array_1[i] - array_2[j])
        return results

    @staticmethod
    @jit(nopython=True)
    def cdist_3d(data: np.ndarray) -> np.ndarray:
        """
        Jitted analogue of meth:`scipy.cdist` for 3D array. Use to calculate Euclidean distances between
        all coordinates in of one array and itself.

        :parameter np.ndarray data: 3D array of body-part coordinates of size len(frames) x -1 x 2.
        :return np.ndarray: 3D array of size data.shape[0], data.shape[1], data.shape[1].
        """

        results = np.full((data.shape[0], data.shape[1], data.shape[1]), np.nan)
        for i in prange(data.shape[0]):
            for j in prange(data[i].shape[0]):
                for k in prange(data[i].shape[0]):
                    results[i][j][k] = np.linalg.norm(data[i][j] - data[i][k])

        return results

    @staticmethod
    # @njit('(float32[:],)')
    def cosine_similarity(data: np.ndarray) -> np.ndarray:
        """
        Jitted analogue of sklearn.metrics.pairwise import cosine_similarity. Similar to scipy.cdist.
        calculates the cosine similarity between all pairs in 2D array.

        :param np.ndarray data: 2D array of observations.
        :returns: Matrix representing the cosine similarity between all observations in ``data``.
        :rtype: np.ndarray

        :example:
        >>> data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32)
        >>> FeatureExtractionMixin().cosine_similarity(data=data)
        >>> [[1.0, 0.974, 0.959][0.974,  1.0, 0.998] [0.959, 0.998, 1.0]
        """

        dot_product = np.dot(data, data.T)
        norms = np.linalg.norm(data, axis=1).reshape(-1, 1)
        similarity = dot_product / (norms * norms.T)
        return similarity

    @staticmethod
    def create_shifted_df(df: pd.DataFrame, periods: int = 1, suffix: str = "_shifted") -> pd.DataFrame:
        """
        Create dataframe including duplicated shifted (1) columns with ``_shifted`` suffix.

        :param pd.DataFrame df: Dataframe to create additional shifted fields from.
        :param periods int: The rows to shift the new fields. 1 denotes that the shifted fields get shifted one row "down". -1 and the fields would be shifted one row "up".
        :param str suffix: The suffix to add to the new, shifted, fields. Default: "shifted".
        :return pd.DataFrame: Dataframe including original and shifted columns.

        :example:
        >>> df = pd.DataFrame(np.random.randint(0,100,size=(3, 1)), columns=['Feature_1'])
        >>> FeatureExtractionMixin.create_shifted_df(df=df)
        >>>             Feature_1  Feature_1_shifted
        >>>    0         76               76.0
        >>>    1         41               76.0
        >>>    2         89               41.0

        """
        data_df_shifted = df.shift(periods=periods)
        data_df_shifted = data_df_shifted.combine_first(df).add_suffix(suffix)
        return pd.concat([df, data_df_shifted], axis=1, join="inner").reset_index(drop=True)

    def check_directionality_viable(self) -> Tuple[bool, List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """
        Check if it is possible to calculate ``directionality`` statistics.

        Specifically, checks if nose and coordinates from pose estimation has to be present

        :return bool: If True, directionality is viable. Else, not viable.
        :return np.ndarray nose_coord: If viable, then 2D array with coordinates of the nose in all frames. Else, empty array.
        :return np.ndarray ear_left_coord: If viable, then 2D array with coordinates of the left ear in all frames. Else, empty array.
        :return np.ndarray ear_right_coord: If viable, then 2D array with coordinates of the right ear in all frames. Else, empty array.
        """
        DELIMITERS = "_ "


        direction_viable = True
        nose_cords, ear_left_cords, ear_right_cords = [], [], []
        for animal_name in self.animal_bp_dict.keys():
            for bp_cord in ["X_bps", "Y_bps"]:
                bp_list = self.animal_bp_dict[animal_name][bp_cord]
                for bp_name in bp_list:
                    if "nose" in bp_name.lower():
                        nose_cords.append(bp_name)
                    elif ("ear" in bp_name.lower()) and ("left" in bp_name.lower()):
                        ear_left_cords.append(bp_name)
                    elif ("ear" in bp_name.lower()) and ("right" in bp_name.lower()):
                        ear_right_cords.append(bp_name)
                    else:
                        pass



        for cord in [nose_cords, ear_left_cords, ear_right_cords]:
            if len(cord) != len(self.animal_bp_dict.keys()) * 2:
                direction_viable = False

        if direction_viable:
            nose_cords = [nose_cords[i * 2 : (i + 1) * 2] for i in range((len(nose_cords) + 2 - 1) // 2)]
            ear_left_cords = [ear_left_cords[i * 2 : (i + 1) * 2] for i in range((len(ear_left_cords) + 2 - 1) // 2)]
            ear_right_cords = [ear_right_cords[i * 2 : (i + 1) * 2] for i in range((len(ear_right_cords) + 2 - 1) // 2)]

        return direction_viable, nose_cords, ear_left_cords, ear_right_cords

    def get_feature_extraction_headers(self, pose: str) -> List[str]:
        """
        Helper to return the headers names (body-part location columns) that should be used during feature extraction.

        :parameter str pose: Pose-estimation setting, e.g., ``16``.
        :return List[str]: The names and order of the pose-estimation columns.
        """
        simba_dir = os.path.dirname(simba.__file__)
        feature_categories_csv_path = os.path.join(
            simba_dir, Paths.SIMBA_FEATURE_EXTRACTION_COL_NAMES_PATH.value
        )
        check_file_exist_and_readable(file_path=feature_categories_csv_path)
        bps = list(pd.read_csv(feature_categories_csv_path)[pose])
        return [x for x in bps if str(x) != "nan"]

    @staticmethod
    @jit(nopython=True)
    def jitted_line_crosses_to_nonstatic_targets(left_ear_array: np.ndarray,
                                                 right_ear_array: np.ndarray,
                                                 nose_array: np.ndarray,
                                                 target_array: np.ndarray) -> np.ndarray:
        """
        Jitted helper to calculate if an animal is directing towards another animals body-part coordinate,
        given the target body-part and the left ear, right ear, and nose coordinates of the observer.

        .. seealso::
           Input left ear, right ear, and nose coordinates of the observer is returned by :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.check_directionality_viable`

           If the target is static, consider :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.jitted_line_crosses_to_static_targets`

        .. image:: _static/img/directing_moving_targets.png
           :width: 400
           :align: center

        :param np.ndarray left_ear_array: 2D array of size len(frames) x 2 with the coordinates of the observer animals left ear
        :param np.ndarray right_ear_array: 2D array of size len(frames) x 2 with the coordinates of the observer animals right ear
        :param np.ndarray nose_array: 2D array of size len(frames) x 2 with the coordinates of the observer animals nose
        :param np.ndarray target_array: 2D array of size len(frames) x 2 with the target body-part location

        :return: 2D array of size len(frames) x 4. First column represent the side of the observer that the target is in view. 0 = Left side, 1 = Right side, 2 = Not in view.
        Second and third column represent the x and y location of the observer animals ``eye`` (half-way between the ear and the nose).
        Fourth column represent if target is in view (bool).
        :rtype: np.ndarray

        """

        results_array = np.zeros((left_ear_array.shape[0], 4))
        for frame_no in prange(results_array.shape[0]):
            Px = np.abs(left_ear_array[frame_no][0] - target_array[frame_no][0])
            Py = np.abs(left_ear_array[frame_no][1] - target_array[frame_no][1])
            Qx = np.abs(right_ear_array[frame_no][0] - target_array[frame_no][0])
            Qy = np.abs(right_ear_array[frame_no][1] - target_array[frame_no][1])
            Nx = np.abs(nose_array[frame_no][0] - target_array[frame_no][0])
            Ny = np.abs(nose_array[frame_no][1] - target_array[frame_no][1])
            Ph = np.sqrt(Px * Px + Py * Py)
            Qh = np.sqrt(Qx * Qx + Qy * Qy)
            Nh = np.sqrt(Nx * Nx + Ny * Ny)
            if Nh < Ph and Nh < Qh and Qh < Ph:
                results_array[frame_no] = [
                    0,
                    right_ear_array[frame_no][0],
                    right_ear_array[frame_no][1],
                    True,
                ]
            elif Nh < Ph and Nh < Qh and Ph < Qh:
                results_array[frame_no] = [
                    1,
                    left_ear_array[frame_no][0],
                    left_ear_array[frame_no][1],
                    True,
                ]
            else:
                results_array[frame_no] = [2, -1, -1, False]

        return results_array

    @staticmethod
    @jit(nopython=True)
    def jitted_line_crosses_to_static_targets(left_ear_array: np.ndarray,
                                              right_ear_array: np.ndarray,
                                              nose_array: np.ndarray,
                                              target_array: np.ndarray) -> np.ndarray:

        """
        Jitted helper to calculate if an animal is directing towards a static location (e.g., ROI centroid),
        given the target location and the left ear, right ear, and nose coordinates of the observer.

        .. image:: _static/img/directing_static_targets.png
           :width: 400
           :align: center

        .. video:: _static/img/ts_example.webm
           :width: 800
           :autoplay:
           :loop:

        ..  youtube:: vqdYUS3bM68
           :width: 640
           :height: 480
           :align: center

        .. note::
           Input left ear, right ear, and nose coordinates of the observer is returned by :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.check_directionality_viable`

           If the target is moving, consider :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.jitted_line_crosses_to_nonstatic_targets`.

        .. seealso::
           For GPU accelerated methods, see :func:`simba.data_processors.cuda.geometry.directionality_to_static_targets`

        :param np.ndarray left_ear_array: 2D array of size len(frames) x 2 with the coordinates of the observer animals left ear
        :param np.ndarray right_ear_array: 2D array of size len(frames) x 2 with the coordinates of the observer animals right ear
        :param np.ndarray nose_array: 2D array of size len(frames) x 2 with the coordinates of the observer animals nose
        :param np.ndarray target_array: 1D array of with x,y of target location

        :return: 2D array of size len(frames) x 4. First column represent the side of the observer that the target is in view. 0 = Left side, 1 = Right side, 2 = Not in view.
        Second and third column represent the x and y location of the observer animals ``eye`` (half-way between the ear and the nose).
        Fourth column represent if target is view (bool).
        :rtype: np.ndarray

        """

        results_array = np.zeros((left_ear_array.shape[0], 4))
        for frame_no in range(results_array.shape[0]):
            Px = np.abs(left_ear_array[frame_no][0] - target_array[0])
            Py = np.abs(left_ear_array[frame_no][1] - target_array[1])
            Qx = np.abs(right_ear_array[frame_no][0] - target_array[0])
            Qy = np.abs(right_ear_array[frame_no][1] - target_array[1])
            Nx = np.abs(nose_array[frame_no][0] - target_array[0])
            Ny = np.abs(nose_array[frame_no][1] - target_array[1])
            Ph = np.sqrt(Px * Px + Py * Py)
            Qh = np.sqrt(Qx * Qx + Qy * Qy)
            Nh = np.sqrt(Nx * Nx + Ny * Ny)
            if Nh < Ph and Nh < Qh and Qh < Ph:
                results_array[frame_no] = [0, right_ear_array[frame_no][0], right_ear_array[frame_no][1], True]
            elif Nh < Ph and Nh < Qh and Ph < Qh:
                results_array[frame_no] = [1, left_ear_array[frame_no][0], left_ear_array[frame_no][1], True]
            else:
                results_array[frame_no] = [2, -1, -1, False]

        return results_array

    @staticmethod
    def minimum_bounding_rectangle(points: np.ndarray) -> np.ndarray:
        """
        Finds the minimum bounding rectangle from convex hull vertices.

        .. image:: _static/img/minimum_bounding_rectangle.png
           :width: 400
           :align: center

        .. note::
           Modified from `JesseBuesking <https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput>`_
           See :func:`simba.mixins.feature_extractors.perimeter_jit.jitted_hull` for computing the convexhull vertices.

        .. seealso::
           For multicore method and improved runtimes, see :func:`simba.mixins.geometry_mixin.GeometryMixin.multiframe_minimum_rotated_rectangle`

        :param np.ndarray points: 2D array representing the convexhull vertices of the animal.
        :return: 2D array representing minimum bounding rectangle of the convexhull vertices of the animal.
        :rtype: np.ndarray



        :example:
        >>>   points = np.random.randint(1, 10, size=(10, 2))
        >>>   FeatureExtractionMixin.minimum_bounding_rectangle(points=points)
        >>> [[10.7260274 ,  3.39726027], [ 1.4109589 , -0.09589041], [-0.31506849,  4.50684932], [ 9., 8. ]]
        """

        pi2 = np.pi / 2.0
        hull_points = points[ConvexHull(points).vertices]
        edges = hull_points[1:] - hull_points[:-1]
        angles = np.arctan2(edges[:, 1], edges[:, 0])
        angles = np.abs(np.mod(angles, pi2))
        angles = np.unique(angles)
        rotations = np.vstack([np.cos(angles), np.cos(angles - pi2), np.cos(angles + pi2), np.cos(angles)]).T
        rotations = rotations.reshape((-1, 2, 2))
        rot_points = np.dot(rotations, hull_points.T)
        min_x, max_x = np.nanmin(rot_points[:, 0], axis=1), np.nanmax(rot_points[:, 0], axis=1)
        min_y, max_y = np.nanmin(rot_points[:, 1], axis=1), np.nanmax(rot_points[:, 1], axis=1)
        areas = (max_x - min_x) * (max_y - min_y)
        best_idx = np.argmin(areas)
        x1, x2 = max_x[best_idx], min_x[best_idx]
        y1, y2 = max_y[best_idx], min_y[best_idx]
        r = rotations[best_idx]
        rval = np.zeros((4, 2))
        rval[0], rval[1] = np.dot([x1, y2], r), np.dot([x2, y2], r)
        rval[2], rval[3] = np.dot([x2, y1], r), np.dot([x1, y1], r)
        return rval

    @staticmethod
    #@jit(nopython=True)
    @njit([types.float64[:](types.float64[:, :], types.float64[:, :], types.float64, types.boolean)], fastmath=True, parallel=True, cache=True)
    def framewise_euclidean_distance(location_1: np.ndarray,
                                     location_2: np.ndarray,
                                     px_per_mm: float,
                                     centimeter: bool) -> np.ndarray:
        """
        Compute frame-wise Euclidean distances between two sets of moving 2D locations.

        This numba-jitted function efficiently calculates the straight-line distance between
        corresponding points in two location arrays for each frame. The distances are converted
        from pixels to real-world units (millimeters or centimeters) using the provided
        pixel-to-millimeter conversion factor.

        Uses the standard Euclidean distance formula: √((x₁-x₂)² + (y₁-y₂)²) / px_per_mm

        .. image:: _static/img/framewise_euclid_dist.png
           :width: 300
           :align: center

        .. image:: _static/img/framewise_euclid_dist.webp
           :width: 300
           :align: center

        .. note::
           This function is optimized with numba JIT parallel execution compilation for high performance on large datasets.

        .. csv-table::
           :header: EXPECTED RUNTIMES
           :file: ../../../docs/tables/bodypart_distance.csv
           :widths: 10, 45, 45
           :align: center
           :header-rows: 1

        .. seealso::
           For GPU CuPy solution, see :func:`simba.data_processors.cuda.statistics.get_euclidean_distance_cupy`.
           For GPU numba CUDA solution, see :func:`simba.data_processors.cuda.statistics.get_euclidean_distance_cuda`.
           For Euclidean distance between one moving and one static target, see :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_euclidean_distance_roi`.
           For wrapper function ensuring dtypes and data validity in this method, see :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.bodypart_distance`.

        :param np.ndarray location_1: First set of 2D coordinates with shape (n_frames, 2), where each row contains [x, y] pixel coordinates for a specific frame.
        :param np.ndarray location_2: Second set of 2D coordinates with shape (n_frames, 2), where each row contains [x, y] pixel coordinates for a specific frame. Must have same shape as location_1.
        :param float px_per_mm: Conversion factor from pixels to millimeters. Must be positive.
        :param bool centimeter: If True, returns distances in centimeters. If False, returns distances in millimeters. Default is False.
        :return: Array of Euclidean distances with shape (n_frames,).
        :rtype: np.ndarray

        :example:
        >>> # Calculate distances between two body parts across frames
        >>> nose_coords = np.array([[100, 150], [102, 148], [105, 145]], dtype=np.float32)
        >>> ear_coords = np.array([[90, 140], [92, 138], [95, 135]], dtype=np.float32)
        >>> distances_mm = FeatureExtractionMixin.framewise_euclidean_distance(location_1=nose_coords, location_2=ear_coords, px_per_mm=4.5, centimeter=False)
        >>> distances_cm = FeatureExtractionMixin.framewise_euclidean_distance(location_1=nose_coords, location_2=ear_coords, px_per_mm=4.5, centimeter=True)
        """

        results = np.full((location_1.shape[0]), np.nan)
        for i in prange(location_1.shape[0]):
            results[i] = np.linalg.norm(location_1[i] - location_2[i]) / px_per_mm
        if centimeter and px_per_mm:
            results = results / 10
        return results

    @staticmethod
    def bodypart_distance(bp1_coords: np.ndarray,
                          bp2_coords: np.ndarray,
                          px_per_mm: float = 1.0,
                          in_centimeters: bool = False) -> np.ndarray:

        """
        Calculate frame-wise Euclidean distances between two sets of body part coordinates.

        The function uses the standard Euclidean distance formula: distance = √((x₁-x₂)² + (y₁-y₂)²) / px_per_mm

        .. seealso::
           Wrapper function (ensuring data validity) for the underlying implementation :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_euclidean_distance`.
           For GPU CuPy solution, see :func:`simba.data_processors.cuda.statistics.get_euclidean_distance_cupy`.
           For GPU numba CUDA solution, see :func:`simba.data_processors.cuda.statistics.get_euclidean_distance_cuda`.
           For Euclidean distance between one moving and one static target, see :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_euclidean_distance_roi`.

        .. image:: _static/img/framewise_euclid_dist.png
           :width: 300
           :align: center

        .. image:: _static/img/framewise_euclid_dist.webp
           :width: 300
           :align: center

        .. csv-table::
           :header: EXPECTED RUNTIMES
           :file: ../../../docs/tables/bodypart_distance.csv
           :widths: 10, 45, 45
           :align: center
           :header-rows: 1

        :param np.ndarray bp1_coords: First body part coordinates with shape (n_frames, 2), where each row  contains [x, y] pixel coordinates for a specific frame.
        :param np.ndarray bp2_coords: Second body part coordinates with shape (n_frames, 2), where each row  contains [x, y] pixel coordinates for a specific frame. Must have the same number of frames as bp1_coords.
        :param float px_per_mm: Conversion factor from pixels to millimeters. Must be positive. Default: 1.0.
        :param bool in_centimeters: If True, returns distances in centimeters. If False, returns distances in millimeters. Default: False.
        :return: Array of Euclidean distances with shape (n_frames,) in the specified units as float32.
        :rtype: np.ndarray[np.float32]

        :example:
        >>> bp1_coords = np.random.randint(0, 500, size=(1000, 2))
        >>> bp2_coords = np.random.randint(0, 500, size=(1000, 2))
        >>> FeatureExtractionMixin().bodypart_distance(bp1_coords=bp1_coords, bp2_coords=bp2_coords, px_per_mm=1.0, in_centimeters=False)
        """

        check_valid_array(data=bp1_coords, source=f'{FeatureExtractionMixin.bodypart_distance.__name__} bp1_coords', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_value=0.0, raise_error=True)
        check_valid_array(data=bp2_coords, source=f'{FeatureExtractionMixin.bodypart_distance.__name__} bp2_coords', accepted_ndims=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_value=0.0, raise_error=True, accepted_axis_0_shape=(bp1_coords.shape[0],))
        check_float(name=f'{FeatureExtractionMixin.bodypart_distance.__name__} px_per_mm', value=px_per_mm, min_value=10e-16, raise_error=True, allow_zero=False)
        check_valid_boolean(value=in_centimeters, source=f'{FeatureExtractionMixin. bodypart_distance.__name__} px_per_mm', raise_error=True)

        bp1_coords = bp1_coords.astype(np.float64)
        bp2_coords = bp2_coords.astype(np.float64)
        px_per_mm = np.float64(px_per_mm)
        results = FeatureExtractionMixin.framewise_euclidean_distance(location_1=bp1_coords,
                                                                      location_2=bp2_coords,
                                                                      px_per_mm=px_per_mm,
                                                                      centimeter=in_centimeters)
        return results.astype(np.float32)

    @staticmethod
    def keypoint_distances(a: np.ndarray,
                           b: np.ndarray,
                           px_per_mm: Optional[float] = 1,
                           in_centimeters: Optional[bool] = False) -> np.ndarray:

        """
        Compute Euclidean distances between corresponding 2D keypoints with unit conversion.

        Given two arrays of 2D coordinates (x, y) sampled across frames, this function computes the
        frame-wise Euclidean distance between matching rows, converts from pixels to millimeters
        using ``px_per_mm``, and optionally reports distances in centimeters. Input validity is checked
        and the output is guaranteed to be ``np.float32``.

        .. seealso::
           For numba decorated function, :func:`simba.mixins.feature_extraction_mixin.FeatureExtractionMixin.framewise_euclidean_distance`
           For GPU CuPy solution, see :func:`simba.data_processors.cuda.statistics.get_euclidean_distance_cupy`.
           For GPU numba CUDA solution, see :func:`simba.data_processors.cuda.statistics.get_euclidean_distance_cuda`.

           Appears faster than numba deocrated method, and slower than GPU methods.

        .. image:: _static/img/framewise_euclid_dist.webp
           :width: 300
           :align: center

        .. csv-table::
           :header: EXPECTED RUNTIMES
           :file: ../../docs/tables/keypoint_distances.csv
           :widths: 20, 20, 20, 20, 20
           :align: center
           :header-rows: 1

        :param np.ndarray a: Array of shape ``(n_frames, 2)`` with non-negative numeric [x, y] coordinates.
        :param np.ndarray b: Array of shape ``(n_frames, 2)`` with non-negative numeric [x, y] coordinates. Must have the same number of rows as ``a``.
        :param float px_per_mm: Pixels-per-millimeter scaling factor (> 0). Distances are divided by this value.
        :param bool in_centimeters: If ``True``, returned distances are reported in centimeters (mm/10).
        :return: Frame-wise distances between corresponding rows in ``a`` and ``b`` (mm or cm).
        :rtype: np.ndarray

        :example:
        >>> a = np.array([[0, 0], [3, 4], [6, 8]], dtype=np.float32)
        >>> b = np.array([[0, 0], [0, 0], [3, 4]], dtype=np.float32)
        >>> # px_per_mm = 1 -> distances reported in millimeters (same numeric scale as pixels)
        >>> d_mm = FeatureExtractionMixin.keypoint_distances(a=a, b=b, px_per_mm=1.0, in_centimeters=False)
        >>> d_cm = FeatureExtractionMixin.keypoint_distances(a=a, b=b, px_per_mm=1.0, in_centimeters=True)
        """

        check_valid_array(data=a, source=f'{FeatureExtractionMixin.keypoint_distances.__name__} a', accepted_ndims=(2,), accepted_axis_1_shape=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_value=0)
        check_valid_array(data=b, source=f'{FeatureExtractionMixin.keypoint_distances.__name__} b', accepted_ndims=(2,), accepted_axis_1_shape=(2,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, min_value=0, accepted_axis_0_shape=(a.shape[0],))
        check_float(name=f'{FeatureExtractionMixin.keypoint_distances.__name__} px_per_mm', value=px_per_mm, allow_zero=False, min_value=10e-16, raise_error=True)
        check_valid_boolean(value=in_centimeters, source=f'{FeatureExtractionMixin.keypoint_distances.__name__} in_centimeters', raise_error=True)

        d = np.linalg.norm(a - b, axis=1) / px_per_mm
        if in_centimeters:
            d = d / 10.0
        return d.astype(np.float32, copy=False)

    def change_in_bodypart_euclidean_distance(
                                              self,
                                              location_1: np.ndarray,
                                              location_2: np.ndarray,
                                              fps: int,
                                              px_per_mm: float,
                                              time_windows: np.ndarray = np.array([0.2, 0.4, 0.8, 1.6]),
    ) -> np.ndarray:
        """
        Computes the difference between the distances of two body-parts in the current frame versus N.N seconds ago.
        Used for computing if animal body-parts are traveling away or towards each other within defined time-windows.
        """
        distances = self.framewise_euclidean_distance(
            location_1=location_1.astype(np.float64), location_2=location_2.astype(np.float64), px_per_mm=np.float64(px_per_mm), centimeter=False
        )
        return self._relative_distances(
            distances=distances, fps=fps, time_windows=time_windows
        )

    def dataframe_gaussian_smoother(
        self, df: pd.DataFrame, fps: int, time_window: int = 100
    ) -> pd.DataFrame:
        """
        Column-wise Gaussian smoothing of dataframe.

        :parameter pd.DataFrame df: Dataframe with un-smoothened data.
        :parameter int fps: The frame-rate of the video representing the data.
        :parameter int time_window: Time-window in milliseconds to use for Gaussian smoothing.
        :return pd.DataFrame: Dataframe with smoothened data

        :references:
        .. [1] `Video expected putput <https://www.youtube.com/watch?v=d9-Bi4_HyfQ>`__.
        """

        frames_in_time_window = int(time_window / (1000 / fps))
        for c in df.columns:
            df[c] = (
                df[c]
                .rolling(
                    window=int(frames_in_time_window), win_type="gaussian", center=True
                )
                .mean(std=5)
                .fillna(df[c])
                .abs()
                .astype(int)
            )
        return df

    def dataframe_savgol_smoother(
        self, df: pd.DataFrame, fps: int, time_window: int = 150
    ) -> pd.DataFrame:
        """
        Column-wise Savitzky-Golay smoothing of dataframe.

        :parameter pd.DataFrame df: Dataframe with un-smoothened data.
        :parameter int fps: The frame-rate of the video representing the data.
        :parameter int time_window: Time-window in milliseconds to use for Gaussian smoothing.
        :return pd.DataFrame: Dataframe with smoothened data

        :references:
        .. [1] `Video expected putput <https://www.youtube.com/watch?v=d9-Bi4_HyfQ>`__.

        """
        frames_in_time_window = int(time_window / (1000 / fps))
        if (frames_in_time_window % 2) == 0:
            frames_in_time_window = frames_in_time_window - 1
        if (frames_in_time_window % 2) <= 3:
            frames_in_time_window = 5
        for c in df.columns:
            df[c] = savgol_filter(
                x=df[c].to_numpy(),
                window_length=frames_in_time_window,
                polyorder=3,
                mode="nearest",
            ).astype(int)
        return df

    def get_bp_headers(self) -> None:
        """
        Helper to create ordered list of all column header fields for SimBA project dataframes.
        """
        self.col_headers = []
        for bp in self.body_parts_lst:
            c1, c2, c3 = (f"{bp}_x", f"{bp}_y", f"{bp}_p")
            self.col_headers.extend((c1, c2, c3))

    def check_directionality_cords(self) -> dict:
        """
        Helper to check if ear and nose body-parts are present within the pose-estimation data.

        :return dict: Body-part names of ear and nose body-parts as values and animal names as keys. If empty, ear and nose body-parts are not present within the pose-estimation data

        """
        results = {}
        for animal in self.animal_bp_dict.keys():
            results[animal] = {'Nose': {}, 'Ear_left': {}, 'Ear_right': {}}
            for dimension in ["X_bps", "Y_bps"]:
                for cord in self.animal_bp_dict[animal][dimension]:
                    if ("nose" in cord.lower()) and ("x" in cord.lower()):
                        results[animal]["Nose"]["X_bps"] = cord
                    elif ("nose" in cord.lower()) and ("y" in cord.lower()):
                        results[animal]["Nose"]["Y_bps"] = cord
                    elif (
                        ("left" in cord.lower())
                        and ("x" in cord.lower())
                        and ("ear" in cord.lower())
                    ):
                        results[animal]["Ear_left"]["X_bps"] = cord
                    elif (
                        ("left" in cord.lower())
                        and ("Y".lower() in cord.lower())
                        and ("ear".lower() in cord.lower())
                    ):
                        results[animal]["Ear_left"]["Y_bps"] = cord
                    elif (
                        ("right" in cord.lower())
                        and ("x" in cord.lower())
                        and ("ear" in cord.lower())
                    ):
                        results[animal]["Ear_right"]["X_bps"] = cord
                    elif (
                        ("right" in cord.lower())
                        and ("y" in cord.lower())
                        and ("ear".lower() in cord.lower())
                    ):
                        results[animal]["Ear_right"]["Y_bps"] = cord
        return results

    def insert_default_headers_for_feature_extraction(
        self, df: pd.DataFrame, headers: List[str], pose_config: str, filename: str
    ) -> pd.DataFrame:
        """
        Helper to insert correct body-part column names prior to defualt feature extraction methods.
        """
        if len(headers) != len(df.columns):
            raise CountError(
                f"Your SimBA project is set to using the default {pose_config} pose-configuration. "
                f"SimBA therefore expects {str(len(headers))} columns of data inside the files within the project_folder. However, "
                f"within file {filename} file, SimBA found {str(len(df.columns))} columns.",
                source=self.__class__.__name__,
            )
        else:
            df.columns = headers
            return df

    @staticmethod
    def line_crosses_to_static_targets(
        p: List[float],
        q: List[float],
        n: List[float],
        M: List[float],
        coord: List[float],
    ) -> (bool, List[float]):
        """
        **Legacy** non-jitted helper to calculate if an animal is directing towards a static coordinate (e.g., ROI centroid).

        .. important:
           For improved runtime, use :func:`simba.mixins.feature_extraction_mixin.jitted_line_crosses_to_static_targets`

        :param list p: left ear coordinates of observing animal.
        :param list q: right ear coordinates of observing animal.
        :param list n: nose coordinates of observing animal.
        :param list M: The location of the target coordinates.
        :param list coord: empty list to store the eye coordinate of the observing animal.
        :return bool: If True, static coordinate is in view.
        :return List: If True, the coordinate of the observing animals ``eye`` (half-way between nose and ear).
        """

        Px = np.abs(p[0] - M[0])
        Py = np.abs(p[1] - M[1])
        Qx = np.abs(q[0] - M[0])
        Qy = np.abs(q[1] - M[1])
        Nx = np.abs(n[0] - M[0])
        Ny = np.abs(n[1] - M[1])
        Ph = np.sqrt(Px * Px + Py * Py)
        Qh = np.sqrt(Qx * Qx + Qy * Qy)
        Nh = np.sqrt(Nx * Nx + Ny * Ny)
        if Nh < Ph and Nh < Qh and Qh < Ph:
            coord.extend((q[0], q[1]))
            return True, coord
        elif Nh < Ph and Nh < Qh and Ph < Qh:
            coord.extend((p[0], p[1]))
            return True, coord
        else:
            return False, coord

    @staticmethod
    @njit("(int64[:,:], int64[:,:], float64)")
    def find_midpoints(bp_1: np.ndarray,
                       bp_2: np.ndarray,
                       percentile: float) -> np.ndarray:
        """
        Compute the midpoints between two sets of 2D points based on a given percentile.

        .. image:: _static/img/find_midpoints.png
           :width: 600
           :align: center

        .. seealso::
           For GPU acceleration, use :func:`simba.data_processors.cuda.geometry.find_midpoints`

        :param np.ndarray bp_1: An array of 2D points representing the first set of points. Rows represent frames. First column represent x coordinates. Second column represent y coordinates.
        :param np.ndarray bp_2: An array of 2D points representing the second set of points. Rows represent frames. First column represent x coordinates. Second column represent y coordinates.
        :param float percentile: The percentile value to determine the distance between the points for calculating midpoints. When set to 0.5 it calculates midpoints at the midpoint of the two points.
        :returns: An array of 2D points representing the midpoints between the points in bp_1 and bp_2 based on the specified percentile.
        :rtype: np.ndarray

        :example:
        >>> bp_1 = np.array([[1, 3], [30, 10]]).astype(np.int64)
        >>> bp_2 = np.array([[10, 4], [20, 1]]).astype(np.int64)
        >>> FeatureExtractionMixin().find_midpoints(bp_1=bp_1, bp_2=bp_2, percentile=0.5)
        >>> [[ 5,  3], [25,  6]]
        """

        result = np.empty_like(bp_1)
        for i in prange(bp_1.shape[0]):
            frm_bp1_x, frm_bp1_y = bp_1[i, 0], bp_1[i, 1]
            frm_bp2_x, frm_bp2_y = bp_2[i, 0], bp_2[i, 1]
            new_x = frm_bp1_x + percentile * (frm_bp2_x - frm_bp1_x)
            new_y = frm_bp1_y + percentile * (frm_bp2_y - frm_bp1_y)
            result[i, 0] = new_x
            result[i, 1] = new_y

        return result