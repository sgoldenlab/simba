__author__ = "Simon Nilsson"

import warnings
warnings.filterwarnings("ignore")
import numpy as np
from numba import jit, prange
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
import math
import os, glob
from scipy import stats
import pandas as pd
from scipy.signal import find_peaks
from simba.utils.enums import Paths, Options
from simba.utils.checks import check_if_filepath_list_is_empty, check_file_exist_and_readable, check_minimum_roll_windows
from simba.utils.read_write import (read_project_path_and_file_type,
                                    read_video_info_csv,
                                    read_config_file,
                                    get_bp_headers)
from simba.utils.errors import CountError
import simba

class FeatureExtractionMixin(object):

    def __init__(self,
                 config_path: str or None=None):

        """
        Methods for featurizing pose-estimation data
        :param configparser.Configparser config_path: path to SimBA project_config.ini
        """

        if config_path:
            self.config_path = config_path
            self.config = read_config_file(config_path=config_path)
            self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
            self.video_info_path = os.path.join(self.project_path, Paths.VIDEO_INFO.value)
            self.video_info_df = read_video_info_csv(file_path=self.video_info_path)
            self.data_in_dir = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
            self.save_dir = os.path.join(self.project_path, Paths.FEATURES_EXTRACTED_DIR.value)
            if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
            bp_path = os.path.join(self.project_path, Paths.BP_NAMES.value)
            check_file_exist_and_readable(file_path=bp_path)
            self.body_parts_lst = list(pd.read_csv(bp_path, header=None)[0])
            self.roll_windows_values = check_minimum_roll_windows(Options.ROLLING_WINDOW_DIVISORS.value, self.video_info_df['fps'].min())
            self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
            check_if_filepath_list_is_empty(filepaths=self.files_found, error_msg=f'No files of type {self.file_type} found in {self.data_in_dir}')
            self.col_headers = get_bp_headers(body_parts_lst=self.body_parts_lst)
            self.col_headers_shifted = [bp + '_shifted' for bp in self.col_headers]

    @staticmethod
    @jit(nopython=True)
    def euclidean_distance(bp_1_x_vals, bp_2_x_vals, bp_1_y_vals, bp_2_y_vals, px_per_mm):
        series = (np.sqrt((bp_1_x_vals - bp_2_x_vals) ** 2 + (bp_1_y_vals - bp_2_y_vals) ** 2)) / px_per_mm
        return series


    @staticmethod
    @jit(nopython=True, fastmath=True)
    def angle3pt(ax, ay, bx, by, cx, cy):
        ang = math.degrees(math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx))
        return ang + 360 if ang < 0 else ang

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def angle3pt_serialized(data: np.array):
        results = np.full((data.shape[0]), 0.0)
        for i in prange(data.shape[0]):
            angle = math.degrees(
                math.atan2(data[i][5] - data[i][3], data[i][4] - data[i][2]) - math.atan2(data[i][1] - data[i][3],
                                                                                          data[i][0] - data[i][2]))
            if angle < 0:
                angle += 360
            results[i] = angle

        return results

    @staticmethod
    def convex_hull_calculator_mp(arr: np.array, px_per_mm: float) -> float:
        arr = np.unique(arr, axis=0).astype(int)
        if arr.shape[0] < 3:
            return 0
        for i in range(1, arr.shape[0]):
            if (arr[i] != arr[0]).all():
                try:
                    return ConvexHull(arr, qhull_options='En').area / px_per_mm
                except QhullError:
                    return 0
            else:
                pass
        return 0

    @staticmethod
    @jit(nopython=True)
    def count_values_in_range(data: np.array, ranges: np.array):
        results = np.full((data.shape[0], ranges.shape[0]), 0)
        for i in prange(data.shape[0]):
            for j in prange(ranges.shape[0]):
                lower_bound, upper_bound = ranges[j][0], ranges[j][1]
                results[i][j] = data[i][np.logical_and(data[i] >= lower_bound, data[i] <= upper_bound)].shape[0]
        return results

    @staticmethod
    @jit(nopython=True)
    def framewise_euclidean_distance_roi(location_1: np.array,
                                         location_2: np.array,
                                         px_per_mm: float,
                                         centimeter: bool = False) -> np.array:

        results = np.full((location_1.shape[0]), np.nan)
        for i in prange(location_1.shape[0]):
            results[i] = np.linalg.norm(location_1[i] - location_2) / px_per_mm
        if centimeter:
            results = results / 10
        return results

    @staticmethod
    @jit(nopython=True)
    def framewise_inside_rectangle_roi(bp_location: np.array,
                                       roi_coords: np.array) -> np.array:

        results = np.full((bp_location.shape[0]), 0)
        within_x_idx = np.argwhere((bp_location[:, 0] <= roi_coords[1][0]) & (bp_location[:, 0] >= roi_coords[0][0])).flatten()
        within_y_idx = np.argwhere((bp_location[:, 1] <= roi_coords[1][1]) & (bp_location[:, 1] >= roi_coords[0][1])).flatten()
        for i in prange(within_x_idx.shape[0]):
            match = np.argwhere(within_y_idx == within_x_idx[i])
            if match.shape[0] > 0:
                results[within_x_idx[i]] = 1
        return results

    @staticmethod
    @jit(nopython=True)
    def framewise_inside_polygon_roi(bp_location: np.array,
                                     roi_coords: np.array) -> np.array:
        results = np.full((bp_location.shape[0]), 0)
        for i in prange(0, results.shape[0]):
            x, y, n = bp_location[i][0], bp_location[i][1], len(roi_coords)
            p2x, p2y, xints, inside = 0.0, 0.0, 0.0, False
            p1x, p1y = roi_coords[0]
            for j in prange(n + 1):
                p2x, p2y = roi_coords[j % n]
                if (y > min(p1y, p2y)) and (y <= max(p1y, p2y)) and (x <= max(p1x, p2x)):
                    if p1y != p2y:
                        xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
                p1x, p1y = p2x, p2y
            if inside:
                results[i] = 1

        return results

    def windowed_frequentist_distribution_tests(self,
                                                data: np.array,
                                                feature_name: str,
                                                fps: int):
        """
        Helper to compare feature value distributions in 1-s sequential time-bins: Kolmogorov-Smirnov and T-tests
        Compares the feature values against a normal distribution: Shapiro.
        Find the number of peaks in *rolling* 1s long feature window.
        """

        ks_results, = np.full((data.shape[0]), -1.0),
        t_test_results = np.full((data.shape[0]), -1.0)
        shapiro_results = np.full((data.shape[0]), -1.0)
        peak_cnt_results = np.full((data.shape[0]), -1.0)

        for i in range(fps, data.shape[0] - fps, fps):
            bin_1_idx, bin_2_idx = [i - fps, i], [i, i + fps]
            bin_1_data, bin_2_data = data[bin_1_idx[0]:bin_1_idx[1]], data[bin_2_idx[0]:bin_2_idx[1]]
            ks_results[i:i + fps + 1] = stats.ks_2samp(data1=bin_1_data, data2=bin_2_data).statistic
            t_test_results[i:i + fps + 1] = stats.ttest_ind(bin_1_data, bin_2_data).statistic

        for i in range(0, data.shape[0] - fps, fps):
            shapiro_results[i:i + fps + 1] = stats.shapiro(data[i:i + fps])[0]

        rolling_idx = np.arange(fps)[None, :] + 1 * np.arange(data.shape[0])[:, None]
        for i in range(rolling_idx.shape[0]):
            bin_start_idx, bin_end_idx = rolling_idx[i][0], rolling_idx[i][-1]
            peaks, _ = find_peaks(data[bin_start_idx:bin_end_idx], height=0)
            peak_cnt_results[i] = len(peaks)

        columns = [f'{feature_name}_KS', f'{feature_name}_TTEST',
                   f'{feature_name}_SHAPIRO', f'{feature_name}_PEAK_CNT']

        return pd.DataFrame(
            np.column_stack((ks_results, t_test_results, shapiro_results, peak_cnt_results)),
            columns=columns).round(4).fillna(0)

    @staticmethod
    @jit(nopython=True, cache=True)
    def cdist(array_1: np.array, array_2: np.array):
        results = np.full((array_1.shape[0], array_2.shape[0]), np.nan)
        for i in prange(array_1.shape[0]):
            for j in prange(array_2.shape[0]):
                results[i][j] = np.linalg.norm(array_1[i] - array_2[j])
        return results

    def check_directionality_viable(self):
        """
        Helper to check if it is possible to calculate ``directionality`` statistics (i.e., nose, and ear coordinates from
        pose estimation has to be present)

        Parameters
        ----------
        None

        Returns
        -------
        directionalitySetting: bool
        NoseCoords: list
        EarLeftCoords: list
        EarRightCoords: list
        """

        direction_viable = True
        nose_cords, ear_left_cords, ear_right_cords = [], [], []
        for animal_name in self.animal_bp_dict.keys():
            for bp_cord in ['X_bps', 'Y_bps']:
                bp_list = self.animal_bp_dict[animal_name][bp_cord]
                for bp_name in bp_list:
                    bp_name_components = bp_name.split('_')
                    bp_name_components = [x.lower() for x in bp_name_components]
                    if ('nose' in bp_name_components):
                        nose_cords.append(bp_name)
                    elif ('ear' in bp_name_components) and ('left' in bp_name_components):
                        ear_left_cords.append(bp_name)
                    elif ('ear' in bp_name_components) and ('right' in bp_name_components):
                        ear_right_cords.append(bp_name)
                    else:
                        pass

        for cord in [nose_cords, ear_left_cords, ear_right_cords]:
            if len(cord) != len(self.animal_bp_dict.keys()) * 2:
                direction_viable = False

        if direction_viable:
            nose_cords = [nose_cords[i * 2:(i + 1) * 2] for i in range((len(nose_cords) + 2 - 1) // 2)]
            ear_left_cords = [ear_left_cords[i * 2:(i + 1) * 2] for i in range((len(ear_left_cords) + 2 - 1) // 2)]
            ear_right_cords = [ear_right_cords[i * 2:(i + 1) * 2] for i in range((len(ear_right_cords) + 2 - 1) // 2)]

        return direction_viable, nose_cords, ear_left_cords, ear_right_cords

    def get_feature_extraction_headers(self,
                                       pose: str):
        """ Helper to return the headers names that should be used during feature extraction"""
        simba_dir = os.path.dirname(simba.__file__)
        feature_categories_csv_path = os.path.join(simba_dir, Paths.SIMBA_FEATURE_EXTRACTION_COL_NAMES_PATH.value)
        check_file_exist_and_readable(file_path=feature_categories_csv_path)
        bps = list(pd.read_csv(feature_categories_csv_path)[pose])
        return [x for x in bps if str(x) != 'nan']

    @staticmethod
    @jit(nopython=True)
    def jitted_line_crosses_to_nonstatic_targets(left_ear_array: np.array,
                                                 right_ear_array: np.array,
                                                 nose_array: np.array,
                                                 target_array: np.array) -> np.array:
        """
        Jitted helper to calculate if an animal is directing towards another animals body-part coordinate.

        Parameters
        ----------
        left_ear_array: array
            left ear coordinates of observing animal.
        right_ear_array: array
            right ear coordinates of observing animal.
        nose_array: array
            nose coordinates of observing animal.
        target_array: array
            The location of the target coordinates.

        Returns
        -------
        np.array
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
            if (Nh < Ph and Nh < Qh and Qh < Ph):
                results_array[frame_no] = [0, right_ear_array[frame_no][0], right_ear_array[frame_no][1], True]
            elif (Nh < Ph and Nh < Qh and Ph < Qh):
                results_array[frame_no] = [1, left_ear_array[frame_no][0], left_ear_array[frame_no][1], True]
            else:
                results_array[frame_no] = [2, -1, -1, False]

        return results_array

    @staticmethod
    @jit(nopython=True)
    def jitted_line_crosses_to_static_targets(left_ear_array: np.array,
                                              right_ear_array: np.array,
                                              nose_array: np.array,
                                              target_array: np.array) -> np.array:
        """
        Jitted helper to calculate if an animal is directing towards a static coordinate
        (e.g., the center of a user-defined ROI)

        Parameters
        ----------
        left_ear_array: array
            left ear coordinates of observing animal.
        right_ear_array: array
            right ear coordinates of observing animal.
        nose_array: array
            nose coordinates of observing animal.
        target_array: array
            The location of the target coordinates.

        Returns
        -------
        np.array
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
            if (Nh < Ph and Nh < Qh and Qh < Ph):
                results_array[frame_no] = [0, right_ear_array[frame_no][0], right_ear_array[frame_no][1], True]
            elif (Nh < Ph and Nh < Qh and Ph < Qh):
                results_array[frame_no] = [1, left_ear_array[frame_no][0], left_ear_array[frame_no][1], True]
            else:
                results_array[frame_no] = [2, -1, -1, False]

        return results_array


    def minimum_bounding_rectangle(self,
                                   points: np.array) -> np.array:

        """
        Finds the minimum bounding rectangle of a convex hull
        https://stackoverflow.com/questions/13542855/algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput
        """

        pi2 = np.pi / 2.
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
    @jit(nopython=True)
    def framewise_euclidean_distance(location_1: np.array,
                                     location_2: np.array,
                                     px_per_mm: float,
                                     centimeter: bool = False) -> np.array:

        results = np.full((location_1.shape[0]), np.nan)
        for i in prange(location_1.shape[0]):
            results[i] = np.linalg.norm(location_1[i] - location_2[i]) / px_per_mm
        if centimeter:
            results = results / 10
        return results


    def get_bp_headers(self):
        """
        Helper to create ordered list of all column header fields for SimBA project dataframes.
        """
        self.col_headers = []
        for bp in self.body_parts_lst:
            c1, c2, c3 = (f'{bp}_x', f'{bp}_y', f'{bp}_p')
            self.col_headers.extend((c1, c2, c3))

    def check_directionality_cords(self) -> dict:

        """
        Helper to check if ear and nose body-parts are present within the pose-estimation data.

        Parameters
        ----------
        animal_bp_dict: dict
            Python dictionary created by ``create_body_part_dictionary``.

        Returns
        -------
        dict
            body-part names of ear and nose body-parts as values and animal names as keys. If empty,
            ear and nose body-parts are not present within the pose-estimation data
        """
        results = {}
        for animal in self.animal_bp_dict.keys():
            results[animal] = {}
            results[animal]['Nose'] = {}
            results[animal]['Ear_left'] = {}
            results[animal]['Ear_right'] = {}
            for dimension in ['X_bps', 'Y_bps']:
                for cord in self.animal_bp_dict[animal][dimension]:
                    if ("nose" in cord.lower()) and ("x" in cord.lower()):
                        results[animal]['Nose']['X_bps'] = cord
                    elif ("nose" in cord.lower()) and ("y" in cord.lower()):
                        results[animal]['Nose']['Y_bps'] = cord
                    elif ("left" in cord.lower()) and ("x" in cord.lower()) and ("ear" in cord.lower()):
                        results[animal]['Ear_left']['X_bps'] = cord
                    elif ("left" in cord.lower()) and ("Y".lower() in cord.lower()) and ("ear".lower() in cord.lower()):
                        results[animal]['Ear_left']['Y_bps'] = cord
                    elif ("right" in cord.lower()) and ("x" in cord.lower()) and ("ear" in cord.lower()):
                        results[animal]['Ear_right']['X_bps'] = cord
                    elif ("right" in cord.lower()) and ("y" in cord.lower()) and ("ear".lower() in cord.lower()):
                        results[animal]['Ear_right']['Y_bps'] = cord
        return results

    def insert_default_headers_for_feature_extraction(self,
                                                      df: pd.DataFrame,
                                                      headers: list,
                                                      pose_config: str,
                                                      filename: str):
        if len(headers) != len(df.columns):
            raise CountError(f'Your SimBA project is set to using the default {pose_config} pose-configuration. '
                             f'SimBA therefore expects {str(len(headers))} columns of data inside the files within the project_folder. However, '
                             f'within file {filename} file, SimBA found {str(len(df.columns))} columns.')
        else:
            df.columns = headers
            return df

    def line_crosses_to_static_targets(p: list,
                                       q: list,
                                       n: list,
                                       M: list,
                                       coord: list):
        """
        Non-jitted helper to calculate if an animal is directing towards a coordinate.
        For improved runtime, use ``simba.mixins.feature_extraction_mixin.jitted_line_crosses_to_static_targets``


        Parameters
        ----------
        p: list
            left ear coordinates of observing animal.
        q: list
            right ear coordinates of observing animal.
        n: list
            nose coordinates of observing animal.
        M: list
            The location of the target coordinates.
        coord: list
            empty list to store the eye coordinate of the observing animal.

        Returns
        -------
        bool
        coord: list
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
        if (Nh < Ph and Nh < Qh and Qh < Ph):
            coord.extend((q[0], q[1]))
            return True, coord
        elif (Nh < Ph and Nh < Qh and Ph < Qh):
            coord.extend((p[0], p[1]))
            return True, coord
        else:
            return False, coord
