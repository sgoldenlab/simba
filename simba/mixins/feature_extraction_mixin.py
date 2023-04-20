import numpy as np
from numba import jit, prange
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
import math
import os, glob
from simba.enums import Paths, Options, ReadConfig, Dtypes
from simba.feature_extractors.unit_tests import (read_video_info_csv,
                                               check_minimum_roll_windows)
from simba.read_config_unit_tests import (read_config_file,
                                          read_config_entry,
                                          read_project_path_and_file_type,
                                          check_if_filepath_list_is_empty)
from simba.drop_bp_cords import (getBpHeaders,
                                 create_body_part_dictionary,
                                 getBpNames)
from simba.misc_tools import check_multi_animal_status

class FeatureExtractionMixin(object):

    def __init__(self,
                 config_path: str):

        self.config = read_config_file(config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.data_in_dir = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.save_dir = os.path.join(self.project_path, Paths.FEATURES_EXTRACTED_DIR.value)
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.no_animals = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.ANIMAL_CNT.value, Dtypes.INT.value)
        self.roll_windows_values = check_minimum_roll_windows(Options.ROLLING_WINDOW_DIVISORS.value,
                                                              self.vid_info_df['fps'].min())
        self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg=f'SIMBA ERROR: No files of type {self.file_type} found in {self.data_in_dir}')
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.xcols, self.ycols, self.pcols = getBpNames(config_path)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.no_animals, self.xcols, self.ycols, self.pcols, [])
        self.col_headers = getBpHeaders(config_path)
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