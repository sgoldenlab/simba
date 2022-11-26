from simba.read_config_unit_tests import (read_config_entry, check_int, check_str, insert_default_headers_for_feature_extraction, check_file_exist_and_readable, read_config_file)
import os, glob
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info, check_minimum_roll_windows
from simba.drop_bp_cords import get_fn_ext
from simba.rw_dfs import read_df, save_df
import pandas as pd
from copy import deepcopy
from scipy.spatial import ConvexHull
import scipy
import numpy as np
from numba import jit
from collections import defaultdict
import math

class ExtractFeaturesFrom14bps(object):
    """
    Class for creating a hard-coded set of features from two animals with 7 tracked body-parts
    each using pose-estimation. Results are stored in the `project_folder/csv/features_extracted`
    directory of the SimBA project.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    ----------
    Feature extraction tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features>`__.


    Examples
    ----------
    >>> feature_extractor = ExtractFeaturesFrom14bps(config_path='MyProjectConfig')
    >>> feature_extractor.extract_features()
    """
    
    def __init__(self,
                 config_path: str):

        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.data_in_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.save_dir = os.path.join(self.project_path, 'csv', 'features_extracted')
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        roll_windows_values = [2, 5, 6, 7.5, 15]
        self.roll_windows_values = check_minimum_roll_windows(roll_windows_values, self.vid_info_df['fps'].min())
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
        self.simba_working_dir = os.getcwd()
        self.in_headers = ["Ear_left_1_x", "Ear_left_1_y", "Ear_left_1_p", "Ear_right_1_x", "Ear_right_1_y",
                         "Ear_right_1_p", "Nose_1_x", "Nose_1_y", "Nose_1_p", "Center_1_x", "Center_1_y", "Center_1_p",
                         "Lat_left_1_x", "Lat_left_1_y",
                         "Lat_left_1_p", "Lat_right_1_x", "Lat_right_1_y", "Lat_right_1_p", "Tail_base_1_x",
                         "Tail_base_1_y", "Tail_base_1_p", "Ear_left_2_x", "Ear_left_2_y", "Ear_left_2_p", "Ear_right_2_x", "Ear_right_2_y", "Ear_right_2_p",
                         "Nose_2_x", "Nose_2_y", "Nose_2_p", "Center_2_x", "Center_2_y", "Center_2_p", "Lat_left_2_x","Lat_left_2_y",
                         "Lat_left_2_p", "Lat_right_2_x", "Lat_right_2_y", "Lat_right_2_p", "Tail_base_2_x",
                         "Tail_base_2_y", "Tail_base_2_p"]
        self.mouse_1_headers, self.mouse_2_headers = self.in_headers[0:21], self.in_headers[21:]
        self.mouse_2_p_headers = [x for x in self.mouse_2_headers if x[-2:] == '_p']
        self.mouse_1_p_headers = [x for x in self.mouse_1_headers if x[-2:] == '_p']
        self.mouse_1_headers = [x for x in self.mouse_1_headers if x[-2:] != '_p']
        self.mouse_2_headers = [x for x in self.mouse_2_headers if x[-2:] != '_p']
        print('Extracting features from {} file(s)...'.format(str(len(self.files_found))))

    @staticmethod
    @jit(nopython=True)
    def __euclidean_distance(bp_1_x_vals, bp_2_x_vals, bp_1_y_vals, bp_2_y_vals, px_per_mm):
        series = (np.sqrt((bp_1_x_vals - bp_2_x_vals) ** 2 + (bp_1_y_vals - bp_2_y_vals) ** 2)) / px_per_mm
        return series

    def __angle3pt(self, ax, ay, bx, by, cx, cy):
        ang = math.degrees(math.atan2(cy - by, cx - bx) - math.atan2(ay - by, ax - bx))
        return ang + 360 if ang < 0 else ang

    def __count_values_in_range(self, series, values_in_range_min, values_in_range_max):
        return series.between(left=values_in_range_min, right=values_in_range_max).sum()

    def extract_features(self):
        """
        Method to compute and save features to disk. Results are saved in the `project_folder/csv/features_extracted`
        directory of the SimBA project.

        Returns
        -------
        None
        """

        for file_cnt, file_path in enumerate(self.files_found):
            roll_windows = []
            _, self.video_name, _ = get_fn_ext(file_path)
            video_settings, self.px_per_mm, fps = read_video_info(self.vid_info_df, self.video_name)
            for window in self.roll_windows_values:
                roll_windows.append(int(fps / window))
            self.in_data = read_df(file_path, self.file_type).fillna(0).apply(pd.to_numeric).reset_index(drop=True)
            self.in_data = insert_default_headers_for_feature_extraction(df=self.in_data, headers=self.in_headers, pose_config='14 body-parts', filename=file_path)
            self.out_data = deepcopy(self.in_data)
            self.out_data['Mouse_1_poly_area'] = self.out_data.apply(lambda x: ConvexHull(np.array([[x['Ear_left_1_x'], x["Ear_left_1_y"]],[x['Ear_right_1_x'], x["Ear_right_1_y"]],[x['Nose_1_x'], x["Nose_1_y"]],[x['Lat_left_1_x'], x["Lat_left_1_y"]],[x['Lat_right_1_x'], x["Lat_right_1_y"]], [x['Tail_base_1_x'], x["Tail_base_1_y"]],[x['Center_1_x'], x["Center_1_y"]]])).area, axis=1)
            self.out_data['Mouse_1_poly_area'] = self.out_data['Mouse_1_poly_area'] / self.px_per_mm
            self.out_data['Mouse_2_poly_area'] = self.out_data.apply(lambda x: ConvexHull(np.array([[x['Ear_left_2_x'], x["Ear_left_2_y"]], [x['Ear_right_2_x'], x["Ear_right_2_y"]], [x['Nose_2_x'], x["Nose_2_y"]], [x['Lat_left_2_x'], x["Lat_left_2_y"]], [x['Lat_right_2_x'], x["Lat_right_2_y"]], [x['Tail_base_2_x'], x["Tail_base_2_y"]], [x['Center_2_x'], x["Center_2_y"]]])).area, axis=1)
            self.out_data['Mouse_2_poly_area'] = self.out_data['Mouse_2_poly_area'] / self.px_per_mm
            self.in_data_shifted = self.out_data.shift(periods=1).add_suffix('_shifted').fillna(0)
            self.in_data = pd.concat([self.in_data, self.in_data_shifted], axis=1, join='inner').fillna(0).reset_index(drop=True)
            self.out_data['Mouse_1_nose_to_tail'] = self.__euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Tail_base_1_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Tail_base_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_nose_to_tail'] = self.__euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Tail_base_2_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Tail_base_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_width'] = self.__euclidean_distance(self.out_data['Lat_left_1_x'].values, self.out_data['Lat_right_1_x'].values, self.out_data['Lat_left_1_y'].values, self.out_data['Lat_right_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_width'] = self.__euclidean_distance(self.out_data['Lat_left_2_x'].values, self.out_data['Lat_right_2_x'].values, self.out_data['Lat_left_2_y'].values, self.out_data['Lat_right_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_Ear_distance'] = self.__euclidean_distance(self.out_data['Ear_left_1_x'].values, self.out_data['Ear_right_1_x'].values, self.out_data['Ear_left_1_y'].values, self.out_data['Ear_right_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_Ear_distance'] = self.__euclidean_distance(self.out_data['Ear_left_2_x'].values, self.out_data['Ear_right_2_x'].values, self.out_data['Ear_left_2_y'].values, self.out_data['Ear_right_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_Nose_to_centroid'] = self.__euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Center_1_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Center_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_Nose_to_centroid'] = self.__euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Center_2_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Center_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_Nose_to_lateral_left'] = self.__euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Lat_left_1_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Lat_left_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_Nose_to_lateral_left'] = self.__euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Lat_left_2_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Lat_left_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_Nose_to_lateral_right'] = self.__euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Lat_right_1_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Lat_right_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_Nose_to_lateral_right'] = self.__euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Lat_right_2_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Lat_right_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_Centroid_to_lateral_left'] = self.__euclidean_distance(self.out_data['Center_1_x'].values, self.out_data['Lat_left_1_x'].values, self.out_data['Center_1_y'].values, self.out_data['Lat_left_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_Centroid_to_lateral_left'] = self.__euclidean_distance(self.out_data['Center_2_x'].values, self.out_data['Lat_left_2_x'].values, self.out_data['Center_2_y'].values, self.out_data['Lat_left_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_Centroid_to_lateral_right'] = self.__euclidean_distance(self.out_data['Center_1_x'].values, self.out_data['Lat_right_1_x'].values, self.out_data['Center_1_y'].values, self.out_data['Lat_right_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_Centroid_to_lateral_right'] = self.__euclidean_distance(self.out_data['Center_2_x'].values, self.out_data['Lat_right_2_x'].values, self.out_data['Center_2_y'].values, self.out_data['Lat_right_2_y'].values, self.px_per_mm)
            self.out_data['Centroid_distance'] = self.__euclidean_distance(self.out_data['Center_2_x'].values, self.out_data['Center_1_x'].values, self.out_data['Center_2_y'].values, self.out_data['Center_1_y'].values, self.px_per_mm)
            self.out_data['Nose_to_nose_distance'] = self.__euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Nose_1_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Nose_1_y'].values, self.px_per_mm)
            self.out_data['M1_Nose_to_M2_lat_left'] = self.__euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Lat_left_2_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Lat_left_2_y'].values, self.px_per_mm)
            self.out_data['M1_Nose_to_M2_lat_left'] = self.__euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Lat_left_2_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Lat_left_2_y'].values, self.px_per_mm)
            self.out_data['M1_Nose_to_M2_lat_right'] = self.__euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Lat_right_2_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Lat_right_2_y'].values, self.px_per_mm)
            self.out_data['M2_Nose_to_M1_lat_left'] = self.__euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Lat_left_1_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Lat_left_1_y'].values, self.px_per_mm)
            self.out_data['M2_Nose_to_M1_lat_right'] = self.__euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Lat_right_1_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Lat_right_1_y'].values, self.px_per_mm)
            self.out_data['M1_Nose_to_M2_tail_base'] = self.__euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Tail_base_2_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Tail_base_2_y'].values, self.px_per_mm)
            self.out_data['M2_Nose_to_M1_tail_base'] = self.__euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Tail_base_1_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Tail_base_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_centroid'] = self.__euclidean_distance(self.in_data['Center_1_x_shifted'].values, self.in_data['Center_1_x'].values, self.in_data['Center_1_y_shifted'].values, self.in_data['Center_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_centroid'] = self.__euclidean_distance(self.in_data['Center_2_x_shifted'].values, self.in_data['Center_2_x'].values, self.in_data['Center_2_y_shifted'].values, self.in_data['Center_2_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_nose'] = self.__euclidean_distance(self.in_data['Nose_1_x_shifted'].values, self.in_data['Nose_1_x'].values, self.in_data['Nose_1_y_shifted'].values, self.in_data['Nose_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_nose'] = self.__euclidean_distance(self.in_data['Nose_2_x_shifted'].values, self.in_data['Nose_2_x'].values, self.in_data['Nose_2_y_shifted'].values, self.in_data['Nose_2_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_tail_base'] = self.__euclidean_distance(self.in_data['Tail_base_1_x_shifted'].values, self.in_data['Tail_base_1_x'].values, self.in_data['Tail_base_1_y_shifted'].values, self.in_data['Tail_base_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_tail_base'] = self.__euclidean_distance(self.in_data['Tail_base_2_x_shifted'].values, self.in_data['Tail_base_2_x'].values, self.in_data['Tail_base_2_y_shifted'].values, self.in_data['Tail_base_2_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_left_ear'] = self.__euclidean_distance(self.in_data['Ear_left_1_x_shifted'].values, self.in_data['Ear_left_1_x'].values, self.in_data['Ear_left_1_y_shifted'].values, self.in_data['Ear_left_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_left_ear'] = self.__euclidean_distance(self.in_data['Ear_left_2_x_shifted'].values, self.in_data['Ear_left_2_x'].values, self.in_data['Ear_left_2_y_shifted'].values, self.in_data['Ear_left_2_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_right_ear'] = self.__euclidean_distance(self.in_data['Ear_right_1_x_shifted'].values, self.in_data['Ear_right_1_x'].values, self.in_data['Ear_right_1_y_shifted'].values, self.in_data['Ear_right_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_right_ear'] = self.__euclidean_distance(self.in_data['Ear_right_2_x_shifted'].values, self.in_data['Ear_right_2_x'].values, self.in_data['Ear_right_2_y_shifted'].values, self.in_data['Ear_right_2_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_lateral_left'] = self.__euclidean_distance(self.in_data['Lat_left_1_x_shifted'].values, self.in_data['Lat_left_1_x'].values, self.in_data['Lat_left_1_y_shifted'].values, self.in_data['Lat_left_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_lateral_left'] = self.__euclidean_distance(self.in_data['Lat_left_2_x_shifted'].values, self.in_data['Lat_left_2_x'].values, self.in_data['Lat_left_2_y_shifted'].values, self.in_data['Lat_left_2_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_lateral_right'] = self.__euclidean_distance(self.in_data['Lat_right_1_x_shifted'].values, self.in_data['Lat_right_1_x'].values, self.in_data['Lat_right_1_y_shifted'].values, self.in_data['Lat_right_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_lateral_right'] = self.__euclidean_distance(self.in_data['Lat_right_2_x_shifted'].values, self.in_data['Lat_right_2_x'].values, self.in_data['Lat_right_2_y_shifted'].values, self.in_data['Lat_right_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_polygon_size_change'] = (self.in_data['Mouse_1_poly_area_shifted'] - self.out_data['Mouse_1_poly_area'])
            self.out_data['Mouse_2_polygon_size_change'] = (self.in_data['Mouse_2_poly_area_shifted'] - self.out_data['Mouse_2_poly_area'])

            print('Calculating hull variables...')
            mouse_1_array, mouse_2_array = self.in_data[self.mouse_1_headers].to_numpy(), self.in_data[self.mouse_2_headers].to_numpy()
            self.hull_dict = defaultdict(list)
            for cnt, (animal_1, animal_2) in enumerate(zip(mouse_1_array, mouse_2_array)):
                animal_1, animal_2 = np.reshape(animal_1, (-1, 2)), np.reshape(animal_2, (-1, 2))
                animal_1_dist = scipy.spatial.distance.cdist(animal_1, animal_1, metric='euclidean')
                animal_2_dist = scipy.spatial.distance.cdist(animal_2, animal_2, metric='euclidean')
                animal_1_dist, animal_2_dist = animal_1_dist[animal_1_dist != 0], animal_2_dist[animal_2_dist != 0]
                for animal, animal_name in zip([animal_1_dist, animal_2_dist], ['M1', 'M2']):
                    self.hull_dict['{}_hull_large_euclidean'.format(animal_name)].append(np.amax(animal) / self.px_per_mm)
                    self.hull_dict['{}_hull_small_euclidean'.format(animal_name)].append(np.min(animal) / self.px_per_mm)
                    self.hull_dict['{}_hull_mean_euclidean'.format(animal_name)].append(np.mean(animal) / self.px_per_mm)
                    self.hull_dict['{}_hull_sum_euclidean'.format(animal_name)].append(np.sum(animal) / self.px_per_mm)
            for k, v in self.hull_dict.items():
                self.out_data[k] = v

            self.out_data['Sum_euclidean_distance_hull_M1_M2'] = self.out_data['M1_hull_sum_euclidean'] + self.out_data['M2_hull_sum_euclidean']
            self.out_data['Total_movement_centroids'] = self.out_data['Movement_mouse_1_centroid'] + self.out_data['Movement_mouse_2_centroid']
            mouse_1_movement_cols = [x for x in self.out_data.columns if x.startswith('Movement_mouse_1_')]
            mouse_2_movement_cols = [x for x in self.out_data.columns if x.startswith('Movement_mouse_2_')]
            self.out_data['Total_movement_all_bodyparts_M1'] = self.out_data[mouse_1_movement_cols].sum(axis=1)
            self.out_data['Total_movement_all_bodyparts_M2'] = self.out_data[mouse_2_movement_cols].sum(axis=1)
            self.out_data['Total_movement_all_bodyparts_both_mice'] = self.out_data['Total_movement_all_bodyparts_M1'] + self.out_data['Total_movement_all_bodyparts_M2']

            print('Calculating rolling windows: medians, medians, and sums...')

            for window in self.roll_windows_values:
                col_name = 'Sum_euclid_distances_hull_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Sum_euclidean_distance_hull_M1_M2'].rolling(int(window), min_periods=1).median()
                col_name = 'Sum_euclid_distances_hull_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Sum_euclidean_distance_hull_M1_M2'].rolling(int(window), min_periods=1).mean()
                col_name = 'Sum_euclid_distances_hull_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Sum_euclidean_distance_hull_M1_M2'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Movement_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Total_movement_centroids'].rolling(int(window), min_periods=1).median()
                col_name = 'Movement_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Total_movement_centroids'].rolling(int(window), min_periods=1).mean()
                col_name = 'Movement_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Total_movement_centroids'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Distance_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Centroid_distance'].rolling(int(window), min_periods=1).median()
                col_name = 'Distance_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Centroid_distance'].rolling(int(window), min_periods=1).mean()
                col_name = 'Distance_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Centroid_distance'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Mouse1_width_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Mouse_1_width'].rolling(int(window), min_periods=1).median()
                col_name = 'Mouse1_width_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Mouse_1_width'].rolling(int(window), min_periods=1).mean()
                col_name = 'Mouse1_width_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Mouse_1_width'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Mouse2_width_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Mouse_2_width'].rolling(int(window), min_periods=1).median()
                col_name = 'Mouse2_width_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Mouse_2_width'].rolling(int(window), min_periods=1).mean()
                col_name = 'Mouse2_width_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Mouse_2_width'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Mouse1_mean_euclid_distances_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_hull_mean_euclidean'].rolling(int(window), min_periods=1).median()
                col_name = 'Mouse1_mean_euclid_distances_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_hull_mean_euclidean'].rolling(int(window), min_periods=1).mean()
                col_name = 'Mouse1_mean_euclid_distances_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_hull_mean_euclidean'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Mouse2_mean_euclid_distances_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M2_hull_mean_euclidean'].rolling(int(window), min_periods=1).median()
                col_name = 'Mouse2_mean_euclid_distances_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M2_hull_mean_euclidean'].rolling(int(window), min_periods=1).mean()
                col_name = 'Mouse2_mean_euclid_distances_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M2_hull_mean_euclidean'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Mouse1_smallest_euclid_distances_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_hull_small_euclidean'].rolling(int(window), min_periods=1).median()
                col_name = 'Mouse1_smallest_euclid_distances_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_hull_small_euclidean'].rolling(int(window), min_periods=1).mean()
                col_name = 'Mouse1_smallest_euclid_distances_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_hull_small_euclidean'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Mouse2_smallest_euclid_distances_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M2_hull_small_euclidean'].rolling(int(window), min_periods=1).median()
                col_name = 'Mouse2_smallest_euclid_distances_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M2_hull_small_euclidean'].rolling(int(window), min_periods=1).mean()
                col_name = 'Mouse2_smallest_euclid_distances_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M2_hull_small_euclidean'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Mouse1_largest_euclid_distances_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_hull_large_euclidean'].rolling(int(window), min_periods=1).median()
                col_name = 'Mouse1_largest_euclid_distances_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_hull_large_euclidean'].rolling(int(window), min_periods=1).mean()
                col_name = 'Mouse1_largest_euclid_distances_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_hull_large_euclidean'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Mouse2_largest_euclid_distances_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M2_hull_large_euclidean'].rolling(int(window), min_periods=1).median()
                col_name = 'Mouse2_largest_euclid_distances_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M2_hull_large_euclidean'].rolling(int(window), min_periods=1).mean()
                col_name = 'Mouse2_largest_euclid_distances_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M2_hull_large_euclidean'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Total_movement_all_bodyparts_both_mice_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Total_movement_all_bodyparts_both_mice'].rolling(int(window), min_periods=1).median()
                col_name = 'Total_movement_all_bodyparts_both_mice_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Total_movement_all_bodyparts_both_mice'].rolling(int(window), min_periods=1).mean()
                col_name = 'Total_movement_all_bodyparts_both_mice_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Total_movement_all_bodyparts_both_mice'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Total_movement_centroids_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Total_movement_centroids'].rolling(int(window), min_periods=1).median()
                col_name = 'Total_movement_centroids_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Total_movement_centroids'].rolling(int(window), min_periods=1).mean()
                col_name = 'Total_movement_centroids_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Total_movement_centroids'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Tail_base_movement_M1_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_1_tail_base'].rolling(int(window), min_periods=1).median()
                col_name = 'Tail_base_movement_M1_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_1_tail_base'].rolling(int(window), min_periods=1).mean()
                col_name = 'Tail_base_movement_M1_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_1_tail_base'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Tail_base_movement_M2_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_2_tail_base'].rolling(int(window), min_periods=1).median()
                col_name = 'Tail_base_movement_M2_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_2_tail_base'].rolling(int(window), min_periods=1).mean()
                col_name = 'Tail_base_movement_M2_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_2_tail_base'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Centroid_movement_M1_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_1_centroid'].rolling(int(window), min_periods=1).median()
                col_name = 'Centroid_movement_M1_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_1_centroid'].rolling(int(window), min_periods=1).mean()
                col_name = 'Centroid_movement_M1_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_1_centroid'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Centroid_movement_M2_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_2_centroid'].rolling(int(window), min_periods=1).median()
                col_name = 'Centroid_movement_M2_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_2_centroid'].rolling(int(window), min_periods=1).mean()
                col_name = 'Centroid_movement_M2_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_2_centroid'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Nose_movement_M1_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_1_nose'].rolling(int(window), min_periods=1).median()
                col_name = 'Nose_movement_M1_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_1_nose'].rolling(int(window), min_periods=1).mean()
                col_name = 'Nose_movement_M1_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_1_nose'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Nose_movement_M2_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_2_nose'].rolling(int(window), min_periods=1).median()
                col_name = 'Nose_movement_M2_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_2_nose'].rolling(int(window), min_periods=1).mean()
                col_name = 'Nose_movement_M2_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_2_nose'].rolling(int(window), min_periods=1).sum()

            print('Calculating angles...')
            self.out_data['Mouse_1_angle'] = self.out_data.apply(lambda x: self.__angle3pt(x['Nose_1_x'], x['Nose_1_y'], x['Center_1_x'], x['Center_1_y'], x['Tail_base_1_x'], x['Tail_base_1_y']), axis=1)
            self.out_data['Mouse_2_angle'] = self.out_data.apply(lambda x: self.__angle3pt(x['Nose_2_x'], x['Nose_2_y'], x['Center_2_x'], x['Center_2_y'], x['Tail_base_2_x'], x['Tail_base_2_y']), axis=1)
            self.out_data['Total_angle_both_mice'] = self.out_data['Mouse_1_angle'] + self.out_data['Mouse_2_angle']

            for window in self.roll_windows_values:
                col_name = 'Total_angle_both_mice_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Total_angle_both_mice'].rolling(int(window), min_periods=1).sum()

            print('Calculating deviations...')
            self.out_data['Total_movement_all_bodyparts_both_mice_deviation'] = (self.out_data['Total_movement_all_bodyparts_both_mice'].mean() - self.out_data['Total_movement_all_bodyparts_both_mice'])
            self.out_data['Sum_euclid_distances_hull_deviation'] = (self.out_data['Sum_euclidean_distance_hull_M1_M2'].mean() - self.out_data['Sum_euclidean_distance_hull_M1_M2'])
            self.out_data['M1_smallest_euclid_distances_hull_deviation'] = (self.out_data['M1_hull_small_euclidean'].mean() - self.out_data['M1_hull_small_euclidean'])
            self.out_data['M1_largest_euclid_distances_hull_deviation'] = (self.out_data['M1_hull_large_euclidean'].mean() - self.out_data['M1_hull_large_euclidean'])
            self.out_data['M1_mean_euclid_distances_hull_deviation'] = (self.out_data['M1_hull_mean_euclidean'].mean() - self.out_data['M1_hull_mean_euclidean'])
            self.out_data['Centroid_distance_deviation'] = (self.out_data['Centroid_distance'].mean() - self.out_data['Centroid_distance'])
            self.out_data['Total_angle_both_mice_deviation'] = (self.out_data['Total_angle_both_mice'].mean() - self.out_data['Total_angle_both_mice'])
            self.out_data['Movement_mouse_1_deviation_centroid'] = (self.out_data['Movement_mouse_1_centroid'].mean() - self.out_data['Movement_mouse_1_centroid'])
            self.out_data['Movement_mouse_2_deviation_centroid'] = (self.out_data['Movement_mouse_2_centroid'].mean() - self.out_data['Movement_mouse_2_centroid'])
            self.out_data['Mouse_1_polygon_deviation'] = (self.out_data['Mouse_1_poly_area'].mean() - self.out_data['Mouse_1_poly_area'])
            self.out_data['Mouse_2_polygon_deviation'] = (self.out_data['Mouse_2_poly_area'].mean() - self.out_data['Mouse_2_poly_area'])

            for window in self.roll_windows_values:
                col_name = 'Total_movement_all_bodyparts_both_mice_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_deviation'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            for window in self.roll_windows_values:
                col_name = 'Sum_euclid_distances_hull_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_deviation'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            for window in self.roll_windows_values:
                col_name = 'Mouse1_smallest_euclid_distances_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_deviation'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            for window in self.roll_windows_values:
                col_name = 'Mouse1_largest_euclid_distances_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_deviation'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            for window in self.roll_windows_values:
                col_name = 'Mouse1_mean_euclid_distances_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_deviation'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            for window in self.roll_windows_values:
                col_name = 'Movement_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_deviation'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            for window in self.roll_windows_values:
                col_name = 'Distance_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_deviation'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            for window in self.roll_windows_values:
                col_name = 'Total_angle_both_mice_{}'.format(str(window))
                deviation_col_name = col_name + '_deviation'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            print('Calculating percentile ranks...')
            self.out_data['Movement_percentile_rank'] = self.out_data['Total_movement_centroids'].rank(pct=True)
            self.out_data['Distance_percentile_rank'] = self.out_data['Centroid_distance'].rank(pct=True)
            self.out_data['Movement_mouse_1_percentile_rank'] = self.out_data['Movement_mouse_1_centroid'].rank(pct=True)
            self.out_data['Movement_mouse_2_percentile_rank'] = self.out_data['Movement_mouse_1_centroid'].rank(pct=True)
            self.out_data['Movement_mouse_1_deviation_percentile_rank'] = self.out_data['Movement_mouse_1_deviation_centroid'].rank(pct=True)
            self.out_data['Movement_mouse_2_deviation_percentile_rank'] = self.out_data['Movement_mouse_2_deviation_centroid'].rank(pct=True)
            self.out_data['Centroid_distance_percentile_rank'] = self.out_data['Centroid_distance'].rank(pct=True)
            self.out_data['Centroid_distance_deviation_percentile_rank'] = self.out_data['Centroid_distance_deviation'].rank(pct=True)

            for window in self.roll_windows_values:
                col_name = 'Total_movement_all_bodyparts_both_mice_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_percentile_rank'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            for window in self.roll_windows_values:
                col_name = 'Sum_euclid_distances_hull_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_percentile_rank'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            for window in self.roll_windows_values:
                col_name = 'Mouse1_mean_euclid_distances_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_percentile_rank'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            for window in self.roll_windows_values:
                col_name = 'Mouse1_smallest_euclid_distances_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_percentile_rank'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            for window in self.roll_windows_values:
                col_name = 'Mouse1_largest_euclid_distances_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_percentile_rank'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            for window in self.roll_windows_values:
                col_name = 'Movement_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_percentile_rank'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            for window in self.roll_windows_values:
                col_name = 'Distance_mean_{}'.format(str(window))
                deviation_col_name = col_name + '_percentile_rank'
                self.out_data[deviation_col_name] = (self.out_data[col_name].mean() - self.out_data[col_name])

            print('Calculating path tortuosities...')
            as_strided = np.lib.stride_tricks.as_strided
            win_size = 3
            centroid_lst_mouse1_x = as_strided(self.out_data.Center_1_x, (len(self.out_data) - (win_size - 1), win_size), (self.out_data.Center_1_x.values.strides * 2))
            centroid_lst_mouse1_y = as_strided(self.out_data.Center_1_y, (len(self.out_data) - (win_size - 1), win_size), (self.out_data.Center_1_y.values.strides * 2))
            centroid_lst_mouse2_x = as_strided(self.out_data.Center_2_x, (len(self.out_data) - (win_size - 1), win_size), (self.out_data.Center_2_x.values.strides * 2))
            centroid_lst_mouse2_y = as_strided(self.out_data.Center_2_y, (len(self.out_data) - (win_size - 1), win_size), (self.out_data.Center_2_y.values.strides * 2))

            for window in self.roll_windows_values:
                start, end = 0, 0 + int(window)
                tortuosities_results = defaultdict(list)
                for frame in range(len(self.out_data)):
                    tortuosities_dict = defaultdict(list)
                    c_centroid_lst_mouse1_x, c_centroid_lst_mouse1_y = centroid_lst_mouse1_x[start:end], centroid_lst_mouse1_y[start:end]
                    c_centroid_lst_mouse2_x, c_centroid_lst_mouse2_y = centroid_lst_mouse2_x[start:end], centroid_lst_mouse2_y[start:end]
                    for frame_in_window in range(len(c_centroid_lst_mouse1_x)):
                        move_angle_mouse_1 = (self.__angle3pt(c_centroid_lst_mouse1_x[frame_in_window][0], c_centroid_lst_mouse1_y[frame_in_window][0],c_centroid_lst_mouse1_x[frame_in_window][1], c_centroid_lst_mouse1_y[frame_in_window][1], c_centroid_lst_mouse1_x[frame_in_window][2], c_centroid_lst_mouse1_y[frame_in_window][2]))
                        move_angle_mouse_2 = (self.__angle3pt(c_centroid_lst_mouse2_x[frame_in_window][0], c_centroid_lst_mouse2_y[frame_in_window][0], c_centroid_lst_mouse2_x[frame_in_window][1], c_centroid_lst_mouse2_y[frame_in_window][1],c_centroid_lst_mouse2_x[frame_in_window][2], c_centroid_lst_mouse2_y[frame_in_window][2]))
                        tortuosities_dict['Animal_1'].append(move_angle_mouse_1)
                        tortuosities_dict['Animal_2'].append(move_angle_mouse_2)
                    tortuosities_results['Animal_1'].append(sum(tortuosities_dict['Animal_1']) / (2 * math.pi))
                    tortuosities_results['Animal_2'].append(sum(tortuosities_dict['Animal_2']) / (2 * math.pi))
                    start += 1
                    end += 1
                col_name = 'Tortuosity_Mouse1_{}'.format(str(window))
                self.out_data[col_name] = tortuosities_results['Animal_1']

            print('Calculating pose probability scores...')
            all_p_columns = self.mouse_2_p_headers + self.mouse_1_p_headers
            self.out_data['Sum_probabilities'] = self.out_data[all_p_columns].sum(axis=1)
            self.out_data['Sum_probabilities_deviation'] = (self.out_data['Sum_probabilities'].mean() - self.out_data['Sum_probabilities'])
            self.out_data['Sum_probabilities_deviation_percentile_rank'] = self.out_data['Sum_probabilities_deviation'].rank(pct=True)
            self.out_data['Sum_probabilities_percentile_rank'] = self.out_data['Sum_probabilities_deviation_percentile_rank'].rank(pct=True)

            p_df = self.out_data.filter(all_p_columns)
            in_range_dict = {'0.1': [0.0, 0.1], '0.5': [0.000000000, 0.5], '0.75': [0.000000000, 0.75]}
            for k, v in in_range_dict.items():
                self.out_data["Low_prob_detections_{}".format(k)] = p_df.apply(func=lambda row: self.__count_values_in_range(row, v[0], v[1]), axis=1)

            self.out_data = self.out_data.reset_index(drop=True).fillna(0)
            save_path = os.path.join(self.save_dir, self.video_name + '.' + self.file_type)
            save_df(self.out_data, self.file_type, save_path)
            print('Feature extraction complete for {} ({}/{})...'.format(self.video_name, str(file_cnt + 1), str(len(self.files_found))))

        print('SIMBA COMPLETE: All features extracted. Results are stored in project_folder/csv/features_extracted directory')

# test = ExtractFeaturesFrom14bps(config_path='/Users/simon/Desktop/train_model_project/project_folder/project_config.ini')
# test.extract_features()











