from simba.read_config_unit_tests import (insert_default_headers_for_feature_extraction)
import os
from simba.features_scripts.unit_tests import (read_video_info)
from simba.misc_tools import get_feature_extraction_headers
from simba.drop_bp_cords import get_fn_ext
from simba.rw_dfs import read_df, save_df
import pandas as pd
from copy import deepcopy
from scipy.spatial import ConvexHull
import scipy
import numpy as np
from collections import defaultdict
import math
from joblib import Parallel, delayed
import time
from simba.features_scripts.feature_extraction_mixin import FeatureExtractionMixin

class ExtractFeaturesFrom14bps(FeatureExtractionMixin):
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
        super().__init__(config_path=config_path)
        self.in_headers = get_feature_extraction_headers(pose='2 animals 14 body-parts')
        self.mouse_1_headers, self.mouse_2_headers = self.in_headers[0:21], self.in_headers[21:]
        self.mouse_2_p_headers = [x for x in self.mouse_2_headers if x[-2:] == '_p']
        self.mouse_1_p_headers = [x for x in self.mouse_1_headers if x[-2:] == '_p']
        self.mouse_1_headers = [x for x in self.mouse_1_headers if x[-2:] != '_p']
        self.mouse_2_headers = [x for x in self.mouse_2_headers if x[-2:] != '_p']
        print('Extracting features from {} file(s)...'.format(str(len(self.files_found))))

    def extract_features(self):
        """
        Method to compute and save features to disk. Results are saved in the `project_folder/csv/features_extracted`
        directory of the SimBA project.

        Returns
        -------
        None
        """
        session_time = 0
        for file_cnt, file_path in enumerate(self.files_found):
            roll_windows, file_start_time = [], time.time()
            _, self.video_name, _ = get_fn_ext(file_path)
            video_settings, self.px_per_mm, fps = read_video_info(self.vid_info_df, self.video_name)
            for window in self.roll_windows_values:
                roll_windows.append(int(fps / window))
            self.in_data = read_df(file_path, self.file_type).fillna(0).apply(pd.to_numeric).reset_index(drop=True)
            self.in_data = insert_default_headers_for_feature_extraction(df=self.in_data, headers=self.in_headers, pose_config='14 body-parts', filename=file_path)
            self.out_data = deepcopy(self.in_data)
            mouse_1_ar = np.reshape(self.out_data[self.mouse_1_headers].values, (len(self.out_data / 2), -1, 2))
            self.out_data['Mouse_1_poly_area'] = Parallel(n_jobs=-1, verbose=0, backend="threading")(delayed(self.convex_hull_calculator_mp)(x, self.px_per_mm) for x in mouse_1_ar)
            mouse_2_ar = np.reshape(self.out_data[self.mouse_2_headers].values, (len(self.out_data / 2), -1, 2))
            self.out_data['Mouse_2_poly_area'] = Parallel(n_jobs=-1, verbose=0, backend="threading")(delayed(self.convex_hull_calculator_mp)(x, self.px_per_mm) for x in mouse_2_ar)
            self.in_data_shifted = self.out_data.shift(periods=1).add_suffix('_shifted').fillna(0)
            self.in_data = pd.concat([self.in_data, self.in_data_shifted], axis=1, join='inner').fillna(0).reset_index(drop=True)
            self.out_data['Mouse_1_nose_to_tail'] = self.euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Tail_base_1_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Tail_base_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_nose_to_tail'] = self.euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Tail_base_2_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Tail_base_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_width'] = self.euclidean_distance(self.out_data['Lat_left_1_x'].values, self.out_data['Lat_right_1_x'].values, self.out_data['Lat_left_1_y'].values, self.out_data['Lat_right_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_width'] = self.euclidean_distance(self.out_data['Lat_left_2_x'].values, self.out_data['Lat_right_2_x'].values, self.out_data['Lat_left_2_y'].values, self.out_data['Lat_right_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_Ear_distance'] = self.euclidean_distance(self.out_data['Ear_left_1_x'].values, self.out_data['Ear_right_1_x'].values, self.out_data['Ear_left_1_y'].values, self.out_data['Ear_right_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_Ear_distance'] = self.euclidean_distance(self.out_data['Ear_left_2_x'].values, self.out_data['Ear_right_2_x'].values, self.out_data['Ear_left_2_y'].values, self.out_data['Ear_right_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_Nose_to_centroid'] = self.euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Center_1_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Center_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_Nose_to_centroid'] = self.euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Center_2_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Center_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_Nose_to_lateral_left'] = self.euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Lat_left_1_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Lat_left_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_Nose_to_lateral_left'] = self.euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Lat_left_2_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Lat_left_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_Nose_to_lateral_right'] = self.euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Lat_right_1_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Lat_right_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_Nose_to_lateral_right'] = self.euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Lat_right_2_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Lat_right_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_Centroid_to_lateral_left'] = self.euclidean_distance(self.out_data['Center_1_x'].values, self.out_data['Lat_left_1_x'].values, self.out_data['Center_1_y'].values, self.out_data['Lat_left_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_Centroid_to_lateral_left'] = self.euclidean_distance(self.out_data['Center_2_x'].values, self.out_data['Lat_left_2_x'].values, self.out_data['Center_2_y'].values, self.out_data['Lat_left_2_y'].values, self.px_per_mm)
            self.out_data['Mouse_1_Centroid_to_lateral_right'] = self.euclidean_distance(self.out_data['Center_1_x'].values, self.out_data['Lat_right_1_x'].values, self.out_data['Center_1_y'].values, self.out_data['Lat_right_1_y'].values, self.px_per_mm)
            self.out_data['Mouse_2_Centroid_to_lateral_right'] = self.euclidean_distance(self.out_data['Center_2_x'].values, self.out_data['Lat_right_2_x'].values, self.out_data['Center_2_y'].values, self.out_data['Lat_right_2_y'].values, self.px_per_mm)
            self.out_data['Centroid_distance'] = self.euclidean_distance(self.out_data['Center_2_x'].values, self.out_data['Center_1_x'].values, self.out_data['Center_2_y'].values, self.out_data['Center_1_y'].values, self.px_per_mm)
            self.out_data['Nose_to_nose_distance'] = self.euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Nose_1_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Nose_1_y'].values, self.px_per_mm)
            self.out_data['M1_Nose_to_M2_lat_left'] = self.euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Lat_left_2_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Lat_left_2_y'].values, self.px_per_mm)
            self.out_data['M1_Nose_to_M2_lat_left'] = self.euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Lat_left_2_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Lat_left_2_y'].values, self.px_per_mm)
            self.out_data['M1_Nose_to_M2_lat_right'] = self.euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Lat_right_2_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Lat_right_2_y'].values, self.px_per_mm)
            self.out_data['M2_Nose_to_M1_lat_left'] = self.euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Lat_left_1_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Lat_left_1_y'].values, self.px_per_mm)
            self.out_data['M2_Nose_to_M1_lat_right'] = self.euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Lat_right_1_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Lat_right_1_y'].values, self.px_per_mm)
            self.out_data['M1_Nose_to_M2_tail_base'] = self.euclidean_distance(self.out_data['Nose_1_x'].values, self.out_data['Tail_base_2_x'].values, self.out_data['Nose_1_y'].values, self.out_data['Tail_base_2_y'].values, self.px_per_mm)
            self.out_data['M2_Nose_to_M1_tail_base'] = self.euclidean_distance(self.out_data['Nose_2_x'].values, self.out_data['Tail_base_1_x'].values, self.out_data['Nose_2_y'].values, self.out_data['Tail_base_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_centroid'] = self.euclidean_distance(self.in_data['Center_1_x_shifted'].values, self.in_data['Center_1_x'].values, self.in_data['Center_1_y_shifted'].values, self.in_data['Center_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_centroid'] = self.euclidean_distance(self.in_data['Center_2_x_shifted'].values, self.in_data['Center_2_x'].values, self.in_data['Center_2_y_shifted'].values, self.in_data['Center_2_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_nose'] = self.euclidean_distance(self.in_data['Nose_1_x_shifted'].values, self.in_data['Nose_1_x'].values, self.in_data['Nose_1_y_shifted'].values, self.in_data['Nose_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_nose'] = self.euclidean_distance(self.in_data['Nose_2_x_shifted'].values, self.in_data['Nose_2_x'].values, self.in_data['Nose_2_y_shifted'].values, self.in_data['Nose_2_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_tail_base'] = self.euclidean_distance(self.in_data['Tail_base_1_x_shifted'].values, self.in_data['Tail_base_1_x'].values, self.in_data['Tail_base_1_y_shifted'].values, self.in_data['Tail_base_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_tail_base'] = self.euclidean_distance(self.in_data['Tail_base_2_x_shifted'].values, self.in_data['Tail_base_2_x'].values, self.in_data['Tail_base_2_y_shifted'].values, self.in_data['Tail_base_2_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_left_ear'] = self.euclidean_distance(self.in_data['Ear_left_1_x_shifted'].values, self.in_data['Ear_left_1_x'].values, self.in_data['Ear_left_1_y_shifted'].values, self.in_data['Ear_left_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_left_ear'] = self.euclidean_distance(self.in_data['Ear_left_2_x_shifted'].values, self.in_data['Ear_left_2_x'].values, self.in_data['Ear_left_2_y_shifted'].values, self.in_data['Ear_left_2_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_right_ear'] = self.euclidean_distance(self.in_data['Ear_right_1_x_shifted'].values, self.in_data['Ear_right_1_x'].values, self.in_data['Ear_right_1_y_shifted'].values, self.in_data['Ear_right_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_right_ear'] = self.euclidean_distance(self.in_data['Ear_right_2_x_shifted'].values, self.in_data['Ear_right_2_x'].values, self.in_data['Ear_right_2_y_shifted'].values, self.in_data['Ear_right_2_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_lateral_left'] = self.euclidean_distance(self.in_data['Lat_left_1_x_shifted'].values, self.in_data['Lat_left_1_x'].values, self.in_data['Lat_left_1_y_shifted'].values, self.in_data['Lat_left_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_lateral_left'] = self.euclidean_distance(self.in_data['Lat_left_2_x_shifted'].values, self.in_data['Lat_left_2_x'].values, self.in_data['Lat_left_2_y_shifted'].values, self.in_data['Lat_left_2_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_1_lateral_right'] = self.euclidean_distance(self.in_data['Lat_right_1_x_shifted'].values, self.in_data['Lat_right_1_x'].values, self.in_data['Lat_right_1_y_shifted'].values, self.in_data['Lat_right_1_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_2_lateral_right'] = self.euclidean_distance(self.in_data['Lat_right_2_x_shifted'].values, self.in_data['Lat_right_2_x'].values, self.in_data['Lat_right_2_y_shifted'].values, self.in_data['Lat_right_2_y'].values, self.px_per_mm)
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
                    self.hull_dict['{}_hull_large_euclidean'.format(animal_name)].append(np.amax(animal, initial=0) / self.px_per_mm)
                    self.hull_dict['{}_hull_small_euclidean'.format(animal_name)].append(np.min(animal, initial=self.hull_dict['{}_hull_large_euclidean'.format(animal_name)][-1]) / self.px_per_mm)
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
            self.out_data['Mouse_1_angle'] = self.angle3pt_serialized(data=self.out_data[['Nose_1_x', 'Nose_1_y', 'Center_1_x', 'Center_1_y', 'Tail_base_1_x', 'Tail_base_1_y']].values)
            self.out_data['Mouse_2_angle'] = self.angle3pt_serialized(data=self.out_data[['Nose_2_x', 'Nose_2_y', 'Center_2_x', 'Center_2_y', 'Tail_base_2_x', 'Tail_base_2_y']].values)
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
                        move_angle_mouse_1 = (self.angle3pt(c_centroid_lst_mouse1_x[frame_in_window][0], c_centroid_lst_mouse1_y[frame_in_window][0],c_centroid_lst_mouse1_x[frame_in_window][1], c_centroid_lst_mouse1_y[frame_in_window][1], c_centroid_lst_mouse1_x[frame_in_window][2], c_centroid_lst_mouse1_y[frame_in_window][2]))
                        move_angle_mouse_2 = (self.angle3pt(c_centroid_lst_mouse2_x[frame_in_window][0], c_centroid_lst_mouse2_y[frame_in_window][0], c_centroid_lst_mouse2_x[frame_in_window][1], c_centroid_lst_mouse2_y[frame_in_window][1],c_centroid_lst_mouse2_x[frame_in_window][2], c_centroid_lst_mouse2_y[frame_in_window][2]))
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
            results = pd.DataFrame(self.count_values_in_range(data=self.out_data.filter(all_p_columns).values, ranges=np.array([[0.0, 0.1], [0.000000000, 0.5], [0.000000000, 0.75]])), columns=['Low_prob_detections_0.1', 'Low_prob_detections_0.5', 'Low_prob_detections_0.75'])
            self.out_data = pd.concat([self.out_data, results], axis=1)
            self.out_data = self.out_data.reset_index(drop=True).fillna(0)
            save_path = os.path.join(self.save_dir, self.video_name + '.' + self.file_type)
            save_df(self.out_data, self.file_type, save_path)
            session_time, file_time = session_time + (time.time() - file_start_time), int(time.time() - file_start_time)
            print('Feature extraction complete for {} ({}/{} (elapsed time: {}s)...'.format(self.video_name,
                                                                                             str(file_cnt + 1),
                                                                                             str(len(self.files_found)),
                                                                                             str(file_time)))

        print('SIMBA COMPLETE: All features extracted (elapsed time: {}s). Results stored in project_folder/csv/features_extracted directory'.format(str(int(session_time))))

# test = ExtractFeaturesFrom14bps(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.extract_features()










