import os, glob
from simba.features_scripts.unit_tests import read_video_info
from simba.read_config_unit_tests import insert_default_headers_for_feature_extraction
from simba.misc_tools import get_feature_extraction_headers
from simba.drop_bp_cords import get_fn_ext
from simba.rw_dfs import read_df, save_df
import pandas as pd
import numpy as np
from copy import deepcopy
import math
from scipy.spatial import ConvexHull
from collections import defaultdict
import scipy
from simba.features_scripts.feature_extraction_mixin import FeatureExtractionMixin

class ExtractFeaturesFrom7bps(FeatureExtractionMixin):
    
    """
    Class for creating a hard-coded set of features from single animals with 7 tracked body-parts
    using pose-estimation. Results are stored in the `project_folder/csv/features_extracted`
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
    >>> feature_extractor = ExtractFeaturesFrom7bps(config_path='MyProjectConfig')
    >>> feature_extractor.extract_features()

    """
     
    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path)
        self.in_headers = get_feature_extraction_headers(pose='1 animal 7 body-parts')
        self.mouse_p_headers = [x for x in self.in_headers if x[-2:] == '_p']
        self.mouse_headers = [x for x in self.in_headers if x[-2:] != '_p']
        print('Extracting features from {} file(s)...'.format(str(len(self.files_found))))

    def extract_features(self):
        """
        Method to compute and save features to disk. Results are saved in the `project_folder/csv/features_extracted`
        directory of the SimBA project.

        Returns
        -------
        None
        """
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            video_settings, self.px_per_mm, fps = read_video_info(self.vid_info_df, self.video_name)
            roll_windows = []
            for window in self.roll_windows_values:
                roll_windows.append(int(fps / window))
            self.in_data = read_df(file_path, self.file_type).fillna(0).apply(pd.to_numeric).reset_index(drop=True)
            self.in_data = insert_default_headers_for_feature_extraction(df=self.in_data, headers=self.in_headers, pose_config='7 body-parts', filename=file_path)
            self.out_data = deepcopy(self.in_data)

            self.out_data['Mouse_poly_area'] = self.out_data.apply(lambda x: ConvexHull(np.array([[x['Ear_left_x'], x["Ear_left_y"]],[x['Ear_right_x'], x["Ear_right_y"]],[x['Nose_x'], x["Nose_y"]],[x['Lat_left_x'], x["Lat_left_y"]],[x['Lat_right_x'], x["Lat_right_y"]], [x['Tail_base_x'], x["Tail_base_y"]],[x['Center_x'], x["Center_y"]]])).area, axis=1)
            self.out_data['Mouse_poly_area'] = self.out_data['Mouse_poly_area'] / self.px_per_mm
            self.in_data_shifted = self.out_data.shift(periods=1).add_suffix('_shifted').fillna(0)
            self.in_data = pd.concat([self.out_data, self.in_data_shifted], axis=1, join='inner').fillna(0).reset_index(drop=True)
            self.out_data['Mouse_nose_to_tail'] = self.euclidean_distance(self.in_data['Nose_x'].values, self.in_data['Tail_base_x'].values, self.in_data['Nose_y'].values, self.in_data['Tail_base_y'].values, self.px_per_mm)
            self.out_data['Mouse_width'] = self.euclidean_distance(self.in_data['Lat_left_x'].values, self.in_data['Lat_right_x'].values, self.in_data['Lat_left_y'].values, self.in_data['Lat_right_y'].values, self.px_per_mm)
            self.out_data['Mouse_Ear_distance'] = self.euclidean_distance(self.in_data['Ear_left_x'].values, self.in_data['Ear_right_x'].values, self.in_data['Ear_left_y'].values, self.in_data['Ear_right_y'].values, self.px_per_mm)
            self.out_data['Mouse_Nose_to_centroid'] = self.euclidean_distance(self.in_data['Nose_x'].values, self.in_data['Center_x'].values, self.in_data['Nose_y'].values, self.in_data['Center_y'].values, self.px_per_mm)
            self.out_data['Mouse_Nose_to_lateral_left'] = self.euclidean_distance(self.in_data['Nose_x'].values, self.in_data['Lat_left_x'].values, self.in_data['Nose_y'].values, self.in_data['Lat_left_y'].values, self.px_per_mm)
            self.out_data['Mouse_Nose_to_lateral_right'] = self.euclidean_distance(self.in_data['Nose_x'].values, self.in_data['Lat_right_x'].values, self.in_data['Nose_y'].values, self.in_data['Lat_right_y'].values, self.px_per_mm)
            self.out_data['Mouse_Centroid_to_lateral_left'] = self.euclidean_distance(self.in_data['Center_x'].values, self.in_data['Lat_left_x'].values, self.in_data['Center_y'].values, self.in_data['Lat_left_y'].values, self.px_per_mm)
            self.out_data['Mouse_Centroid_to_lateral_right'] = self.euclidean_distance(self.in_data['Center_x'].values, self.in_data['Lat_right_x'].values, self.in_data['Center_y'].values, self.in_data['Lat_right_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_centroid'] = self.euclidean_distance(self.in_data['Center_x_shifted'].values, self.in_data['Center_x'].values, self.in_data['Center_y_shifted'].values, self.in_data['Center_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_nose'] = self.euclidean_distance(self.in_data['Nose_x_shifted'].values, self.in_data['Nose_x'].values, self.in_data['Nose_y_shifted'].values, self.in_data['Nose_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_tail_base'] = self.euclidean_distance(self.in_data['Tail_base_x_shifted'].values, self.in_data['Tail_base_x'].values, self.in_data['Tail_base_y_shifted'].values, self.in_data['Tail_base_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_left_ear'] = self.euclidean_distance(self.in_data['Ear_left_x_shifted'].values, self.in_data['Ear_left_x'].values, self.in_data['Ear_left_y_shifted'].values, self.in_data['Ear_left_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_right_ear'] = self.euclidean_distance(self.in_data['Ear_right_x_shifted'].values, self.in_data['Ear_right_x'].values, self.in_data['Ear_right_y_shifted'].values, self.in_data['Ear_right_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_lateral_left'] = self.euclidean_distance(self.in_data['Lat_left_x_shifted'].values, self.in_data['Lat_left_x'].values, self.in_data['Lat_left_y_shifted'].values, self.in_data['Lat_left_y'].values, self.px_per_mm)
            self.out_data['Movement_mouse_lateral_right'] = self.euclidean_distance(self.in_data['Lat_right_x_shifted'].values, self.in_data['Lat_right_x'].values, self.in_data['Lat_right_y_shifted'].values, self.in_data['Lat_right_y'].values, self.px_per_mm)

            self.out_data['Mouse_polygon_size_change'] = (self.in_data['Mouse_poly_area_shifted'] - self.in_data['Mouse_poly_area'])

            print('Calculating hull variables...')
            mouse_array = self.in_data[self.mouse_headers].to_numpy()
            self.hull_dict = defaultdict(list)
            for cnt, animal_frm in enumerate(mouse_array):
                animal_frm = np.reshape(animal_frm, (-1, 2))
                animal_dists = scipy.spatial.distance.cdist(animal_frm, animal_frm, metric='euclidean')
                animal_dists = animal_dists[animal_dists != 0]
                self.hull_dict['M1_largest_euclidean_distance_hull'].append(np.amax(animal_dists, initial=0) / self.px_per_mm)
                self.hull_dict['M1_smallest_euclidean_distance_hull'].append(np.amin(animal_dists, initial=self.hull_dict['M1_largest_euclidean_distance_hull'][-1]) / self.px_per_mm)
                self.hull_dict['M1_mean_euclidean_distance_hull'].append(np.mean(animal_dists) / self.px_per_mm)
                self.hull_dict['M1_sum_euclidean_distance_hull'].append(np.sum(animal_dists) / self.px_per_mm)
            for k, v in self.hull_dict.items():
                self.out_data[k] = v

            self.out_data['Total_movement_all_bodyparts_M1'] = self.out_data[['Movement_mouse_centroid', 'Movement_mouse_nose', 'Movement_mouse_left_ear', 'Movement_mouse_right_ear', 'Movement_mouse_lateral_left', 'Movement_mouse_lateral_right']].sum(axis=1)

            print('Calculating rolling windows: medians, medians, and sums...')
            for window in self.roll_windows_values:
                col_name = 'Mouse1_width_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Mouse_width'].rolling(int(window), min_periods=1).median()
                col_name = 'Mouse1_width_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Mouse_width'].rolling(int(window), min_periods=1).mean()
                col_name = 'Mouse1_width_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Mouse_width'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Mouse1_mean_euclid_distances_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_mean_euclidean_distance_hull'].rolling(int(window), min_periods=1).median()
                col_name = 'Mouse1_mean_euclid_distances_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_mean_euclidean_distance_hull'].rolling(int(window), min_periods=1).mean()
                col_name = 'Mouse1_mean_euclid_distances_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_mean_euclidean_distance_hull'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Mouse1_smallest_euclid_distances_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_smallest_euclidean_distance_hull'].rolling(int(window), min_periods=1).median()
                col_name = 'Mouse1_smallest_euclid_distances_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_smallest_euclidean_distance_hull'].rolling(int(window), min_periods=1).mean()
                col_name = 'Mouse1_smallest_euclid_distances_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_smallest_euclidean_distance_hull'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Mouse1_largest_euclid_distances_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_largest_euclidean_distance_hull'].rolling(int(window), min_periods=1).median()
                col_name = 'Mouse1_largest_euclid_distances_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_largest_euclidean_distance_hull'].rolling(int(window), min_periods=1).mean()
                col_name = 'Mouse1_largest_euclid_distances_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['M1_largest_euclidean_distance_hull'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Tail_base_movement_M1_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_tail_base'].rolling(int(window), min_periods=1).median()
                col_name = 'Tail_base_movement_M1_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_tail_base'].rolling(int(window), min_periods=1).mean()
                col_name = 'Tail_base_movement_M1_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_tail_base'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Centroid_movement_M1_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_centroid'].rolling(int(window), min_periods=1).median()
                col_name = 'Centroid_movement_M1_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_centroid'].rolling(int(window), min_periods=1).mean()
                col_name = 'Centroid_movement_M1_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_centroid'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Nose_movement_M1_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_nose'].rolling(int(window), min_periods=1).median()
                col_name = 'Nose_movement_M1_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_nose'].rolling(int(window), min_periods=1).mean()
                col_name = 'Nose_movement_M1_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Movement_mouse_nose'].rolling(int(window), min_periods=1).sum()

            for window in self.roll_windows_values:
                col_name = 'Total_movement_M1_median_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Total_movement_all_bodyparts_M1'].rolling(int(window), min_periods=1).median()
                col_name = 'Total_movement_M1_mean_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Total_movement_all_bodyparts_M1'].rolling(int(window), min_periods=1).mean()
                col_name = 'Total_movement_M1_sum_{}'.format(str(window))
                self.out_data[col_name] = self.out_data['Total_movement_all_bodyparts_M1'].rolling(int(window), min_periods=1).sum()

            print('Calculating angles...')
            self.out_data['Mouse_angle'] = self.angle3pt_serialized(data=self.out_data[['Nose_x', 'Nose_y', 'Center_x', 'Center_y', 'Tail_base_x', 'Tail_base_y']].values)

            print('Calculating deviations...')
            self.out_data['Total_movement_all_bodyparts_deviation'] = (self.out_data['Total_movement_all_bodyparts_M1'].mean() - self.out_data['Total_movement_all_bodyparts_M1'])
            self.out_data['M1_smallest_euclid_distances_hull_deviation'] = (self.out_data['M1_smallest_euclidean_distance_hull'].mean() - self.out_data['M1_smallest_euclidean_distance_hull'])
            self.out_data['M1_largest_euclid_distances_hull_deviation'] = (self.out_data['M1_largest_euclidean_distance_hull'].mean() - self.out_data['M1_largest_euclidean_distance_hull'])
            self.out_data['M1_mean_euclid_distances_hull_deviation'] = (self.out_data['M1_mean_euclidean_distance_hull'].mean() - self.out_data['M1_mean_euclidean_distance_hull'])
            self.out_data['Total_angle_deviation'] = (self.out_data['Mouse_angle'].mean() - self.out_data['Mouse_angle'])
            self.out_data['Movement_mouse_deviation_centroid'] = (self.out_data['Movement_mouse_centroid'].mean() - self.out_data['Movement_mouse_centroid'])
            self.out_data['Mouse_polygon_deviation'] = (self.out_data['Mouse_poly_area'].mean() - self.out_data['Mouse_poly_area'])

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

            self.out_data['Movement_mouse_percentile_rank'] = self.out_data['Movement_mouse_centroid'].rank(pct=True)
            self.out_data['Movement_mouse_deviation_percentile_rank'] = self.out_data['Movement_mouse_deviation_centroid'].rank(pct=True)

            for window in self.roll_windows_values:
                col_name = 'Total_movement_M1_mean_{}'.format(str(window))
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

            print('Calculating path tortuosities...')
            as_strided = np.lib.stride_tricks.as_strided
            win_size = 3
            centroid_lst_mouse_x = as_strided(self.out_data['Center_x'], (len(self.out_data) - (win_size - 1), win_size), (self.out_data['Center_x'].values.strides * 2))
            centroid_lst_mouse_y = as_strided(self.out_data['Center_y'], (len(self.out_data) - (win_size - 1), win_size), (self.out_data['Center_y'].values.strides * 2))

            for window in self.roll_windows_values:
                start, end = 0, 0 + int(window)
                tortuosities_results = defaultdict(list)
                for frame in range(len(self.out_data)):
                    tortuosities_dict = defaultdict(list)
                    c_centroid_lst_mouse_x, c_centroid_lst_mouse_y = centroid_lst_mouse_x[start:end], centroid_lst_mouse_y[start:end]
                    for frame_in_window in range(len(c_centroid_lst_mouse_x)):
                        move_angle_mouse_ = (self.angle3pt(c_centroid_lst_mouse_x[frame_in_window][0], c_centroid_lst_mouse_y[frame_in_window][0],c_centroid_lst_mouse_x[frame_in_window][1],c_centroid_lst_mouse_y[frame_in_window][1],c_centroid_lst_mouse_x[frame_in_window][2],c_centroid_lst_mouse_y[frame_in_window][2]))
                        tortuosities_dict['Animal_1'].append(move_angle_mouse_)
                    tortuosities_results['Animal_1'].append(sum(tortuosities_dict['Animal_1']) / (2 * math.pi))
                    start += 1
                    end += 1
                col_name = 'Tortuosity_Mouse1_{}'.format(str(window))
                self.out_data[col_name] = tortuosities_results['Animal_1']

            print('Calculating pose probability scores...')
            self.out_data['Sum_probabilities'] = (self.out_data['Ear_left_p'] + self.out_data['Ear_right_p'] + self.out_data['Nose_p'] + self.out_data['Tail_base_p'])
            self.out_data['Sum_probabilities_deviation'] = (self.out_data['Sum_probabilities'].mean() - self.out_data['Sum_probabilities'])
            self.out_data['Sum_probabilities_deviation_percentile_rank'] = self.out_data['Sum_probabilities_deviation'].rank(pct=True)
            self.out_data['Sum_probabilities_percentile_rank'] = self.out_data['Sum_probabilities_deviation_percentile_rank'].rank(pct=True)
            results = pd.DataFrame(self.count_values_in_range(data=self.out_data.filter(self.mouse_p_headers).values, ranges=np.array([[0.0, 0.1], [0.000000000, 0.5], [0.000000000, 0.75]])), columns=['Low_prob_detections_0.1', 'Low_prob_detections_0.5','Low_prob_detections_0.75'])
            self.out_data = pd.concat([self.out_data, results], axis=1)

            self.out_data = self.out_data.reset_index(drop=True).fillna(0)
            save_path = os.path.join(self.save_dir, self.video_name + '.' + self.file_type)
            save_df(self.out_data, self.file_type, save_path)

            print('Feature extraction complete for {} ({}/{})...'.format(self.video_name, str(file_cnt + 1),str(len(self.files_found))))
        print('SIMBA COMPLETE: All features extracted. Results are stored in the project_folder/csv/features_extracted directory')

# test = ExtractFeaturesFrom7bps(config_path='/Users/simon/Desktop/train_model_project/project_folder/project_config.ini')
# test.extract_features()




