from simba.misc_tools import check_multi_animal_status, get_fn_ext
from simba.misc_tools import read_config_file
from simba.read_config_unit_tests import (read_config_entry, check_int, check_str)
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info, check_minimum_roll_windows
from simba.rw_dfs import read_df, save_df
from simba.drop_bp_cords import getBpHeaders, create_body_part_dictionary, getBpNames
import os, glob
import pandas as pd
import numpy as np
from itertools import product
from numba import jit, prange

class UserDefinedFeatureExtractor(object):
    """
    Class for featurizing data within SimBA project using user-defined body-parts in the pose-estimation data.
    Results are stored in the `project_folder/csv/features_extracted` directory of the SimBA project.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    ----------
    Feature extraction tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features>`__.

    Examples
    ----------
    >>> feature_extractor = UserDefinedFeatureExtractor(config_path='MyProjectConfig')
    >>> feature_extractor.extract_features()

    """

    def __init__(self,
                 config_path: str):

        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='str')
        self.in_folder = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.out_folder = os.path.join(self.project_path, 'csv', 'features_extracted')
        if not os.path.exists(self.out_folder): os.makedirs(self.out_folder)
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, 'logs', 'video_info.csv'))
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.xcols, self.ycols, self.pcols = getBpNames(config_path)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.no_animals, self.xcols, self.ycols, self.pcols, [])
        self.col_headers = getBpHeaders(config_path)
        self.col_headers_shifted = [bp + '_shifted' for bp in self.col_headers]
        self.roll_windows_values = check_minimum_roll_windows([2, 5, 6, 7.5, 15], self.vid_info_df ['fps'].min())
        self.files_found = glob.glob(self.in_folder + '/*.' + self.file_type)
        print('Extracting features from {} file(s)...'.format(str(len(self.files_found))))

    @staticmethod
    @jit(nopython=True)
    def __count_values_in_range(data: np.array, ranges: np.array):
        results = np.full((data.shape[0], ranges.shape[0]), 0)
        for i in prange(data.shape[0]):
            for j in prange(ranges.shape[0]):
                lower_bound, upper_bound = ranges[j][0], ranges[j][1]
                results[i][j] = data[i][np.logical_and(data[i] >= lower_bound, data[i] <= upper_bound)].shape[0]
        return results

    def __euclid_dist_between_bps_of_other_animals(self):
        print('Calculating euclidean distances...')
        self.distance_col_names = []
        for animal_name, animal_data in self.animal_bp_dict.items():
            current_animal_bp_xs, current_animal_bp_ys = animal_data['X_bps'], animal_data['Y_bps']
            other_animals = {i: self.animal_bp_dict[i] for i in self.animal_bp_dict if i != animal_name}
            for current_animal_bp_x, current_animal_bp_y in zip(current_animal_bp_xs, current_animal_bp_ys):
                for other_animal_name, other_animal_data in other_animals.items():
                    other_animal_bp_xs, other_animal_bp_ys = other_animal_data['X_bps'], other_animal_data['Y_bps']
                    for other_animal_bp_x, other_animal_bp_y in zip(other_animal_bp_xs, other_animal_bp_ys):
                        current_bp_name, other_bp_name = current_animal_bp_x.strip('_x'), other_animal_bp_x.strip('_x')
                        col_name = 'Euclidean_distance_{}_{}'.format(current_bp_name, other_bp_name)
                        reverse_col_name = 'Euclidean_distance_{}_{}'.format(other_bp_name, current_bp_name)
                        if not reverse_col_name in self.data_df.columns:
                            self.data_df[col_name] = (np.sqrt((self.data_df[current_animal_bp_x] - self.data_df[other_animal_bp_x]) ** 2 + (self.data_df[current_animal_bp_y] - self.data_df[other_animal_bp_y]) ** 2)) / self.px_per_mm
                            self.distance_col_names.append(col_name)

    def __movement_of_all_bps(self):
        print('Calculating movements of all body-parts...')
        self.mean_movement_cols, self.sum_movement_cols = [], []
        for animal_name, animal_data in self.animal_bp_dict.items():
            animal_cols = []
            current_animal_bp_xs, current_animal_bp_ys = animal_data['X_bps'], animal_data['Y_bps']
            for bp_x, bp_y in zip(current_animal_bp_xs, current_animal_bp_ys):
                shifted_bp_x, shifted_bp_y = bp_x + '_shifted', bp_y + '_shifted'
                col_name = 'Movement_' + bp_x.strip('_x')
                self.data_df[col_name] = (np.sqrt((self.data_df_comb[bp_x] - self.data_df_comb[shifted_bp_x]) ** 2 + (self.data_df_comb[bp_y] - self.data_df_comb[shifted_bp_y]) ** 2)) / self.px_per_mm
                animal_cols.append(col_name)
            self.data_df['All_bp_movements_' + animal_name + '_sum'] = self.data_df[animal_cols].sum(axis=1)
            self.data_df['All_bp_movements_' + animal_name + '_mean'] = self.data_df[animal_cols].mean(axis=1)
            self.data_df['All_bp_movements_' + animal_name + '_min'] = self.data_df[animal_cols].min(axis=1)
            self.data_df['All_bp_movements_' + animal_name + '_max'] = self.data_df[animal_cols].max(axis=1)
            self.mean_movement_cols.append('All_bp_movements_' + animal_name + '_mean')
            self.sum_movement_cols.append('All_bp_movements_' + animal_name + '_sum')

    def __rolling_windows_bp_distances(self):
        print('Calculating rolling windows data: distances between body-parts...')
        for i in product(self.roll_windows_values, self.distance_col_names):
            self.data_df['Mean_{}_{}'.format(i[1], i[0])] = self.data_df[i[1]].rolling(int(i[0]), min_periods=1).mean()
            self.data_df['Sum_{}_{}'.format(i[1], i[0])] = self.data_df[i[1]].rolling(int(i[0]), min_periods=1).sum()

    def __rolling_windows_movement(self):
        print('Calculating rolling windows data: animal movements...')
        for i in product(self.roll_windows_values, self.mean_movement_cols):
            self.data_df['Mean_{}_{}'.format(i[1], i[0])] = self.data_df[i[1]].rolling(int(i[0]), min_periods=1).mean()
            self.data_df['Sum_{}_{}'.format(i[1], i[0])] = self.data_df[i[1]].rolling(int(i[0]), min_periods=1).sum()

    def __pose_probability_filters(self):
        p_df = self.data_df.filter(self.pcols, axis=1)
        self.data_df['Sum_probabilities'] = p_df.sum(axis=1)
        self.data_df['Mean_probabilities'] = p_df.mean(axis=1)

        results = pd.DataFrame(self.__count_values_in_range(data=self.data_df.filter(self.pcols).values, ranges=np.array([[0.0, 0.1], [0.000000000, 0.5], [0.000000000, 0.75]])), columns=['Low_prob_detections_0.1', 'Low_prob_detections_0.5','Low_prob_detections_0.75'])
        self.out_data = pd.concat([self.data_df, results], axis=1)

    def extract_features(self):
        """
        Method to compute and save features to disk. Results are saved in the `project_folder/csv/features_extracted`
        directory of the SimBA project.

        Returns
        -------
        None
        """

        for file_cnt, file_path in enumerate(self.files_found):
            print('Extracting features for video {} / {}'.format(str(file_cnt+1), str(len(self.files_found))))
            _, file_name, _ = get_fn_ext(file_path)
            check_str('file name', file_name)
            video_settings, self.px_per_mm, fps = read_video_info(self.vid_info_df, file_name)
            roll_windows = []
            for i in range(len(self.roll_windows_values)):
                roll_windows.append(int(fps / self.roll_windows_values[i]))
            self.data_df = read_df(file_path, self.file_type)
            self.data_df.columns = self.col_headers
            self.data_df = self.data_df.fillna(0).apply(pd.to_numeric)
            self.data_df_shifted = self.data_df.shift(periods=1)
            self.data_df_shifted.columns = self.col_headers_shifted
            self.data_df_comb = pd.concat([self.data_df, self.data_df_shifted], axis=1, join='inner').fillna(0).reset_index(drop=True)
            self.__euclid_dist_between_bps_of_other_animals()
            self.__movement_of_all_bps()
            self.__rolling_windows_bp_distances()
            self.__rolling_windows_movement()
            self.__pose_probability_filters()
            save_path = os.path.join(self.out_folder, file_name + '.' + self.file_type)
            self.data_df = self.data_df.reset_index(drop=True).fillna(0)
            save_df(self.data_df, self.file_type, save_path)
            print('Saving features for video {}...'.format(file_name))
            print('Feature extraction complete for video {}'.format(file_name))

        print('SIMBA COMPLETE: Feature extraction complete for {} video(s). Results are saved inside the project_folder/csv/features_extracted director.'.format(len(self.files_found)))

# test = UserDefinedFeatureExtractor(config_path='/Users/simon/Desktop/train_model_project/project_folder/project_config.ini')
# test.extract_features()