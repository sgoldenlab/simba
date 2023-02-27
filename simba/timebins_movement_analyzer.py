__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
from simba.read_config_unit_tests import (read_config_entry,
                                          check_that_column_exist,
                                          read_config_file,
                                          check_if_filepath_list_is_empty,
                                          read_project_path_and_file_type)
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from datetime import datetime
from simba.misc_tools import (check_multi_animal_status,
                              SimbaTimer,
                              framewise_euclidean_distance)
from simba.enums import (ReadConfig,
                         Paths,
                         Dtypes)
from simba.drop_bp_cords import create_body_part_dictionary, getBpNames
import os, glob
from simba.rw_dfs import read_df
from simba.drop_bp_cords import get_fn_ext
from numba import jit
import numpy as np
import itertools


class TimeBinsMovementAnalyzer(object):
    """
    Class for calculating and aggregating movement statistics into time-bins.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    bin_length: int
        Integer representing the time bin size in seconds

    Example
    ----------

    >>> timebin_movement_analyzer = TimeBinsMovementAnalyzer(config_path='MyConfigPath', bin_length=15)
    >>> timebin_movement_analyzer.analyze_movement()

    """

    def __init__(self,
                 config_path: str,
                 bin_length: int):

        self.timer = SimbaTimer()
        self.timer.start_timer()
        self.bin_length = bin_length
        self.config = read_config_file(config_path)
        self.datetime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.data_in_dir = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.vid_info_df = read_video_info_csv(os.path.join(self.project_path, Paths.VIDEO_INFO.value))
        self.animal_cnt = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.ANIMAL_CNT.value, Dtypes.INT.value)
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.animal_cnt)
        self.col_headers = []
        for animal_no in range(self.animal_cnt):
            animal_bp = read_config_entry(self.config, ReadConfig.PROCESS_MOVEMENT_SETTINGS.value, 'animal_{}_bp'.format(str(animal_no + 1)), Dtypes.STR.value)
            self.col_headers.extend((animal_bp + '_x', animal_bp + '_y'))
        self.x_cols, self.y_cols, self.p_cols = getBpNames(config_path)
        self.x_cols = [x for x in self.x_cols if x in self.col_headers]
        self.y_cols = [x for x in self.y_cols if x in self.col_headers]
        self.files_found = glob.glob(self.data_in_dir + '/*.' + self.file_type)
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg='SIMBA ERROR: Cannot analyze movement in time-bins, data directory {} is empty.'.format(self.data_in_dir))
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.animal_cnt, self.x_cols, self.y_cols, [], [])
        self.animal_combinations = list(itertools.combinations(self.animal_bp_dict, 2))
        print('Processing {} video(s)...'.format(str(len(self.files_found))))

    def analyze_movement(self):
        """
        Method for running the movement time-bin analysis. Results are stored in the ``project_folder/logs`` directory
        of the SimBA project.

        Returns
        ----------
        None
        """
        video_dict = {}
        self.out_df_lst = []
        for file_cnt, file_path in enumerate(self.files_found):
            result_df = pd.DataFrame()
            _, video_name, _ = get_fn_ext(file_path)
            video_dict[video_name] = {}
            video_settings, px_per_mm, fps = read_video_info(vid_info_df=self.vid_info_df, video_name=video_name)
            fps = int(fps)
            bin_length_frames = int(fps * self.bin_length)
            data_df = read_df(file_path, self.file_type)
            data_df_sliced = pd.DataFrame()
            for animal, data in self.animal_bp_dict.items():
                check_that_column_exist(df=data_df, column_name=data['X_bps'][0], file_name=file_path)
                check_that_column_exist(df=data_df, column_name=data['Y_bps'][0], file_name=file_path)
                data_df_sliced[data['X_bps'][0]] = data_df[data['X_bps'][0]]
                data_df_sliced[data['Y_bps'][0]] = data_df[data['Y_bps'][0]]
            data_df_shifted = data_df.shift(periods=1).add_suffix('_shifted').fillna(0)
            data_df = pd.concat([data_df_sliced, data_df_shifted], axis=1, join='inner').fillna(0).reset_index(drop=True)
            for animal, data in self.animal_bp_dict.items():
                movement_col_name = 'Movement {}'.format(animal)
                x_col, y_col = data['X_bps'][0], data['Y_bps'][0]
                bp_time_1 = data_df[[x_col, y_col]].values
                bp_time_2 = data_df[[x_col + '_shifted', y_col + '_shifted']].values
                result_df[movement_col_name] = pd.Series(framewise_euclidean_distance(location_1=bp_time_1, location_2=bp_time_2, px_per_mm=px_per_mm, centimeter=True))
                result_df.loc[0, movement_col_name] = 0
            for animal_c in self.animal_combinations:
                distance_col_name = 'Distance {} {}'.format(animal_c[0], animal_c[1])
                bp_1_x, bp_1_y = self.animal_bp_dict[animal_c[0]]['X_bps'][0], self.animal_bp_dict[animal_c[0]]['Y_bps'][0]
                bp_2_x, bp_2_y = self.animal_bp_dict[animal_c[0]]['X_bps'][0], self.animal_bp_dict[animal_c[1]]['Y_bps'][0]
                bp_1 = data_df[[bp_1_x, bp_1_y]].values.astype(float)
                bp_2 = data_df[[bp_2_x, bp_2_y]].values.astype(float)
                result_df[distance_col_name] = pd.Series(framewise_euclidean_distance(location_1=bp_1, location_2=bp_2, px_per_mm=px_per_mm))
            results_df_lists = [result_df[i:i + bin_length_frames] for i in range(0, result_df.shape[0], bin_length_frames)]
            indexed_df_lst = []
            for bin, results in enumerate(results_df_lists):
                timbe_bin_per_s = [results[i:i + fps] for i in range(0, results.shape[0], fps)]
                for second, df in enumerate(timbe_bin_per_s):
                    df['Time bin #'], df['Second'] = bin, second
                    indexed_df_lst.append(df)
            indexed_df = pd.concat(indexed_df_lst, axis=0)
            movement_cols = [x for x in indexed_df.columns if x.startswith('Movement ')]
            distance_cols = [x for x in indexed_df.columns if x.startswith('Distance ')]
            for movement_col in movement_cols:
                movement_sum = indexed_df.groupby(['Time bin #'])[movement_col].sum().reset_index()
                movement_velocity = indexed_df.groupby(['Time bin #', 'Second'])[movement_col].sum().reset_index()
                movement_velocity = movement_velocity.groupby(['Time bin #'])[movement_col].mean().reset_index()
                video_dict[video_name][movement_col + ' (cm)'] = movement_sum
                video_dict[video_name][movement_col+ ' velocity (cm/s)'] = movement_velocity
            for distance_col in distance_cols:
                video_dict[video_name][distance_col + ' (cm)'] = indexed_df.groupby(['Time bin #'])[distance_col].mean().reset_index()

        for video_name, video_info in video_dict.items():
            for measurement, bin_data in video_info.items():
                data_df = pd.DataFrame.from_dict(bin_data).reset_index(drop=True)
                data_df.columns = ['Time bin #', measurement]
                data_df = pd.melt(data_df, id_vars=['Time bin #']).drop('variable', axis=1).rename(columns={'value': 'Value'})
                data_df.insert(loc=0, column='Measurement', value=measurement)
                data_df.insert(loc=0, column='Video', value=video_name)
                self.out_df_lst.append(data_df)
        video_df = pd.concat(self.out_df_lst, axis= 0).sort_values(by=['Video', 'Time bin #']).set_index('Video')
        save_path = os.path.join(self.project_path, 'logs', 'Time_bins_movement_results_' + self.datetime + '.csv')
        video_df.to_csv(save_path)
        self.timer.stop_timer()
        print('SIMBA COMPLETE: Movement time-bins results saved at {} (elapsed time: {}s)'.format(save_path, self.timer.elapsed_time_str))

# test = TimeBinsMovementAnalyzer(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini', bin_length=20)
# test.analyze_movement()