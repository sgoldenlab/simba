__author__ = "Simon Nilsson", "JJ Choong"

import pandas as pd
from simba.misc_tools import check_multi_animal_status, get_fn_ext
from datetime import datetime
from simba.features_scripts.unit_tests import read_video_info_csv, read_video_info
from simba.read_config_unit_tests import read_config_entry, check_that_column_exist, read_config_file
import os, glob
from collections import defaultdict
from simba.rw_dfs import read_df
import numpy as np
from statistics import mean

class MovementProcessor(object):
    """
    Class for computing aggregate movement statistics.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    ----------

    Examples
    ----------
    >>> movement_processor = MovementProcessor(config_path='MyConfigPath')
    >>> movement_processor.process_movement()

    """

    def __init__(self,
                 config_path: str):

        self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.logs_path = os.path.join(self.project_path, 'logs')
        self.save_path = os.path.join(self.logs_path, 'Movement_log_{}.csv'.format(self.timestamp))
        self.in_dir = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.vid_info_df = read_video_info_csv(os.path.join(self.logs_path, 'video_info.csv'))
        self.file_type = read_config_entry(self.config, 'General settings', 'workflow_file_type', 'str', 'csv')
        self.animal_cnt = read_config_entry(self.config, 'process movements', 'no_of_animals', 'int')
        self.p_threshold = read_config_entry(self.config, 'process movements', 'probability_threshold', 'float')
        self.multiAnimalStatus, self.multiAnimalIDList = check_multi_animal_status(self.config, self.animal_cnt)
        self.bp_dict = defaultdict(list)
        self.bp_columns = []
        for cnt, animal in enumerate(self.multiAnimalIDList):
            bp_name = read_config_entry(self.config, 'process movements', 'animal_{}_bp'.format(cnt+1), 'str')
            if bp_name == 'None':
                print('SIMBA ERROR: No body-parts found in config [process movements][animal_N_bp]')
                raise ValueError
            for c in ['_x', '_y', '_p']:
                self.bp_dict[animal].append(bp_name + c)
                self.bp_columns.append(bp_name + c)
        self.files_found = glob.glob(self.in_dir + '/*.' + self.file_type)
        print('Processing {} video(s)...'.format(str(len(self.files_found))))

    def __euclidean_distance(self, bp_1_x_vals, bp_2_x_vals, bp_1_y_vals, bp_2_y_vals, px_per_mm):
        series = (np.sqrt((bp_1_x_vals - bp_2_x_vals) ** 2 + (bp_1_y_vals - bp_2_y_vals) ** 2)) / px_per_mm
        return series

    def process_movement(self):
        """
        Method to run movement aggregation computations.

        Returns
        ----------
        Attribute: dict
            self.results

        """
        self.results = {}
        self.movement_dict = {}
        for file_path in self.files_found:
            _, video_name, _ = get_fn_ext(file_path)
            print('Analysing {}...'.format(video_name))
            self.data_df = read_df(file_path, self.file_type)[self.bp_columns]
            self.video_info, self.px_per_mm, self.fps = read_video_info(vid_info_df=self.vid_info_df, video_name=video_name)
            self.results[video_name] = {}
            self.movement_dict[video_name] = {}
            for animal_name, animal_bps in self.bp_dict.items():
                self.results[video_name][animal_name] = {}
                animal_df = self.data_df[animal_bps]
                if self.p_threshold > 0.00:
                    animal_df = animal_df[animal_df[animal_bps[2]] >= self.p_threshold]
                animal_df = animal_df.iloc[:, 0:2].reset_index(drop=True)
                df_shifted = animal_df.shift(1)
                df_shifted = df_shifted.combine_first(animal_df).add_suffix('_shifted')
                animal_df = pd.concat([animal_df, df_shifted], axis=1)
                self.movement = self.__euclidean_distance(animal_df[animal_bps[0]], animal_df[animal_bps[0] + '_shifted'], animal_df[animal_bps[1]], animal_df[animal_bps[1] + '_shifted'], self.px_per_mm)
                self.movement_dict[video_name][animal_name] = self.movement
                self.results[video_name][animal_name]['Distance (cm)'] = round((self.movement.sum() / 10), 4)
                velocity_lst = []
                for df in np.array_split(self.movement, self.fps):
                    velocity_lst.append(df.sum())
                self.results[video_name][animal_name]['Velocity (cm/s)'] = round((mean(velocity_lst) / 10), 4)

    def save_results(self):
        """
        Method to save movement aggregation data into the `project_folder/logs` directory
        of the SimBA project.

        Returns
        ----------
        None

        """

        self.out_df = pd.DataFrame(columns=['Video', 'Animal', 'Measurement', 'Value'])
        for video, video_data in self.results.items():
            for animal, animal_data in video_data.items():
                for measure, mesure_val in animal_data.items():
                    self.out_df.loc[len(self.out_df)] = [video, animal, measure, mesure_val]
        self.out_df.set_index('Video').to_csv(self.save_path)
        print('SIMBA COMPLETE: Movement log saved in {}'.format(self.save_path))

#
# test = MovementProcessor(config_path='/Users/simon/Desktop/troubleshooting/train_model_project/project_folder/project_config.ini')
# test.process_movement()
