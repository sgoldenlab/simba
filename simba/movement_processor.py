__author__ = "Simon Nilsson"

import pandas as pd
from simba.misc_tools import (get_fn_ext,
                              framewise_euclidean_distance)
from simba.feature_extractors.unit_tests import read_video_info
from simba.read_config_unit_tests import (read_config_entry,
                                          check_if_filepath_list_is_empty)
from simba.enums import ReadConfig
import os
from simba.mixins.config_reader import ConfigReader
from collections import defaultdict
from simba.rw_dfs import read_df
import numpy as np
from statistics import mean

class MovementProcessor(ConfigReader):
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
    >>> movement_processor.save_results()

    """

    def __init__(self,
                 config_path: str,
                 visualization: bool = False,
                 files: list=None):

        super().__init__(config_path=config_path)
        self.save_path = os.path.join(self.logs_path, 'Movement_log_{}.csv'.format(self.datetime))
        try:
            self.animal_cnt = read_config_entry(self.config, ReadConfig.PROCESS_MOVEMENT_SETTINGS.value, 'no_of_animals', 'int')
            self.p_threshold = read_config_entry(self.config, ReadConfig.PROCESS_MOVEMENT_SETTINGS.value, 'probability_threshold', 'float')
        except:
            print('Please analyze movements before visualizing data.')
            raise ValueError('Please analyze movements before visualizing data.')
        self.bp_dict = defaultdict(list)
        self.bp_columns = []
        if not visualization:
            for cnt, animal in enumerate(self.multi_animal_id_list):
                bp_name = read_config_entry(self.config, ReadConfig.PROCESS_MOVEMENT_SETTINGS.value, 'animal_{}_bp'.format(cnt+1), 'str')
                if bp_name == 'None':
                    print('SIMBA ERROR: No body-parts found in config [process movements][animal_N_bp]')
                    raise ValueError()
                for c in ['_x', '_y', '_p']:
                    self.bp_dict[animal].append(bp_name + c)
                    self.bp_columns.append(bp_name + c)
            check_if_filepath_list_is_empty(filepaths=self.outlier_corrected_paths,
                                            error_msg=f'SIMBA ERROR: Cannot process movement. ZERO data files found in the {self.outlier_corrected_dir} directory.')
            self.files_found = self.outlier_corrected_paths
        else:
            for cnt in range(self.animal_cnt):
                bp_name = read_config_entry(self.config, ReadConfig.PROCESS_MOVEMENT_SETTINGS.value, 'animal_{}_bp'.format(cnt+1), 'str')
                if bp_name == 'None':
                    print('SIMBA ERROR: No body-parts found in config [process movements][animal_N_bp]')
                    raise ValueError()
                for c in ['_x', '_y', '_p']:
                    self.bp_dict[self.multi_animal_id_list[cnt]].append(bp_name + c)
                    self.bp_columns.append(bp_name + c)
            self.files_found = files
        print(f'Processing {str(len(self.outlier_corrected_paths))} video(s)...')

    def process_movement(self):
        """
        Method to run movement aggregation computations.

        Returns
        ----------
        Attribute: dict
            results

        """
        self.results = {}
        self.movement_dict = {}
        for file_path in self.files_found:
            _, video_name, _ = get_fn_ext(file_path)
            print('Analysing {}...'.format(video_name))
            self.data_df = read_df(file_path, self.file_type)[self.bp_columns]
            self.video_info, self.px_per_mm, self.fps = read_video_info(vid_info_df=self.video_info_df, video_name=video_name)
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
                bp_time_1 = animal_df[[animal_bps[0], animal_bps[1]]].values.astype(float)
                bp_time_2 = animal_df[[animal_bps[0] + '_shifted', animal_bps[1] + '_shifted']].values.astype(float)
                self.movement = pd.Series(framewise_euclidean_distance(location_1=bp_time_1, location_2=bp_time_2, px_per_mm=self.px_per_mm))
                self.movement.loc[0] = 0
                self.movement_dict[video_name][animal_name] = self.movement
                self.results[video_name][animal_name]['Distance (cm)'] = round((self.movement.sum() / 10), 4)
                velocity_lst = []
                for df in np.array_split(self.movement, int(len(self.movement) / self.fps)):
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
        self.timer.stop_timer()
        print('SIMBA COMPLETE: Movement log saved in {} (elapsed time: {}s)'.format(self.save_path, self.timer.elapsed_time_str))

# test = MovementProcessor(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
# test.process_movement()
# test.save_results()
