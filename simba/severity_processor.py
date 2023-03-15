__author__ = "Simon Nilsson", "JJ Choong"

import numpy as np
import glob, os

import pandas as pd

from simba.rw_dfs import read_df
from simba.features_scripts.unit_tests import read_video_info, read_video_info_csv
from simba.read_config_unit_tests import (read_config_file,
                                          read_config_entry,
                                          read_project_path_and_file_type,
                                          check_if_filepath_list_is_empty)
from datetime import datetime
from numba import jit
from simba.drop_bp_cords import getBpNames, create_body_part_dictionary
from simba.enums import ReadConfig, Dtypes, Paths
from simba.misc_tools import (check_multi_animal_status,
                              get_fn_ext,
                              SimbaTimer)


class SeverityProcessor(object):
    """
    Class for analyzing the `severity` of classification frame events based on how much
    the animals are moving. Frames are scored as less or more severe at lower and higher movements.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    settings: dict
        how to calculate the severity. E.g., {'brackets': 10, 'clf': 'Attack', 'animals': ['Simon', 'JJ'], 'time': True, 'frames': False}.

    Notes
    ----------
    `GitHub documentation <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md__.

    Examples
    ----------
    >>> settings = {'brackets': 10, 'clf': 'Attack', 'animals': ['Simon', 'JJ'], 'time': True, 'frames': False}
    >>> processor = SeverityProcessor(config_path='project_folder/project_config.ini', settings=settings)
    >>> processor.run()
    >>> processor.save()
    """

    def __init__(self,
                 config_path: str,
                 settings: dict):

        self.timer = SimbaTimer()
        self.timer.start_timer()
        self.config = read_config_file(ini_path=config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        log_dir = os.path.join(self.project_path, 'logs')
        self.settings = settings
        self.in_dir = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.files_found = glob.glob(self.in_dir + '/*.' + self.file_type)
        check_if_filepath_list_is_empty(filepaths=self.files_found,
                                        error_msg=f'SIMBA ERROR: Cannot process severity. {self.in_dir} directory is empty')
        save_name = os.path.join(f'severity_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
        self.save_path = os.path.join(log_dir, save_name)
        self.results = {}
        self.x_cols, self.y_cols, self.p_cols = getBpNames(config_path)
        self.video_info_df = read_video_info_csv(file_path=os.path.join(log_dir, 'video_info.csv'))
        self.no_animals = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.ANIMAL_CNT.value, Dtypes.INT.value)
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.no_animals, self.x_cols, self.y_cols, self.p_cols, [])

    @staticmethod
    @jit(nopython=True)
    def __euclidean_distance(bp_1_x_vals, bp_2_x_vals, bp_1_y_vals, bp_2_y_vals):
        return (np.sqrt((bp_1_x_vals - bp_2_x_vals) ** 2 + (bp_1_y_vals - bp_2_y_vals) ** 2))

    def run(self):
        for file_path in self.files_found:
            _, video_name, _ = get_fn_ext(file_path)
            self.results[video_name] = {}
            df = read_df(file_path=file_path, file_type=self.file_type)
            if self.settings['clf'] not in df.columns:
                print(f'SIMBA ERROR: Skipping file {video_name} - {self.settings["clf"]} data not present in file')
                continue
            _, _, fps = read_video_info(vid_info_df=self.video_info_df, video_name=video_name)
            for animal_name, animal_bodyparts in self.animal_bp_dict.items():
                animal_df = df[animal_bodyparts['X_bps'] + animal_bodyparts['Y_bps']]
                shifted = animal_df.shift(periods=1).fillna(0)
                movement = pd.DataFrame()
                for (bp_x, bp_y) in zip(animal_bodyparts['X_bps'], animal_bodyparts['Y_bps']):
                    movement[bp_x.rstrip('_x')] = self.__euclidean_distance(animal_df[bp_x].values, shifted[bp_x].values, animal_df[bp_y].values, shifted[bp_y].values)
                movement['sum'] = movement.sum(axis=1)
                movement['sum'].iloc[0] = 0
                df[animal_name] = movement['sum']
            df['movement'] = df[self.settings['animals']].sum(axis=1)
            df['bin'] = pd.qcut(x=df['movement'], q=self.settings['brackets'], labels=list(range(1, self.settings['brackets']+1)))
            clf_df = df['bin'][df[self.settings['clf']] == 1].astype(int).reset_index(drop=True)
            for i in range(0, self.settings['brackets']):
                if self.settings['frames']:
                    self.results[video_name][f'Grade {str(i + 1)} (frames)'] = len(clf_df[clf_df == i])
                if self.settings['time']:
                    self.results[video_name][f'Grade {str(i + 1)} (s)'] = round((len(clf_df[clf_df == i])/ fps), 4)

    def save(self):
        out_df = pd.DataFrame(columns=['VIDEO', 'MEASUREMENT', 'VALUE'])
        for video_name, video_data in self.results.items():
            for grade, grade_data in video_data.items():
                out_df.loc[len(out_df)] = [video_name, grade, grade_data]
        out_df.to_csv(self.save_path)
        self.timer.stop_timer()
        print(f'SIMBA COMPLETE: Severity data saved at {self.save_path} (elapsed time {self.timer.elapsed_time_str}s)')
