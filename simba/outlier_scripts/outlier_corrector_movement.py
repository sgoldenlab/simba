from datetime import datetime
from simba.drop_bp_cords import (getBpNames,
                                 create_body_part_dictionary,
                                 getBpHeaders)
from simba.misc_tools import (get_fn_ext,
                              check_multi_animal_status)
from simba.read_config_unit_tests import (read_config_entry,
                                          read_config_file,
                                          read_project_path_and_file_type)
from simba.train_model_functions import insert_column_headers_for_outlier_correction
from simba.enums import Paths, ReadConfig, Dtypes
import os, glob
from simba.rw_dfs import read_df, save_df
import pandas as pd
import numpy as np

class OutlierCorrecterMovement(object):
    """
    Class for detecting and amending outliers in pose-estimation data based on movement sizes of the body-parts
    in the current frame.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    ----------
    `Outlier correction documentation <https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf>`__.

    Examples
    ----------
    >>> outlier_correcter_movement = OutlierCorrecterMovement(config_path='MyProjectConfig')
    >>> outlier_correcter_movement.correct_movement_outliers()

    """

    def __init__(self,
                 config_path: str):

        self.config = read_config_file(config_path)
        self.project_path, self.file_type = read_project_path_and_file_type(config=self.config)
        self.datetime_str = datetime.now().strftime('%Y%m%d%H%M%S')
        self.dir_in = os.path.join(self.project_path, Paths.INPUT_CSV.value)
        self.dir_out = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement')
        if not os.path.exists(self.dir_out): os.makedirs(self.dir_out)
        self.log_path = os.path.join(self.project_path, 'logs')
        self.files_found = glob.glob(self.dir_in + '/*.' + self.file_type)
        x_cols, y_cols, p_cols = getBpNames(config_path)
        self.headers = getBpHeaders(config_path)
        self.body_parts = list(set([x[:-2] for x in self.headers]))
        self.no_animals = read_config_entry(self.config, ReadConfig.GENERAL_SETTINGS.value, ReadConfig.ANIMAL_CNT.value, Dtypes.INT.value)
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.no_animals, x_cols, y_cols, p_cols,[])
        if self.no_animals == 1:
            self.animal_id = read_config_entry(self.config, ReadConfig.MULTI_ANIMAL_ID_SETTING.value, ReadConfig.MULTI_ANIMAL_IDS.value, Dtypes.STR.value)
            if self.animal_id != 'None':
                self.animal_bp_dict[self.animal_id] = self.animal_bp_dict.pop('Animal_1')
        self.above_criterion_dict_dict = {}
        self.criterion = read_config_entry(self.config, ReadConfig.OUTLIER_SETTINGS.value, ReadConfig.MOVEMENT_CRITERION.value, Dtypes.FLOAT.value)
        self.outlier_bp_dict = {}
        for animal_name in self.animal_bp_dict.keys():
            self.outlier_bp_dict[animal_name] = {}
            self.outlier_bp_dict[animal_name]['bp_1'] = read_config_entry(self.config, 'Outlier settings', 'movement_bodypart1_{}'.format(animal_name.lower()), 'str')
            self.outlier_bp_dict[animal_name]['bp_2'] = read_config_entry(self.config, 'Outlier settings', 'movement_bodypart2_{}'.format(animal_name.lower()), 'str')


    def __outlier_replacer(self, bp_lst=None, animal_name=None):
        self.above_criterion_dict = {}
        self.below_criterion_dict = {}
        self.above_criterion_dict_dict[self.video_name][animal_name] = {}
        for c in self.body_parts:
            self.data_df_combined[c + '_movement'] = np.sqrt((self.data_df_combined[c + '_x'] - self.data_df_combined[c + '_x_shifted']) ** 2 + (self.data_df_combined[c + '_y'] - self.data_df_combined[c + '_y_shifted']) ** 2)
            self.above_criterion_dict[c] = list(self.data_df_combined.index[self.data_df_combined[c + '_movement'] > self.animal_criteria[animal_name]])
            self.below_criterion_dict[c] = list(self.data_df_combined.index[self.data_df_combined[c + '_movement'] <= self.animal_criteria[animal_name]])
            self.above_criterion_dict_dict[self.video_name][animal_name][c] = self.above_criterion_dict[c]
        for body_part, body_part_idx in self.above_criterion_dict.items():
            body_part_x, body_part_y = body_part + '_x', body_part + '_y'
            for idx in body_part_idx:
                try:
                    closest_idx = max([i for i in self.below_criterion_dict[body_part] if idx > i])
                except ValueError:
                    closest_idx = idx
                self.data_df.loc[[idx], body_part_x] = self.data_df.loc[[closest_idx], body_part_x].values[0]
                self.data_df.loc[[idx], body_part_y] = self.data_df.loc[[closest_idx], body_part_y].values[0]

    def correct_movement_outliers(self):
        """
        Runs outlier detection and correction. Results are stored in the
        ``project_folder/csv/outlier_corrected_movement`` directory of the SimBA project.
        """
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            print('Processing video {}. Video {}/{}..'.format(self.video_name, str(file_cnt+1), str(len(self.files_found))))
            self.above_criterion_dict_dict[self.video_name] = {}
            save_path = os.path.join(self.dir_out, self.video_name + '.' + self.file_type)
            self.data_df = read_df(file_path, self.file_type)
            try:
                self.data_df = self.data_df.drop(self.data_df.index[[0, 1]]).apply(pd.to_numeric).reset_index(drop=True)
            except ValueError as e:
                print(e.args)
                print('SIMBA WARNING: SimBA found more than the expected two header columns. SimBA will try to proceed '
                      'by removing one additional column header level. This can happen when you import multi-animal DLC data as standard DLC data.')
                self.data_df = self.data_df.drop(self.data_df.index[[0, 1, 2]]).apply(pd.to_numeric).reset_index(drop=True)
            self.data_df = insert_column_headers_for_outlier_correction(data_df=self.data_df, new_headers=list(self.headers), filepath=file_path)
            self.data_df_shifted = self.data_df.shift(periods=1).add_suffix('_shifted').fillna(0)
            self.data_df_combined = pd.concat([self.data_df, self.data_df_shifted], axis=1, join='inner').fillna(0)
            self.animal_criteria = {}
            for animal_name, animal_bps in self.outlier_bp_dict.items():
                animal_bp_distances = np.sqrt((self.data_df[animal_bps['bp_1'] + '_x'] - self.data_df[animal_bps['bp_2'] + '_x']) ** 2 + (self.data_df[animal_bps['bp_1'] + '_y'] - self.data_df[animal_bps['bp_2'] + '_y']) ** 2)
                self.animal_criteria[animal_name] = animal_bp_distances.mean() * self.criterion
            for animal_name in self.outlier_bp_dict.keys():
                animal_bps = [x for x in self.headers if x.startswith(animal_name + '_')]
                self.__outlier_replacer(bp_lst=animal_bps, animal_name=animal_name)
            save_df(self.data_df, self.file_type, save_path)
            print('Corrected movement outliers for file {} ...'.format(self.video_name))
        self.__save_log_file()

    def __save_log_file(self):
        out_df_lst = []
        for video_name, video_data in self.above_criterion_dict_dict.items():
            for animal_name, animal_data in video_data.items():
                for bp_name, vid_idx_lst in animal_data.items():
                    correction_ratio = round(len(vid_idx_lst) / len(self.data_df), 6)
                    out_df_lst.append(pd.DataFrame([[video_name, animal_name, bp_name, len(vid_idx_lst), correction_ratio]], columns=['Video', 'Animal', 'Body-part', 'Corrections', 'Correction ratio (%)']))
        out_df = pd.concat(out_df_lst, axis=0).reset_index(drop=True)
        log_fn = os.path.join(self.log_path, 'Outliers_movement_{}.csv'.format(self.datetime_str))
        out_df.to_csv(log_fn)
        print('Log for corrected "movement outliers" saved in project_folder/logs')

# test = OutlierCorrecterMovement(config_path='/Users/simon/Desktop/troubleshooting/User_def_2/project_folder/project_config.ini')
# test.correct_movement_outliers()


