from simba.read_config_unit_tests import read_config_entry
import os, glob
from simba.misc_tools import get_fn_ext
from simba.rw_dfs import (read_df,
                          save_df)
from simba.enums import ReadConfig, Dtypes
import numpy as np
import pandas as pd
from simba.mixins.config_reader import ConfigReader

class OutlierCorrecterLocation(ConfigReader):
    """
    Class for detecting and amending outliers in pose-estimation data based in the location of the body-parts
    in the current frame relative to the location of the body-part in the preceding frame.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format

    Notes
    ----------
    Outlier correction documentation <https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf>`__.


    Examples
    ----------
    >>> outlier_correcter_location = OutlierCorrecterLocation(config_path='MyProjectConfig')
    >>> outlier_correcter_location.correct_location_outliers()
    """

    def __init__(self,
                 config_path: str):

        super().__init__(config_path=config_path)

        if not os.path.exists(self.outlier_corrected_dir): os.makedirs(self.outlier_corrected_dir)
        self.files_found = glob.glob(self.outlier_corrected_movement_dir + '/*.' + self.file_type)
        self.body_parts = list(set([x[:-2] for x in self.column_headers]))
        if self.animal_cnt == 1:
            self.animal_id = read_config_entry(self.config, ReadConfig.MULTI_ANIMAL_ID_SETTING.value, ReadConfig.MULTI_ANIMAL_IDS.value, Dtypes.STR.value)
            if self.animal_id != 'None':
                self.animal_bp_dict[self.animal_id] = self.animal_bp_dict.pop('Animal_1')
        self.above_criterion_dict_dict = {}
        self.below_criterion_dict_dict = {}
        self.criterion = read_config_entry(self.config, ReadConfig.OUTLIER_SETTINGS.value, ReadConfig.LOCATION_CRITERION.value, Dtypes.FLOAT.value)
        self.outlier_bp_dict = {}
        for animal_name in self.animal_bp_dict.keys():
            self.outlier_bp_dict[animal_name] = {}
            self.outlier_bp_dict[animal_name]['bp_1'] = read_config_entry(self.config, ReadConfig.OUTLIER_SETTINGS.value, 'location_bodypart1_{}'.format(animal_name.lower()), 'str')
            self.outlier_bp_dict[animal_name]['bp_2'] = read_config_entry(self.config, ReadConfig.OUTLIER_SETTINGS.value, 'location_bodypart2_{}'.format(animal_name.lower()), 'str')

    def __find_location_outliers(self):
        for animal_name, animal_data in self.bp_dict.items():
            animal_criterion = self.animal_criteria[animal_name]
            self.above_criterion_dict_dict[self.video_name][animal_name] = {}
            self.below_criterion_dict_dict[self.video_name][animal_name] = {}
            for body_part_name, body_part_data in animal_data.items():
                self.above_criterion_dict_dict[self.video_name][animal_name][body_part_name] = []
                self.below_criterion_dict_dict[self.video_name][animal_name][body_part_name] = []
                for frame in range(body_part_data.shape[0]):
                    second_bp_names = [x for x in list(animal_data.keys()) if x != body_part_name]
                    first_bp_cord = body_part_data[frame]
                    distance_above_criterion_counter = 0
                    for second_bp in second_bp_names:
                        second_bp_cord = animal_data[second_bp][frame]
                        distance = np.sqrt((first_bp_cord[0] - second_bp_cord[0]) ** 2 + (first_bp_cord[1] - second_bp_cord[1]) ** 2)
                        if distance > animal_criterion:
                            distance_above_criterion_counter += 1
                    if distance_above_criterion_counter > 1:
                        self.above_criterion_dict_dict[self.video_name][animal_name][body_part_name].append(frame)
                    else:
                        self.below_criterion_dict_dict[self.video_name][animal_name][body_part_name].append(frame)


    def __correct_outliers(self):
        above_citeria_dict = self.above_criterion_dict_dict[self.video_name]
        for animal_name, animal_bp_data in above_citeria_dict.items():
            for bp_name, outlier_idx_lst in animal_bp_data.items():
                body_part_x, body_part_y = bp_name + '_x', bp_name + '_y'
                for outlier_idx in outlier_idx_lst:
                    try:
                        closest_idx = max([i for i in self.below_criterion_dict_dict[self.video_name][animal_name][bp_name] if outlier_idx > i])
                    except ValueError:
                        closest_idx = outlier_idx
                    self.data_df.loc[[outlier_idx], body_part_x] = self.data_df.loc[[closest_idx], body_part_x].values[0]
                    self.data_df.loc[[outlier_idx], body_part_y] = self.data_df.loc[[closest_idx], body_part_y].values[0]

    def correct_location_outliers(self):
        """
        Runs outlier detection and correction. Results are stored in the
        ``project_folder/csv/outlier_corrected_movement_location`` directory of the SimBA project.
        """

        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            print('Processing video {}. Video {}/{}..'.format(self.video_name, str(file_cnt+1), str(len(self.files_found))))
            self.above_criterion_dict_dict[self.video_name] = {}
            self.below_criterion_dict_dict[self.video_name] = {}
            save_path = os.path.join(self.outlier_corrected_dir, self.video_name + '.' + self.file_type)
            self.data_df = read_df(file_path, self.file_type)
            self.animal_criteria = {}
            for animal_name, animal_bps in self.outlier_bp_dict.items():
                animal_bp_distances = np.sqrt((self.data_df[animal_bps['bp_1'] + '_x'] - self.data_df[animal_bps['bp_2'] + '_x']) ** 2 + (self.data_df[animal_bps['bp_1'] + '_y'] - self.data_df[animal_bps['bp_2'] + '_y']) ** 2)
                self.animal_criteria[animal_name] = animal_bp_distances.mean() * self.criterion
            self.bp_dict = {}
            for animal_name, animal_bps in self.animal_bp_dict.items():
                bp_col_names = np.array([[i, j] for i, j in zip(animal_bps['X_bps'], animal_bps['Y_bps'])]).ravel()
                animal_arr = self.data_df[bp_col_names].to_numpy()
                self.bp_dict[animal_name] = {}
                for bp_cnt, bp_col_start in enumerate(range(0, animal_arr.shape[1], 2)):
                    bp_name = animal_bps['X_bps'][bp_cnt][:-2]
                    self.bp_dict[animal_name][bp_name] = animal_arr[:, bp_col_start:bp_col_start+2]
            self.__find_location_outliers()
            self.__correct_outliers()
            save_df(self.data_df, self.file_type, save_path)
            print('Corrected location outliers for file {} ...'.format(self.video_name))
        self.__save_log_file()

    def __save_log_file(self):
        out_df_lst = []
        for video_name, video_data in self.above_criterion_dict_dict.items():
            for animal_name, animal_data in video_data.items():
                for bp_name, vid_idx_lst in animal_data.items():
                    correction_ratio = round(len(vid_idx_lst) / len(self.data_df), 6)
                    out_df_lst.append(pd.DataFrame([[video_name, animal_name, bp_name, len(vid_idx_lst), correction_ratio]], columns=['Video', 'Animal', 'Body-part', 'Corrections', 'Correction ratio (%)']))
        out_df = pd.concat(out_df_lst, axis=0).reset_index(drop=True)
        log_fn = os.path.join(self.logs_path, 'Outliers_location_{}.csv'.format(self.datetime))
        out_df.to_csv(log_fn)
        self.timer.stop_timer()
        print(f'SIMBA COMPLETE: Log for corrected "location outliers" saved in project_folder/logs (elapsed time {self.timer.elapsed_time_str}s)')

# test = OutlierCorrecterLocation(config_path='/Users/simon/Desktop/troubleshooting/Zebrafish/project_folder/project_config.ini')
# test.correct_location_outliers()
#
# test = OutlierCorrecterLocation(config_path='/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini')
# test.correct_location_outliers()