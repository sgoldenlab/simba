__author__ = "Simon Nilsson", "JJ Choong"

from simba.ROI_analyzer import ROIAnalyzer
import pandas as pd
from simba.read_config_unit_tests import (check_int,
                                          check_str,
                                          check_float,
                                          read_config_entry,
                                          read_config_file)
import os, glob
from copy import deepcopy
from simba.drop_bp_cords import (getBpNames,
                                 create_body_part_dictionary,
                                 checkDirectionalityCords,
                                 get_fn_ext)
from simba.misc_tools import (check_multi_animal_status,
                              line_length_numba_to_static_location)
import numpy as np

class DirectingROIAnalyzer(object):
    """
    Class for computing aggregate statistics for animals are directing towards ROIs.

    Parameters
    ----------
    config_path: str
        Path to SimBA project config file in Configparser format

    Notes
    ----------
    `ROI tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial_new.md>`__.

    Examples
    ----------
    >>> roi_directing_calculator = DirectingROIAnalyzer(config_path='MyProjectConfig')
    >>> roi_directing_calculator.calc_directing_to_ROIs()

    """

    def __init__(self,
                 config_path: str=None,
                 data_path: str=None):


        self.config = read_config_file(config_path)
        self.project_path = read_config_entry(self.config, 'General settings', 'project_path', data_type='folder_path')
        self.data_path = os.path.join(self.project_path, 'csv', 'outlier_corrected_movement_location')
        self.roi_analyzer = ROIAnalyzer(ini_path=config_path, data_path=self.data_path)
        self.roi_analyzer.read_roi_dfs()
        self.files_found = deepcopy(self.roi_analyzer.files_found)
        self.x_cols, self.y_cols, self.pcols = getBpNames(config_path)
        self.no_animals = read_config_entry(self.config, 'General settings', 'animal_no', 'int')
        self.multi_animal_status, self.multi_animal_id_lst = check_multi_animal_status(self.config, self.no_animals)
        self.animal_bp_dict = create_body_part_dictionary(self.multi_animal_status, self.multi_animal_id_lst, self.no_animals, self.x_cols, self.y_cols, [], [])
        self.direct_bp_dict = checkDirectionalityCords(self.animal_bp_dict)

    def __format_direction_data(self):
        x_min = np.minimum(self.direction_data[:, 1], self.nose_arr[:, 0])
        y_min = np.minimum(self.direction_data[:, 2], self.nose_arr[:, 1])
        delta_x = abs((self.direction_data[:, 1] - self.nose_arr[:, 0]) / 2)
        delta_y = abs((self.direction_data[:, 2] - self.nose_arr[:, 1]) / 2)
        x_middle, y_middle = np.add(x_min, delta_x), np.add(y_min, delta_y)
        direction_data = np.concatenate((y_middle.reshape(-1, 1), self.direction_data), axis=1)
        direction_data = np.concatenate((x_middle.reshape(-1, 1), direction_data), axis=1)
        direction_data = np.delete(direction_data, [2, 3, 4], 1)
        bp_data = pd.DataFrame(direction_data, columns=['Eye_x', 'Eye_y', 'Directing_BOOL'])
        bp_data['ROI_x'] = self.center_cord[0]
        bp_data['ROI_y'] = self.center_cord[1]
        bp_data = bp_data[['Eye_x', 'Eye_y', 'ROI_x', 'ROI_y', 'Directing_BOOL']]
        bp_data.insert(loc=0, column='ROI', value=self.shape_info['Name'])
        bp_data.insert(loc=0, column='Animal', value=self.animal_name)
        bp_data.insert(loc=0, column='Video', value=self.video_name)
        bp_data = bp_data.reset_index().rename(columns={'index': 'Frame'})
        bp_data = bp_data[bp_data['Directing_BOOL'] == 1].reset_index(drop=True)
        self.results_lst.append(bp_data)

    def calc_directing_to_ROIs(self):
        """
        Method to calculate directing-towards ROI data

        Returns
        -------
        Attribute: pd.DataFrame
            results_df
        """

        self.results_lst = []
        for file_cnt, file_path in enumerate(self.files_found):
            _, self.video_name, _ = get_fn_ext(file_path)
            self.roi_analyzer.files_found = [file_path]
            self.roi_analyzer.analyze_ROIs()
            video_data_df = self.roi_analyzer.data_df
            for animal_name in self.roi_analyzer.multiAnimalIDList:
                self.animal_name = animal_name
                animal_bp_names = self.animal_bp_dict[animal_name]['X_bps'] + self.animal_bp_dict[animal_name]['Y_bps']
                animal_data_df = video_data_df[animal_bp_names]
                animal_direct_bps = self.direct_bp_dict[animal_name]
                self.ear_left_arr = animal_data_df[[animal_direct_bps['Ear_left']['X_bps'], animal_direct_bps['Ear_left']['Y_bps']]].to_numpy()
                self.ear_right_arr = animal_data_df[[animal_direct_bps['Ear_right']['X_bps'], animal_direct_bps['Ear_right']['Y_bps']]].to_numpy()
                self.nose_arr = animal_data_df[[animal_direct_bps['Nose']['X_bps'], animal_direct_bps['Nose']['Y_bps']]].to_numpy()
                for _, row in self.roi_analyzer.video_recs.iterrows():
                    self.shape_info = row
                    center_cord = ((int(row['topLeftX'] + (row['width'] / 2))), (int(row['topLeftY'] + (row['height'] / 2))))
                    self.center_cord = np.asarray(center_cord)
                    self.direction_data = line_length_numba_to_static_location(left_ear_array=self.ear_left_arr,
                                                                               right_ear_array=self.ear_right_arr,
                                                                               nose_array=self.nose_arr,
                                                                               target_array=self.center_cord)
                    self.__format_direction_data()
                for _, row in self.roi_analyzer.video_circs.iterrows():
                    self.shape_info = row
                    self.center_cord = np.asarray((row['centerX'], row['centerY']))
                    self.direction_data = line_length_numba_to_static_location(left_ear_array=self.ear_left_arr,
                                                                               right_ear_array=self.ear_right_arr,
                                                                               nose_array=self.nose_arr,
                                                                               target_array=self.center_cord)
                    self.__format_direction_data()
                for _, row in self.roi_analyzer.video_polys.iterrows():
                    self.shape_info = row
                    self.center_cord = np.asarray((row['Center_X'], row['Center_Y']))
                    self.direction_data = line_length_numba_to_static_location(left_ear_array=self.ear_left_arr,
                                                                               right_ear_array=self.ear_right_arr,
                                                                               nose_array=self.nose_arr,
                                                                               target_array=self.center_cord)
                    self.__format_direction_data()

        self.results_df = pd.concat(self.results_lst, axis=0)

# test = DirectingROIAnalyzer(config_path='/Users/simon/Desktop/train_model_project/project_folder/project_config.ini')
# test.calc_directing_to_ROIs()





