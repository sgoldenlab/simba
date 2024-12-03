__author__ = "Simon Nilsson"


import functools
import multiprocessing
import os
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import check_float, check_if_dir_exists
from simba.utils.enums import ConfigKey, Dtypes
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_config_entry, read_df,
                                    write_df)


class OutlierCorrecterLocation(ConfigReader, FeatureExtractionMixin):
    """
    Detect and amend outliers in pose-estimation data based in the location of the body-parts
    in the current frame relative to the location of the body-part in the preceding frame using heuristic rules.

    Uses heuristic rules critera is grabbed from the SimBA project project_config.ini under the [Outlier settings] header.

    .. note::
       `Documentation <https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf>`_.

    .. image:: _static/img/location_outlier.png
       :width: 500
       :align: center

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param Optional[Union[str, os.PathLike]] data_dir: The directory storing the input data. If None, then the ``outlier_corrected_movement`` directory of the SimBA project.
    :param Optional[Union[str, os.PathLike]] save_dir: The directory to store the results. If None, then the ``outlier_corrected_movement_location`` directory of the SimBA project.
    :param Optional[Dict[str, Dict[str, str]]] animal_dict: Dictionary holding the animal names, and the two body-parts to use to measure the mean or median size of the animals. If None, grabs the info from the SimBA project config.
    :param Optional[float] criterion: The criterion multiplier. If None, grabs the info from the SimBA project config.

    :example:
    >>> _ = OutlierCorrecterLocation(config_path='MyProjectConfig').run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 animal_dict: Optional[Dict[str, Dict[str, str]]] = None,
                 criterion: Optional[float] = None):

        ConfigReader.__init__(self, config_path=config_path, create_logger=False, read_video_info=False)
        FeatureExtractionMixin.__init__(self)
        if not os.path.exists(self.outlier_corrected_dir):
            os.makedirs(self.outlier_corrected_dir)
        if criterion is None:
            self.criterion = read_config_entry(self.config, ConfigKey.OUTLIER_SETTINGS.value, ConfigKey.LOCATION_CRITERION.value, Dtypes.FLOAT.value)
        else:
            check_float(name=f'{criterion} criterion', value=criterion, min_value=10e-10)
            self.criterion = criterion
        if data_dir is not None:
            check_if_dir_exists(in_dir=data_dir, source=self.__class__.__name__)
            self.data_dir = data_dir
        else:
            self.data_dir = self.outlier_corrected_movement_dir
        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir, source=self.__class__.__name__)
            self.save_dir = save_dir
        else:
            self.save_dir = self.outlier_corrected_dir

        self.above_criterion_dict_dict, self.below_criterion_dict_dict = {},{}
        if animal_dict is None:
            self.outlier_bp_dict = {}
            if self.animal_cnt == 1:
                self.animal_id = read_config_entry(self.config, ConfigKey.MULTI_ANIMAL_ID_SETTING.value, ConfigKey.MULTI_ANIMAL_IDS.value, Dtypes.STR.value)
                if self.animal_id != "None":
                    self.animal_bp_dict[self.animal_id] = self.animal_bp_dict.pop("Animal_1")

            for animal_name in self.animal_bp_dict.keys():
                self.outlier_bp_dict[animal_name] = {}
                self.outlier_bp_dict[animal_name]["bp_1"] = read_config_entry(self.config, ConfigKey.OUTLIER_SETTINGS.value, "location_bodypart1_{}".format(animal_name.lower()),"str")
                self.outlier_bp_dict[animal_name]["bp_2"] = read_config_entry(self.config, ConfigKey.OUTLIER_SETTINGS.value, "location_bodypart2_{}".format(animal_name.lower()),"str")
        else:
            self.outlier_bp_dict = animal_dict

    def __find_location_outliers(self, bp_dict: dict, animal_criteria: dict):
        above_criteria_dict, below_criteria_dict = {}, {}
        for animal_name, animal_data in bp_dict.items():
            animal_criterion = animal_criteria[animal_name]
            above_criteria_dict[animal_name]= {}
            for first_bp_cnt, (first_body_part_name, first_bp_cords) in enumerate(animal_data.items()):
                second_bp_names = [x for x in list(animal_data.keys()) if x != first_body_part_name]
                above_criterion_frms = []
                for second_bp_cnt, second_bp in enumerate(second_bp_names):
                    second_bp_cords = animal_data[second_bp]
                    distances = self.framewise_euclidean_distance(location_1=first_bp_cords, location_2=second_bp_cords, px_per_mm=1.0, centimeter=False)
                    above_criterion_frms.extend(np.argwhere(distances > animal_criterion).flatten())
                unique, counts = np.unique(above_criterion_frms, return_counts=True)
                above_criteria_dict[animal_name][first_body_part_name] = np.sort(unique[counts > 1])
        return above_criteria_dict


    def __correct_outliers(self, df: pd.DataFrame, above_criteria_dict: dict):
        for animal_name, animal_data in above_criteria_dict.items():
            for body_part_name, frm_idx in animal_data.items():
                col_names = [f'{body_part_name}_x', f'{body_part_name}_y']
                if len(frm_idx) > 0:
                    df.loc[frm_idx, col_names] = np.nan
        return df.fillna(method='ffill', axis=0).fillna(0)

    def run(self):
        self.logs, self.frm_cnts = {}, {}
        data_paths = find_files_of_filetypes_in_directory(directory=self.data_dir, extensions=[f'.{self.file_type}'], raise_error=True)
        for file_cnt, data_path in enumerate(data_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(data_path)
            print(f"Processing video {video_name}..")
            save_path = os.path.join(self.save_dir, f"{video_name}.{self.file_type}")
            above_criterion_dict, below_criterion_dict, animal_criteria, bp_dict = {}, {}, {}, {}
            df = read_df(data_path, self.file_type)
            for animal_name, animal_bps in self.outlier_bp_dict.items():
                animal_bp_distances = np.sqrt((df[animal_bps["bp_1"] + "_x"] - df[animal_bps["bp_2"] + "_x"]) ** 2 + (df[animal_bps["bp_1"] + "_y"] - df[animal_bps["bp_2"] + "_y"]) ** 2)
                animal_criteria[animal_name] = (animal_bp_distances.mean() * self.criterion)
            for animal_name, animal_bps in self.animal_bp_dict.items():
                bp_col_names = np.array([[i, j] for i, j in zip(animal_bps["X_bps"], animal_bps["Y_bps"])]).ravel()
                animal_arr = df[bp_col_names].to_numpy()
                bp_dict[animal_name] = {}
                for bp_cnt, bp_col_start in enumerate(range(0, animal_arr.shape[1], 2)):
                    bp_name = animal_bps["X_bps"][bp_cnt][:-2]
                    bp_dict[animal_name][bp_name] = animal_arr[:, bp_col_start: bp_col_start + 2]
            above_criteria_dict = self.__find_location_outliers(bp_dict=bp_dict, animal_criteria=animal_criteria)
            df = self.__correct_outliers(df=df, above_criteria_dict=above_criteria_dict)
            write_df(df=df, file_type=self.file_type, save_path=save_path)
            self.logs[video_name], self.frm_cnts[video_name] = above_criteria_dict, len(df)
            video_timer.stop_timer()
            print(f"Corrected location outliers for file {video_name} (elapsed time: {video_timer.elapsed_time_str}s)...")
        self.__save_log_file()

    def __save_log_file(self):
        out_df = pd.DataFrame(columns=['VIDEO', 'ANIMAL', 'BODY-PART', 'CORRECTION COUNT', 'CORRECTION RATIO'])
        for video_name, video_data in self.logs.items():
            for animal_name, animal_data in video_data.items():
                for bp_name, bp_data in animal_data.items():
                    correction_ratio = round(len(bp_data) / self.frm_cnts[video_name], 6)
                    out_df.loc[len(out_df)] = [video_name, animal_name, bp_name, len(bp_data), correction_ratio]
        self.logs_path = os.path.join(self.logs_path, f"Outliers_location_{self.datetime}.csv")
        out_df.to_csv(self.logs_path)
        self.timer.stop_timer()
        stdout_success(msg='Log for corrected "location outliers" saved in project_folder/logs', elapsed_time=self.timer.elapsed_time_str)



# test = OutlierCorrecterLocation(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")
# test.run()

# test = OutlierCorrecterLocation(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.correct_location_outliers()
