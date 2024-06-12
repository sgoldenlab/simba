__author__ = "Simon Nilsson"

import os
from typing import Union

import numpy as np
import pandas as pd
from numba import jit

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.enums import ConfigKey, Dtypes
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (get_fn_ext, read_config_entry, read_df,
                                    write_df)


class OutlierCorrecterMovement(ConfigReader, FeatureExtractionMixin):
    """
    Detect and ammend outliers in pose-estimation data based on movement lenghth (Euclidean) of the body-parts
    in the current frame from preceeding frame. Uses critera stored in the SimBA project project_config.ini
    under the [Outlier settings] header.

    :param str config_path: path to SimBA project config file in Configparser format

    .. image:: _static/img/movement_outlier.png
       :width: 500
       :align: center

    .. note::
       `Outlier correction documentation <https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf>`__.

    Examples
    ----------
    >>> outlier_correcter_movement = OutlierCorrecterMovement(config_path='MyProjectConfig')
    >>> outlier_correcter_movement.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path)
        FeatureExtractionMixin.__init__(self)
        if not os.path.exists(self.outlier_corrected_movement_dir): os.makedirs(self.outlier_corrected_movement_dir)
        if self.animal_cnt == 1:
            self.animal_id = read_config_entry(self.config, ConfigKey.MULTI_ANIMAL_ID_SETTING.value, ConfigKey.MULTI_ANIMAL_IDS.value, Dtypes.STR.value)
            if self.animal_id != "None":
                self.animal_bp_dict[self.animal_id] = self.animal_bp_dict.pop("Animal_1")
        self.above_criterion_dict_dict = {}
        self.criterion = read_config_entry(
            self.config,
            ConfigKey.OUTLIER_SETTINGS.value,
            ConfigKey.MOVEMENT_CRITERION.value,
            Dtypes.FLOAT.value,
        )
        self.outlier_bp_dict = {}
        for animal_name in self.animal_bp_dict.keys():
            self.outlier_bp_dict[animal_name] = {}
            self.outlier_bp_dict[animal_name]["bp_1"] = read_config_entry(
                self.config,
                "Outlier settings",
                "movement_bodypart1_{}".format(animal_name.lower()),
                "str",
            )
            self.outlier_bp_dict[animal_name]["bp_2"] = read_config_entry(
                self.config,
                "Outlier settings",
                "movement_bodypart2_{}".format(animal_name.lower()),
                "str",
            )

    @staticmethod
    @jit(nopython=True)
    def __corrector(data=np.ndarray, criterion=float):
        results, current_value, cnt = np.full(data.shape, np.nan), data[0, :], 0
        for i in range(data.shape[0]):
            dist = abs(np.linalg.norm(current_value - data[i, :]))
            if dist <= criterion:
                current_value = data[i, :]
                cnt += 1
            results[i, :] = current_value
        return results, cnt

    def __outlier_replacer(self):
        for animal_name, animal_body_parts in self.animal_bp_dict.items():
            for bp_x_name, bp_y_name in zip(
                animal_body_parts["X_bps"], animal_body_parts["Y_bps"]
            ):
                vals, cnt = self.__corrector(
                    data=self.data_df[[bp_x_name, bp_y_name]].values,
                    criterion=self.animal_criteria[animal_name],
                )
                df = pd.DataFrame(vals, columns=[bp_x_name, bp_y_name])
                self.data_df.update(df)
                self.log.loc[len(self.log)] = [
                    self.video_name,
                    animal_name,
                    bp_x_name[:-2],
                    cnt,
                    round(cnt / len(df), 6),
                ]

    def run(self):
        """
        Runs outlier detection and correction. Results are stored in the
        ``project_folder/csv/outlier_corrected_movement`` directory of the SimBA project.
        """
        self.log = pd.DataFrame(
            columns=[
                "VIDEO",
                "ANIMAL",
                "BODY-PART",
                "CORRECTION COUNT",
                "CORRECTION PCT",
            ]
        )
        for file_cnt, file_path in enumerate(self.input_csv_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            print(f"Processing video {self.video_name}. Video {file_cnt+1}/{len(self.input_csv_paths)}...")
            self.above_criterion_dict_dict[self.video_name] = {}
            save_path = os.path.join(self.outlier_corrected_movement_dir, f"{self.video_name}.{self.file_type}")
            self.data_df = read_df(file_path, self.file_type, check_multiindex=True)
            self.data_df = self.insert_column_headers_for_outlier_correction(data_df=self.data_df, new_headers=self.bp_headers, filepath=file_path)
            self.data_df_combined = self.create_shifted_df(df=self.data_df)
            self.animal_criteria = {}
            for animal_name, animal_bps in self.outlier_bp_dict.items():
                animal_bp_distances = np.sqrt(
                    (
                        self.data_df[animal_bps["bp_1"] + "_x"]
                        - self.data_df[animal_bps["bp_2"] + "_x"]
                    )
                    ** 2
                    + (
                        self.data_df[animal_bps["bp_1"] + "_y"]
                        - self.data_df[animal_bps["bp_2"] + "_y"]
                    )
                    ** 2
                )
                self.animal_criteria[animal_name] = (animal_bp_distances.mean() * self.criterion)
            self.__outlier_replacer()
            write_df(df=self.data_df, file_type=self.file_type, save_path=save_path)
            video_timer.stop_timer()
            print(f"Corrected movement outliers for file {self.video_name} (elapsed time: {video_timer.elapsed_time_str}s)...")
        self.__save_log_file()

    def __save_log_file(self):
        self.log_fn = os.path.join(self.logs_path, f"Outliers_movement_{self.datetime}.csv")
        self.log.to_csv(self.log_fn)
        self.timer.stop_timer()
        stdout_success(msg=f'Log for corrected "movement outliers" saved in {self.logs_path}', elapsed_time=self.timer.elapsed_time_str)


#
# test = OutlierCorrecterMovement(config_path='/Users/simon/Desktop/envs/troubleshooting/dorian_2/project_folder/project_config.ini')
# test.run()

# test = OutlierCorrecterMovement(config_path='/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini')
# test.correct_movement_outliers()

# test = OutlierCorrecterMovement(config_path='/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini')
# test.run()

# test = OutlierCorrecterMovement(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.correct_movement_outliers()
#
# test = OutlierCorrecterMovement(config_path='/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini')
# test.correct_movement_outliers()
