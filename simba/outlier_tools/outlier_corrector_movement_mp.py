__author__ = "Simon Nilsson"

import functools
import multiprocessing
import os
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from numba import jit

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import check_float, check_if_dir_exists, check_int
from simba.utils.enums import ConfigKey, Dtypes
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_config_entry, read_df,
                                    write_df)


def _movement_outlier_corrector(data_path: str,
                                config: ConfigReader,
                                animal_bp_dict: dict,
                                outlier_dict: dict,
                                file_type: str,
                                save_dir: str,
                                criterion: float):


    @jit(nopython=True)
    def _corrector(data: np.ndarray, criterion: float):
        results, current_value, cnt = np.full(data.shape, np.nan), data[0, :], 0
        for i in range(data.shape[0]):
            dist = abs(np.linalg.norm(current_value - data[i, :]))
            if dist <= criterion:
                current_value = data[i, :]
            else:
                cnt += 1
            results[i, :] = current_value
        return results, cnt


    def _outlier_replacer(data_df: pd.DataFrame,
                          animal_criteria: dict,
                          video_name: str):

        log = pd.DataFrame(columns=["VIDEO", "ANIMAL", "BODY-PART", "CORRECTION COUNT", "CORRECTION PCT"])
        for animal_name, animal_body_parts in animal_bp_dict.items():
            for bp_x_name, bp_y_name in zip(animal_body_parts["X_bps"], animal_body_parts["Y_bps"]):
                vals, cnt = _corrector(data=data_df[[bp_x_name, bp_y_name]].values, criterion=animal_criteria[animal_name])
                df = pd.DataFrame(vals, columns=[bp_x_name, bp_y_name])
                data_df.update(df)
                log.loc[len(log)] = [video_name, animal_name, bp_x_name[:-2], cnt, round(cnt / len(df), 6)]

        return data_df, log


    video_timer = SimbaTimer(start=True)
    _, video_name, _  = get_fn_ext(filepath=data_path)
    save_path = os.path.join(save_dir, f"{video_name}.{file_type}")
    df = read_df(data_path, file_type, check_multiindex=True)
    df = config.insert_column_headers_for_outlier_correction(data_df=df, new_headers=config.bp_headers, filepath=data_path)
    animal_criteria = {}
    for animal_name, animal_bps in outlier_dict.items():
        animal_bp_distances = np.sqrt((df[animal_bps["bp_1"] + "_x"] - df[animal_bps["bp_2"] + "_x"]) ** 2 + (df[animal_bps["bp_1"] + "_y"] - df[animal_bps["bp_2"] + "_y"]) ** 2)
        animal_criteria[animal_name] = (animal_bp_distances.mean() * criterion)
    df, log = _outlier_replacer(animal_criteria=animal_criteria, data_df=df, video_name=video_name)
    write_df(df=df, file_type=file_type, save_path=save_path)
    video_timer.stop_timer()
    print(f"Corrected movement outliers for file {video_name} (elapsed time: {video_timer.elapsed_time_str}s)...")

    return video_name, log



class OutlierCorrecterMovementMultiProcess(ConfigReader, FeatureExtractionMixin):
    """
    Detect and ammend outliers in pose-estimation data based on movement lenghth (Euclidean) of the body-parts
    in the current frame from preceeding frame. If not passed, then uses critera stored in the SimBA project project_config.ini
    under the [Outlier settings] headed. Uses multiprocessing.

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param Optional[Union[str, os.PathLike]] data_dir: The directory storing the input data. If None, then the ``input_csv`` directory of the SimBA project.
    :param Optional[Union[str, os.PathLike]] save_dir: The directory to store the results. If None, then the ``outlier_corrected_movement`` directory of the SimBA project.
    :param Optional[int] core_cnt: The number of cores to use. If -1, then all available cores. Default: -1.
    :param Optional[Dict[str, Dict[str, str]]] animal_dict: Dictionary holding the animal names, and the two body-parts to use to measure the mean or median size of the animals. If None, grabs the info from the SimBA project config.
    :param Optional[float] criterion: The criterion multiplier. If None, grabs the info from the SimBA project config.

    .. image:: _static/img/movement_outlier.png
       :width: 500
       :align: center

    .. note::
       `Outlier correction documentation <https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf>`__.

    :example:
    >>> outlier_correcter_movement = OutlierCorrecterMovementMultiProcess(config_path='MyProjectConfig')
    >>> outlier_correcter_movement.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_dir: Optional[Union[str, os.PathLike]] = None,
                 save_dir: Optional[Union[str, os.PathLike]] = None,
                 core_cnt: Optional[int] = -1,
                 animal_dict: Optional[Dict[str, Dict[str, str]]] = None,
                 criterion: Optional[float] = None):

        ConfigReader.__init__(self, config_path=config_path, create_logger=False)
        FeatureExtractionMixin.__init__(self)
        if not os.path.exists(self.outlier_corrected_movement_dir):
            os.makedirs(self.outlier_corrected_movement_dir)
        if criterion is None:
            self.criterion = read_config_entry(self.config, ConfigKey.OUTLIER_SETTINGS.value, ConfigKey.MOVEMENT_CRITERION.value, Dtypes.FLOAT.value)
        else:
            check_float(name=f'{criterion} criterion', value=criterion, min_value=10e-10)
            self.criterion = criterion
        if data_dir is not None:
            check_if_dir_exists(in_dir=data_dir, source=self.__class__.__name__)
            self.data_dir = data_dir
        else:
            self.data_dir = self.input_csv_dir

        if save_dir is not None:
            check_if_dir_exists(in_dir=save_dir, source=self.__class__.__name__)
            self.save_dir = save_dir
        else:
            self.save_dir = self.outlier_corrected_movement_dir

        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0])
        self.core_cnt = core_cnt
        if self.core_cnt == -1:
            self.core_cnt = find_core_cnt()[0]

        self.outlier_bp_dict, self.above_criterion_dict_dict = {}, {}
        if animal_dict is None:
            if self.animal_cnt == 1:
                self.animal_id = read_config_entry(self.config, ConfigKey.MULTI_ANIMAL_ID_SETTING.value, ConfigKey.MULTI_ANIMAL_IDS.value, Dtypes.STR.value)
                if self.animal_id != "None":
                    self.animal_bp_dict[self.animal_id] = self.animal_bp_dict.pop("Animal_1")

            for animal_name in self.animal_bp_dict.keys():
                self.outlier_bp_dict[animal_name] = {}
                self.outlier_bp_dict[animal_name]["bp_1"] = read_config_entry(self.config,"Outlier settings", "movement_bodypart1_{}".format(animal_name.lower()),"str")
                self.outlier_bp_dict[animal_name]["bp_2"] = read_config_entry(self.config,"Outlier settings", "movement_bodypart2_{}".format(animal_name.lower()),"str")
        else:
            self.outlier_bp_dict = animal_dict

    def run(self):
        self.logs = []
        data_paths = find_files_of_filetypes_in_directory(directory=self.data_dir, extensions=[f'.{self.file_type}'], raise_error=True)
        data_path_tuples = [(x) for x in data_paths]
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(_movement_outlier_corrector,
                                          config=self,
                                          animal_bp_dict=self.animal_bp_dict,
                                          outlier_dict=self.outlier_bp_dict,
                                          save_dir=self.save_dir,
                                          file_type=self.file_type,
                                          criterion=self.criterion)
            for cnt, (video_name, log) in enumerate(pool.imap(constants, data_path_tuples, chunksize=1)):
                print(f"Video {video_name} complete...")
                self.logs.append(log)

        self.__save_log_file()

    def __save_log_file(self):
        log_fn = os.path.join(self.logs_path, f"Outliers_movement_{self.datetime}.csv")
        self.logs = pd.concat(self.logs, axis=0)
        self.logs.to_csv(log_fn)
        self.timer.stop_timer()
        stdout_success(msg=f'Log for corrected "movement outliers" saved in {self.logs_path}', elapsed_time=self.timer.elapsed_time_str)

#
# if __name__ == "__main__":
#     #test = OutlierCorrecterMovementMultiProcess(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")
#     test = OutlierCorrecterMovementMultiProcess(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
#     test.run()
#
