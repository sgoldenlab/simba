__author__ = "Simon Nilsson"

import glob
import os
from copy import deepcopy
from typing import Union

import numpy as np
import pandas as pd
from numba import jit, prange

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.enums import TagNames
from simba.utils.errors import InvalidInputError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df, write_df)


class OutlierCorrecterLocationAdvanced(ConfigReader, FeatureExtractionMixin):
    """
    Performs location outlier correction that allows different heuristic outlier criteria for different animals or body-parts.
    For example, correct some outlier body-parts with a location heuristic criteria of 4x above the mean movement,
    and other body-parts with a heuristic critera of 2x above the mean movement.

    .. note::
       See `notebook <https://simba-uw-tf-dev.readthedocs.io/en/latest/nb/advanced_outlier_correction.html>`__. for example use-case.

    .. image:: _static/img/location_outlier.png
       :width: 400
       :align: center

    :parameter str config_path: path to SimBA project config file in Configparser format.
    :parameter str input_dir: Directory containing input files. E.g., `project_folder/csv/input_csv` directory.
    :parameter Literal['animal', 'body-part'] type: If the rules are defined on animal or body-part level. E.g., If one heuristic rule per animal then `animal`. If one heurstic rule prr body-part then `body-parts`
    :parameter Literal['mean', 'median'] agg_method: If to use the mean or median to compute criterion.
    :parameter dict criterion_body_parts: The body-parts used to calculate the size of the animals.
    :parameter dict settings: The heuristic multiplier rules to use for each body part or animal.

    :example:
    >>> settings = {'Simon': 1.1, 'JJ': 1.2}
    >>> criterion_body_parts = {'Simon': ['Nose_1', 'Tail_base_1'], 'JJ': ['Nose_2', 'Tail_base_2']}
    >>> outlier_corrector = OutlierCorrecterLocationAdvanced(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', input_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/input_csv', type='animal', agg_method='mean', criterion_body_parts=criterion_body_parts, settings=settings)
    >>> outlier_corrector.run()
    >>> settings = {'Simon': {'Ear_left_1': 1.1, 'Ear_right_1': 5.1, 'Nose_1': 2.1, 'Center_1': 1.5, 'Lat_left_1': 3.1, 'Lat_right_1': 1.9, 'Tail_base_1': 2.3, 'Tail_end_1': 1.4}, 'JJ': {'Ear_left_2': 1.2, 'Ear_right_2': 1.2, 'Nose_2': 2, 'Center_2': 4.1, 'Lat_left_2': 9, 'Lat_right_2': 1.2, 'Tail_base_2': 1.6, 'Tail_end_2': 2.2}}
    >>> outlier_corrector = OutlierCorrecterLocationAdvanced(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', input_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/input_csv', type='body-part', agg_method='mean', criterion_body_parts=criterion_body_parts, settings=settings)
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        input_dir: Union[str, os.PathLike],
        type: str,
        agg_method: str,
        criterion_body_parts: dict,
        settings: dict,
    ):
        ConfigReader.__init__(self, config_path=config_path)
        FeatureExtractionMixin.__init__(self)
        log_event(
            logger_name=str(self.__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        self.data_files = find_files_of_filetypes_in_directory(
            directory=input_dir, extensions=["." + self.file_type]
        )
        if len(self.data_files) == 0:
            raise NoFilesFoundError(
                msg=f"No data of filetype {input_dir} for in directory {self.file_type}",
                source=self.__class__.__name__,
            )
        if type not in ["animal", "body-parts"]:
            raise InvalidInputError(
                msg=f"Type {type} not supported. Valid options: {['animal', 'body-parts']}",
                source=self.__class__.__name__,
            )
        if agg_method not in ["mean", "median"]:
            raise InvalidInputError(
                msg=f"Aggregation method {agg_method} not supported. Valid options: {['mean', 'median']}",
                source=self.__class__.__name__,
            )
        self.settings, self.type, self.agg_method, self.criterion_body_parts = (
            settings,
            type,
            agg_method,
            criterion_body_parts,
        )

    def fix_settings_for_animal_input(self):
        new_settings = {}
        for animal_name, animal_bps in self.animal_bp_dict.items():
            if animal_name in self.settings.keys():
                if animal_name not in new_settings.keys():
                    new_settings[animal_name] = {}
                for body_part in animal_bps["X_bps"]:
                    new_settings[animal_name][body_part[:-2]] = self.settings[
                        animal_name
                    ]
        self.settings = new_settings

    @staticmethod
    @jit(nopython=True)
    def three_dim_cdist(data: np.ndarray) -> np.ndarray:
        results = np.full((data.shape[0], data.shape[1], data.shape[1]), np.nan)
        for i in prange(data.shape[0]):
            for j in prange(data[i].shape[0]):
                for k in prange(data[i].shape[0]):
                    results[i][j][k] = np.linalg.norm(data[i][j] - data[i][k])

        return results

    def _save_data_log(self, data_log: dict):
        out = pd.DataFrame(
            columns=[
                "VIDEO",
                "ANIMAL",
                "BODY-PART",
                "OUTLIER COUNT",
                "OUTLIER RATIO (%)",
            ]
        )
        save_path = os.path.join(
            self.logs_path, f"location_outliers_{self.datetime}.csv"
        )
        for video_name, video_data in data_log.items():
            for animal_name, animal_data in video_data.items():
                for body_part_name, body_part_data in animal_data.items():
                    out.loc[len(out)] = [
                        video_name,
                        animal_name,
                        body_part_name,
                        body_part_data["outlier_cnt"],
                        body_part_data["outlier_ratio"],
                    ]
        write_df(df=out, file_type="csv", save_path=save_path)
        stdout_success(
            msg=f"Location outlier log saved at {save_path}",
            source=self.__class__.__name__,
        )

    def run(self):
        data_log = {}
        if self.type == "animal":
            self.fix_settings_for_animal_input()

        for file_cnt, file_path in enumerate(self.data_files):
            video_timer, video_log = SimbaTimer(start=True), {}
            _, self.video_name, _ = get_fn_ext(file_path)
            print(
                "Processing video {}. Video {}/{}...".format(
                    self.video_name, str(file_cnt + 1), str(len(self.data_files))
                )
            )
            self.data_df = read_df(file_path, self.file_type, check_multiindex=True)
            self.data_df = self.insert_column_headers_for_outlier_correction(
                data_df=self.data_df, new_headers=self.bp_headers, filepath=file_path
            )
            self.results = deepcopy(self.data_df)
            self.data_df_combined = self.create_shifted_df(df=self.data_df)
            animal_movement_agg = {}
            for animal_name, animal_bps in self.animal_bp_dict.items():
                if animal_name in self.criterion_body_parts.keys():
                    animal_criterion_bps = self.criterion_body_parts[animal_name]
                    bp_1_headers = [
                        animal_criterion_bps[0] + "_x",
                        animal_criterion_bps[0] + "_y",
                    ]
                    bp_2_headers = [
                        animal_criterion_bps[1] + "_x",
                        animal_criterion_bps[1] + "_y",
                    ]
                    distances = self.framewise_euclidean_distance(
                        location_1=self.data_df[bp_1_headers].values,
                        location_2=self.data_df[bp_2_headers].values,
                        px_per_mm=1,
                    )
                    if self.agg_method == "mean":
                        animal_movement_agg[animal_name] = np.mean(distances).astype(
                            int
                        )
                    if self.agg_method == "median":
                        animal_movement_agg[animal_name] = np.median(distances).astype(
                            int
                        )

            for animal_name, animal_bps in self.settings.items():
                video_log[animal_name] = {}
                animal_bp_headers = np.array(
                    [
                        item
                        for pair in zip(
                            self.animal_bp_dict[animal_name]["X_bps"],
                            self.animal_bp_dict[animal_name]["Y_bps"],
                        )
                        for item in pair
                    ]
                )
                animal_body_part_names = [
                    x[:-2] for x in self.animal_bp_dict[animal_name]["X_bps"]
                ]
                animal_body_part_dict = {
                    k: v
                    for (v, k) in zip(
                        animal_body_part_names, range(0, len(animal_body_part_names))
                    )
                }
                animal_arr = self.data_df[animal_bp_headers].values.reshape(
                    len(self.data_df), -1, 2
                )
                data = self.three_dim_cdist(data=animal_arr)
                df = (
                    pd.concat(
                        [pd.DataFrame(arr) for arr in data], keys=np.arange(len(data))
                    )
                    .reset_index()
                    .rename(columns={"level_0": "frame", "level_1": "body_part"})
                )
                df["body_part"] = df["body_part"].map(animal_body_part_dict)
                above_criterion_dict = {}
                for bp, criterion_multiplier in animal_bps.items():
                    if animal_name in animal_movement_agg.keys():
                        bp_criterion = (
                            animal_movement_agg[animal_name] * criterion_multiplier
                        )
                        bp_df = df[df["body_part"] == bp]
                        bp_df["counts_above_criterion"] = self.count_values_in_range(
                            data=bp_df.drop(["frame", "body_part"], axis=1).values,
                            ranges=np.array([[bp_criterion, np.inf]]),
                        )
                        above_criterion_dict[bp] = list(
                            bp_df["frame"][bp_df["counts_above_criterion"] > 1]
                        )

                for (
                    bp,
                    body_part_frames_above_criterion,
                ) in above_criterion_dict.items():
                    self.results.loc[
                        body_part_frames_above_criterion, [f"{bp}_x", f"{bp}_y"]
                    ] = np.nan
                    self.results[[f"{bp}_x", f"{bp}_y"]] = (
                        self.results[[f"{bp}_x", f"{bp}_y"]].ffill().bfill().astype(int)
                    )
                    video_log[animal_name][bp] = {
                        "outlier_cnt": len(body_part_frames_above_criterion),
                        "outlier_ratio": len(body_part_frames_above_criterion)
                        / len(self.results),
                    }

            df_save_path = os.path.join(
                self.outlier_corrected_dir, self.video_name + f"{self.file_type}."
            )
            write_df(df=self.results, file_type=self.file_type, save_path=df_save_path)
            video_timer.stop_timer()
            stdout_success(
                msg=f"Location outliers complete for video {self.video_name}.",
                elapsed_time=video_timer.elapsed_time_str,
                source=self.__class__.__name__,
            )
            data_log[self.video_name] = video_log

        self._save_data_log(data_log=data_log)
        stdout_success(
            msg=f"{len(self.data_files)} video(s) corrected for location outliers. Saved in {self.outlier_corrected_dir}",
            source=self.__class__.__name__,
        )


# settings = {'Simon': 2.5} #
# # settings = {'Simon': {'Ear_left_1': 1.1,
# #                       'Ear_right_1': 5.1,
# #                       'Nose_1': 2.1,
# #                       'Center_1': 1.5,
# #                       'Lat_left_1': 3.1,
# #                       'Lat_right_1': 1.9,
# #                       'Tail_base_1': 2.3},
# #                       #'Tail_end_1': 1.4},
# #                'JJ': {'Ear_left_2': 1.2,
# #                       'Ear_right_2': 1.2,
# #                       'Nose_2': 2,
# #                       'Center_2': 4.1,
# #                       'Lat_left_2': 9,
# #                       'Lat_right_2': 1.2,
# #                       'Tail_base_2': 1.6,
# #                       'Tail_end_2': 2.2}}
# criterion_body_parts = {'Simon': ['Nose_1', 'Tail_base_1'], } #'JJ': ['Nose_2', 'Tail_base_2']
# test = OutlierCorrecterLocationAdvanced(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                         input_dir='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/csv/input_csv',
#                                         type='animal',
#                                         agg_method='mean',
#                                         criterion_body_parts=criterion_body_parts,
#                                         settings=settings)
# test.run()


# test = OutlierCorrecterLocation(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.correct_location_outliers()
