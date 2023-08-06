__author__ = "Simon Nilsson"

import glob
import os

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.utils.enums import ConfigKey, Dtypes
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (get_fn_ext, read_config_entry, read_df,
                                    write_df)


class OutlierCorrecterLocation(ConfigReader):
    """
    Detect and amend outliers in pose-estimation data based in the location of the body-parts
    in the current frame relative to the location of the body-part in the preceding frame. Uses critera
    stored in the SimBA project project_config.ini under the [Outlier settings] header.

    .. note::
       `Documentation <https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf>`_.

    .. image:: _static/img/location_outlier.png
       :width: 500
       :align: center

    :parameter str config_path: path to SimBA project config file in Configparser format

    Examples
    ----------
    >>> _ = OutlierCorrecterLocation(config_path='MyProjectConfig').run()
    """

    def __init__(self, config_path: str):
        super().__init__(config_path=config_path)
        if not os.path.exists(self.outlier_corrected_dir):
            os.makedirs(self.outlier_corrected_dir)
        if self.animal_cnt == 1:
            self.animal_id = read_config_entry(
                self.config,
                ConfigKey.MULTI_ANIMAL_ID_SETTING.value,
                ConfigKey.MULTI_ANIMAL_IDS.value,
                Dtypes.STR.value,
            )
            if self.animal_id != "None":
                self.animal_bp_dict[self.animal_id] = self.animal_bp_dict.pop(
                    "Animal_1"
                )
        self.above_criterion_dict_dict = {}
        self.below_criterion_dict_dict = {}
        self.criterion = read_config_entry(
            self.config,
            ConfigKey.OUTLIER_SETTINGS.value,
            ConfigKey.LOCATION_CRITERION.value,
            Dtypes.FLOAT.value,
        )
        self.outlier_bp_dict = {}
        for animal_name in self.animal_bp_dict.keys():
            self.outlier_bp_dict[animal_name] = {}
            self.outlier_bp_dict[animal_name]["bp_1"] = read_config_entry(
                self.config,
                ConfigKey.OUTLIER_SETTINGS.value,
                "location_bodypart1_{}".format(animal_name.lower()),
                "str",
            )
            self.outlier_bp_dict[animal_name]["bp_2"] = read_config_entry(
                self.config,
                ConfigKey.OUTLIER_SETTINGS.value,
                "location_bodypart2_{}".format(animal_name.lower()),
                "str",
            )

    def __find_location_outliers(self):
        for animal_name, animal_data in self.bp_dict.items():
            animal_criterion = self.animal_criteria[animal_name]
            self.above_criterion_dict_dict[self.video_name][animal_name] = {}
            self.below_criterion_dict_dict[self.video_name][animal_name] = {}
            for body_part_name, body_part_data in animal_data.items():
                self.above_criterion_dict_dict[self.video_name][animal_name][
                    body_part_name
                ] = []
                self.below_criterion_dict_dict[self.video_name][animal_name][
                    body_part_name
                ] = []
                for frame in range(body_part_data.shape[0]):
                    second_bp_names = [
                        x for x in list(animal_data.keys()) if x != body_part_name
                    ]
                    first_bp_cord = body_part_data[frame]
                    distance_above_criterion_counter = 0
                    for second_bp in second_bp_names:
                        second_bp_cord = animal_data[second_bp][frame]
                        distance = np.sqrt(
                            (first_bp_cord[0] - second_bp_cord[0]) ** 2
                            + (first_bp_cord[1] - second_bp_cord[1]) ** 2
                        )
                        if distance > animal_criterion:
                            distance_above_criterion_counter += 1
                    if distance_above_criterion_counter > 1:
                        self.above_criterion_dict_dict[self.video_name][animal_name][
                            body_part_name
                        ].append(frame)
                    else:
                        self.below_criterion_dict_dict[self.video_name][animal_name][
                            body_part_name
                        ].append(frame)

    def __correct_outliers(self):
        above_citeria_dict = self.above_criterion_dict_dict[self.video_name]
        for animal_name, animal_bp_data in above_citeria_dict.items():
            for bp_name, outlier_idx_lst in animal_bp_data.items():
                body_part_x, body_part_y = bp_name + "_x", bp_name + "_y"
                for outlier_idx in outlier_idx_lst:
                    try:
                        closest_idx = max(
                            [
                                i
                                for i in self.below_criterion_dict_dict[
                                    self.video_name
                                ][animal_name][bp_name]
                                if outlier_idx > i
                            ]
                        )
                    except ValueError:
                        closest_idx = outlier_idx
                    self.data_df.loc[[outlier_idx], body_part_x] = self.data_df.loc[
                        [closest_idx], body_part_x
                    ].values[0]
                    self.data_df.loc[[outlier_idx], body_part_y] = self.data_df.loc[
                        [closest_idx], body_part_y
                    ].values[0]

    def run(self):
        """
        Runs outlier detection and correction. Results are stored in the
        ``project_folder/csv/outlier_corrected_movement_location`` directory of the SimBA project.
        """

        for file_cnt, file_path in enumerate(self.outlier_corrected_movement_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(file_path)
            print(
                f"Processing video {self.video_name}. Video {file_cnt+1}/{len(self.outlier_corrected_movement_paths)}.."
            )
            self.above_criterion_dict_dict[self.video_name] = {}
            self.below_criterion_dict_dict[self.video_name] = {}
            save_path = os.path.join(
                self.outlier_corrected_dir, self.video_name + "." + self.file_type
            )
            self.data_df = read_df(file_path, self.file_type)
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
                self.animal_criteria[animal_name] = (
                    animal_bp_distances.mean() * self.criterion
                )
            self.bp_dict = {}
            for animal_name, animal_bps in self.animal_bp_dict.items():
                bp_col_names = np.array(
                    [[i, j] for i, j in zip(animal_bps["X_bps"], animal_bps["Y_bps"])]
                ).ravel()
                animal_arr = self.data_df[bp_col_names].to_numpy()
                self.bp_dict[animal_name] = {}
                for bp_cnt, bp_col_start in enumerate(range(0, animal_arr.shape[1], 2)):
                    bp_name = animal_bps["X_bps"][bp_cnt][:-2]
                    self.bp_dict[animal_name][bp_name] = animal_arr[
                        :, bp_col_start : bp_col_start + 2
                    ]
            self.__find_location_outliers()
            self.__correct_outliers()
            write_df(df=self.data_df, file_type=self.file_type, save_path=save_path)
            video_timer.stop_timer()
            print(
                f"Corrected location outliers for file {self.video_name} (elapsed time: {video_timer.elapsed_time_str}s)..."
            )
        self.__save_log_file()

    def __save_log_file(self):
        out_df_lst = []
        for video_name, video_data in self.above_criterion_dict_dict.items():
            for animal_name, animal_data in video_data.items():
                for bp_name, vid_idx_lst in animal_data.items():
                    correction_ratio = round(len(vid_idx_lst) / len(self.data_df), 6)
                    out_df_lst.append(
                        pd.DataFrame(
                            [
                                [
                                    video_name,
                                    animal_name,
                                    bp_name,
                                    len(vid_idx_lst),
                                    correction_ratio,
                                ]
                            ],
                            columns=[
                                "Video",
                                "Animal",
                                "Body-part",
                                "Corrections",
                                "Correction ratio (%)",
                            ],
                        )
                    )
        out_df = pd.concat(out_df_lst, axis=0).reset_index(drop=True)
        self.logs_path = os.path.join(
            self.logs_path, f"Outliers_location_{self.datetime}.csv"
        )
        out_df.to_csv(self.logs_path)
        self.timer.stop_timer()
        stdout_success(
            msg='Log for corrected "location outliers" saved in project_folder/logs',
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = OutlierCorrecterLocation(config_path='/Users/simon/Desktop/envs/troubleshooting/naresh/project_folder/project_config.ini')
# test.run()

# test = OutlierCorrecterLocation(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.correct_location_outliers()
