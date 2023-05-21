__author__ = "Simon Nilsson"


import os
from itertools import product

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import check_str
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df


class UserDefinedFeatureExtractor(ConfigReader, FeatureExtractionMixin):
    """
    Generic featurizer of data within SimBA project using user-defined body-parts in the pose-estimation data.
    Results are stored in the `project_folder/csv/features_extracted` directory of the SimBA project.

    :parameter str config_path: path to SimBA project config file in Configparser format

    .. note::
       `Feature extraction tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features>`__.

    Examples
    ----------
    >>> feature_extractor = UserDefinedFeatureExtractor(config_path='MyProjectConfig')
    >>> feature_extractor.run()
    """

    def __init__(self, config_path: str):
        FeatureExtractionMixin.__init__(self, config_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        print(
            "Extracting features from {} file(s)...".format(str(len(self.files_found)))
        )

    def __euclid_dist_between_bps_of_other_animals(self):
        print("Calculating euclidean distances...")
        self.distance_col_names = []
        for animal_name, animal_data in self.animal_bp_dict.items():
            current_animal_bp_xs, current_animal_bp_ys = (
                animal_data["X_bps"],
                animal_data["Y_bps"],
            )
            other_animals = {
                i: self.animal_bp_dict[i]
                for i in self.animal_bp_dict
                if i != animal_name
            }
            for current_animal_bp_x, current_animal_bp_y in zip(
                current_animal_bp_xs, current_animal_bp_ys
            ):
                for other_animal_name, other_animal_data in other_animals.items():
                    other_animal_bp_xs, other_animal_bp_ys = (
                        other_animal_data["X_bps"],
                        other_animal_data["Y_bps"],
                    )
                    for other_animal_bp_x, other_animal_bp_y in zip(
                        other_animal_bp_xs, other_animal_bp_ys
                    ):
                        current_bp_name, other_bp_name = current_animal_bp_x.strip(
                            "_x"
                        ), other_animal_bp_x.strip("_x")
                        col_name = "Euclidean_distance_{}_{}".format(
                            current_bp_name, other_bp_name
                        )
                        reverse_col_name = "Euclidean_distance_{}_{}".format(
                            other_bp_name, current_bp_name
                        )
                        if not reverse_col_name in self.data_df.columns:
                            self.data_df[col_name] = (
                                np.sqrt(
                                    (
                                        self.data_df[current_animal_bp_x]
                                        - self.data_df[other_animal_bp_x]
                                    )
                                    ** 2
                                    + (
                                        self.data_df[current_animal_bp_y]
                                        - self.data_df[other_animal_bp_y]
                                    )
                                    ** 2
                                )
                            ) / self.px_per_mm
                            self.distance_col_names.append(col_name)

    def __movement_of_all_bps(self):
        print("Calculating movements of all body-parts...")
        self.mean_movement_cols, self.sum_movement_cols = [], []
        for animal_name, animal_data in self.animal_bp_dict.items():
            animal_cols = []
            current_animal_bp_xs, current_animal_bp_ys = (
                animal_data["X_bps"],
                animal_data["Y_bps"],
            )
            for bp_x, bp_y in zip(current_animal_bp_xs, current_animal_bp_ys):
                shifted_bp_x, shifted_bp_y = bp_x + "_shifted", bp_y + "_shifted"
                col_name = "Movement_" + bp_x.strip("_x")
                self.data_df[col_name] = (
                    np.sqrt(
                        (self.data_df_comb[bp_x] - self.data_df_comb[shifted_bp_x]) ** 2
                        + (self.data_df_comb[bp_y] - self.data_df_comb[shifted_bp_y])
                        ** 2
                    )
                ) / self.px_per_mm
                animal_cols.append(col_name)
            self.data_df["All_bp_movements_" + animal_name + "_sum"] = self.data_df[
                animal_cols
            ].sum(axis=1)
            self.data_df["All_bp_movements_" + animal_name + "_mean"] = self.data_df[
                animal_cols
            ].mean(axis=1)
            self.data_df["All_bp_movements_" + animal_name + "_min"] = self.data_df[
                animal_cols
            ].min(axis=1)
            self.data_df["All_bp_movements_" + animal_name + "_max"] = self.data_df[
                animal_cols
            ].max(axis=1)
            self.mean_movement_cols.append("All_bp_movements_" + animal_name + "_mean")
            self.sum_movement_cols.append("All_bp_movements_" + animal_name + "_sum")

    def __rolling_windows_bp_distances(self):
        print("Calculating rolling windows data: distances between body-parts...")
        for i in product(self.roll_windows_values, self.distance_col_names):
            self.data_df["Mean_{}_{}".format(i[1], i[0])] = (
                self.data_df[i[1]].rolling(int(i[0]), min_periods=1).mean()
            )
            self.data_df["Sum_{}_{}".format(i[1], i[0])] = (
                self.data_df[i[1]].rolling(int(i[0]), min_periods=1).sum()
            )

    def __rolling_windows_movement(self):
        print("Calculating rolling windows data: animal movements...")
        for i in product(self.roll_windows_values, self.mean_movement_cols):
            self.data_df["Mean_{}_{}".format(i[1], i[0])] = (
                self.data_df[i[1]].rolling(int(i[0]), min_periods=1).mean()
            )
            self.data_df["Sum_{}_{}".format(i[1], i[0])] = (
                self.data_df[i[1]].rolling(int(i[0]), min_periods=1).sum()
            )

    def __pose_probability_filters(self):
        p_df = self.data_df.filter(self.p_cols, axis=1)
        self.data_df["Sum_probabilities"] = p_df.sum(axis=1)
        self.data_df["Mean_probabilities"] = p_df.mean(axis=1)
        results = pd.DataFrame(
            self.count_values_in_range(
                data=self.data_df.filter(self.p_cols).values,
                ranges=np.array([[0.0, 0.1], [0.0, 0.5], [0.0, 0.75]]),
            ),
            columns=[
                "Low_prob_detections_0.1",
                "Low_prob_detections_0.5",
                "Low_prob_detections_0.75",
            ],
        )
        self.data_df = pd.concat([self.data_df, results], axis=1)

    def run(self):
        """
        Method to compute and save features to disk. Results are saved in the `project_folder/csv/features_extracted`
        directory of the SimBA project.

        Returns
        -------
        None
        """

        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            print(
                "Extracting features for video {}/{}...".format(
                    str(file_cnt + 1), str(len(self.files_found))
                )
            )
            _, file_name, _ = get_fn_ext(file_path)
            check_str("file name", file_name)
            video_settings, self.px_per_mm, fps = self.read_video_info(
                video_name=file_name
            )
            roll_windows = []
            for i in range(len(self.roll_windows_values)):
                roll_windows.append(int(fps / self.roll_windows_values[i]))
            self.data_df = read_df(file_path, self.file_type)
            self.data_df.columns = self.col_headers
            self.data_df = self.data_df.fillna(0).apply(pd.to_numeric)
            self.data_df_shifted = self.data_df.shift(periods=1)
            self.data_df_shifted.columns = self.col_headers_shifted
            self.data_df_comb = (
                pd.concat([self.data_df, self.data_df_shifted], axis=1, join="inner")
                .fillna(0)
                .reset_index(drop=True)
            )
            self.__euclid_dist_between_bps_of_other_animals()
            self.__movement_of_all_bps()
            self.__rolling_windows_bp_distances()
            self.__rolling_windows_movement()
            self.__pose_probability_filters()
            save_path = os.path.join(self.save_dir, file_name + "." + self.file_type)
            self.data_df = self.data_df.reset_index(drop=True).fillna(0)
            write_df(df=self.data_df, file_type=self.file_type, save_path=save_path)
            video_timer.stop_timer()
            print(
                f"Feature extraction complete for video {file_name} (elapsed time: {video_timer.elapsed_time_str}s)"
            )

        self.timer.stop_timer()
        stdout_success(
            f"Feature extraction complete for {str(len(self.files_found))} video(s). Results are saved inside the project_folder/csv/features_extracted directory",
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = UserDefinedFeatureExtractor(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.run()
