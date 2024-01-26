__author__ = "Simon Nilsson"

import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

from simba.feature_extractors.perimeter_jit import jitted_hull
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.enums import Formats
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df


class ExtractFeaturesFrom8bps2Animals(ConfigReader, FeatureExtractionMixin):
    """
    Extracts hard-coded set of features from pose-estimation data from two animals with 4 tracked body-parts each.
    Results are stored in the `project_folder/csv/features_extracted` directory of the SimBA project.

    :parameter str config_path: path to SimBA project config file in Configparser format

    .. note::
       `Feature extraction tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features>`__.
       `Expected pose configuration <https://github.com/sgoldenlab/simba/blob/master/simba/pose_configurations/schematics/5.png>`_

       .. image:: _static/img/pose_configurations/5.png
          :width: 150
          :align: center


    Examples
    ----------
    >>> feature_extractor = ExtractFeaturesFrom8bps2Animals(config_path='MyProjectConfig')
    >>> feature_extractor.run()
    """

    def __init__(self, config_path: str):
        FeatureExtractionMixin.__init__(self, config_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        self.in_headers = self.get_feature_extraction_headers(
            pose="2 animals 8 body-parts"
        )
        self.mouse_1_headers, self.mouse_2_headers = (
            self.in_headers[0:12],
            self.in_headers[12:],
        )
        self.mouse_2_p_headers = [x for x in self.mouse_2_headers if x[-2:] == "_p"]
        self.mouse_1_p_headers = [x for x in self.mouse_1_headers if x[-2:] == "_p"]
        self.mouse_1_headers = [x for x in self.mouse_1_headers if x[-2:] != "_p"]
        self.mouse_2_headers = [x for x in self.mouse_2_headers if x[-2:] != "_p"]
        print(
            "Extracting features from {} file(s)...".format(str(len(self.files_found)))
        )

    def run(self):
        """
        Method to compute and save feature battery to disk. Results are saved in the `project_folder/csv/features_extracted`
        directory of the SimBA project.

        Returns
        -------
        None
        """

        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            roll_windows = []
            _, self.video_name, _ = get_fn_ext(file_path)
            video_settings, self.px_per_mm, fps = self.read_video_info(
                video_name=self.video_name
            )
            for window in self.roll_windows_values:
                roll_windows.append(int(fps / window))
            self.in_data = (
                read_df(file_path, self.file_type)
                .fillna(0)
                .apply(pd.to_numeric)
                .reset_index(drop=True)
            )
            print(
                "Processing {} ({} frames)...".format(
                    self.video_name, str(len(self.in_data))
                )
            )
            self.in_data = self.insert_default_headers_for_feature_extraction(
                df=self.in_data,
                headers=self.in_headers,
                pose_config="8 body-parts from 2 animals",
                filename=file_path,
            )
            self.out_data = deepcopy(self.in_data)
            mouse_1_ar = np.reshape(
                self.out_data[self.mouse_1_headers].values,
                (len(self.out_data / 2), -1, 2),
            ).astype(np.float32)
            self.out_data["Mouse_1_poly_area"] = (
                jitted_hull(points=mouse_1_ar, target=Formats.PERIMETER.value)
                / self.px_per_mm
            )
            mouse_2_ar = np.reshape(
                self.out_data[self.mouse_2_headers].values,
                (len(self.out_data / 2), -1, 2),
            ).astype(np.float32)
            self.out_data["Mouse_2_poly_area"] = (
                jitted_hull(points=mouse_2_ar, target=Formats.PERIMETER.value)
                / self.px_per_mm
            )
            self.in_data_shifted = (
                self.out_data.shift(periods=1).add_suffix("_shifted").fillna(0)
            )
            self.in_data = (
                pd.concat([self.in_data, self.in_data_shifted], axis=1, join="inner")
                .fillna(0)
                .reset_index(drop=True)
            )
            self.out_data["Mouse_1_nose_to_tail"] = self.euclidean_distance(
                self.out_data["Nose_1_x"].values,
                self.out_data["Tail_base_1_x"].values,
                self.out_data["Nose_1_y"].values,
                self.out_data["Tail_base_1_y"].values,
                self.px_per_mm,
            )
            self.out_data["Mouse_2_nose_to_tail"] = self.euclidean_distance(
                self.out_data["Nose_2_x"].values,
                self.out_data["Tail_base_2_x"].values,
                self.out_data["Nose_2_y"].values,
                self.out_data["Tail_base_2_y"].values,
                self.px_per_mm,
            )
            self.out_data["Mouse_1_Ear_distance"] = self.euclidean_distance(
                self.out_data["Ear_left_1_x"].values,
                self.out_data["Ear_right_1_x"].values,
                self.out_data["Ear_left_1_y"].values,
                self.out_data["Ear_right_1_y"].values,
                self.px_per_mm,
            )
            self.out_data["Mouse_2_Ear_distance"] = self.euclidean_distance(
                self.out_data["Ear_left_2_x"].values,
                self.out_data["Ear_right_2_x"].values,
                self.out_data["Ear_left_2_y"].values,
                self.out_data["Ear_right_2_y"].values,
                self.px_per_mm,
            )
            self.out_data["Nose_to_nose_distance"] = self.euclidean_distance(
                self.out_data["Nose_2_x"].values,
                self.out_data["Nose_1_x"].values,
                self.out_data["Nose_2_y"].values,
                self.out_data["Nose_1_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_mouse_1_nose"] = self.euclidean_distance(
                self.in_data["Nose_1_x_shifted"].values,
                self.in_data["Nose_1_x"].values,
                self.in_data["Nose_1_y_shifted"].values,
                self.in_data["Nose_1_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_mouse_2_nose"] = self.euclidean_distance(
                self.in_data["Nose_2_x_shifted"].values,
                self.in_data["Nose_2_x"].values,
                self.in_data["Nose_2_y_shifted"].values,
                self.in_data["Nose_2_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_mouse_1_tail_base"] = self.euclidean_distance(
                self.in_data["Tail_base_1_x_shifted"].values,
                self.in_data["Tail_base_1_x"].values,
                self.in_data["Tail_base_1_y_shifted"].values,
                self.in_data["Tail_base_1_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_mouse_2_tail_base"] = self.euclidean_distance(
                self.in_data["Tail_base_2_x_shifted"].values,
                self.in_data["Tail_base_2_x"].values,
                self.in_data["Tail_base_2_y_shifted"].values,
                self.in_data["Tail_base_2_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_mouse_1_left_ear"] = self.euclidean_distance(
                self.in_data["Ear_left_1_x_shifted"].values,
                self.in_data["Ear_left_1_x"].values,
                self.in_data["Ear_left_1_y_shifted"].values,
                self.in_data["Ear_left_1_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_mouse_2_left_ear"] = self.euclidean_distance(
                self.in_data["Ear_left_2_x_shifted"].values,
                self.in_data["Ear_left_2_x"].values,
                self.in_data["Ear_left_2_y_shifted"].values,
                self.in_data["Ear_left_2_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_mouse_1_right_ear"] = self.euclidean_distance(
                self.in_data["Ear_right_1_x_shifted"].values,
                self.in_data["Ear_right_1_x"].values,
                self.in_data["Ear_right_1_y_shifted"].values,
                self.in_data["Ear_right_1_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_mouse_2_right_ear"] = self.euclidean_distance(
                self.in_data["Ear_right_2_x_shifted"].values,
                self.in_data["Ear_right_2_x"].values,
                self.in_data["Ear_right_2_y_shifted"].values,
                self.in_data["Ear_right_2_y"].values,
                self.px_per_mm,
            )
            self.out_data["Mouse_1_polygon_size_change"] = (
                self.in_data["Mouse_1_poly_area_shifted"]
                - self.out_data["Mouse_1_poly_area"]
            )
            self.out_data["Mouse_2_polygon_size_change"] = (
                self.in_data["Mouse_2_poly_area_shifted"]
                - self.out_data["Mouse_2_poly_area"]
            )

            print("Calculating hull variables...")
            self.hull_dict = defaultdict(list)
            mouse_1_array, mouse_2_array = (
                self.in_data[self.mouse_1_headers].to_numpy(),
                self.in_data[self.mouse_2_headers].to_numpy(),
            )
            for cnt, (animal_1, animal_2) in enumerate(
                zip(mouse_1_array, mouse_2_array)
            ):
                animal_1, animal_2 = np.reshape(animal_1, (-1, 2)), np.reshape(
                    animal_2, (-1, 2)
                )
                animal_1_dist, animal_2_dist = self.cdist(
                    animal_1, animal_1
                ), self.cdist(animal_2, animal_2)
                animal_1_dist, animal_2_dist = (
                    animal_1_dist[animal_1_dist != 0],
                    animal_2_dist[animal_2_dist != 0],
                )
                for animal, animal_name in zip(
                    [animal_1_dist, animal_2_dist], ["M1", "M2"]
                ):
                    self.hull_dict[
                        "{}_hull_large_euclidean".format(animal_name)
                    ].append(np.amax(animal, initial=0) / self.px_per_mm)
                    self.hull_dict[
                        "{}_hull_small_euclidean".format(animal_name)
                    ].append(
                        np.min(
                            animal,
                            initial=self.hull_dict[
                                "{}_hull_large_euclidean".format(animal_name)
                            ][-1],
                        )
                        / self.px_per_mm
                    )
                    self.hull_dict["{}_hull_mean_euclidean".format(animal_name)].append(
                        np.mean(animal) / self.px_per_mm
                    )
                    self.hull_dict["{}_hull_sum_euclidean".format(animal_name)].append(
                        np.sum(animal, initial=0) / self.px_per_mm
                    )

            for k, v in self.hull_dict.items():
                self.out_data[k] = v

            self.out_data["Sum_euclidean_distance_hull_M1_M2"] = (
                self.out_data["M1_hull_sum_euclidean"]
                + self.out_data["M2_hull_sum_euclidean"]
            )

            self.out_data["Total_movement_nose"] = self.out_data.eval(
                "Movement_mouse_1_nose + Movement_mouse_2_nose"
            )
            self.out_data["Total_movement_tail_base"] = self.out_data.eval(
                "Movement_mouse_1_tail_base + Movement_mouse_2_tail_base"
            )
            self.out_data["Total_movement_all_bodyparts_M1"] = self.out_data.eval(
                "Movement_mouse_1_nose + Movement_mouse_1_tail_base + Movement_mouse_1_left_ear + Movement_mouse_1_right_ear"
            )
            self.out_data["Total_movement_all_bodyparts_M2"] = self.out_data.eval(
                "Movement_mouse_2_nose + Movement_mouse_2_tail_base + Movement_mouse_2_left_ear + Movement_mouse_2_right_ear"
            )
            self.out_data["Total_movement_all_bodyparts_both_mice"] = (
                self.out_data.eval(
                    "Total_movement_all_bodyparts_M1 + Total_movement_all_bodyparts_M2"
                )
            )

            for window in self.roll_windows_values:
                col_name = "Sum_euclid_distances_hull_median_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Sum_euclidean_distance_hull_M1_M2"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Sum_euclid_distances_hull_mean_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Sum_euclidean_distance_hull_M1_M2"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Sum_euclid_distances_hull_sum_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Sum_euclidean_distance_hull_M1_M2"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            for window in self.roll_windows_values:
                col_name = "Movement_median_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Total_movement_all_bodyparts_both_mice"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Movement_mean_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Total_movement_all_bodyparts_both_mice"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Movement_sum_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Total_movement_all_bodyparts_both_mice"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            for window in self.roll_windows_values:
                col_name = "Distance_median_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Nose_to_nose_distance"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Distance_mean_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Nose_to_nose_distance"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Distance_sum_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Nose_to_nose_distance"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            for window in self.roll_windows_values:
                col_name = "Mouse1_mean_euclid_distances_median_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["M1_hull_mean_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Mouse1_mean_euclid_distances_mean_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["M1_hull_mean_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Mouse1_mean_euclid_distances_sum_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["M1_hull_mean_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            for window in self.roll_windows_values:
                col_name = "Mouse2_mean_euclid_distances_median_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["M2_hull_mean_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Mouse2_mean_euclid_distances_mean_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["M2_hull_mean_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Mouse2_mean_euclid_distances_sum_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["M2_hull_mean_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            for window in self.roll_windows_values:
                col_name = "Mouse1_smallest_euclid_distances_median_{}".format(
                    str(window)
                )
                self.out_data[col_name] = (
                    self.out_data["M1_hull_small_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Mouse1_smallest_euclid_distances_mean_{}".format(
                    str(window)
                )
                self.out_data[col_name] = (
                    self.out_data["M1_hull_small_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Mouse1_smallest_euclid_distances_sum_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["M1_hull_small_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            for window in self.roll_windows_values:
                col_name = "Mouse2_smallest_euclid_distances_median_{}".format(
                    str(window)
                )
                self.out_data[col_name] = (
                    self.out_data["M2_hull_small_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Mouse2_smallest_euclid_distances_mean_{}".format(
                    str(window)
                )
                self.out_data[col_name] = (
                    self.out_data["M2_hull_small_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Mouse2_smallest_euclid_distances_sum_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["M2_hull_small_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            for window in self.roll_windows_values:
                col_name = "Mouse1_largest_euclid_distances_median_{}".format(
                    str(window)
                )
                self.out_data[col_name] = (
                    self.out_data["M1_hull_large_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Mouse1_largest_euclid_distances_mean_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["M1_hull_large_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Mouse1_largest_euclid_distances_sum_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["M1_hull_large_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            for window in self.roll_windows_values:
                col_name = "Mouse2_largest_euclid_distances_median_{}".format(
                    str(window)
                )
                self.out_data[col_name] = (
                    self.out_data["M2_hull_large_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Mouse2_largest_euclid_distances_mean_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["M2_hull_large_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Mouse2_largest_euclid_distances_sum_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["M2_hull_large_euclidean"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            for window in self.roll_windows_values:
                col_name = "Total_movement_all_bodyparts_both_mice_median_{}".format(
                    str(window)
                )
                self.out_data[col_name] = (
                    self.out_data["Total_movement_all_bodyparts_both_mice"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Total_movement_all_bodyparts_both_mice_mean_{}".format(
                    str(window)
                )
                self.out_data[col_name] = (
                    self.out_data["Total_movement_all_bodyparts_both_mice"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Total_movement_all_bodyparts_both_mice_sum_{}".format(
                    str(window)
                )
                self.out_data[col_name] = (
                    self.out_data["Total_movement_all_bodyparts_both_mice"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            for window in self.roll_windows_values:
                col_name = "Tail_base_movement_M1_median_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Movement_mouse_1_tail_base"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Tail_base_movement_M1_mean_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Movement_mouse_1_tail_base"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Tail_base_movement_M1_sum_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Movement_mouse_1_tail_base"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            for window in self.roll_windows_values:
                col_name = "Tail_base_movement_M2_median_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Movement_mouse_2_tail_base"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Tail_base_movement_M2_mean_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Movement_mouse_2_tail_base"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Tail_base_movement_M2_sum_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Movement_mouse_2_tail_base"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            for window in self.roll_windows_values:
                col_name = "Nose_movement_M1_median_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Movement_mouse_1_nose"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Nose_movement_M1_mean_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Movement_mouse_1_nose"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Nose_movement_M1_sum_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Movement_mouse_1_nose"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            for window in self.roll_windows_values:
                col_name = "Nose_movement_M2_median_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Movement_mouse_2_nose"]
                    .rolling(int(window), min_periods=1)
                    .median()
                )
                col_name = "Nose_movement_M2_mean_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Movement_mouse_2_nose"]
                    .rolling(int(window), min_periods=1)
                    .mean()
                )
                col_name = "Nose_movement_M2_sum_{}".format(str(window))
                self.out_data[col_name] = (
                    self.out_data["Movement_mouse_2_nose"]
                    .rolling(int(window), min_periods=1)
                    .sum()
                )

            self.out_data["Total_movement_all_bodyparts_both_mice_deviation"] = (
                self.out_data["Total_movement_all_bodyparts_both_mice"].mean()
                - self.out_data["Total_movement_all_bodyparts_both_mice"]
            )
            self.out_data["Sum_euclid_distances_hull_deviation"] = (
                self.out_data["Sum_euclidean_distance_hull_M1_M2"].mean()
                - self.out_data["Sum_euclidean_distance_hull_M1_M2"]
            )
            self.out_data["M1_smallest_euclid_distances_hull_deviation"] = (
                self.out_data["M1_hull_small_euclidean"].mean()
                - self.out_data["M1_hull_small_euclidean"]
            )
            self.out_data["M1_largest_euclid_distances_hull_deviation"] = (
                self.out_data["M1_hull_large_euclidean"].mean()
                - self.out_data["M1_hull_large_euclidean"]
            )
            self.out_data["M1_mean_euclid_distances_hull_deviation"] = (
                self.out_data["M1_hull_mean_euclidean"].mean()
                - self.out_data["M1_hull_mean_euclidean"]
            )
            self.out_data["Movement_mouse_1_deviation_nose"] = (
                self.out_data["Movement_mouse_1_nose"].mean()
                - self.out_data["Movement_mouse_1_nose"]
            )
            self.out_data["Movement_mouse_2_deviation_nose"] = (
                self.out_data["Movement_mouse_2_nose"].mean()
                - self.out_data["Movement_mouse_2_nose"]
            )
            self.out_data["Mouse_1_polygon_deviation"] = (
                self.out_data["Mouse_1_poly_area"].mean()
                - self.out_data["Mouse_1_poly_area"]
            )
            self.out_data["Mouse_2_polygon_deviation"] = (
                self.out_data["Mouse_2_poly_area"].mean()
                - self.out_data["Mouse_2_poly_area"]
            )

            for window in self.roll_windows_values:
                col_name = "Total_movement_all_bodyparts_both_mice_mean_{}".format(
                    str(window)
                )
                deviation_col_name = col_name + "_deviation"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            for window in self.roll_windows_values:
                col_name = "Sum_euclid_distances_hull_mean_{}".format(str(window))
                deviation_col_name = col_name + "_deviation"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            for window in self.roll_windows_values:
                col_name = "Mouse1_smallest_euclid_distances_mean_{}".format(
                    str(window)
                )
                deviation_col_name = col_name + "_deviation"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            for window in self.roll_windows_values:
                col_name = "Mouse1_largest_euclid_distances_mean_{}".format(str(window))
                deviation_col_name = col_name + "_deviation"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            for window in self.roll_windows_values:
                col_name = "Mouse1_mean_euclid_distances_mean_{}".format(str(window))
                deviation_col_name = col_name + "_deviation"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            for window in self.roll_windows_values:
                col_name = "Movement_mean_{}".format(str(window))
                deviation_col_name = col_name + "_deviation"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            for window in self.roll_windows_values:
                col_name = "Distance_mean_{}".format(str(window))
                deviation_col_name = col_name + "_deviation"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            print("Calculating percentile ranks...")
            self.out_data["Movement_percentile_rank"] = self.out_data[
                "Total_movement_nose"
            ].rank(pct=True)
            self.out_data["Distance_percentile_rank"] = self.out_data[
                "Nose_to_nose_distance"
            ].rank(pct=True)
            self.out_data["Movement_mouse_1_percentile_rank"] = self.out_data[
                "Movement_mouse_1_nose"
            ].rank(pct=True)
            self.out_data["Movement_mouse_2_percentile_rank"] = self.out_data[
                "Movement_mouse_2_nose"
            ].rank(pct=True)
            self.out_data["Movement_mouse_1_deviation_percentile_rank"] = self.out_data[
                "Movement_mouse_1_deviation_nose"
            ].rank(pct=True)
            self.out_data["Movement_mouse_2_deviation_percentile_rank"] = self.out_data[
                "Movement_mouse_1_deviation_nose"
            ].rank(pct=True)

            for window in self.roll_windows_values:
                col_name = "Total_movement_all_bodyparts_both_mice_mean_{}".format(
                    str(window)
                )
                deviation_col_name = col_name + "_percentile_rank"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            for window in self.roll_windows_values:
                col_name = "Sum_euclid_distances_hull_mean_{}".format(str(window))
                deviation_col_name = col_name + "_percentile_rank"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            for window in self.roll_windows_values:
                col_name = "Mouse1_mean_euclid_distances_mean_{}".format(str(window))
                deviation_col_name = col_name + "_percentile_rank"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            for window in self.roll_windows_values:
                col_name = "Mouse1_smallest_euclid_distances_mean_{}".format(
                    str(window)
                )
                deviation_col_name = col_name + "_percentile_rank"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            for window in self.roll_windows_values:
                col_name = "Mouse1_largest_euclid_distances_mean_{}".format(str(window))
                deviation_col_name = col_name + "_percentile_rank"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            for window in self.roll_windows_values:
                col_name = "Movement_mean_{}".format(str(window))
                deviation_col_name = col_name + "_percentile_rank"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            for window in self.roll_windows_values:
                col_name = "Distance_mean_{}".format(str(window))
                deviation_col_name = col_name + "_percentile_rank"
                self.out_data[deviation_col_name] = (
                    self.out_data[col_name].mean() - self.out_data[col_name]
                )

            print("Calculating pose probability scores...")
            all_p_columns = self.mouse_2_p_headers + self.mouse_1_p_headers
            self.out_data["Sum_probabilities"] = self.out_data[all_p_columns].sum(
                axis=1
            )
            self.out_data["Sum_probabilities_deviation"] = (
                self.out_data["Sum_probabilities"].mean()
                - self.out_data["Sum_probabilities"]
            )
            self.out_data["Sum_probabilities_deviation_percentile_rank"] = (
                self.out_data["Sum_probabilities_deviation"].rank(pct=True)
            )
            self.out_data["Sum_probabilities_percentile_rank"] = self.out_data[
                "Sum_probabilities_deviation_percentile_rank"
            ].rank(pct=True)
            results = pd.DataFrame(
                self.count_values_in_range(
                    data=self.out_data.filter(all_p_columns).values,
                    ranges=np.array([[0.0, 0.1], [0.0, 0.5], [0.0, 0.75]]),
                ),
                columns=[
                    "Low_prob_detections_0.1",
                    "Low_prob_detections_0.5",
                    "Low_prob_detections_0.75",
                ],
            )
            self.out_data = pd.concat([self.out_data, results], axis=1)
            self.out_data = self.out_data.reset_index(drop=True).fillna(0)
            save_path = os.path.join(
                self.save_dir, self.video_name + "." + self.file_type
            )
            write_df(df=self.out_data, file_type=self.file_type, save_path=save_path)
            video_timer.stop_timer()
            print(
                f"Feature extraction complete for {self.video_name} ({file_cnt + 1}/{len(self.files_found)} (elapsed time: {video_timer.elapsed_time_str}s))..."
            )
        self.timer.stop_timer()
        stdout_success(
            msg="All features extracted. Results stored in project_folder/csv/features_extracted directory.",
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = ExtractFeaturesFrom8bps2Animals(config_path='/Users/simon/Desktop/envs/troubleshooting/8Bp_2_animals/project_folder/project_config.ini')
# test.run()
