__author__ = "Simon Nilsson"

import glob
import math
import os
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd

from simba.feature_extractors.perimeter_jit import jitted_hull
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df


class ExtractFeaturesFrom9bps(ConfigReader, FeatureExtractionMixin):
    """
    Extracts hard-coded set of features from pose-estimation data from single animals with 9 tracked body-parts.
    Results are stored in the `project_folder/csv/features_extracted` directory of the SimBA project.

    :parameter str config_path: path to SimBA project config file in Configparser format

    .. note::
       `Feature extraction tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features>`__.
       `Expected pose configuration <https://github.com/sgoldenlab/simba/blob/master/simba/pose_configurations/schematics/4.png>`_

       .. image:: _static/img/pose_configurations/4.png
          :width: 150
          :align: center


    Examples
    ----------
    >>> feature_extractor = ExtractFeaturesFrom9bps(config_path='MyProjectConfig')
    >>> feature_extractor.run()
    """

    def __init__(self, config_path: str):
        FeatureExtractionMixin.__init__(self, config_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        self.in_headers = self.get_feature_extraction_headers(
            pose="1 animal 9 body-parts"
        )
        self.mouse_p_headers = [x for x in self.in_headers if x[-2:] == "_p"]
        self.mouse_headers = [x for x in self.in_headers if x[-2:] != "_p"]

    def run(self):
        for file_cnt, file_path in enumerate(self.outlier_corrected_paths):
            video_timer = SimbaTimer(start=True)
            roll_windows = []
            _, self.video_name, _ = get_fn_ext(file_path)
            print(f"Extracting features for video {self.video_name}...")
            video_settings, self.px_per_mm, fps = self.read_video_info(
                video_name=self.video_name
            )
            for window in self.roll_windows_values:
                roll_windows.append(int(fps / window))
            self.roll_windows_values = [int(x) for x in self.roll_windows_values]
            self.in_data = (
                read_df(file_path, self.file_type)
                .fillna(0)
                .apply(pd.to_numeric)
                .reset_index(drop=True)
            )
            self.in_data = self.insert_default_headers_for_feature_extraction(
                df=self.in_data,
                headers=self.in_headers,
                pose_config="1 animal 9 body-parts",
                filename=file_path,
            )
            self.out_data = deepcopy(self.in_data)
            self.in_data_shifted = (
                self.out_data.shift(periods=1).add_suffix("_shifted").fillna(0)
            )
            self.in_data = (
                pd.concat([self.in_data, self.in_data_shifted], axis=1, join="inner")
                .fillna(0)
                .reset_index(drop=True)
            )
            mouse_arr = np.reshape(
                self.out_data[self.mouse_headers].values,
                (len(self.out_data / 2), -1, 2),
            ).astype(np.float32)
            print("Calculating hull features...")
            self.out_data["Mouse_poly_area"] = (
                jitted_hull(points=mouse_arr, target="perimeter") / self.px_per_mm
            )

            print("Calculating Euclidean distances... ")
            self.out_data["Nose_to_tail"] = self.euclidean_distance(
                self.in_data["Mouse1_nose_x"].values,
                self.in_data["Mouse1_tail_x"].values,
                self.in_data["Mouse1_nose_y"].values,
                self.in_data["Mouse1_tail_y"].values,
                self.px_per_mm,
            )
            self.out_data["Distance_feet"] = self.euclidean_distance(
                self.in_data["Mouse1_left_foot_x"].values,
                self.in_data["Mouse1_right_foot_x"].values,
                self.in_data["Mouse1_left_foot_y"].values,
                self.in_data["Mouse1_right_foot_y"].values,
                self.px_per_mm,
            )
            self.out_data["Distance_hands"] = self.euclidean_distance(
                self.in_data["Mouse1_left_hand_x"].values,
                self.in_data["Mouse1_right_hand_x"].values,
                self.in_data["Mouse1_left_hand_y"].values,
                self.in_data["Mouse1_right_hand_y"].values,
                self.px_per_mm,
            )
            self.out_data["Distance_ears"] = self.euclidean_distance(
                self.in_data["Mouse1_left_ear_x"].values,
                self.in_data["Mouse1_right_ear_x"].values,
                self.in_data["Mouse1_left_ear_y"].values,
                self.in_data["Mouse1_right_ear_y"].values,
                self.px_per_mm,
            )

            self.out_data["Distance_unilateral_left_hands_feet"] = (
                self.euclidean_distance(
                    self.in_data["Mouse1_left_foot_x"].values,
                    self.in_data["Mouse1_left_hand_x"].values,
                    self.in_data["Mouse1_left_foot_y"].values,
                    self.in_data["Mouse1_left_hand_y"].values,
                    self.px_per_mm,
                )
            )
            self.out_data["Distance_unilateral_right_hands_feet"] = (
                self.euclidean_distance(
                    self.in_data["Mouse1_right_foot_x"].values,
                    self.in_data["Mouse1_right_hand_x"].values,
                    self.in_data["Mouse1_right_foot_y"].values,
                    self.in_data["Mouse1_right_hand_y"].values,
                    self.px_per_mm,
                )
            )
            self.out_data["Distance_bilateral_left_foot_right_hand"] = (
                self.euclidean_distance(
                    self.in_data["Mouse1_left_foot_x"].values,
                    self.in_data["Mouse1_right_hand_x"].values,
                    self.in_data["Mouse1_left_foot_y"].values,
                    self.in_data["Mouse1_right_hand_y"].values,
                    self.px_per_mm,
                )
            )
            self.out_data["Distance_bilateral_right_foot_left_hand"] = (
                self.euclidean_distance(
                    self.in_data["Mouse1_right_foot_x"].values,
                    self.in_data["Mouse1_left_hand_x"].values,
                    self.in_data["Mouse1_right_foot_y"].values,
                    self.in_data["Mouse1_left_hand_y"].values,
                    self.px_per_mm,
                )
            )
            self.out_data["Distance_back_tail"] = self.euclidean_distance(
                self.in_data["Mouse1_back_x"].values,
                self.in_data["Mouse1_tail_x"].values,
                self.in_data["Mouse1_back_y"].values,
                self.in_data["Mouse1_tail_y"].values,
                self.px_per_mm,
            )
            self.out_data["Distance_back_nose"] = self.euclidean_distance(
                self.in_data["Mouse1_back_x"].values,
                self.in_data["Mouse1_nose_x"].values,
                self.in_data["Mouse1_back_y"].values,
                self.in_data["Mouse1_nose_y"].values,
                self.px_per_mm,
            )

            self.out_data["Movement_nose"] = self.euclidean_distance(
                self.in_data["Mouse1_nose_x_shifted"].values,
                self.in_data["Mouse1_nose_x"].values,
                self.in_data["Mouse1_nose_y_shifted"].values,
                self.in_data["Mouse1_nose_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_back"] = self.euclidean_distance(
                self.in_data["Mouse1_back_x_shifted"].values,
                self.in_data["Mouse1_back_x"].values,
                self.in_data["Mouse1_back_y_shifted"].values,
                self.in_data["Mouse1_back_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_left_ear"] = self.euclidean_distance(
                self.in_data["Mouse1_left_ear_x_shifted"].values,
                self.in_data["Mouse1_left_ear_x"].values,
                self.in_data["Mouse1_left_ear_y_shifted"].values,
                self.in_data["Mouse1_left_ear_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_right_ear"] = self.euclidean_distance(
                self.in_data["Mouse1_right_ear_x_shifted"].values,
                self.in_data["Mouse1_right_ear_x"].values,
                self.in_data["Mouse1_right_ear_y_shifted"].values,
                self.in_data["Mouse1_right_ear_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_left_foot"] = self.euclidean_distance(
                self.in_data["Mouse1_left_foot_x_shifted"].values,
                self.in_data["Mouse1_left_foot_x"].values,
                self.in_data["Mouse1_left_foot_y_shifted"].values,
                self.in_data["Mouse1_left_foot_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_right_foot"] = self.euclidean_distance(
                self.in_data["Mouse1_right_foot_x_shifted"].values,
                self.in_data["Mouse1_right_foot_x"].values,
                self.in_data["Mouse1_right_foot_y_shifted"].values,
                self.in_data["Mouse1_right_foot_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_tail"] = self.euclidean_distance(
                self.in_data["Mouse1_tail_x_shifted"].values,
                self.in_data["Mouse1_tail_x"].values,
                self.in_data["Mouse1_tail_y_shifted"].values,
                self.in_data["Mouse1_tail_y"].values,
                self.px_per_mm,
            )

            self.out_data["Movement_left_hand"] = self.euclidean_distance(
                self.in_data["Mouse1_left_hand_x_shifted"].values,
                self.in_data["Mouse1_left_hand_x"].values,
                self.in_data["Mouse1_left_hand_y_shifted"].values,
                self.in_data["Mouse1_left_hand_y"].values,
                self.px_per_mm,
            )
            self.out_data["Movement_right_hand"] = self.euclidean_distance(
                self.in_data["Mouse1_right_hand_x_shifted"].values,
                self.in_data["Mouse1_right_hand_x"].values,
                self.in_data["Mouse1_right_hand_y_shifted"].values,
                self.in_data["Mouse1_right_hand_y"].values,
                self.px_per_mm,
            )
            self.out_data["Mouse_polygon_size_change"] = (
                self.out_data["Mouse_poly_area"].shift(periods=1)
                - self.out_data["Mouse_poly_area"]
            )

            mouse_array = self.in_data[self.mouse_headers].to_numpy()
            self.hull_dict = defaultdict(list)
            for cnt, animal_frm in enumerate(mouse_array):
                animal_frm = np.reshape(animal_frm, (-1, 2))
                animal_dists = self.cdist(animal_frm, animal_frm)
                animal_dists = animal_dists[animal_dists != 0]
                self.hull_dict["Largest_euclidean_distance_hull"].append(
                    np.amax(animal_dists, initial=0) / self.px_per_mm
                )
                self.hull_dict["Smallest_euclidean_distance_hull"].append(
                    np.min(animal_dists, initial=0) / self.px_per_mm
                )
                self.hull_dict["Mean_euclidean_distance_hull"].append(
                    np.mean(animal_dists) / self.px_per_mm
                )
                self.hull_dict["Sum_euclidean_distance_hull"].append(
                    np.sum(animal_dists) / self.px_per_mm
                )
            for k, v in self.hull_dict.items():
                self.out_data[k] = v

            self.out_data["Total_movement_all_bodyparts"] = (
                self.out_data["Movement_nose"]
                + self.out_data["Movement_back"]
                + self.out_data["Movement_left_ear"]
                + self.out_data["Movement_right_ear"]
                + self.out_data["Movement_left_foot"]
                + self.out_data["Movement_right_foot"]
                + self.out_data["Movement_tail"]
                + self.out_data["Movement_left_hand"]
                + self.out_data["Movement_right_hand"]
            )

            print("Calculating rolling windows features...")
            for i in self.roll_windows_values:
                self.out_data[f"Nose_to_tail_median_{i}"] = (
                    self.out_data["Nose_to_tail"].rolling(i, min_periods=1).median()
                )
                self.out_data[f"Nose_to_tail_mean_{i}"] = (
                    self.out_data["Nose_to_tail"].rolling(i, min_periods=1).mean()
                )
                self.out_data[f"Nose_to_tail_sum_{i}"] = (
                    self.out_data["Nose_to_tail"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Distance_feet_median_{i}"] = (
                    self.out_data["Distance_feet"].rolling(i, min_periods=1).median()
                )
                self.out_data[f"Distance_feet_mean_{i}"] = (
                    self.out_data["Distance_feet"].rolling(i, min_periods=1).mean()
                )
                self.out_data[f"Distance_feet_sum_{i}"] = (
                    self.out_data["Distance_feet"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Distance_ears_median_{i}"] = (
                    self.out_data["Distance_ears"].rolling(i, min_periods=1).median()
                )
                self.out_data[f"Distance_ears_mean_{i}"] = (
                    self.out_data["Distance_ears"].rolling(i, min_periods=1).mean()
                )
                self.out_data[f"Distance_ears_sum_{i}"] = (
                    self.out_data["Distance_ears"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Distance_unilateral_left_hands_feet_median_{i}"] = (
                    self.out_data["Distance_unilateral_left_hands_feet"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Distance_unilateral_left_hands_feet_mean_{i}"] = (
                    self.out_data["Distance_unilateral_left_hands_feet"]
                    .rolling(i, min_periods=1)
                    .mean()
                )
                self.out_data[f"Distance_unilateral_left_hands_feet_sum_{i}"] = (
                    self.out_data["Distance_unilateral_left_hands_feet"]
                    .rolling(i, min_periods=1)
                    .sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Distance_unilateral_right_hands_feet_median_{i}"] = (
                    self.out_data["Distance_unilateral_right_hands_feet"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Distance_unilateral_right_hands_feet_mean_{i}"] = (
                    self.out_data["Distance_unilateral_right_hands_feet"]
                    .rolling(i, min_periods=1)
                    .mean()
                )
                self.out_data[f"Distance_unilateral_right_hands_feet_sum_{i}"] = (
                    self.out_data["Distance_unilateral_right_hands_feet"]
                    .rolling(i, min_periods=1)
                    .sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Distance_bilateral_left_foot_right_hand_median_{i}"] = (
                    self.out_data["Distance_bilateral_left_foot_right_hand"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Distance_bilateral_left_foot_right_hand_mean_{i}"] = (
                    self.out_data["Distance_bilateral_left_foot_right_hand"]
                    .rolling(i, min_periods=1)
                    .mean()
                )
                self.out_data[f"Distance_bilateral_left_foot_right_hand_sum_{i}"] = (
                    self.out_data["Distance_bilateral_left_foot_right_hand"]
                    .rolling(i, min_periods=1)
                    .sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Distance_bilateral_right_foot_left_hand_median_{i}"] = (
                    self.out_data["Distance_bilateral_right_foot_left_hand"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Distance_bilateral_right_foot_left_hand_mean_{i}"] = (
                    self.out_data["Distance_bilateral_right_foot_left_hand"]
                    .rolling(i, min_periods=1)
                    .mean()
                )
                self.out_data[f"Distance_bilateral_right_foot_left_hand_sum_{i}"] = (
                    self.out_data["Distance_bilateral_right_foot_left_hand"]
                    .rolling(i, min_periods=1)
                    .sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Distance_back_tail_median_{i}"] = (
                    self.out_data["Distance_back_tail"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Distance_back_tail_mean_{i}"] = (
                    self.out_data["Distance_back_tail"].rolling(i, min_periods=1).mean()
                )
                self.out_data[f"Distance_back_tail_sum_{i}"] = (
                    self.out_data["Distance_back_tail"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Distance_back_nose_median_{i}"] = (
                    self.out_data["Distance_back_nose"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Distance_back_nose_mean_{i}"] = (
                    self.out_data["Distance_back_nose"].rolling(i, min_periods=1).mean()
                )
                self.out_data[f"Distance_back_nose_sum_{i}"] = (
                    self.out_data["Distance_back_nose"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Movement_nose_median_{i}"] = (
                    self.out_data["Movement_nose"].rolling(i, min_periods=1).median()
                )
                self.out_data[f"Movement_nose_mean_{i}"] = (
                    self.out_data["Movement_nose"].rolling(i, min_periods=1).mean()
                )
                self.out_data[f"Movement_nose_sum_{i}"] = (
                    self.out_data["Movement_nose"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Movement_back_median_{i}"] = (
                    self.out_data["Movement_back"].rolling(i, min_periods=1).median()
                )
                self.out_data[f"Movement_back_mean_{i}"] = (
                    self.out_data["Movement_back"].rolling(i, min_periods=1).mean()
                )
                self.out_data[f"Movement_back_sum_{i}"] = (
                    self.out_data["Movement_back"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Movement_left_ear_median_{i}"] = (
                    self.out_data["Movement_left_ear"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Movement_left_ear_mean_{i}"] = (
                    self.out_data["Movement_left_ear"].rolling(i, min_periods=1).mean()
                )
                self.out_data[f"Movement_left_ear_sum_{i}"] = (
                    self.out_data["Movement_left_ear"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Movement_right_ear_median_{i}"] = (
                    self.out_data["Movement_right_ear"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Movement_right_ear_mean_{i}"] = (
                    self.out_data["Movement_right_ear"].rolling(i, min_periods=1).mean()
                )
                self.out_data[f"Movement_right_ear_sum_{i}"] = (
                    self.out_data["Movement_right_ear"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Movement_left_foot_median_{i}"] = (
                    self.out_data["Movement_left_foot"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Movement_left_foot_mean_{i}"] = (
                    self.out_data["Movement_left_foot"].rolling(i, min_periods=1).mean()
                )
                self.out_data[f"Movement_left_foot_sum_{i}"] = (
                    self.out_data["Movement_left_foot"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Movement_right_foot_median_{i}"] = (
                    self.out_data["Movement_right_foot"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Movement_right_foot_mean_{i}"] = (
                    self.out_data["Movement_right_foot"]
                    .rolling(i, min_periods=1)
                    .mean()
                )
                self.out_data[f"Movement_right_foot_sum_{i}"] = (
                    self.out_data["Movement_right_foot"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Movement_right_foot_median_{i}"] = (
                    self.out_data["Movement_right_foot"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Movement_right_foot_mean_{i}"] = (
                    self.out_data["Movement_right_foot"]
                    .rolling(i, min_periods=1)
                    .mean()
                )
                self.out_data[f"Movement_right_foot_sum_{i}"] = (
                    self.out_data["Movement_right_foot"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Movement_tail_median_{i}"] = (
                    self.out_data["Movement_tail"].rolling(i, min_periods=1).median()
                )
                self.out_data[f"Movement_tail_mean_{i}"] = (
                    self.out_data["Movement_tail"].rolling(i, min_periods=1).mean()
                )
                self.out_data[f"Movement_tail_sum_{i}"] = (
                    self.out_data["Movement_tail"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Movement_left_hand_median_{i}"] = (
                    self.out_data["Movement_left_hand"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Movement_left_hand_mean_{i}"] = (
                    self.out_data["Movement_left_hand"].rolling(i, min_periods=1).mean()
                )
                self.out_data[f"Movement_left_hand_sum_{i}"] = (
                    self.out_data["Movement_left_hand"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Movement_right_hand_median_{i}"] = (
                    self.out_data["Movement_right_hand"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Movement_right_hand_mean_{i}"] = (
                    self.out_data["Movement_right_hand"]
                    .rolling(i, min_periods=1)
                    .mean()
                )
                self.out_data[f"Movement_right_hand_sum_{i}"] = (
                    self.out_data["Movement_right_hand"].rolling(i, min_periods=1).sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Total_movement_all_bodyparts_median_{i}"] = (
                    self.out_data["Total_movement_all_bodyparts"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Total_movement_all_bodyparts_mean_{i}"] = (
                    self.out_data["Total_movement_all_bodyparts"]
                    .rolling(i, min_periods=1)
                    .mean()
                )
                self.out_data[f"Total_movement_all_bodyparts_sum_{i}"] = (
                    self.out_data["Total_movement_all_bodyparts"]
                    .rolling(i, min_periods=1)
                    .sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Mean_euclidean_distance_hull_median_{i}"] = (
                    self.out_data["Mean_euclidean_distance_hull"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Mean_euclidean_distance_hull_mean_{i}"] = (
                    self.out_data["Mean_euclidean_distance_hull"]
                    .rolling(i, min_periods=1)
                    .mean()
                )
                self.out_data[f"Mean_euclidean_distance_hull_sum_{i}"] = (
                    self.out_data["Mean_euclidean_distance_hull"]
                    .rolling(i, min_periods=1)
                    .sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Smallest_euclidean_distance_hull_median_{i}"] = (
                    self.out_data["Smallest_euclidean_distance_hull"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Smallest_euclidean_distance_hull_mean_{i}"] = (
                    self.out_data["Smallest_euclidean_distance_hull"]
                    .rolling(i, min_periods=1)
                    .mean()
                )
                self.out_data[f"Smallest_euclidean_distance_hull_sum_{i}"] = (
                    self.out_data["Smallest_euclidean_distance_hull"]
                    .rolling(i, min_periods=1)
                    .sum()
                )

            for i in self.roll_windows_values:
                self.out_data[f"Largest_euclidean_distance_hull_median_{i}"] = (
                    self.out_data["Largest_euclidean_distance_hull"]
                    .rolling(i, min_periods=1)
                    .median()
                )
                self.out_data[f"Largest_euclidean_distance_hull_mean_{i}"] = (
                    self.out_data["Largest_euclidean_distance_hull"]
                    .rolling(i, min_periods=1)
                    .mean()
                )
                self.out_data[f"Largest_euclidean_distance_hull_sum_{i}"] = (
                    self.out_data["Largest_euclidean_distance_hull"]
                    .rolling(i, min_periods=1)
                    .sum()
                )

            print("Calculating angles...")
            self.out_data["Mouse_angle"] = self.angle3pt_serialized(
                data=self.out_data[
                    [
                        "Mouse1_nose_x",
                        "Mouse1_nose_y",
                        "Mouse1_back_x",
                        "Mouse1_back_y",
                        "Mouse1_tail_x",
                        "Mouse1_tail_y",
                    ]
                ].values
            )

            ########### DEVIATIONS ###########################################
            print("Calculating deviations...")
            self.out_data["Total_movement_all_bodyparts_both_deviation"] = (
                self.out_data["Total_movement_all_bodyparts"].mean()
                - self.out_data["Total_movement_all_bodyparts"]
            )
            self.out_data["Smallest_euclid_distances_hull_deviation"] = (
                self.out_data["Smallest_euclidean_distance_hull"].mean()
                - self.out_data["Smallest_euclidean_distance_hull"]
            )
            self.out_data["Largest_euclid_distances_hull_deviation"] = (
                self.out_data["Largest_euclidean_distance_hull"].mean()
                - self.out_data["Largest_euclidean_distance_hull"]
            )
            self.out_data["Mean_euclid_distances_hull_deviation"] = (
                self.out_data["Mean_euclidean_distance_hull"].mean()
                - self.out_data["Mean_euclidean_distance_hull"]
            )
            self.out_data["Movement_deviation_back"] = (
                self.out_data["Movement_back"].mean() - self.out_data["Movement_back"]
            )
            self.out_data["Polygon_deviation"] = (
                self.out_data["Mouse_poly_area"].mean()
                - self.out_data["Mouse_poly_area"]
            )

            for i in self.roll_windows_values:
                self.out_data[
                    f"Smallest_euclidean_distance_hull_mean_{i}_deviation"
                ] = (
                    self.out_data[f"Smallest_euclidean_distance_hull_mean_{i}"].mean()
                    - self.out_data[f"Smallest_euclidean_distance_hull_mean_{i}"]
                )

            for i in self.roll_windows_values:
                self.out_data[f"Largest_euclidean_distance_hull_mean_{i}_deviation"] = (
                    self.out_data[f"Largest_euclidean_distance_hull_mean_{i}"].mean()
                    - self.out_data[f"Largest_euclidean_distance_hull_mean_{i}"]
                )

            for i in self.roll_windows_values:
                self.out_data[f"Mean_euclidean_distance_hull_mean_{i}_deviation"] = (
                    self.out_data[f"Mean_euclidean_distance_hull_mean_{i}"].mean()
                    - self.out_data[f"Mean_euclidean_distance_hull_mean_{i}"]
                )

            for i in self.roll_windows_values:
                self.out_data[f"Total_movement_all_bodyparts_mean_{i}_deviation"] = (
                    self.out_data[f"Total_movement_all_bodyparts_mean_{i}"].mean()
                    - self.out_data[f"Total_movement_all_bodyparts_mean_{i}"]
                )

            self.out_data["Movement_percentile_rank"] = self.out_data[
                "Movement_back"
            ].rank(pct=True)

            for i in self.roll_windows_values:
                self.out_data[
                    f"Mean_euclidean_distance_hull_mean_{i}_percentile_rank"
                ] = (
                    self.out_data[f"Mean_euclidean_distance_hull_mean_{i}"].mean()
                    - self.out_data[f"Mean_euclidean_distance_hull_mean_{i}"]
                )

            for i in self.roll_windows_values:
                self.out_data[
                    f"Smallest_euclidean_distance_hull_mean_{i}_percentile_rank"
                ] = (
                    self.out_data[f"Smallest_euclidean_distance_hull_mean_{i}"].mean()
                    - self.out_data[f"Smallest_euclidean_distance_hull_mean_{i}"]
                )

            for i in self.roll_windows_values:
                self.out_data[
                    f"Largest_euclidean_distance_hull_mean_{i}_percentile_rank"
                ] = (
                    self.out_data[f"Largest_euclidean_distance_hull_mean_{i}"].mean()
                    - self.out_data[f"Largest_euclidean_distance_hull_mean_{i}"]
                )

            for i in self.roll_windows_values:
                self.out_data[
                    f"Total_movement_all_bodyparts_mean_{i}_percentile_rank"
                ] = (
                    self.out_data[f"Total_movement_all_bodyparts_mean_{i}"].mean()
                    - self.out_data[f"Total_movement_all_bodyparts_mean_{i}"]
                )

            ########### CALCULATE STRAIGHTNESS OF POLYLINE PATH: tortuosity  ###########################################
            print("Calculating path tortuosities...")
            as_strided = np.lib.stride_tricks.as_strided
            win_size = 3
            centroidList_Mouse1_x = as_strided(
                self.out_data.Mouse1_nose_x,
                (len(self.out_data) - (win_size - 1), win_size),
                (self.out_data.Mouse1_nose_x.values.strides * 2),
            )
            centroidList_Mouse1_y = as_strided(
                self.out_data.Mouse1_nose_y,
                (len(self.out_data) - (win_size - 1), win_size),
                (self.out_data.Mouse1_nose_y.values.strides * 2),
            )

            for k in range(len(self.roll_windows_values)):
                start = 0
                end = start + int(self.roll_windows_values[k])
                tortuosity_M1 = []
                for y in range(len(self.out_data)):
                    tortuosity_List_M1 = []
                    CurrCentroidList_Mouse1_x = centroidList_Mouse1_x[start:end]
                    CurrCentroidList_Mouse1_y = centroidList_Mouse1_y[start:end]
                    for i in range(len(CurrCentroidList_Mouse1_x)):
                        currMovementAngle_mouse1 = self.angle3pt(
                            CurrCentroidList_Mouse1_x[i][0],
                            CurrCentroidList_Mouse1_y[i][0],
                            CurrCentroidList_Mouse1_x[i][1],
                            CurrCentroidList_Mouse1_y[i][1],
                            CurrCentroidList_Mouse1_x[i][2],
                            CurrCentroidList_Mouse1_y[i][2],
                        )
                        tortuosity_List_M1.append(currMovementAngle_mouse1)
                    tortuosity_M1.append(sum(tortuosity_List_M1) / (2 * math.pi))
                    start += 1
                    end += 1
                self.out_data[f"Tortuosity_Mouse1_{self.roll_windows_values[k]}"] = (
                    tortuosity_M1
                )

            ########### CALC THE NUMBER OF LOW PROBABILITY DETECTIONS & TOTAL PROBABILITY VALUE FOR ROW###########################################
            print("Calculating pose probability scores...")
            self.out_data["Sum_probabilities"] = (
                self.out_data["Mouse1_left_ear_p"]
                + self.out_data["Mouse1_right_ear_p"]
                + self.out_data["Mouse1_left_hand_p"]
                + self.out_data["Mouse1_right_hand_p"]
                + self.out_data["Mouse1_left_foot_p"]
                + self.out_data["Mouse1_tail_p"]
                + self.out_data["Mouse1_right_foot_p"]
                + self.out_data["Mouse1_back_p"]
                + self.out_data["Mouse1_nose_p"]
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
                    data=self.out_data.filter(self.mouse_p_headers).values,
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
                f"Feature extraction complete for {self.video_name} ({(file_cnt + 1)}/{len(self.files_found)} (elapsed time: {video_timer.elapsed_time_str}s)..."
            )

        self.timer.stop_timer()
        stdout_success(
            msg="All features extracted. Results stored in project_folder/csv/features_extracted directory",
            elapsed_time=self.timer.elapsed_time_str,
        )


#
# test = ExtractFeaturesFrom9bps(config_path='/Users/simon/Desktop/envs/troubleshooting/Emergence/project_folder/project_config.ini')
# test.run()
