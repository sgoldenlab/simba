__author__ = "Tzuk Polinsky"

import os
from itertools import product

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import check_str
from simba.utils.enums import Paths
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df


class MiceFreezingFeatureExtractor(ConfigReader, FeatureExtractionMixin):
    """
    Generic featurizer of data within SimBA project using user-defined body-parts in the pose-estimation data.
    Results are stored in the `project_folder/csv/features_extracted` directory of the SimBA project.

    :parameter str config_path: path to SimBA project config file in Configparser format

    .. note::
       `Feature extraction tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial.md#step-5-extract-features>`__.

    Examples
    ----------
    >>> feature_extractor = MiceFreezingFeatureExtractor(config_path='MyProjectConfig')
    >>> feature_extractor.run()
    """

    def __init__(self, config_path: str):
        FeatureExtractionMixin.__init__(self, config_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        print(
            "Extracting features from {} file(s)...".format(str(len(self.files_found)))
        )

    # Function to calculate the direction vector
    def angle_between_vectors(self, v1, v2):
        unit_vector_1 = v1 / np.linalg.norm(v1)
        unit_vector_2 = v2 / np.linalg.norm(v2)
        dot_product = unit_vector_2.dot(unit_vector_1.T)
        angle = np.arccos(dot_product)
        return np.degrees(angle)

    def calculate_direction_vector(self, from_point, to_point):
        return np.array(to_point) - np.array(from_point)

    def extract_features(self, input_file_path: str, window_size: int, video_center: [int, int], pixel_mm: float,
                         directionality_data: pd.DataFrame):
        print("Calculating freezing features ...")

        input_data = pd.read_csv(input_file_path)
        output_data = pd.DataFrame(
            columns=["activity"])
        columns_to_drop = [col for col in input_data.columns if ('bug' in col) or ("_p" in col)]
        columns_to_drop.append("Unnamed: 0")
        without_bug = input_data.drop(columns_to_drop, axis=1)

        body_parts_diffs = without_bug.diff(axis=0)
        time_point_diff = body_parts_diffs.sum(axis=1)
        #second_time_point_diff = time_point_diff.diff()
        rolling_windows = time_point_diff.rolling(window=window_size, min_periods=1).sum()
        output_data["activity"] = rolling_windows.abs().fillna(500)
        bug_cols = [colName for colName in input_data.columns if ("bug" in colName) and ("_p") not in colName]
        center_cols = [colName for colName in without_bug.columns if ("center" in colName) and ("_p") not in colName]
        #tails_cols = [colName for colName in without_bug.columns if ("tail" in colName) and ("_p") not in colName]
        nose_cols = [colName for colName in without_bug.columns if ("nose" in colName) and ("_p") not in colName]
        centers = without_bug[center_cols].to_numpy()
        #tails = without_bug[tails_cols].to_numpy()
        noses = without_bug[nose_cols].to_numpy()
        bug = input_data[bug_cols].to_numpy()
        distances_from_bug = np.linalg.norm(bug - noses,axis=1)
        video_centers = np.array([video_center]*len(centers))
        distances_from_center = np.linalg.norm(video_centers - noses,axis=1)
        #body_size = np.insert(np.diff(np.linalg.norm(tails - noses,axis=1), axis=0),0,0)
        output_data["distances_from_bug"] = pd.DataFrame(distances_from_bug).rolling(window=window_size, min_periods=1).mean().fillna(100).to_numpy()
        output_data["distances_from_center"] = pd.DataFrame(distances_from_center).rolling(window=window_size, min_periods=1).mean().fillna(100).to_numpy()
        #output_data["body_size"] = pd.DataFrame(body_size).rolling(window=window_size, min_periods=1).sum().abs().fillna(100).to_numpy()
        angles = []
        for i, center in enumerate(centers):
            nose = noses[i]
            vector_fixed_to_center = self.calculate_direction_vector(video_center, center)
            vector_center_to_nose = self.calculate_direction_vector(center, nose)
            angles.append(self.angle_between_vectors(vector_center_to_nose, vector_fixed_to_center))
        # output_data["nose_direction"] = angles
        angles_df = pd.DataFrame(angles)
        angles_diff = angles_df.diff()
        angles_diff_sum = angles_diff.rolling(window=window_size, min_periods=1).sum()
        output_data["nose_direction_sum_of_diffs"] = angles_diff_sum.abs().fillna(0)
        # output_data["nose_direction_avg"] = angles_df.rolling(window=window_size, min_periods=1).mean().fillna(0)
        directionality_rolling = directionality_data.rolling(window=window_size, min_periods=1)
        output_data["amount_of_looking_at_bug"] = directionality_rolling.sum().fillna(0)
        onsets = [-1] * len(output_data["amount_of_looking_at_bug"])
        for j, rol in enumerate(directionality_rolling):
            for i, r in enumerate(rol):
                if r:
                    onsets[j] = i
                    break
        output_data["looking_at_bug_onset"] = onsets
        return output_data

    def run(self):
        """
        Method to compute and save features to disk. Results are saved in the `project_folder/csv/features_extracted`
        directory of the SimBA project.

        Returns
        -------
        None
        """
        self.roi_coordinates_path = os.path.join(
            self.logs_path, Paths.ROI_DEFINITIONS.value
        )
        polygons = pd.read_hdf(self.roi_coordinates_path, key="polygons")
        directionality_dir_path = os.path.join(self.body_part_directionality_df_dir, "bug")
        for file_cnt, file_path in enumerate(self.files_found):
            video_timer = SimbaTimer(start=True)
            print(
                "Extracting features for video {}/{}...".format(
                    str(file_cnt + 1), str(len(self.files_found))
                )
            )
            _, file_name, _ = get_fn_ext(file_path)
            current_polygon = polygons[polygons["Video"] == file_name]
            directionality_data_path = os.path.join(directionality_dir_path, file_name + ".csv")
            directionality_data = pd.read_csv(directionality_data_path)["Directing_BOOL"]
            check_str("file name", file_name)
            video_settings, self.px_per_mm, fps = self.read_video_info(
                video_name=file_name
            )
            self.data_df = self.extract_features(file_path, 25, (
                current_polygon["Center_X"].values[0], current_polygon["Center_Y"].values[0]),
                                                 video_settings["pixels/mm"].values[0], directionality_data)
            save_path = os.path.join(self.save_dir, file_name + "." + self.file_type)
            self.data_df = self.data_df.reset_index(drop=True).fillna(0)
            write_df(df=self.data_df, file_type=self.file_type, save_path=save_path)
            video_timer.stop_timer()
            print(
                f"Feature extraction complete for video {file_name} (elapsed time: {video_timer.elapsed_time_str}s)"
            )
            print(
                f"Feature extraction file for video {file_name} saved to {save_path})"
            )

        self.timer.stop_timer()
        stdout_success(
            f"Feature extraction complete for {str(len(self.files_found))} video(s). Results are saved inside the project_folder/csv/features_extracted directory",
            elapsed_time=self.timer.elapsed_time_str,
        )

# test = UserDefinedFeatureExtractor(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.run()
