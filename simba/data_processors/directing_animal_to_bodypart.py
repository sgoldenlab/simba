__author__ = "Tzuk Polinsky"

import itertools
import os
from typing import Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.enums import ConfigKey, Dtypes
from simba.utils.errors import AnimalNumberError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_df


class DirectingAnimalsToBodyPartAnalyzer(ConfigReader, FeatureExtractionMixin):
    """
    Calculate when animals are directing towards their own body-parts. Results are stored in
    the ``project_folder/logs/directionality_dataframes`` directory of the SimBA project.

    :parameter str config_path: path to SimBA project config file in Configparser format

    .. important::
       Requires the pose-estimation data for the ``left ear``, ``right ear`` and ``nose`` of each individual animals.
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data>`__.
       `Expected output <https://github.com/sgoldenlab/simba/blob/master/misc/Direction_data_example.csv>`__.

    Examples
    -----
    >>> directing_analyzer = DirectingAnimalsToBodyPartAnalyzer(config_path='MyProjectConfig')
    >>> directing_analyzer.process_directionality()
    >>> directing_analyzer.create_directionality_dfs()
    >>> directing_analyzer.save_directionality_dfs()
    >>> directing_analyzer.summary_statistics()
    """

    def __init__(self, config_path: Union[str, os.PathLike]):
        super().__init__(config_path=config_path)
        if not os.path.exists(self.directionality_df_dir):
            os.makedirs(self.directionality_df_dir)
        check_if_filepath_list_is_empty(
            filepaths=self.outlier_corrected_paths,
            error_msg=f"SIMBA ERROR: No data found in the {self.outlier_corrected_dir} directory",
        )

        print(f"Processing {str(len(self.outlier_corrected_paths))} video(s)...")

    def process_directionality(self):
        """
        Method to compute when animals are directing towards their own body-parts.

        Returns
        -------
        Attribute: dict
            results_dict
        """

        self.results_dict = {}
        bp_x_name = self.bodypart_direction + "_x"
        bp_y_name = self.bodypart_direction + "_y"
        for file_cnt, file_path in enumerate(self.outlier_corrected_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(file_path)
            self.results_dict[video_name] = {}
            data_df = read_df(file_path, self.file_type)
            direct_bp_dict = self.check_directionality_cords()
            for key, animal in direct_bp_dict.items():
                result_key = "{} {} {}".format(
                    key,
                    "directing towards",
                    self.bodypart_direction,
                )
                self.results_dict[video_name][result_key] = {}
                ear_left_arr = data_df[
                    [
                        animal["Ear_left"]["X_bps"],
                        animal["Ear_left"]["Y_bps"],
                    ]
                ].to_numpy()
                ear_right_arr = data_df[
                    [
                        animal["Ear_right"]["X_bps"],
                        animal["Ear_right"]["Y_bps"],
                    ]
                ].to_numpy()
                nose_arr = data_df[
                    [
                        animal["Nose"]["X_bps"],
                        animal["Nose"]["Y_bps"],
                    ]
                ].to_numpy()
                target_cord_arr = data_df[[bp_x_name, bp_y_name]].to_numpy()
                direction_data = self.jitted_line_crosses_to_nonstatic_targets(
                    left_ear_array=ear_left_arr,
                    right_ear_array=ear_right_arr,
                    nose_array=nose_arr,
                    target_array=target_cord_arr,
                )
                x_min = np.minimum(direction_data[:, 1], nose_arr[:, 0])
                y_min = np.minimum(direction_data[:, 2], nose_arr[:, 1])
                delta_x = abs((direction_data[:, 1] - nose_arr[:, 0]) / 2)
                delta_y = abs((direction_data[:, 2] - nose_arr[:, 1]) / 2)
                x_middle, y_middle = np.add(x_min, delta_x), np.add(y_min, delta_y)
                direction_data = np.concatenate(
                    (y_middle.reshape(-1, 1), direction_data), axis=1
                )
                direction_data = np.concatenate(
                    (x_middle.reshape(-1, 1), direction_data), axis=1
                )
                direction_data = np.delete(direction_data, [2, 3, 4], 1)
                direction_data = np.hstack((direction_data, target_cord_arr))
                bp_data = pd.DataFrame(
                    direction_data,
                    columns=["Eye_x", "Eye_y", "Directing_BOOL", bp_x_name, bp_y_name],
                )
                bp_data = bp_data[
                    ["Eye_x", "Eye_y", bp_x_name, bp_y_name, "Directing_BOOL"]
                ]
                self.results_dict[video_name][result_key][bp_x_name[:-2]] = bp_data
            video_timer.stop_timer()
            print(
                "Direction analysis complete for video {} ({}/{}, elapsed time: {}s)...".format(
                    video_name,
                    str(file_cnt + 1),
                    str(len(self.outlier_corrected_paths)),
                    video_timer.elapsed_time_str,
                )
            )

    def create_directionality_dfs(self):
        """
        Method to transpose results created by :meth:`~simba.DirectingOtherAnimalsAnalyzer.process_directionality`.
        into dict of dataframes

        Returns
        -------
        Attribute: dict
            directionality_df_dict
        """

        print("Transposing body part directionality data...")
        self.directionality_df_dict = {}
        for video_name, video_data in self.results_dict.items():
            out_df_lst = []
            for animal_permutation, permutation_data in video_data.items():
                for bp_name, bp_data in permutation_data.items():
                    directing_df = bp_data.reset_index().rename(  # [bp_data["Directing_BOOL"] == 1]
                        columns={
                            "index": "Frame_#",
                            bp_name
                            + "_x": "Animal_{}_x".format(self.bodypart_direction),
                            bp_name
                            + "_y": "Animal_{}_y".format(self.bodypart_direction),
                        }
                    )
                    directing_df.insert(loc=0, column="Video", value=video_name)
                    out_df_lst.append(directing_df)
            self.directionality_df_dict[video_name] = pd.concat(out_df_lst, axis=0)
        stdout_success(msg="Transposing body part directionality data completed")

    def read_directionality_dfs(self):
        results = {}
        for file_cnt, file_path in enumerate(self.body_part_directionality_paths):
            video_timer = SimbaTimer(start=True)
            _, file_name, _ = get_fn_ext(file_path)
            results[file_name] = pd.read_csv(file_path)
            video_timer.stop_timer()
            print(
                "read body part directionality data completed for video {} ({}/{}, elapsed time: {}s)...".format(
                    file_name,
                    str(file_cnt + 1),
                    str(len(self.outlier_corrected_paths)),
                    video_timer.elapsed_time_str,
                )
            )
        stdout_success(msg="reading body part directionality data completed")
        return results

    def save_directionality_dfs(self):
        """
        Method to save result created by :meth:`~simba.DirectingOtherAnimalsAnalyzer.create_directionality_dfs`.
        into CSV files on disk. Results are stored in `project_folder/logs` directory of the SimBA project.

        Returns
        -------
        None
        """
        if not os.path.exists(self.body_part_directionality_df_dir):
            os.makedirs(self.body_part_directionality_df_dir)
        for video_name, video_data in self.directionality_df_dict.items():
            save_name = os.path.join(
                self.body_part_directionality_df_dir, video_name + ".csv"
            )
            video_data.to_csv(save_name)
            print(f"Detailed directional data saved for video {video_name}...")
        stdout_success(
            msg=f"All detailed directional data saved in the {self.body_part_directionality_df_dir} directory"
        )

    def summary_statistics(self):
        """
        Method to save aggregate statistics of data created by :meth:`~simba.DirectingOtherAnimalsAnalyzer.create_directionality_dfs`.
        into CSV files on disk. Results are stored in `project_folder/logs` directory of the SimBA project.

        Returns
        -------
        None
        """

        print("Computing summary statistics...")
        out_df_lst = []
        for video_name, video_data in self.results_dict.items():
            _, _, fps = self.read_video_info(video_name=video_name)
            for animal_permutation, permutation_data in video_data.items():
                idx_directing = set()
                for bp_name, bp_data in permutation_data.items():
                    idx_directing.update(
                        list(bp_data.index[bp_data["Directing_BOOL"] == 1])
                    )
                value = round(len(idx_directing) / fps, 3)
                out_df_lst.append(
                    pd.DataFrame(
                        [[video_name, animal_permutation, value]],
                        columns=["Video", "Animal permutation", "Value (s)"],
                    )
                )
        self.summary_df = (
            pd.concat(out_df_lst, axis=0)
            .sort_values(by=["Video", "Animal permutation"])
            .set_index("Video")
        )
        self.save_path = os.path.join(
            self.logs_path,
            "Body_part_directions_data_{}.csv".format(str(self.datetime)),
        )
        self.summary_df.to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(
            msg=f"Summary body part directional statistics saved at {self.save_path}"
        )
        stdout_success(
            msg="All directional data saved in SimBA project",
            elapsed_time=self.timer.elapsed_time_str,
        )


# test = DirectingOtherAnimalsAnalyzer(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# test.process_directionality()
# test.create_directionality_dfs()
# test.save_directionality_dfs()
# test.summary_statistics()
