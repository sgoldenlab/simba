__author__ = "Simon Nilsson"

import itertools
import os
from typing import Optional, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import (check_if_filepath_list_is_empty,
                                check_that_dir_has_list_of_filenames)
from simba.utils.enums import TagNames
from simba.utils.errors import AnimalNumberError, CountError, InvalidInputError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import get_fn_ext, read_df, write_df


class DirectingOtherAnimalsAnalyzer(ConfigReader, FeatureExtractionMixin):
    """
    Calculate when animals are directing towards body-parts of other animals. Results are stored in
    the ``project_folder/logs/directionality_dataframes`` directory of the SimBA project.

    .. note:
       `Example expected bool table <https://github.com/sgoldenlab/simba/blob/master/misc/boolean_directionaly_example.csv>`__.
       `Example expected summary table <https://github.com/sgoldenlab/simba/blob/master/misc/detailed_summary_directionality_example.csv>`__.
       `Example expected aggregate statistics table <https://github.com/sgoldenlab/simba/blob/master/misc/direction_data_aggregates_example.csv>`__.

    .. important::
       Requires the pose-estimation data for the ``left ear``, ``right ear`` and ``nose`` of each individual animals.
       `Github Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/ROI_tutorial.md#part-3-generating-features-from-roi-data>`__.
       `Expected output <https://github.com/sgoldenlab/simba/blob/master/misc/Direction_data_example.csv>`__.

    :parameter str config_path: path to SimBA project config file in Configparser format.
    :parameter bool bool_tables: If True, creates boolean output tables.
    :parameter bool summary_tables: If True, creates summary tables including approximate location of eye of observer and the location of observed body-parts and frames when observation was detected.
    :parameter bool aggregate_statistics_tables: If True, summary statistics tables of how much time each animal spent observation the other animals.

    :examples:
    >>> directing_analyzer = DirectingOtherAnimalsAnalyzer(config_path='MyProjectConfig')
    >>> directing_analyzer.run()
    """

    def __init__(
        self,
        config_path: Union[str, os.PathLike],
        bool_tables: Optional[bool] = True,
        summary_tables: Optional[bool] = False,
        append_bool_tables_to_features: Optional[bool] = False,
        aggregate_statistics_tables: Optional[bool] = False,
    ):

        super().__init__(config_path=config_path)
        log_event(
            logger_name=str(self.__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        if self.animal_cnt < 2:
            raise AnimalNumberError(
                "Cannot analyze directionality between animals in a 1 animal project.",
                source=self.__class__.__name__,
            )
        check_if_filepath_list_is_empty(
            filepaths=self.outlier_corrected_paths,
            error_msg=f"SIMBA ERROR: No data found in the {self.outlier_corrected_dir} directory",
        )
        self.animal_permutations = list(itertools.permutations(self.animal_bp_dict, 2))
        (
            self.bool_tables,
            self.summary_tables,
            self.aggregate_statistics_tables,
            self.append_bool_tables_to_features,
        ) = (
            bool_tables,
            summary_tables,
            aggregate_statistics_tables,
            append_bool_tables_to_features,
        )
        if self.append_bool_tables_to_features:
            check_that_dir_has_list_of_filenames(
                dir=self.features_dir,
                file_name_lst=self.outlier_corrected_paths,
                file_type=self.file_type,
            )
        print(f"Processing {str(len(self.outlier_corrected_paths))} video(s)...")

    def run(self):
        self.results_dict = {}
        for file_cnt, file_path in enumerate(self.outlier_corrected_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(file_path)
            print(f"Analyzing directionality between animals in video {video_name}...")
            self.results_dict[video_name] = {}
            data_df = read_df(file_path, self.file_type)
            direct_bp_dict = self.check_directionality_cords()
            for animal_permutation in self.animal_permutations:
                self.results_dict[video_name][
                    "{} {} {}".format(
                        animal_permutation[0],
                        "directing towards",
                        animal_permutation[1],
                    )
                ] = {}
                first_animal_bps, second_animal_bps = (
                    direct_bp_dict[animal_permutation[0]],
                    self.animal_bp_dict[animal_permutation[1]],
                )
                first_ear_left_arr = data_df[
                    [
                        first_animal_bps["Ear_left"]["X_bps"],
                        first_animal_bps["Ear_left"]["Y_bps"],
                    ]
                ].to_numpy()
                first_ear_right_arr = data_df[
                    [
                        first_animal_bps["Ear_right"]["X_bps"],
                        first_animal_bps["Ear_right"]["Y_bps"],
                    ]
                ].to_numpy()
                first_nose_arr = data_df[
                    [
                        first_animal_bps["Nose"]["X_bps"],
                        first_animal_bps["Nose"]["Y_bps"],
                    ]
                ].to_numpy()
                other_animal_x_bps, other_animal_y_bps = (
                    second_animal_bps["X_bps"],
                    second_animal_bps["Y_bps"],
                )
                for x_bp, y_bp in zip(other_animal_x_bps, other_animal_y_bps):
                    target_cord_arr = data_df[[x_bp, y_bp]].to_numpy()
                    direction_data = self.jitted_line_crosses_to_nonstatic_targets(
                        left_ear_array=first_ear_left_arr,
                        right_ear_array=first_ear_right_arr,
                        nose_array=first_nose_arr,
                        target_array=target_cord_arr,
                    )
                    x_min = np.minimum(direction_data[:, 1], first_nose_arr[:, 0])
                    y_min = np.minimum(direction_data[:, 2], first_nose_arr[:, 1])
                    delta_x = abs((direction_data[:, 1] - first_nose_arr[:, 0]) / 2)
                    delta_y = abs((direction_data[:, 2] - first_nose_arr[:, 1]) / 2)
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
                        columns=["Eye_x", "Eye_y", "Directing_BOOL", x_bp, y_bp],
                    )
                    bp_data = bp_data[["Eye_x", "Eye_y", x_bp, y_bp, "Directing_BOOL"]]
                    bp_data.insert(loc=0, column="Animal_2_body_part", value=x_bp[:-2])
                    bp_data.insert(
                        loc=0, column="Animal_2", value=animal_permutation[1]
                    )
                    bp_data.insert(
                        loc=0, column="Animal_1", value=animal_permutation[0]
                    )
                    self.results_dict[video_name][
                        "{} {} {}".format(
                            animal_permutation[0],
                            "directing towards",
                            animal_permutation[1],
                        )
                    ][x_bp[:-2]] = bp_data
            video_timer.stop_timer()
            print(
                f"Direction analysis complete for video {video_name} ({file_cnt + 1}/{len(self.outlier_corrected_paths)}, elapsed time: {video_timer.elapsed_time_str}s)..."
            )
        if self.bool_tables:
            self.create_bool_tables()
        if self.summary_tables:
            self._transpose_results_to_df()
            self._save_directionality_dfs()
        if self.aggregate_statistics_tables:
            self.summary_statistics()

    def create_bool_tables(self):
        save_dir = os.path.join(
            self.logs_path, f"Animal_directing_animal_booleans_{self.datetime}"
        )
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        for video_cnt, (video_name, video_data) in enumerate(self.results_dict.items()):
            video_df = pd.DataFrame()
            print(
                f"Saving boolean directing tables for video {video_name} (Video {video_cnt+1}/{len(self.results_dict.keys())})..."
            )
            for animal_permutation, animal_permutation_data in video_data.items():
                for body_part_name, body_part_data in animal_permutation_data.items():
                    video_df[f"{animal_permutation}_{body_part_name}"] = body_part_data[
                        "Directing_BOOL"
                    ]
            if self.append_bool_tables_to_features:
                print(
                    f"Adding directionality tables to features data for video {video_name}..."
                )
                df = read_df(
                    file_path=os.path.join(
                        self.features_dir, f"{video_name}.{self.file_type}"
                    ),
                    file_type=self.file_type,
                )
                if len(df) != len(video_df):
                    raise CountError(
                        msg=f"Failed to join data files as they contains different number of frames: the file representing video {video_name} in directory {self.outlier_corrected_dir} contains {len(video_df)} frames, and the file representing video {video_name} in directory {self.features_dir} contains {len(df)} frames."
                    )
                else:
                    df = pd.concat(
                        [df.reset_index(drop=True), video_df.reset_index(drop=True)],
                        axis=1,
                    )
                    write_df(
                        df=df,
                        file_type=self.file_type,
                        save_path=os.path.join(
                            self.features_dir, f"{video_name}.{self.file_type}"
                        ),
                    )
            video_df.to_csv(os.path.join(save_dir, f"{video_name}.csv"))
        stdout_success(
            msg=f"All boolean tables saved in {save_dir}!",
            source=self.__class__.__name__,
        )

    def _transpose_results_to_df(self):
        """
        Privet method to transpose results created by :meth:`~simba.DirectingOtherAnimalsAnalyzer.process_directionality`.
        into dict of dataframes.
        """

        print("Transposing directionality data for summary tables...")
        self.directionality_df_dict = {}
        for video_name, video_data in self.results_dict.items():
            out_df_lst = []
            for animal_permutation, permutation_data in video_data.items():
                for bp_name, bp_data in permutation_data.items():
                    directing_df = (
                        bp_data[bp_data["Directing_BOOL"] == 1]
                        .reset_index()
                        .rename(
                            columns={
                                "index": "Frame_#",
                                bp_name + "_x": "Animal_2_bodypart_x",
                                bp_name + "_y": "Animal_2_bodypart_y",
                            }
                        )
                    )
                    directing_df.insert(loc=0, column="Video", value=video_name)
                    out_df_lst.append(directing_df)
            self.directionality_df_dict[video_name] = pd.concat(
                out_df_lst, axis=0
            ).drop("Directing_BOOL", axis=1)

    def _save_directionality_dfs(self):
        """
        Privat method to save result created by :meth:`~simba.DirectingOtherAnimalsAnalyzer.create_directionality_dfs`.
        into CSV files on disk. Results are stored in `project_folder/logs` directory of the SimBA project.
        """
        save_dir = os.path.join(
            self.logs_path,
            f"detailed_directionality_summary_dataframes_{self.datetime}",
        )
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for video_cnt, (video_name, video_data) in enumerate(
            self.directionality_df_dict.items()
        ):
            save_name = os.path.join(save_dir, f"{video_name}.csv")
            video_data.to_csv(save_name)
            print(
                f"Detailed directional summary tables saved for video {video_name} (Video {video_cnt+1}/{len(list(self.directionality_df_dict.keys()))})..."
            )
        stdout_success(
            f"All detailed directional data saved in the {save_dir} directory!",
            source=self.__class__.__name__,
        )

    def summary_statistics(self):
        """
        Method to save aggregate statistics of data created by :meth:`~simba.DirectingOtherAnimalsAnalyzer.create_directionality_dfs`.
        into CSV files on disk. Results are stored in `project_folder/logs` directory of the SimBA project.
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
            self.logs_path, f"Direction_aggregate_summary_data_{self.datetime}.csv"
        )
        self.summary_df.to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(
            msg=f"Summary directional statistics saved at {self.save_path}",
            source=self.__class__.__name__,
        )
        stdout_success(
            msg="All directional data saved in SimBA project",
            elapsed_time=self.timer.elapsed_time_str,
            source=self.__class__.__name__,
        )


# test = DirectingOtherAnimalsAnalyzer(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                                      bool_tables=True,
#                                      summary_tables=False,
#                                      aggregate_statistics_tables=False,
#                                      append_bool_tables_to_features=True)
# test.run()
