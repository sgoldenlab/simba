__author__ = "Simon Nilsson"

import itertools
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.feature_extraction_supplement_mixin import \
    FeatureExtractionSupplemental
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_float,
    check_if_filepath_list_is_empty, check_int)
from simba.utils.enums import TagNames
from simba.utils.errors import FrameRangeError
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import get_fn_ext, read_df


class TimeBinsMovementCalculator(ConfigReader, FeatureExtractionMixin):
    """
    Computes aggregate movement statistics in user-defined time-bins.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter int bin_length: Integer representing the time bin size in seconds.
    :parameter bool plots: If True, creates time-bin line plots representing the movement in each time-bin per video. Results are saved in the ``project_folder/logs/`` sub-directory.

    .. note::
        `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results>`__.

    .. image:: _static/img/TimeBinsMovementCalculator.png
       :width: 500
       :align: center

    :example:
    >>> calculator = TimeBinsMovementCalculator(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', bin_length=0.04,plots=True, body_parts=['Nose_1', 'Nose_2'])
    >>> calculator.run()
    """

    def __init__(
        self,
        config_path: str,
        bin_length: Union[int, float],
        body_parts: List[str],
        plots: Optional[bool] = False,
    ):

        ConfigReader.__init__(self, config_path=config_path)
        log_event(
            logger_name=str(self.__class__.__name__),
            log_type=TagNames.CLASS_INIT.value,
            msg=self.create_log_msg_from_init_args(locals=locals()),
        )
        self.bin_length, self.plots = bin_length, plots
        check_float(name="TIME BIN", value=bin_length, min_value=10e-6)
        self.col_headers, self.bp_dict = [], {}
        for bp_cnt, bp in enumerate(body_parts):
            self.col_headers.extend((f"{bp}_x", f"{bp}_y"))
            animal_name = self.find_animal_name_from_body_part_name(
                bp_name=bp, bp_dict=self.animal_bp_dict
            )
            self.bp_dict[bp_cnt] = {animal_name: [f"{bp}_x", f"{bp}_y"]}
        check_if_filepath_list_is_empty(
            filepaths=self.outlier_corrected_paths,
            error_msg=f"SIMBA ERROR: Cannot analyze movement in time-bins, data directory {self.outlier_corrected_dir} is empty.",
        )
        self.animal_combinations = list(itertools.combinations(self.animal_bp_dict, 2))
        print(f"Processing {len(self.outlier_corrected_paths)} video(s)...")

    def __create_plots(self):
        timer = SimbaTimer(start=True)
        print("Creating time-bin movement plots...")
        plots_dir = os.path.join(
            self.project_path, "logs", f"time_bin_movement_plots_{self.datetime}"
        )
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)
        for video_name in self.results["VIDEO"].unique():
            video_df = self.results.loc[
                (self.results["VIDEO"] == video_name)
                & (self.results["MEASUREMENT"] == "Movement (cm)")
            ]
            video_df["TIME BIN #"] = video_df["TIME BIN #"].astype(int)
            for body_part in video_df["BODY-PART"].unique():
                body_part_df = (
                    video_df[video_df["BODY-PART"] == body_part]
                    .reset_index(drop=True)
                    .sort_values(by=["TIME BIN #"])
                )
                body_part_df[f"Time bin # (bin length {self.bin_length}s)"] = (
                    body_part_df["TIME BIN #"]
                )
                body_part_df["VALUE"] = body_part_df["VALUE"].astype(float)
                _ = PlottingMixin.line_plot(
                    df=body_part_df,
                    x=f"Time bin # (bin length {self.bin_length}s)",
                    y="VALUE",
                    y_label="Distance (cm)",
                    save_path=os.path.join(plots_dir, f"{video_name}_{body_part}.png"),
                )
        timer.stop_timer()
        stdout_success(
            msg=f"Time bin movement plots saved in {plots_dir}",
            elapsed_time=timer.elapsed_time_str,
            source=self.__class__.__name__,
        )

    def run(self):
        video_dict, self.out_df_lst = {}, []
        self.movement_dict = {}
        self.save_path = os.path.join(
            self.project_path,
            "logs",
            f"Time_bins_{self.bin_length}s_movement_results_{self.datetime}.csv",
        )
        check_all_file_names_are_represented_in_video_log(
            video_info_df=self.video_info_df, data_paths=self.outlier_corrected_paths
        )
        for file_cnt, file_path in enumerate(self.outlier_corrected_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(file_path)
            print(
                f"Processing time-bin movements for video {video_name} ({str(file_cnt+1)}/{str(len(self.outlier_corrected_paths))})..."
            )
            video_dict[video_name] = {}
            video_settings, px_per_mm, fps = self.read_video_info(video_name=video_name)
            fps, self.movement_cols, self.velocity_cols = int(fps), set(), set()
            bin_length_frames = int(fps * self.bin_length)
            if bin_length_frames == 0:
                raise FrameRangeError(
                    msg=f"The specified time-bin length of {self.bin_length} is TOO SHORT for video {video_name} which has a specified FPS of {fps}. This results in time bins that are LESS THAN a single frame.",
                    source=self.__class__.__name__,
                )
            self.data_df = read_df(file_path, self.file_type, usecols=self.col_headers)
            self.data_df = self.create_shifted_df(df=self.data_df)
            results = []
            for animal_data in self.bp_dict.values():
                name, bps = list(animal_data.keys())[0], list(animal_data.values())[0]
                bp_time_1, bp_time_2 = (
                    self.data_df[bps].values,
                    self.data_df[[f"{bps[0]}_shifted", f"{bps[1]}_shifted"]].values,
                )
                movement_data = pd.DataFrame(
                    self.framewise_euclidean_distance(
                        location_1=bp_time_1,
                        location_2=bp_time_2,
                        px_per_mm=px_per_mm,
                        centimeter=True,
                    ),
                    columns=["VALUE"],
                )
                self.movement_dict[video_name] = movement_data
                movement_df_lists = [
                    movement_data[i : i + bin_length_frames]
                    for i in range(0, movement_data.shape[0], bin_length_frames)
                ]
                for bin, movement_df in enumerate(movement_df_lists):
                    movement, velocity = (
                        FeatureExtractionSupplemental.distance_and_velocity(
                            x=movement_df["VALUE"].values,
                            fps=fps,
                            pixels_per_mm=1,
                            centimeters=False,
                        )
                    )
                    results.append(
                        {
                            "VIDEO": video_name,
                            "TIME BIN #": bin,
                            "ANIMAL": name,
                            "BODY-PART": bps[0][:-2],
                            "MEASUREMENT": "Movement (cm)",
                            "VALUE": round(movement, 4),
                        }
                    )
                    results.append(
                        {
                            "VIDEO": video_name,
                            "TIME BIN #": bin,
                            "ANIMAL": name,
                            "BODY-PART": bps[0][:-2],
                            "MEASUREMENT": "Velocity (cm/s)",
                            "VALUE": round(velocity, 4),
                        }
                    )
            results = pd.DataFrame(results).reset_index(drop=True)
            self.out_df_lst.append(results)
            video_timer.stop_timer()
            print(
                f"Video {video_name} complete (elapsed time: {video_timer.elapsed_time_str}s)..."
            )

    def save(self):
        self.results = pd.concat(self.out_df_lst, axis=0).sort_values(
            by=["VIDEO", "TIME BIN #", "MEASUREMENT", "ANIMAL"]
        )[["VIDEO", "TIME BIN #", "ANIMAL", "BODY-PART", "MEASUREMENT", "VALUE"]]
        self.results.set_index("VIDEO").to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(
            msg=f"Movement time-bins results saved at {self.save_path}",
            elapsed_time=self.timer.elapsed_time_str,
            source=self.__class__.__name__,
        )
        if self.plots:
            self.__create_plots()


# test = TimeBinsMovementCalculator(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini",
#                                   bin_length=0.1,
#                                   plots=True,
#                                   body_parts=['Nose_1'])
# test.run()
# test.save()
