__author__ = "Simon Nilsson"

import functools
import itertools
import multiprocessing
import os
import platform
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.feature_extraction_supplement_mixin import \
    FeatureExtractionSupplemental
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_float, check_int,
    check_that_column_exist, check_valid_boolean, check_valid_lst)
from simba.utils.enums import TagNames
from simba.utils.errors import FrameRangeError, InvalidInputError, NoDataError
from simba.utils.lookups import get_current_time
from simba.utils.printing import SimbaTimer, log_event, stdout_success
from simba.utils.read_write import (create_directory, find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    find_time_stamp_from_frame_numbers,
                                    get_fn_ext, read_df, read_video_info)


def _time_bin_movement_helper(data: list,
                              bin_length: int,
                              verbose: bool,
                              video_info_df: pd.DataFrame,
                              bp_headers: list,
                              bp_dict: dict,
                              distance: bool,
                              velocity: bool):

    batch_id, file_paths = data
    video_dict, movement_dict, batch_results = {}, {}, []
    for file_cnt, file_path in enumerate(file_paths):
        video_timer = SimbaTimer(start=True)
        _, video_name, _ = get_fn_ext(file_path)
        if verbose: print(f"Processing time-bin movements ({bin_length}s) for video {video_name} ({str(file_cnt+1)}/{str(len(file_paths))}, core batch: {batch_id})...")
        video_dict[video_name] = {}
        video_settings, px_per_mm, fps = read_video_info(video_name=video_name, video_info_df=video_info_df)
        #fps, px_per_mm = 30, 4
        fps, movement_cols, velocity_cols = fps, set(), set()
        bin_length_frames = int(fps * bin_length)
        if bin_length_frames == 0:  raise FrameRangeError(msg=f"The specified time-bin length of {bin_length} is TOO SHORT for video {video_name} which has a specified FPS of {fps}. This results in time bins that are LESS THAN a single frame.", source=_time_bin_movement_helper.__name__,)
        data_df = read_df(file_path)
        check_that_column_exist(df=data_df, column_name=bp_headers, file_name=file_path)
        data_df = data_df[bp_headers]
        data_df = FeatureExtractionMixin().create_shifted_df(df=data_df)
        video_results = []
        for animal_data in bp_dict.values():
            name, bps = list(animal_data.keys())[0], list(animal_data.values())[0]
            bp_time_1, bp_time_2 = (data_df[bps].values, data_df[[f"{bps[0]}_shifted", f"{bps[1]}_shifted"]].values,)
            movement_data = pd.DataFrame(FeatureExtractionMixin().framewise_euclidean_distance(location_1=bp_time_1.astype(np.float64), location_2=bp_time_2.astype(np.float64), px_per_mm=np.float64(px_per_mm), centimeter=True), columns=["VALUE"])
            movement_dict[video_name] = movement_data
            movement_df_lists = [movement_data[i : i + bin_length_frames] for i in range(0, movement_data.shape[0], bin_length_frames)]
            for bin, movement_df in enumerate(movement_df_lists):
                bin_times = find_time_stamp_from_frame_numbers(start_frame=int(bin_length_frames * bin), end_frame=min(int(bin_length_frames * (bin + 1)), len(data_df)), fps=fps)
                movement_data, velocity_data = (FeatureExtractionSupplemental.distance_and_velocity(x=movement_df["VALUE"].values, fps=fps, pixels_per_mm=px_per_mm, centimeters=False))
                if distance:
                    video_results.append({"VIDEO": video_name,"TIME BIN #": bin, "START TIME": bin_times[0], "END TIME": bin_times[1], "ANIMAL": name,"BODY-PART": bps[0][:-2],"MEASUREMENT": "Movement (cm)","VALUE": round(movement_data, 4)})
                if velocity:
                    video_results.append({"VIDEO": video_name,"TIME BIN #": bin, "START TIME": bin_times[0], "END TIME": bin_times[1], "ANIMAL": name,"BODY-PART": bps[0][:-2],"MEASUREMENT": "Velocity (cm/s)","VALUE": round(velocity_data, 4)})
        results = pd.DataFrame(video_results).reset_index(drop=True)
        batch_results.append(results)
        video_timer.stop_timer()
        if verbose:
            print(f"Time-bin movement calculations for video {video_name} complete (elapsed time: {video_timer.elapsed_time_str}s)...")

    return batch_results, video_dict, movement_dict


class TimeBinsMovementCalculatorMultiprocess(ConfigReader, FeatureExtractionMixin):
    """
    Computes aggregate movement statistics in user-defined time-bins using multiprocessing for improved performance.

    .. note::
        `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results>`__.

    .. note::
        On macOS (Darwin), multiprocessing start method is automatically set to 'spawn' for compatibility.

    .. image:: _static/img/TimeBinsMovementCalculator.png
       :width: 500
       :align: center

    .. seealso::
       For single core class, see :func:`simba.data_processors.timebins_movement_calculator.TimeBinsMovementCalculator`.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format.
    :param Union[int, float] bin_length: Time bin size in seconds. Must be greater than 0.
    :param List[str] body_parts: List of body part names to calculate movement for (e.g., ['Nose_1', 'Nose_2']). Body parts must exist in the project's body part configuration.
    :param Optional[List[Union[str, os.PathLike]]] data_path: Optional list of specific file paths to process. If None, processes all files in the project's outlier corrected directory. Can also be a single file path or a directory path containing CSV files.
    :param Optional[bool] plots: If True, creates time-bin line plots representing the movement in each time-bin per video. Results are saved in the ``project_folder/logs/`` sub-directory. Default: False.
    :param bool verbose: If True, prints progress information during processing. Default: True.
    :param int core_cnt: Number of CPU cores to use for multiprocessing. If -1, uses all available cores. If greater than available cores, uses all available cores. Must be greater than 0. Default: -1.

    :example:
    >>> calculator = TimeBinsMovementCalculatorMultiprocess(
    ...     config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
    ...     bin_length=5.0,
    ...     body_parts=['Nose_1', 'Nose_2'],
    ...     plots=True,
    ...     core_cnt=4
    ... )
    >>> calculator.run()
    >>> calculator.save()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 bin_length: Union[int, float],
                 body_parts: List[str],
                 data_path: Optional[Union[List[Union[str, os.PathLike]], str, os.PathLike]] = None,
                 plots: bool = False,
                 verbose: bool = True,
                 core_cnt: int = -1,
                 distance: bool = True,
                 velocity: bool = True,
                 transpose: bool = False,
                 include_timestamp: bool = False):

        ConfigReader.__init__(self, config_path=config_path)
        log_event(logger_name=str(self.__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()),)
        check_float(name=f"{self.__class__.__name__} TIME BIN", value=bin_length, min_value=10e-6)
        check_valid_lst(data=body_parts, source=f'{self.__class__.__name__} file_paths', min_len=1, valid_dtypes=(str,), valid_values=self.body_parts_lst)
        if data_path is None:
            if len(self.outlier_corrected_paths) == 0: raise NoDataError(msg=f'No data files found in {self.outlier_corrected_dir}', source=self.__class__.__name__)
            self.file_paths = self.outlier_corrected_paths
        elif os.path.isfile(data_path):
            self.file_paths = [data_path]
        elif os.path.isdir(data_path):
            self.file_paths = find_files_of_filetypes_in_directory(directory=self.file_paths, extensions=('.csv',), raise_warning=False, raise_error=True, as_dict=False)
        else:
            self.file_paths = data_path
        check_valid_boolean(value=[plots], source=f'{self.__class__.__name__} plots', raise_error=True)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=distance, source=f'{self.__class__.__name__} distance', raise_error=True)
        check_valid_boolean(value=velocity, source=f'{self.__class__.__name__} velocity', raise_error=True)
        check_valid_boolean(value=transpose, source=f'{self.__class__.__name__} transpose', raise_error=True)
        check_valid_boolean(value=include_timestamp, source=f'{self.__class__.__name__} include_timestamp', raise_error=True)
        check_int(f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0], raise_error=True)
        self.core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
        self.verbose, self.distance, self.velocity, self.transpose, self.include_timestamp = verbose, distance, velocity, transpose, include_timestamp
        if not distance and not velocity:
            raise InvalidInputError(msg='distance AND velocity are both False. To compute movement metrics, set at least one value to True.', source=self.__class__.__name__)
        self.col_headers, self.bp_dict = [], {}
        for bp_cnt, bp in enumerate(body_parts):
            self.col_headers.extend((f"{bp}_x", f"{bp}_y"))
            animal_name = self.find_animal_name_from_body_part_name(bp_name=bp, bp_dict=self.animal_bp_dict)
            self.bp_dict[bp_cnt] = {animal_name: [f"{bp}_x", f"{bp}_y"]}
        self.animal_combinations = list(itertools.combinations(self.animal_bp_dict, 2))
        self.bin_length, self.plots = bin_length, plots
        if platform.system() == "Darwin": multiprocessing.set_start_method("spawn", force=True)
        if verbose: print(f"Processing {len(self.file_paths)} video(s) for time-bins movement data... ({get_current_time()})")

    def __create_plots(self):
        timer = SimbaTimer(start=True)
        print("Creating time-bin movement plots...")
        plots_dir = os.path.join( self.project_path, "logs", f"time_bin_movement_plots_{self.datetime}")
        create_directory(paths=plots_dir, overwrite=True)
        for video_name in self.results["VIDEO"].unique():
            video_timer = SimbaTimer(start=True)
            if self.verbose: print(f'Creating plots for video {video_name}...')
            video_df = self.results.loc[(self.results["VIDEO"] == video_name) & (self.results["MEASUREMENT"] == "Movement (cm)")]
            video_df["TIME BIN #"] = video_df["TIME BIN #"].astype(int)
            for body_part in video_df["BODY-PART"].unique():
                body_part_df = (video_df[video_df["BODY-PART"] == body_part] .reset_index(drop=True) .sort_values(by=["TIME BIN #"]))
                body_part_df[f"Time bin # (bin length {self.bin_length}s)"] = (body_part_df["TIME BIN #"])
                body_part_df["VALUE"] = body_part_df["VALUE"].astype(float)
                _ = PlottingMixin.line_plot(df=body_part_df,
                                            x=f"Time bin # (bin length {self.bin_length}s)",
                                            y="VALUE", y_label="Distance (cm)",
                                            save_path=os.path.join(plots_dir, f"{video_name}_{body_part}.png"))
                video_timer.stop_timer()
                if self.verbose: print(f'Plot for video {video_name} complete (elapsed time: {video_timer.elapsed_time_str}s)')
        timer.stop_timer()
        stdout_success(msg=f"Time bin movement plots saved in {plots_dir}", elapsed_time=timer.elapsed_time_str, source=self.__class__.__name__)

    def run(self):
        self.video_dict, self.movement_dict, self.out_df_lst = {}, {}, []
        self.save_path = os.path.join( self.project_path, "logs", f"Time_bins_{self.bin_length}s_movement_results_{self.datetime}.csv")
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.file_paths)
        split_data_paths = [self.file_paths[i * len(self.file_paths) // self.core_cnt: (i + 1) * len(self.file_paths) // self.core_cnt] for i in range(self.core_cnt)]
        split_data_paths = [(x, i) for x, i in enumerate(split_data_paths)]

        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(_time_bin_movement_helper,
                                          bin_length=self.bin_length,
                                          verbose=self.verbose,
                                          video_info_df=self.video_info_df,
                                          bp_headers=self.col_headers,
                                          bp_dict=self.bp_dict,
                                          distance=self.distance,
                                          velocity=self.velocity)
            for cnt, result in enumerate(pool.imap(constants, split_data_paths, chunksize=self.multiprocess_chunksize)):
                results_df, video_dict, movement_dict = result
                self.out_df_lst.append(results_df)
                self.movement_dict.update(movement_dict)
                self.video_dict.update(video_dict)
            self.out_df_lst = [item for sub in self.out_df_lst for item in sub]

    def save(self):
        self.results = pd.concat(self.out_df_lst, axis=0).sort_values(by=["VIDEO", "TIME BIN #", "MEASUREMENT", "ANIMAL"])[["VIDEO", "TIME BIN #", "START TIME", "END TIME", "ANIMAL", "BODY-PART", "MEASUREMENT", "VALUE"]]
        if not self.include_timestamp:
            self.results = self.results.drop(["START TIME", "END TIME"], axis=1)
        if self.plots: self.__create_plots()
        if self.transpose:
            self.results = self.results.pivot_table(index=["VIDEO", "ANIMAL", "BODY-PART", "MEASUREMENT"], columns="TIME BIN #", values="VALUE").reset_index()
        self.results.set_index("VIDEO").to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(msg=f"Movement time-bins results for {len(self.file_paths)} videos ({self.bin_length}s bin size) saved at {self.save_path}", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)



# if __name__ == "__main__":
#     test = TimeBinsMovementCalculatorMultiprocess(config_path=r"/Users/simon/Desktop/envs/simba/troubleshooting/mitra/project_folder/project_config.ini",
#                                                   body_parts=['Nose'], #['Simon CENTER OF GRAVITY', 'JJ CENTER OF GRAVITY', 'Animal_1 CENTER OF GRAVITY']
#                                                   bin_length=0.5,
#                                                   plots=True)
#     test.run()
#     test.save()



# test = TimeBinsMovementCalculator(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini",
#                                   bin_length=0.1,
#                                   plots=True,
#                                   body_parts=['Nose_1'])
# test.run()
# test.save()
