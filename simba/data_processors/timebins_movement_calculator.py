__author__ = "Simon Nilsson; sronilsson@gmail.com"

import itertools
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.feature_extraction_supplement_mixin import \
    FeatureExtractionSupplemental
from simba.mixins.plotting_mixin import PlottingMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_float, check_that_column_exist,
    check_valid_boolean, check_valid_lst, check_valid_tuple)
from simba.utils.enums import TagNames
from simba.utils.errors import FrameRangeError, InvalidInputError, NoDataError
from simba.utils.printing import (SimbaTimer, log_event, stdout_information,
                                  stdout_success)
from simba.utils.read_write import (create_directory,
                                    find_files_of_filetypes_in_directory,
                                    find_time_stamp_from_frame_numbers,
                                    get_fn_ext, read_df)


class TimeBinsMovementCalculator(ConfigReader, FeatureExtractionMixin):
    """
    Compute aggregate movement and/or velocity statistics in user-defined time-bins.

    .. note::
        `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results>`__.

    .. image:: _static/img/TimeBinsMovementCalculator.png
       :width: 500
       :align: center

    .. seealso::
       For multicore processing, see :class:`simba.data_processors.timebins_movement_calculator_mp.TimeBinsMovementCalculator`.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file.
    :param Union[int, float] bin_length: Time-bin size in seconds.
    :param Union[List[str], Tuple[str]] body_parts: Body-part names to include in the movement calculations.
    :param Optional[Union[List[Union[str, os.PathLike]], Union[str, os.PathLike]]] data_path: Optional file path(s) to process. If ``None``, all outlier-corrected files in the project are used.
    :param bool plots: If ``True``, create per-video movement line plots for each body-part. Default: ``False``.
    :param verbose (bool): If True, prints progress messages during processing. Default: True.
    :param float threshold: Confidence threshold used when filtering low-confidence positions. Default: ``0.0``.
    :param bool distance: If ``True``, compute movement distance per time-bin. Default: ``True``.
    :param bool velocity: If ``True``, compute velocity per time-bin. Default: ``True``.
    :param bool transpose: If ``True``, save output in transposed format with one column per time-bin. Default: ``False``.
    :param bool include_timestamp: If ``True``, include start/end timestamps for each time-bin in saved results. Default: ``False``.

    :example:
    >>> calculator = TimeBinsMovementCalculator(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', bin_length=0.04, plots=True, body_parts=['Nose_1', 'Nose_2'])
    >>> calculator.run()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 bin_length: Union[int, float],
                 body_parts: Union[List[str], Tuple[str]],
                 data_path: Optional[Union[List[Union[str, os.PathLike]], Union[str, os.PathLike]]] = None,
                 plots: bool = False,
                 verbose: bool = True,
                 threshold: float = 0.00,
                 distance: bool = True,
                 velocity: bool = True,
                 transpose: bool = False,
                 include_timestamp: bool = False):

        ConfigReader.__init__(self, config_path=config_path)
        log_event(logger_name=str(self.__class__.__name__), log_type=TagNames.CLASS_INIT.value, msg=self.create_log_msg_from_init_args(locals=locals()),)
        check_float(name=f"{self.__class__.__name__} TIME BIN", value=bin_length, allow_negative=False, allow_zero=False)
        if isinstance(body_parts, list):
            check_valid_lst(data=body_parts, source=f'{self.__class__.__name__} body_parts', min_len=1, valid_dtypes=(str,), valid_values=self.body_parts_lst)
        elif isinstance(body_parts, tuple):
            check_valid_tuple(x=body_parts, source=f'{self.__class__.__name__} body_parts', minimum_length=1, valid_dtypes=(str,), accepted_values=self.body_parts_lst)
        else:
            raise InvalidInputError(msg='Body-parts has to be a list of tuple of strings', source=f'{self.__class__.__name__} body_parts')
        if data_path is None:
            if len(self.outlier_corrected_paths) == 0: raise NoDataError(msg=f'No data files found in {self.outlier_corrected_dir}', source=self.__class__.__name__)
            self.file_paths = self.outlier_corrected_paths
        elif isinstance(data_path, list):
            _ = [check_file_exist_and_readable(file_path=x, raise_error=True) for x in data_path]
            self.file_paths = data_path
        elif os.path.isdir(data_path):
            self.file_paths = find_files_of_filetypes_in_directory(directory=self.file_paths, extensions=('.csv',), raise_warning=False, raise_error=True, as_dict=False)
        elif isinstance(data_path, str):
            check_file_exist_and_readable(file_path=data_path, raise_error=True)
            self.file_paths = [data_path]
        check_valid_boolean(value=[plots], source=f'{self.__class__.__name__} plots', raise_error=True)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=distance, source=f'{self.__class__.__name__} distance', raise_error=True)
        check_valid_boolean(value=velocity, source=f'{self.__class__.__name__} velocity', raise_error=True)
        check_valid_boolean(value=transpose, source=f'{self.__class__.__name__} transpose', raise_error=True)
        check_valid_boolean(value=include_timestamp, source=f'{self.__class__.__name__} include_timestamp', raise_error=True)
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, allow_negative=False)
        self.verbose, self.distance, self.velocity, self.transpose, self.include_timestamp = verbose, distance, velocity, transpose, include_timestamp
        self.threshold = threshold
        if not distance and not velocity:
            raise InvalidInputError(msg='distance AND velocity are both False. To compute movement metrics, set at least one value to True.', source=self.__class__.__name__)
        self.col_headers, self.bp_dict = [], {}
        for bp_cnt, bp in enumerate(body_parts):
            self.col_headers.extend((f"{bp}_x", f"{bp}_y", f"{bp}_p"))
            animal_name = self.find_animal_name_from_body_part_name(bp_name=bp, bp_dict=self.animal_bp_dict)
            self.bp_dict[bp_cnt] = {animal_name: [f"{bp}_x", f"{bp}_y", f"{bp}_p"]}
        self.animal_combinations = list(itertools.combinations(self.animal_bp_dict, 2))
        self.bin_length, self.plots = bin_length, plots
        if verbose:
            stdout_information(msg=f"Processing {len(self.file_paths)} video(s) for time-bins movement data...")

    def __create_plots(self):
        timer = SimbaTimer(start=True)
        stdout_information(msg="Creating time-bin movement plots...")
        plots_dir = os.path.join( self.project_path, "logs", f"time_bin_movement_plots_{self.datetime}")
        create_directory(paths=plots_dir, overwrite=True)
        y_max = -np.inf
        for video_name in self.results["VIDEO"].unique():
            video_df = self.results.loc[(self.results["VIDEO"] == video_name) & (self.results["MEASUREMENT"] == "Movement (cm)")]
            for body_part in video_df["BODY-PART"].unique():
                body_part_df = (video_df[video_df["BODY-PART"] == body_part].reset_index(drop=True).sort_values(by=["TIME BIN #"]))
                body_part_df["VALUE"] = body_part_df["VALUE"].astype(float)
                y_max = max(y_max, np.max(body_part_df["VALUE"]))

        for video_name in self.results["VIDEO"].unique():
            video_df = self.results.loc[(self.results["VIDEO"] == video_name) & (self.results["MEASUREMENT"] == "Movement (cm)")]
            video_df["TIME BIN #"] = video_df["TIME BIN #"].astype(int)
            for body_part in video_df["BODY-PART"].unique():
                body_part_df = (video_df[video_df["BODY-PART"] == body_part] .reset_index(drop=True) .sort_values(by=["TIME BIN #"]))
                body_part_df[f"Time bin # (bin length {self.bin_length}s)"] = (body_part_df["TIME BIN #"])
                body_part_df["VALUE"] = body_part_df["VALUE"].astype(float)
                _ = PlottingMixin.make_line_plot(data=[body_part_df["VALUE"].astype(float).values],
                                                 colors=['Green'],
                                                 save_path=os.path.join(plots_dir, f"{video_name}_{body_part}.png"),
                                                 title=video_name,
                                                 y_max=int(y_max),
                                                 line_opacity=0.8,
                                                 y_lbl="DISTANCE (CM)",
                                                 x_lbl=f"TIME BIN # (BIN LENGTH {self.bin_length}s)",
                                                 x_tick_lbls_as_int=True,
                                                 y_tick_lbls_as_int=True,
                                                 x_tick_cnt=body_part_df["VALUE"].astype(float).values.shape[0]+1)
        timer.stop_timer()
        stdout_success(msg=f"Time bin movement plots saved in {plots_dir}", elapsed_time=timer.elapsed_time_str, source=self.__class__.__name__)

    def _remove_low_confidence_positions(self, arr, threshold):
        arr = arr.copy()
        valid = arr[:, -1] >= threshold

        last_valid = None
        for i in range(len(arr)):
            if valid[i]:
                last_valid = arr[i, :-1]
            elif last_valid is not None:
                arr[i, :-1] = last_valid

        return arr


    def run(self):
        video_dict, self.out_df_lst = {}, []
        self.movement_dict = {}
        self.save_path = os.path.join( self.project_path, "logs", f"Time_bins_{self.bin_length}s_movement_results_{self.datetime}.csv")
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.file_paths)
        for file_cnt, file_path in enumerate(self.file_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(file_path)
            if self.verbose:
                stdout_information(msg=f"Processing time-bin movements ({self.bin_length}s) for video {video_name} ({str(file_cnt+1)}/{str(len(self.file_paths))})...")
            video_dict[video_name] = {}
            video_settings, px_per_mm, fps = self.read_video_info(video_name=video_name)
            fps, self.movement_cols, self.velocity_cols = int(fps), set(), set()
            bin_length_frames = int(fps * self.bin_length)
            if bin_length_frames == 0:
                raise FrameRangeError(msg=f"The specified time-bin length of {self.bin_length} is TOO SHORT for video {video_name} which has a specified FPS of {fps}. This results in time bins that are LESS THAN a single frame.", source=self.__class__.__name__,)
            self.data_df = read_df(file_path, self.file_type)
            check_that_column_exist(df=self.data_df, column_name=self.col_headers, file_name=file_path)
            self.data_df, results = self.data_df[self.col_headers], []
            self.shifted_df = self.create_shifted_df(df=self.data_df)
            for animal_data in self.bp_dict.values():
                animal_name, animal_bps = list(animal_data.keys())[0], list(animal_data.values())[0]
                bp_time_1, bp_time_2 = (self.shifted_df[animal_bps].values[:, :2], self.shifted_df[[f"{animal_bps[0]}_shifted", f"{animal_bps[1]}_shifted"]].values,)
                self.movement_dict[video_name] = pd.DataFrame(self.framewise_euclidean_distance(location_1=bp_time_1.astype(np.float64), location_2=bp_time_2.astype(np.float64), px_per_mm=np.float64(px_per_mm), centimeter=True), columns=["VALUE"])
                animal_data = self.data_df[animal_bps].values.astype(np.float32)
                movement_lists = [animal_data[i : i + bin_length_frames] for i in range(0, animal_data.shape[0], bin_length_frames)]
                for bin, movement_bin_positions in enumerate(movement_lists):
                    bin_times = find_time_stamp_from_frame_numbers(start_frame=int(bin_length_frames*bin), end_frame=min(int(bin_length_frames*(bin+1)), len(self.data_df)), fps=fps)
                    if self.threshold > 0.0:
                        movement_bin_positions = self._remove_low_confidence_positions(arr=movement_bin_positions, threshold=self.threshold)
                    movement_bin_positions = movement_bin_positions[:, :2]
                    movement_bin_positions_shifted = self.create_shifted_array(data=movement_bin_positions, periods=1)
                    movement_df = pd.DataFrame(self.framewise_euclidean_distance(location_1=movement_bin_positions.astype(np.float64), location_2=movement_bin_positions_shifted.astype(np.float64), px_per_mm=np.float64(px_per_mm), centimeter=True), columns=["VALUE"])
                    movement, velocity = (FeatureExtractionSupplemental.distance_and_velocity(x=movement_df["VALUE"].values, fps=fps, pixels_per_mm=px_per_mm, centimeters=False))
                    if self.distance:
                        results.append({"VIDEO": video_name,"TIME BIN #": bin, "START TIME": bin_times[0], "END TIME": bin_times[1], "ANIMAL": animal_name,"BODY-PART": animal_bps[0][:-2],"MEASUREMENT": "Movement (cm)","VALUE": round(movement, 4)})
                    if self.velocity:
                        results.append({"VIDEO": video_name,"TIME BIN #": bin, "START TIME": bin_times[0], "END TIME": bin_times[1], "ANIMAL": animal_name,"BODY-PART": animal_bps[0][:-2],"MEASUREMENT": "Velocity (cm/s)","VALUE": round(velocity, 4)})
            results = pd.DataFrame(results).reset_index(drop=True)
            self.out_df_lst.append(results)
            video_timer.stop_timer()
            if self.verbose:
                stdout_information(msg=f"Time-bin movement calculations for video {video_name} complete...", elapsed_time=video_timer.elapsed_time_str)

    def save(self):
        self.results = pd.concat(self.out_df_lst, axis=0).sort_values(by=["VIDEO", "TIME BIN #", "MEASUREMENT", "ANIMAL"])[["VIDEO", "TIME BIN #", "START TIME", "END TIME", "ANIMAL", "BODY-PART", "MEASUREMENT", "VALUE"]]
        if not self.include_timestamp:
            self.results = self.results.drop(["START TIME", "END TIME"], axis=1)
        if self.plots: self.__create_plots()
        if self.transpose:
            self.results = self.results.pivot_table(index=["VIDEO", "ANIMAL", "BODY-PART", "MEASUREMENT"], columns="TIME BIN #", values="VALUE").reset_index()
        self.results.set_index("VIDEO").to_csv(self.save_path)
        self.timer.stop_timer()
        if self.verbose: stdout_success(msg=f"Movement time-bins results saved at {self.save_path}", elapsed_time=self.timer.elapsed_time_str, source=self.__class__.__name__)



# test = TimeBinsMovementCalculator(config_path=r"E:\troubleshooting\mitra_pbn\mitra_pbn\project_folder\project_config.ini",
#                                   body_parts= ('center',),
#                                   bin_length=60,
#                                   transpose=True,
#                                   threshold=0.859,
#                                   velocity=False,
#                                   plots=True)
# test.run()
# test.save()


# test = TimeBinsMovementCalculator(config_path=r"E:\troubleshooting\mitra_emergence_hour\project_folder\project_config.ini",
#                                   body_parts=['center'],
#                                   bin_length=600,
#                                   plots=False,
#                                   include_timestamp=False,
#                                   transpose=True,
#                                   velocity=False)
# test.run()
# test.save()



# test = TimeBinsMovementCalculator(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                           body_parts=['Nose'], #['Simon CENTER OF GRAVITY', 'JJ CENTER OF GRAVITY', 'Animal_1 CENTER OF GRAVITY']
#                           bin_length=10, plots=True)
# test.run()
# test.save()



# test = TimeBinsMovementCalculator(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini",
#                                   bin_length=0.1,
#                                   plots=True,
#                                   body_parts=['Nose_1'])
# test.run()
# test.save()
