import itertools
import os
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from simba.data_processors.timebins_movement_calculator import \
    TimeBinsMovementCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_supplement_mixin import \
    FeatureExtractionSupplemental
from simba.roi_tools.roi_aggregate_statistics_analyzer import \
    ROIAggregateStatisticsAnalyzer
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_float, check_if_dir_exists,
    check_valid_boolean, check_valid_lst)
from simba.utils.data import detect_bouts, slice_roi_dict_for_video
from simba.utils.errors import (CountError, FrameRangeError,
                                ROICoordinatesNotFoundError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_data_paths, read_df
from simba.utils.warnings import ROIWarning

SHAPE_TYPE = "Shape_type"
TOTAL_ROI_TIME = 'TOTAL ROI TIME (S)'
ENTRY_COUNTS = 'ROI ENTRIES (COUNTS)'
FIRST_ROI_ENTRY_TIME = 'FIRST ROI ENTRY TIME (S)'
LAST_ROI_ENTRY_TIME = 'LAST ROI ENTRY TIME (S)'
MEAN_BOUT_TIME = 'MEAN ROI BOUT TIME (S)'
VELOCITY = 'AVERAGE ROI VELOCITY (CM/S)'
MOVEMENT = 'TOTAL ROI MOVEMENT (CM)'
MEASUREMENT = 'MEASUREMENT'
VIDEO_FPS = 'VIDEO FPS'
VIDEO_LENGTH = 'VIDEO LENGTH (S)'
PIX_PER_MM = 'PIXEL TO MILLIMETER CONVERSION FACTOR'
OUTSIDE_ROI = 'OUTSIDE REGIONS OF INTEREST'


class ROITimebinAnalyzer(ConfigReader):
    """
    Analyzes region-of-interest (ROI) data from video tracking experiments conditioned on time-bin.

    This class computes various statistics related to body-part movements inside defined ROIs,
    including entry counts, total time spent, and bout durations.

    :param config_path (str | os.PathLike): Path to the configuration file.
    :param data_path (str | os.PathLike | List[str], optional): Path(s) to the data files.
    :param threshold (float): Probability threshold for body-part inclusion.
    :param body_parts (List[str], optional): List of body parts to analyze.
    :param detailed_bout_data (bool): Whether to compute detailed bout data.
    :param calculate_distances (bool): Whether to compute distances traveled.
    :param total_time (bool): Whether to calculate total time spent in ROIs.
    :param entry_counts (bool): Whether to count entries into ROIs.
    :param first_entry_time (bool): Whether to record the first entry time.
    :param include_time_stamps (bool): Whether to include time bin start times, start frame, end time and end frame in output.
    :param outside_rois (bool): If checked, SimBA will treat all areas NOT covered by a ROI drawing as a single additional ROI and compute the chosen metrics for this, single, ROI.
    :param last_entry_time (bool): Whether to record the last entry time.
    :param transpose (bool): Whether to transpose the final results.
    :param detailed_bout_data_save_path (str | os.PathLike, optional): Path to save detailed bout data.
    :param save_path (str | os.PathLike, optional): Path to save summary statistics.

    .. note::

    :example:
    >>> test = ROITimebinAnalyzer(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", bin_size=61, body_parts=['Nose'], detailed_bout_data=True, calculate_distances=True, transpose=True)
    >>> test.run()
    >>> test.save()

    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 bin_size: float,
                 data_path: Optional[Union[str, os.PathLike, List[str]]] = None,
                 threshold: float = 0.0,
                 body_parts: Optional[List[str]] = None,
                 detailed_bout_data: bool = False,
                 calculate_distances: bool = False,
                 total_time: bool = True,
                 entry_counts: bool = True,
                 first_entry_time: bool = False,
                 last_entry_time: bool = False,
                 transpose: bool = False,
                 outside_rois: bool = False,
                 verbose: bool = True,
                 include_fps: bool = False,
                 include_video_length: bool = False,
                 include_px_per_mm: bool = False,
                 include_time_stamps: bool = False,
                 detailed_bout_data_save_path: Optional[Union[str, os.PathLike]] = None,
                 save_path: Optional[Union[str, os.PathLike]] = None):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(expected_file_path=self.roi_coordinates_path)
        check_valid_lst(data=body_parts, source=f"{self.__class__.__name__} body-parts", valid_dtypes=(str,), valid_values=self.project_bps)
        check_float(name="Body-part probability threshold", value=threshold, min_value=0.0, max_value=1.0)
        check_float(name="bin_size", value=bin_size, min_value=10e-6)
        if len(set(body_parts)) != len(body_parts):
            raise CountError(msg=f"All body-part entries have to be unique. Got {body_parts}", source=self.__class__.__name__)
        if detailed_bout_data_save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(detailed_bout_data_save_path))
        else:
            detailed_bout_data_save_path = os.path.join(self.logs_path, f'{"Detailed_ROI_data"}_{self.datetime}.csv')
        if save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(save_path))
        else:
            save_path = os.path.join(self.logs_path, f"ROI_time_bins_{bin_size}s_data_{self.datetime}.csv")
        self.detailed_bout_data_save_path, self.save_path = detailed_bout_data_save_path, save_path
        check_valid_boolean(value=[detailed_bout_data], source=f'{self.__class__.__name__} detailed_bout_data', raise_error=True)
        check_valid_boolean(value=[total_time], source=f'{self.__class__.__name__} total_time', raise_error=True)
        check_valid_boolean(value=[entry_counts], source=f'{self.__class__.__name__} entry_counts', raise_error=True)
        check_valid_boolean(value=[first_entry_time], source=f'{self.__class__.__name__} first_entry_time', raise_error=True)
        check_valid_boolean(value=[last_entry_time], source=f'{self.__class__.__name__} last_entry_time', raise_error=True)
        check_valid_boolean(value=[calculate_distances], source=f'{self.__class__.__name__} calculate_distances', raise_error=True)
        check_valid_boolean(value=[transpose], source=f'{self.__class__.__name__} transpose', raise_error=True)
        check_valid_boolean(value=[include_fps], source=f'{self.__class__.__name__} include_fps', raise_error=True)
        check_valid_boolean(value=[include_video_length], source=f'{self.__class__.__name__} include_video_length', raise_error=True)
        check_valid_boolean(value=[include_px_per_mm], source=f'{self.__class__.__name__} include_px_per_mm', raise_error=True)
        check_valid_boolean(value=[include_time_stamps], source=f'{self.__class__.__name__} include_time_stamps', raise_error=True)
        check_valid_boolean(value=[outside_rois], source=f'{self.__class__.__name__} outside_rois', raise_error=True)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose', raise_error=True)
        self.read_roi_data()
        self.data_paths = read_data_paths(path=data_path, default=self.outlier_corrected_paths, default_name=self.outlier_corrected_dir, file_type=self.file_type)
        self.bp_dict, self.bp_lk = {}, {}
        for bp in body_parts:
            animal = self.find_animal_name_from_body_part_name(bp_name=bp, bp_dict=self.animal_bp_dict)
            self.bp_dict[animal] = [f'{bp}_{"x"}', f'{bp}_{"y"}', f'{bp}_{"p"}']
            self.bp_lk[animal] = bp
        self.roi_headers = [v for k, v in self.bp_dict.items()]
        self.roi_headers = [item for sublist in self.roi_headers for item in sublist]
        self.calculate_distances, self.threshold = calculate_distances, threshold
        self.detailed_bout_data = detailed_bout_data
        self.total_time, self.entry_counts, self.body_parts, self.include_time_stamps = total_time, entry_counts, body_parts, include_time_stamps
        self.first_entry_time, self.last_entry_time, self.include_px_per_mm = first_entry_time, last_entry_time, include_px_per_mm
        self.transpose, self.include_fps, self.include_video_length, self.bin_size, self.non_roi_zone = transpose, include_fps, include_video_length, bin_size, outside_rois
        self.detailed_dfs, self.detailed_df, self.verbose = [], [], verbose
        self.results = pd.DataFrame(columns=["VIDEO", "SHAPE", "ANIMAL", "BODY-PART", "TIME-BIN #", 'TIME-BIN START TIME (S)', 'TIME-BIN END TIME (S)', 'TIME-BIN START FRAME (#)', 'TIME-BIN END FRAME (#)', "MEASUREMENT", 'VALUE'])
        self.movement_timebins = TimeBinsMovementCalculator(config_path=config_path, bin_length=bin_size, body_parts=body_parts, plots=False, verbose=self.verbose)
        self.movement_timebins.run()

    def __clean_results(self):
        if not self.total_time:
            self.results = self.results[self.results[MEASUREMENT] != TOTAL_ROI_TIME]
        if not self.entry_counts:
            self.results = self.results[self.results[MEASUREMENT] != ENTRY_COUNTS]
        if not self.first_entry_time:
            self.results = self.results[self.results[MEASUREMENT] != FIRST_ROI_ENTRY_TIME]
        if not self.last_entry_time:
            self.results = self.results[self.results[MEASUREMENT] != LAST_ROI_ENTRY_TIME]
        if not self.calculate_distances:
            self.results = self.results[self.results[MEASUREMENT] != VELOCITY]
            self.results = self.results[self.results[MEASUREMENT] != MOVEMENT]
        if not self.include_fps:
            self.results = self.results[self.results[MEASUREMENT] != VIDEO_FPS]
        if not self.include_video_length:
            self.results = self.results[self.results[MEASUREMENT] != VIDEO_LENGTH]
        if not self.include_px_per_mm:
            self.results = self.results[self.results[MEASUREMENT] != PIX_PER_MM]
        if not self.include_time_stamps:
            self.results = self.results.drop(['TIME-BIN START TIME (S)', 'TIME-BIN END TIME (S)', 'TIME-BIN START FRAME (#)', 'TIME-BIN END FRAME (#)'], axis=1)
        self.results = self.results.sort_values(by=["VIDEO", "ANIMAL", "BODY-PART", "SHAPE", "TIME-BIN #", "MEASUREMENT"]).reset_index(drop=True)
        if self.transpose:
            self.results['VALUE'] = pd.to_numeric(self.results['VALUE'], errors='coerce')
            if not self.include_time_stamps:
                self.results = self.results.pivot_table(index=["VIDEO"], columns=["ANIMAL", "SHAPE", "TIME-BIN #", "MEASUREMENT"], values="VALUE")
            else:
                self.results = self.results.pivot_table(index=["VIDEO"], columns=["ANIMAL", "SHAPE", "TIME-BIN #", 'TIME-BIN START TIME (S)', 'TIME-BIN END TIME (S)', 'TIME-BIN START FRAME (#)', 'TIME-BIN END FRAME (#)',  "MEASUREMENT"], values="VALUE")
            self.results = self.results.fillna(value='None')
        else:
            self.results = self.results.set_index('VIDEO')

        if self.detailed_bout_data and (len(self.detailed_df) > 0):
            self.detailed_df = self.detailed_df.rename(columns={"Event": "SHAPE NAME", "Start_time": "START TIME", "End Time": "END TIME", "Start_frame": "START FRAME", "End_frame": "END FRAME", "Bout_time": "DURATION (S)"})
            self.detailed_df["BODY-PART"] = self.detailed_df["ANIMAL"].map(self.bp_lk)
            self.detailed_df = self.detailed_df[["VIDEO", "ANIMAL", "BODY-PART", "SHAPE NAME", "START TIME", "END TIME", "START FRAME", "END FRAME", "DURATION (S)"]].reset_index(drop=True)

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        roi_analyzer = ROIAggregateStatisticsAnalyzer(config_path=self.config_path,
                                                      data_path=self.data_paths,
                                                      threshold=self.threshold,
                                                      body_parts=self.body_parts,
                                                      detailed_bout_data=True,
                                                      verbose=self.verbose)
        roi_analyzer.run()
        self.detailed_df = pd.concat(roi_analyzer.detailed_dfs, axis=0)
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(filepath=file_path)
            if self.verbose: print(f"Analysing ROI data for video {video_name}... (Video {file_cnt + 1}/{len(self.data_paths)})")
            _, px_per_mm, fps = self.read_video_info(video_name=video_name)
            video_df = self.detailed_df[self.detailed_df['VIDEO'] == video_name].reset_index(drop=True)
            _, video_shape_names = slice_roi_dict_for_video(data=self.roi_dict, video_name=video_name)
            if len(video_shape_names) == 0:
                ROIWarning(msg=f'Video {video_name} has no drawn ROIs. Skipping video ROI time-bin analysis.', source=self.__class__.__name__)
            frames_per_bin = int(fps * self.bin_size)
            if frames_per_bin == 0:
                raise FrameRangeError(msg=f"The specified time-bin length of {self.bin_size} is TOO SHORT for video {video_name} which has a specified FPS of {fps}. This combination produces time bins that are LESS THAN a single frame.", source=self.__class__.__name__)
            video_frms = list(range(0, len(read_df(file_path=file_path, file_type=self.file_type))))
            video_length = len(video_frms) / fps
            frame_bins = [video_frms[i: i + (frames_per_bin)] for i in range(0, len(video_frms), frames_per_bin)]
            for animal_name, roi_name in list(itertools.product(self.bp_dict.keys(), video_shape_names)):
                bp_name = self.bp_lk[animal_name]
                video_roi_animal_df = video_df.loc[(video_df["Event"] == roi_name) & (video_df["ANIMAL"] == animal_name)]
                entry_frms = list(video_roi_animal_df['Start_frame'])
                inside_roi_frm_idx = [list(range(x, y)) for x, y in zip(list(video_roi_animal_df['Start_frame'].astype(int)), list(video_roi_animal_df["End_frame"].astype(int) + 1))]
                inside_roi_frm_idx = [i for s in inside_roi_frm_idx for i in s]
                for bin_cnt, bin_frms in enumerate(frame_bins):
                    if self.verbose: print(f"Analysing time-bin {bin_cnt+1} (bin: {bin_cnt+1}/{len(frame_bins)}, roi: {roi_name}, animal: {animal_name}, video: {file_cnt + 1}/{len(self.data_paths)}, video name: {video_name})")
                    bin_start_time, bin_end_time = bin_frms[0] / fps, bin_frms[-1] / fps
                    bin_start_frm, bin_end_frm = bin_frms[0], bin_frms[-1]
                    frms_inside_roi_in_timebin_idx = [x for x in inside_roi_frm_idx if x in bin_frms]
                    entry_roi_in_timebin_idx = [x for x in entry_frms if x in bin_frms]
                    if len(frms_inside_roi_in_timebin_idx) > 0:
                        total_time = len(frms_inside_roi_in_timebin_idx) / fps
                        bin_move = (self.movement_timebins.movement_dict[video_name].iloc[frms_inside_roi_in_timebin_idx].values.flatten().astype(np.float32))
                        distance, velocity = FeatureExtractionSupplemental.distance_and_velocity(x=bin_move, fps=fps, pixels_per_mm=1, centimeters=False)
                    else:
                        total_time = 0
                        distance, velocity = 0, None
                    if len(entry_roi_in_timebin_idx) > 0:
                        entry_counts = len(entry_roi_in_timebin_idx)
                        first_entry_time = entry_roi_in_timebin_idx[0] / fps
                        last_entry_time = entry_roi_in_timebin_idx[-1] / fps
                    else:
                        entry_counts = 0
                        first_entry_time = None
                        last_entry_time = None

                    self.results.loc[len(self.results)] = [video_name, roi_name, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, TOTAL_ROI_TIME, total_time]
                    self.results.loc[len(self.results)] = [video_name, roi_name, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, ENTRY_COUNTS, entry_counts]
                    self.results.loc[len(self.results)] = [video_name, roi_name, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, FIRST_ROI_ENTRY_TIME, first_entry_time]
                    self.results.loc[len(self.results)] = [video_name, roi_name, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, LAST_ROI_ENTRY_TIME, last_entry_time]
                    self.results.loc[len(self.results)] = [video_name, roi_name, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, MOVEMENT,  distance]
                    self.results.loc[len(self.results)] = [video_name, roi_name, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, VELOCITY, velocity]
                    self.results.loc[len(self.results)] = [video_name, roi_name, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, VIDEO_FPS, fps]
                    self.results.loc[len(self.results)] = [video_name, roi_name, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, VIDEO_LENGTH, video_length]
                    self.results.loc[len(self.results)] = [video_name, roi_name, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, PIX_PER_MM, px_per_mm]

            if self.non_roi_zone:
                for animal_name in self.bp_dict.keys():
                    video_animal_df = video_df.loc[video_df["ANIMAL"] == animal_name]
                    video_animal_df = video_animal_df.rename(columns={"Event": "SHAPE NAME", "Start_time": "START TIME", "End Time": "END TIME", "Start_frame": "START FRAME", "End_frame": "END FRAME", "Bout_time": "DURATION (S)"})
                    inside_rois_frm = [i for s, e in zip(video_animal_df['START FRAME'], video_animal_df['END FRAME'])  for i in range(s, e + 1)]
                    outside_rois_frm = pd.DataFrame(data=[1 if i not in inside_rois_frm else 0 for i in range(len(video_frms))], columns=[OUTSIDE_ROI])
                    outside_roi_bouts = detect_bouts(data_df=outside_rois_frm, target_lst=OUTSIDE_ROI, fps=fps)
                    outside_roi_frm_idx = [list(range(x, y)) for x, y in zip(list(outside_roi_bouts['Start_frame'].astype(int)), list(outside_roi_bouts["End_frame"].astype(int) + 1))]
                    outside_roi_frm_idx = [i for s in outside_roi_frm_idx for i in s]


                    for bin_cnt, bin_frms in enumerate(frame_bins):
                        if self.verbose: print(f"Analysing OUTSIDE ROI time-bin {bin_cnt + 1} (bin: {bin_cnt + 1}/{len(frame_bins)}, roi: {OUTSIDE_ROI}, animal: {animal_name}, video: {file_cnt + 1}/{len(self.data_paths)}, video name: {video_name})")
                        bin_start_time, bin_end_time = bin_frms[0] / fps, bin_frms[-1] / fps
                        bin_start_frm, bin_end_frm = bin_frms[0], bin_frms[-1]
                        frms_outside_roi_in_timebin_idx = [x for x in outside_roi_frm_idx if x in bin_frms]
                        entry_roi_in_timebin_idx = [x for x in entry_frms if x in bin_frms]
                        if len(frms_outside_roi_in_timebin_idx) > 0:
                            total_time = len(frms_outside_roi_in_timebin_idx) / fps
                            bin_move = (self.movement_timebins.movement_dict[video_name].iloc[frms_outside_roi_in_timebin_idx].values.flatten().astype(np.float32))
                            distance, velocity = FeatureExtractionSupplemental.distance_and_velocity(x=bin_move, fps=fps, pixels_per_mm=1, centimeters=False)
                        else:
                            total_time = 0
                            distance, velocity = 0, None
                        if len(entry_roi_in_timebin_idx) > 0:
                            entry_counts = len(entry_roi_in_timebin_idx)
                            first_entry_time = entry_roi_in_timebin_idx[0] / fps
                            last_entry_time = entry_roi_in_timebin_idx[-1] / fps
                        else:
                            entry_counts = 0
                            first_entry_time = None
                            last_entry_time = None

                        self.results.loc[len(self.results)] = [video_name, OUTSIDE_ROI, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, TOTAL_ROI_TIME, total_time]
                        self.results.loc[len(self.results)] = [video_name, OUTSIDE_ROI, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, ENTRY_COUNTS, entry_counts]
                        self.results.loc[len(self.results)] = [video_name, OUTSIDE_ROI, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, FIRST_ROI_ENTRY_TIME, first_entry_time]
                        self.results.loc[len(self.results)] = [video_name, OUTSIDE_ROI, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, LAST_ROI_ENTRY_TIME, last_entry_time]
                        self.results.loc[len(self.results)] = [video_name, OUTSIDE_ROI, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, MOVEMENT,  distance]
                        self.results.loc[len(self.results)] = [video_name, OUTSIDE_ROI, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, VELOCITY, velocity]
                        self.results.loc[len(self.results)] = [video_name, OUTSIDE_ROI, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, VIDEO_FPS, fps]
                        self.results.loc[len(self.results)] = [video_name, OUTSIDE_ROI, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, VIDEO_LENGTH, video_length]
                        self.results.loc[len(self.results)] = [video_name, OUTSIDE_ROI, animal_name, bp_name, bin_cnt, bin_start_time, bin_end_time, bin_start_frm, bin_end_frm, PIX_PER_MM, px_per_mm]

            video_timer.stop_timer()
            if self.verbose: print(f'ROI analysis video {video_name} complete... (elapsed time: {video_timer.elapsed_time_str}s)')


    def save(self):
        self.__clean_results()
        if self.detailed_bout_data and len(self.detailed_df) > 0:
            self.detailed_df.to_csv(self.detailed_bout_data_save_path)
            print(f"Detailed ROI data saved at {self.detailed_bout_data_save_path}...")
        self.results.to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(f'ROI statistics saved at {self.save_path}', elapsed_time=self.timer.elapsed_time_str)


# test = ROITimebinAnalyzer(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                           bin_size=500,
#                           body_parts=['Nose'],
#                           detailed_bout_data=True,
#                           calculate_distances=True,
#                           transpose=False,
#                           outside_rois=True,
#                           verbose=True)
# test.run()
# test.save()
# #



# test = ROITimebinCalculator(config_path=r"C:\troubleshooting\spontenous_alternation\project_folder\project_config.ini",
#                             bin_length=0.5,
#                             body_parts=['nose'],
#                             threshold=0.00,
#                             movement=True)
# test.run()
# test.save()


# test = ROITimebinCalculator(config_path=r"C:\troubleshooting\ROI_movement_test\project_folder\project_config.ini",
#                             bin_length=0.5,
#                             body_parts=['Head'],
#                             threshold=0.00,
#                             movement=True)
# test.run()
# test.save()

# test = ROITimebinCalculator(config_path=r"/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini",
#                             bin_length=1,
#                             body_parts=['Nose_1'],
#                             threshold=0.00,
#                             movement=True)
# test.run()
# test.save()
