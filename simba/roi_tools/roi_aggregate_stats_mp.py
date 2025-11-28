__author__ = "Simon Nilsson"

import os
from typing import List, Optional, Tuple, Union
import functools
import multiprocessing
import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.feature_extraction_supplement_mixin import \
    FeatureExtractionSupplemental
from simba.roi_tools.roi_utils import get_roi_dict_from_dfs
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log,
    check_file_exist_and_readable, check_float, check_if_dir_exists,
    check_that_column_exist, check_valid_boolean, check_valid_lst, check_int)
from simba.utils.data import detect_bouts, slice_roi_dict_for_video
from simba.utils.enums import ROI_SETTINGS, Keys, Formats
from simba.utils.errors import CountError, ROICoordinatesNotFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_data_paths, read_df, find_core_cnt, read_video_info
from simba.utils.warnings import NoDataFoundWarning

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

def _agg_roi_stats_helper(data_paths: list,
                          roi_dict: dict,
                          threshold: float,
                          verbose: bool,
                          non_roi_zone: bool,
                          video_info_df: pd.DataFrame,
                          roi_headers: list,
                          bp_dict: dict):

    batch_id, data_paths = data_paths
    results = pd.DataFrame(columns=["VIDEO", "ANIMAL", "BODY-PART", "SHAPE", "SHAPE TYPE", "MEASUREMENT", "VALUE"])
    detailed_dfs = []
    for file_cnt, file_path in enumerate(data_paths):
        video_timer = SimbaTimer(start=True)
        _, video_name, _ = get_fn_ext(file_path)
        if verbose: print(f"Analysing ROI data for video {video_name}... (Video {file_cnt+1}/{len(data_paths)}, core: {batch_id}")
        video_settings, pix_per_mm, fps = read_video_info(video_name=video_name, video_info_df=video_info_df)
        sliced_roi_dict, video_shape_names = slice_roi_dict_for_video(data=roi_dict, video_name=video_name)
        if len(video_shape_names) == 0:
            NoDataFoundWarning(msg=f"Skipping video {video_name}: No user-defined ROI data found for this video...")
            continue
        sliced_roi_dict = get_roi_dict_from_dfs(rectangle_df=sliced_roi_dict[Keys.ROI_RECTANGLES.value],circle_df=sliced_roi_dict[Keys.ROI_CIRCLES.value],polygon_df=sliced_roi_dict[Keys.ROI_POLYGONS.value])
        data_df = read_df(file_path, Formats.CSV.value).reset_index(drop=True)
        check_that_column_exist(df=data_df, column_name=roi_headers, file_name=file_path)
        for animal_name, bp_cols in bp_dict.items():
            p_arr = data_df[bp_cols[2]].values.astype(np.float32)
            below_threshold_idx = np.argwhere(p_arr < threshold)
            bp_arr = data_df[[bp_cols[0], bp_cols[1]]].values.astype(np.float32)
            animal_roi_bouts = []
            for roi_cnt, (roi_name, roi_data) in enumerate(sliced_roi_dict.items()):
                if roi_data[SHAPE_TYPE].lower() == ROI_SETTINGS.RECTANGLE.value.lower():
                    roi_coords = np.array([[roi_data['topLeftX'], roi_data['topLeftY']], [roi_data['Bottom_right_X'], roi_data['Bottom_right_Y']]])
                    r = FeatureExtractionMixin.framewise_inside_rectangle_roi(bp_location=bp_arr, roi_coords=roi_coords)
                elif roi_data[SHAPE_TYPE].lower() == ROI_SETTINGS.CIRCLE.value.lower():
                    circle_center = np.array([roi_data['Center_X'], roi_data['Center_Y']]).astype(np.int32)
                    r = FeatureExtractionMixin.is_inside_circle(bp=bp_arr, roi_center=circle_center, roi_radius=roi_data['radius'])
                else:
                    vertices = roi_data['vertices'].astype(np.int32)
                    r = FeatureExtractionMixin.framewise_inside_polygon_roi(bp_location=bp_arr, roi_coords=vertices)
                r[below_threshold_idx] = 0
                data_df[roi_data['Name']] = r
                roi_bouts = detect_bouts(data_df=data_df, target_lst=[roi_data["Name"]], fps=fps)
                if len(roi_bouts) > 0:
                    total_time = roi_bouts['Bout_time'].sum()
                    entry_counts = len(roi_bouts)
                    first_entry_time = roi_bouts['Start_time'].values[0]
                    last_entry_time = roi_bouts['Start_time'].values[-1]
                    mean_bout_time = roi_bouts['Bout_time'].mean()
                    movement, velocity = FeatureExtractionSupplemental.movement_stats_from_bouts_df(bp_data=bp_arr, event_name=roi_data["Name"], bout_df=roi_bouts, fps=fps, px_per_mm=pix_per_mm)
                    roi_bouts['VIDEO'] = video_name
                    roi_bouts['ANIMAL'] = animal_name
                    detailed_dfs.append(roi_bouts)
                    animal_roi_bouts.append(roi_bouts)
                else:
                    total_time, entry_counts = 0, 0
                    first_entry_time, last_entry_time = None, None
                    mean_bout_time = None
                    movement, velocity = 0, None
                results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], roi_name, roi_data[SHAPE_TYPE], TOTAL_ROI_TIME, total_time]
                results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], roi_name, roi_data[SHAPE_TYPE], ENTRY_COUNTS, entry_counts]
                results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], roi_name, roi_data[SHAPE_TYPE], FIRST_ROI_ENTRY_TIME, first_entry_time]
                results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], roi_name, roi_data[SHAPE_TYPE], LAST_ROI_ENTRY_TIME, last_entry_time]
                results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], roi_name, roi_data[SHAPE_TYPE], MEAN_BOUT_TIME, mean_bout_time]
                results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], roi_name, roi_data[SHAPE_TYPE], VELOCITY, velocity]
                results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], roi_name, roi_data[SHAPE_TYPE], MOVEMENT, movement]
                results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], roi_name, roi_data[SHAPE_TYPE], VIDEO_FPS, fps]
                results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], roi_name, roi_data[SHAPE_TYPE], VIDEO_LENGTH, len(data_df) / fps]
                results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], roi_name, roi_data[SHAPE_TYPE], PIX_PER_MM, pix_per_mm]

            if non_roi_zone and len(animal_roi_bouts) > 0:
                animal_roi_bouts = pd.concat(animal_roi_bouts, axis=0)
                animal_roi_bouts = animal_roi_bouts.rename(columns={"Event": "SHAPE NAME", "Start_time": "START TIME", "End Time": "END TIME", "Start_frame": "START FRAME", "End_frame": "END FRAME", "Bout_time": "DURATION (S)"})
                inside_rois_frm = [i for s, e in zip(animal_roi_bouts['START FRAME'], animal_roi_bouts['END FRAME']) for i in range(s, e+1)]
                data_df[OUTSIDE_ROI] = 1
                data_df.loc[inside_rois_frm, OUTSIDE_ROI] = 0
                outside_roi_bouts = detect_bouts(data_df=data_df, target_lst=OUTSIDE_ROI, fps=fps)
                if len(outside_roi_bouts) > 0:
                    total_time = outside_roi_bouts['Bout_time'].sum()
                    entry_counts = len(outside_roi_bouts)
                    first_entry_time = outside_roi_bouts['Start_time'].values[0]
                    last_entry_time = outside_roi_bouts['Start_time'].values[-1]
                    mean_bout_time = outside_roi_bouts['Bout_time'].mean()
                    movement, velocity = FeatureExtractionSupplemental.movement_stats_from_bouts_df(bp_data=bp_arr, event_name=OUTSIDE_ROI, bout_df=outside_roi_bouts, fps=fps, px_per_mm=pix_per_mm)
                    outside_roi_bouts['VIDEO'] = video_name
                    outside_roi_bouts['ANIMAL'] = animal_name
                    detailed_dfs.append(outside_roi_bouts)
                    results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], OUTSIDE_ROI, 'NONE', TOTAL_ROI_TIME, total_time]
                    results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], OUTSIDE_ROI, 'NONE', ENTRY_COUNTS, entry_counts]
                    results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], OUTSIDE_ROI, 'NONE', FIRST_ROI_ENTRY_TIME, first_entry_time]
                    results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], OUTSIDE_ROI, 'NONE', LAST_ROI_ENTRY_TIME, last_entry_time]
                    results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], OUTSIDE_ROI, 'NONE', MEAN_BOUT_TIME, mean_bout_time]
                    results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], OUTSIDE_ROI, 'NONE', VELOCITY, velocity]
                    results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], OUTSIDE_ROI, 'NONE', MOVEMENT, movement]
                    results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], OUTSIDE_ROI, 'NONE', VIDEO_FPS, fps]
                    results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], OUTSIDE_ROI, 'NONE', VIDEO_LENGTH, len(data_df) / fps]
                    results.loc[len(results)] = [video_name, animal_name, bp_cols[0][:-2], OUTSIDE_ROI, 'NONE', PIX_PER_MM, pix_per_mm]
        video_timer.stop_timer()
        if verbose: print(f'ROI analysis video {video_name} complete... (elapsed time: {video_timer.elapsed_time_str}s)')
    detailed_dfs = pd.concat(detailed_dfs, axis=0).reset_index(drop=True)
    return batch_id, results, detailed_dfs


class ROIAggregateStatisticsAnalyzer(ConfigReader, FeatureExtractionMixin):
    """
    Analyzes region-of-interest (ROI) data from video tracking experiments.

    This class computes various statistics related to body-part movements inside defined ROIs, including entry counts, total time spent, and bout durations.

    :param config_path (str | os.PathLike): Path to the configuration file.
    :param data_path (str | os.PathLike | List[str], optional): Path(s) to the data files.
    :param threshold (float): Probability threshold for body-part inclusion.
    :param body_parts (List[str], optional): List of body parts to analyze.
    :param detailed_bout_data (bool): Whether to compute detailed bout data.
    :param calculate_distances (bool): Whether to compute distances traveled.
    :param total_time (bool): Whether to calculate total time spent in ROIs.
    :param entry_counts (bool): Whether to count entries into ROIs.
    :param first_entry_time (bool): Whether to record the first entry time.
    :param outside_rois (bool): If checked, SimBA will treat all areas NOT covered by a ROI drawing as a single additional ROI and compute the chosen metrics for this, single, ROI.
    :param last_entry_time (bool): Whether to record the last entry time.
    :param mean_bout_time (bool): Whether to compute mean bout duration.
    :param transpose (bool): Whether to transpose the final results.
    :param detailed_bout_data_save_path (str | os.PathLike, optional): Path to save detailed bout data.
    :param save_path (str | os.PathLike, optional): Path to save summary statistics.

    :example:
    >>> analyzer = ROIAggregateStatisticsAnalyzer(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", body_parts=['Center'], first_entry_time=True, threshold=0.0, calculate_distances=True, transpose=False, detailed_bout_data=True)
    >>> analyzer.run()
    >>> analyzer.save()
    """


    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 data_path: Optional[Union[str, os.PathLike, List[str]]] = None,
                 threshold: float = 0.0,
                 body_parts: Optional[List[str]] = None,
                 detailed_bout_data: bool = False,
                 calculate_distances: bool = False,
                 total_time: bool = True,
                 entry_counts: bool = True,
                 first_entry_time: bool = False,
                 last_entry_time: bool = False,
                 mean_bout_time: bool = False,
                 outside_rois: bool = False,
                 transpose: bool = False,
                 include_fps: bool = False,
                 include_video_length: bool = False,
                 include_px_per_mm: bool = False,
                 verbose: bool = True,
                 detailed_bout_data_save_path: Optional[Union[str, os.PathLike]] = None,
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 core_cnt: int = -1):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(expected_file_path=self.roi_coordinates_path)
        check_valid_lst(data=body_parts, source=f"{self.__class__.__name__} body-parts", valid_dtypes=(str,), valid_values=self.project_bps)
        check_float(name="Body-part probability threshold", value=threshold, min_value=0.0, max_value=1.0)
        if len(set(body_parts)) != len(body_parts):
            raise CountError(msg=f"All body-part entries have to be unique. Got {body_parts}", source=self.__class__.__name__)
        if detailed_bout_data_save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(detailed_bout_data_save_path))
        else:
            detailed_bout_data_save_path = os.path.join(self.logs_path, f'{"Detailed_ROI_data"}_{self.datetime}.csv')
        if save_path is not None:
            check_if_dir_exists(in_dir=os.path.dirname(save_path))
        else:
            save_path = os.path.join(self.logs_path, f'{"ROI_descriptive_statistics"}_{self.datetime}.csv')
        self.detailed_bout_data_save_path, self.save_path = detailed_bout_data_save_path, save_path
        check_valid_boolean(value=[detailed_bout_data], source=f'{self.__class__.__name__} detailed_bout_data', raise_error=True)
        check_valid_boolean(value=[total_time], source=f'{self.__class__.__name__} total_time', raise_error=True)
        check_valid_boolean(value=[entry_counts], source=f'{self.__class__.__name__} entry_counts', raise_error=True)
        check_valid_boolean(value=[first_entry_time], source=f'{self.__class__.__name__} first_entry_time', raise_error=True)
        check_valid_boolean(value=[last_entry_time], source=f'{self.__class__.__name__} last_entry_time', raise_error=True)
        check_valid_boolean(value=[mean_bout_time], source=f'{self.__class__.__name__} mean_bout_time', raise_error=True)
        check_valid_boolean(value=[calculate_distances], source=f'{self.__class__.__name__} calculate_distances', raise_error=True)
        check_valid_boolean(value=[transpose], source=f'{self.__class__.__name__} transpose', raise_error=True)
        check_valid_boolean(value=[include_fps], source=f'{self.__class__.__name__} include_fps', raise_error=True)
        check_valid_boolean(value=[include_video_length], source=f'{self.__class__.__name__} include_video_length', raise_error=True)
        check_valid_boolean(value=[include_px_per_mm], source=f'{self.__class__.__name__} include_px_per_mm', raise_error=True)
        check_valid_boolean(value=[outside_rois], source=f'{self.__class__.__name__} outside_rois', raise_error=True)
        check_valid_boolean(value=[verbose], source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_int(name=f'{self.__class__.__name__} core_cnt', value=core_cnt, min_value=-1, unaccepted_vals=[0], raise_error=True)
        self.core_cnt = find_core_cnt()[0] if core_cnt == -1 or core_cnt > find_core_cnt()[0] else core_cnt
        self.read_roi_data()
        FeatureExtractionMixin.__init__(self)
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
        self.total_time, self.entry_counts = total_time, entry_counts
        self.first_entry_time, self.last_entry_time, self.mean_bout_time, self.include_px_per_mm = first_entry_time, last_entry_time, mean_bout_time, include_px_per_mm
        self.transpose, self.include_fps, self.include_video_length, self.non_roi_zone = transpose, include_fps, include_video_length, outside_rois
        self.detailed_df, self.verbose = [], verbose


    def __clean_results(self):
        if not self.total_time:
            self.results = self.results[self.results[MEASUREMENT] != TOTAL_ROI_TIME]
        if not self.entry_counts:
            self.results = self.results[self.results[MEASUREMENT] != ENTRY_COUNTS]
        if not self.first_entry_time:
            self.results = self.results[self.results[MEASUREMENT] != FIRST_ROI_ENTRY_TIME]
        if not self.last_entry_time:
            self.results = self.results[self.results[MEASUREMENT] != LAST_ROI_ENTRY_TIME]
        if not self.mean_bout_time:
            self.results = self.results[self.results[MEASUREMENT] != MEAN_BOUT_TIME]
        if not self.calculate_distances:
            self.results = self.results[self.results[MEASUREMENT] != VELOCITY]
            self.results = self.results[self.results[MEASUREMENT] != MOVEMENT]
        if not self.include_fps:
            self.results = self.results[self.results[MEASUREMENT] != VIDEO_FPS]
        if not self.include_video_length:
            self.results = self.results[self.results[MEASUREMENT] != VIDEO_LENGTH]
        if not self.include_px_per_mm:
            self.results = self.results[self.results[MEASUREMENT] != PIX_PER_MM]
        self.results = self.results.sort_values(by=["VIDEO", "ANIMAL", "BODY-PART", "SHAPE", "SHAPE TYPE", "MEASUREMENT"]).reset_index(drop=True)
        if self.transpose:
            self.results['VALUE'] = pd.to_numeric(self.results['VALUE'], errors='coerce')
            self.results = self.results.pivot_table(index=["VIDEO"], columns=["ANIMAL", "SHAPE", "MEASUREMENT"], values="VALUE")
            self.results = self.results.fillna(value='None')
        else:
            self.results = self.results.set_index('VIDEO')

        if self.detailed_bout_data and (len(self.detailed_dfs) > 0):
            self.detailed_df = pd.concat(self.detailed_dfs, axis=0)
            self.detailed_df = self.detailed_df.rename(columns={"Event": "SHAPE NAME", "Start_time": "START TIME", "End Time": "END TIME", "Start_frame": "START FRAME", "End_frame": "END FRAME", "Bout_time": "DURATION (S)"})
            self.detailed_df["BODY-PART"] = self.detailed_df["ANIMAL"].map(self.bp_lk)
            self.detailed_df = self.detailed_df[["VIDEO", "ANIMAL", "BODY-PART", "SHAPE NAME", "START TIME", "END TIME", "START FRAME", "END FRAME", "DURATION (S)"]].reset_index(drop=True)

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.data_paths)
        chunked_data_paths = [self.data_paths[i:i + ((len(self.data_paths) + self.core_cnt - 1) // self.core_cnt)] for i in range(0, len(self.data_paths), (len(self.data_paths) + self.core_cnt - 1) // self.core_cnt)]
        chunked_data_paths = [(i, x) for i, x in enumerate(chunked_data_paths)]
        self.results, self.detailed_dfs = [], []
        with multiprocessing.Pool(self.core_cnt, maxtasksperchild=self.maxtasksperchild) as pool:
            constants = functools.partial(_agg_roi_stats_helper,
                                          roi_dict=self.roi_dict,
                                          threshold=self.threshold,
                                          verbose=self.verbose,
                                          video_info_df=self.video_info_df,
                                          roi_headers=self.roi_headers,
                                          bp_dict=self.bp_dict,
                                          non_roi_zone=self.non_roi_zone)
            for cnt, (batch_id, result, detailed_dfs) in enumerate(pool.map(constants, chunked_data_paths, chunksize=self.multiprocess_chunksize)):
                self.results.append(result); self.detailed_dfs.append(detailed_dfs)
                print(f"Data batch core {batch_id} / {self.core_cnt} complete...")
        self.results = pd.concat(self.results, axis=0).reset_index(drop=True)
        pool.join()
        pool.terminate()

    def save(self):
        self.__clean_results()
        if self.detailed_bout_data and len(self.detailed_df) > 0:
            self.detailed_df.to_csv(self.detailed_bout_data_save_path)
            if self.verbose: print(f"Detailed ROI data saved at {self.detailed_bout_data_save_path}...")
        self.results.to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(f'ROI statistics saved at {self.save_path}', elapsed_time=self.timer.elapsed_time_str)

if __name__ == "__main__":
    analyzer = ROIAggregateStatisticsAnalyzer(config_path=r"D:\troubleshooting\maplight_ri\project_folder\project_config.ini",
                                              body_parts=['resident_NOSE'],
                                              include_fps=False,
                                              threshold=0.5,
                                              calculate_distances=True,
                                              transpose=True,
                                              detailed_bout_data=True,
                                              outside_rois=True,
                                              verbose=True,
                                              core_cnt=16)
    analyzer.run()
    analyzer.save()