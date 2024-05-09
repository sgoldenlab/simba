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
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
from simba.utils.checks import check_float, check_if_filepath_list_is_empty
from simba.utils.errors import (BodypartColumnNotFoundError, DuplicationError,
                                FrameRangeError, ROICoordinatesNotFoundError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, read_data_paths, read_df


class ROITimebinCalculator(ConfigReader):
    """
    Compute how much time and how many entries animals are making into user-defined ROIs
    within user-defined time bins. Also compute the average velocity and distance moved within user-defined ROIs
    split by time-bins.

    Results are stored in the ``project_folder/logs`` directory of
    the SimBA project.

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param float bin_length: length of time bins in seconds.
    :param List[str] body_parts: List of body-parts to use as proxy of animal locations.
    :param float threshold: Filter pose-estimation data detected below defined threshold.
    :param Optional[bool] movement: If True, compute the distances and velocities within time-bins. Default False.

    .. note::
       `Example anticipated ROI time-bins entry results <https://github.com/sgoldenlab/simba/blob/master/misc/ROI_time_bins_5.2s_entry_data_20240331125343.csv>`__.
       `Example anticipated ROI time-bins latency results <https://github.com/sgoldenlab/simba/blob/master/misc/ROI_time_bins_5.2s_time_data_20240331125343.csv>`__.
       `Example anticipated movement results <https://github.com/sgoldenlab/simba/blob/master/misc/Time_bins_0.5s_movement_results_20240330143150.csv>`__.

    :example:
    >>> calculator = ROITimebinCalculator(config_path=r"/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini", bin_length=1.0, body_parts=['Nose_1'], threshold=0.00, movement=True)
    >>> calculator.run()
    >>> calculator.save()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 bin_length: float,
                 body_parts: List[str],
                 data_path: Optional[Union[str, os.PathLike, List[str]]] = None,
                 threshold: Optional[float] = 0.0,
                 movement: Optional[bool] = False):

        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(expected_file_path=self.roi_coordinates_path)
        check_float(name="bin_length", value=bin_length, min_value=10e-6)
        check_float(name="threshold", value=threshold, min_value=0.0, max_value=1.0)
        self.data_paths = read_data_paths(path=data_path, default=self.outlier_corrected_paths, default_name=self.outlier_corrected_dir, file_type=self.file_type)
        self.read_roi_data()
        self.bin_length, self.body_parts, self.threshold = (bin_length, body_parts, threshold)
        self.save_path_time = os.path.join(self.logs_path, f"ROI_time_bins_{bin_length}s_time_data_{self.datetime}.csv")
        self.save_path_entries = os.path.join(self.logs_path, f"ROI_time_bins_{bin_length}s_entry_data_{self.datetime}.csv")
        for bp in body_parts:
            if bp not in self.body_parts_lst:
                raise BodypartColumnNotFoundError(msg=f'The body-part {bp} is not a valid body-part in the SimBA project. Options: {self.body_parts_lst}', source=self.__class__.__name__)
        if len(set(body_parts)) != len(body_parts):
            raise DuplicationError(msg=f'All body-part entries have to be unique. Got {body_parts}', source=self.__class__.__name__)
        self.roi_analyzer = ROIAnalyzer(config_path=self.config_path, data_path=self.outlier_corrected_dir, calculate_distances=False, threshold=threshold, body_parts=body_parts)
        self.roi_analyzer.run()
        self.animal_names = list(self.roi_analyzer.bp_dict.keys())
        self.bp_dict = self.roi_analyzer.bp_dict
        self.entries_exits_df = self.roi_analyzer.detailed_df
        self.movement = movement
        if movement:
            self.save_path_movement_velocity = os.path.join(self.logs_path, f"ROI_time_bins_{bin_length}s_movement_velocity_data_{self.datetime}.csv")
            self.movement_timebins = TimeBinsMovementCalculator(config_path=config_path, bin_length=bin_length, body_parts=body_parts, plots=False)
            self.movement_timebins.run()

    def run(self):
        self.results_entries = pd.DataFrame(columns=["VIDEO","SHAPE","ANIMAL","BODY-PART","TIME BIN #","ENTRY COUNT",])
        self.results_time = pd.DataFrame(columns=["VIDEO","SHAPE","ANIMAL","BODY-PART","TIME BIN #","TIME INSIDE SHAPE (S)"])
        self.results_movement_velocity = pd.DataFrame(columns=["VIDEO","SHAPE","ANIMAL","BODY-PART","TIME BIN #","DISTANCE (CM)","VELOCITY (CM/S)"])
        print(f"Analyzing time-bin data for {len(self.data_paths)} video(s)...")
        for file_cnt, file_path in enumerate(self.data_paths):
            video_timer = SimbaTimer(start=True)
            _, self.video_name, _ = get_fn_ext(filepath=file_path)
            _, px_per_mm, fps = self.read_video_info(video_name=self.video_name)
            frames_per_bin = int(fps * self.bin_length)
            if frames_per_bin == 0:
                raise FrameRangeError(msg=f"The specified time-bin length of {self.bin_length} is TOO SHORT for video {self.video_name} which has a specified FPS of {fps}. This results in time bins that are LESS THAN a single frame.", source=self.__class__.__name__)
            video_frms = list(range(0, len(read_df(file_path=file_path, file_type=self.file_type))))
            frame_bins = [video_frms[i : i + (frames_per_bin)] for i in range(0, len(video_frms), frames_per_bin)]
            self.video_data = self.entries_exits_df[self.entries_exits_df["VIDEO"] == self.video_name]
            for animal_name, shape_name in list(itertools.product(self.animal_names, self.shape_names)):
                data_df = self.video_data.loc[(self.video_data["SHAPE NAME"] == shape_name) & (self.video_data["ANIMAL"] == animal_name)]
                body_part = self.bp_dict[animal_name][0][:-2]
                entry_frms = list(data_df["START FRAME"])
                inside_shape_frms = [list(range(x, y)) for x, y in zip(list(data_df["START FRAME"].astype(int)), list(data_df["END FRAME"].astype(int) + 1))]
                inside_shape_frms = [i for s in inside_shape_frms for i in s]
                for bin_cnt, bin_frms in enumerate(frame_bins):
                    frms_inside_roi_in_timebin = [x for x in inside_shape_frms if x in bin_frms]
                    entry_roi_in_timebin = [x for x in entry_frms if x in bin_frms]
                    self.results_time.loc[len(self.results_time)] = [self.video_name,shape_name,animal_name,body_part,bin_cnt,len(frms_inside_roi_in_timebin) / fps]
                    self.results_entries.loc[len(self.results_entries)] = [self.video_name,shape_name,animal_name,body_part,bin_cnt,len(entry_roi_in_timebin)]
                    if self.movement:
                        if len(frms_inside_roi_in_timebin) > 0:
                            bin_move = (self.movement_timebins.movement_dict[self.video_name].iloc[frms_inside_roi_in_timebin].values.flatten().astype(np.float32))
                            _, velocity = (FeatureExtractionSupplemental.distance_and_velocity(x=bin_move,fps=fps, pixels_per_mm=1, centimeters=True))
                            self.results_movement_velocity.loc[len(self.results_movement_velocity)] = [self.video_name,
                                                                                                       shape_name,
                                                                                                       animal_name,
                                                                                                       body_part,
                                                                                                       bin_cnt,
                                                                                                       bin_move[1:].sum() / 10,
                                                                                                       velocity]
                        else:
                            self.results_movement_velocity.loc[len(self.results_movement_velocity)] = [self.video_name,
                                                                                                       shape_name,
                                                                                                       animal_name,
                                                                                                       body_part,
                                                                                                       bin_cnt,
                                                                                                       0,
                                                                                                       0]
            video_timer.stop_timer()
            print(f"Video {self.video_name} complete (elapsed time {video_timer.elapsed_time_str}s)")

    def save(self):
        self.results_time.sort_values(by=["VIDEO", "SHAPE", "ANIMAL", "TIME BIN #"]).set_index("VIDEO").to_csv(self.save_path_time)
        self.results_entries.sort_values(by=["VIDEO", "SHAPE", "ANIMAL", "TIME BIN #"]).set_index("VIDEO").to_csv(self.save_path_entries)
        self.timer.stop_timer()
        stdout_success(msg=f"ROI time bin entry data saved at {self.save_path_entries}", elapsed_time=self.timer.elapsed_time_str)
        stdout_success(msg=f"ROI time bin time data saved at {self.save_path_time}", elapsed_time=self.timer.elapsed_time_str)
        if self.movement:
            self.results_movement_velocity.sort_values(
                by=["VIDEO", "SHAPE", "ANIMAL", "TIME BIN #"]
            ).set_index("VIDEO").to_csv(self.save_path_movement_velocity)
            stdout_success(
                msg=f"ROI time-bin movement data saved at {self.save_path_movement_velocity}",
                elapsed_time=self.timer.elapsed_time_str,
            )


# test = ROITimebinCalculator(config_path=r"/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini",
#                             bin_length=1,
#                             body_parts=['Nose_1'],
#                             threshold=0.00,
#                             movement=True)
# test.run()
# test.save()
