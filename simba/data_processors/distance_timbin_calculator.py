__author__ = "Simon Nilsson; sronilsson@gmail.com"
import os
from typing import List, Optional, Union, Tuple, Iterable

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import (check_all_file_names_are_represented_in_video_log, check_float, check_str, check_that_column_exist, check_valid_boolean, check_valid_lst, check_instance, check_valid_tuple)
from simba.utils.errors import NoDataError, InvalidInputError, FrameRangeError
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.warnings import NotEnoughDataWarning
from simba.utils.read_write import (find_files_of_filetypes_in_directory,  get_fn_ext, read_df, find_time_stamp_from_frame_numbers)


class DistanceTimeBinCalculator(ConfigReader, FeatureExtractionMixin):
    """
    Compute body-part pair distance statistics per time bin.

    For each input video/data file and each selected body-part pair, computes frame-wise Euclidean distances (converted using project pixel/mm calibration), splits
    them into fixed-duration time bins, and summarizes the selected statistics per bin.

    :param Union[str, os.PathLike] config_path: Path to the SimBA project config file.
    :param Iterable[Tuple[str, str]] body_parts: Iterable of 2-tuples defining body-part pairs to compare, e.g. ``(("Nose_1", "Nose_2"), ("Center_1", "Center_2"))``.
    :param float time_bin: Time-bin size in seconds. Must be > 0. Bins are converted to  frame windows using each video's FPS.
    :param float threshold: Optional confidence threshold in [0.0, 1.0]. If > 0, frames are excluded for a pair when either body-part probability is below threshold.
    :param Optional[List[str]] file_paths: Optional data source. Can be: (i) list of CSV paths, (ii) single CSV path, or (iii) directory with CSVs. If None, uses project outlier-corrected files.
    :param Optional[Union[str, os.PathLike]] save_path: Output CSV path. If None, saves to a timestamped file in the project logs directory.
    :param bool distance_mean: Include mean distance per time bin.
    :param bool distance_median: Include median distance per time bin.
    :param bool distance_var: Include variance-distance metric per time bin.
    :param bool verbose: If True, print progress updates.
    :param bool transpose: If True, output is pivoted to wide format with one row per video.

    :raises InvalidInputError: If all metrics are disabled or inputs are invalid.
    :raises FrameRangeError: If ``time_bin`` is too short for a video's FPS (bin < 1 frame).
    :raises NoDataError: If no valid input data files are found.

    :example:
    >>> runner = DistanceTimeBinCalculator(
    ...     config_path=r"C:\\my_project\\project_config.ini",
    ...     body_parts=(("Nose_1", "Nose_2"),),
    ...     time_bin=60.0,
    ...     threshold=0.0,
    ...     distance_mean=True,
    ...     distance_median=True,
    ...     distance_var=False,
    ...     transpose=False
    ... )
    >>> runner.run()
    >>> runner.save()
    """
    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 body_parts: Iterable[Tuple[str, str]],
                 time_bin: float,
                 threshold: float = 0.00,
                 file_paths: Optional[List[str]] = None,
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 distance_mean: bool = True,
                 distance_median: bool = True,
                 distance_var: bool = True,
                 verbose: bool = True,
                 transpose: bool = False):

        ConfigReader.__init__(self, config_path=config_path)
        FeatureExtractionMixin.__init__(self)
        if file_paths is not None:
            if isinstance(file_paths, list):
                check_valid_lst(data=file_paths, source=f'{self.__class__.__name__} file_paths', min_len=1, valid_dtypes=(str,))
                self.file_paths = file_paths
            if isinstance(file_paths, str):
                if os.path.isfile(file_paths):
                    self.file_paths = [file_paths]
                elif os.path.isdir(file_paths):
                    self.file_paths = find_files_of_filetypes_in_directory(directory=file_paths, extensions=['.csv'], as_dict=False, raise_error=True)
                else:
                    raise NoDataError(msg=f'{file_paths} is not a valid data file path or data directory path', source=self.__class__.__name__)
            else:
                raise NoDataError(msg=f'{file_paths} is not a valid data file path or data directory path', source=self.__class__.__name__)
        else:
            if len(self.outlier_corrected_paths) == 0:
                raise NoDataError(msg=f'No data files found in {self.outlier_corrected_dir}', source=self.__class__.__name__)
            self.file_paths = self.outlier_corrected_paths
        if save_path is None:
            self.save_path = os.path.join(self.logs_path, f"Distance_log_{self.datetime}.csv")
        else:
            check_str(name=f'{self.__class__.__name__} save_path', value=save_path, raise_error=True)
            self.save_path = save_path
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_valid_boolean(value=distance_mean, source=f'{self.__class__.__name__} distance_mean', raise_error=True)
        check_valid_boolean(value=distance_median, source=f'{self.__class__.__name__} distance_median', raise_error=True)
        check_valid_boolean(value=transpose, source=f'{self.__class__.__name__} transpose', raise_error=True)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=distance_var, source=f'{self.__class__.__name__} distance_var', raise_error=True)
        check_float(name=f'{self.__class__.__name__} time_bin', value=time_bin, allow_zero=False, allow_negative=False)
        if not distance_mean and not distance_median and not distance_var:
            raise InvalidInputError(msg='All metrics are un-checked. To compute distance metrics, check at least one output variable.', source=self.__class__.__name__)

        self.distance_mean, self.distance_median, self.distance_var, self.time_bin = distance_mean, distance_median, distance_var, time_bin
        self.threshold, self.body_parts, self.transpose, self.verbose = threshold, body_parts, transpose, verbose
        check_instance(source=f'{self.__class__.__name__} body_parts', accepted_types=(list, tuple,), instance=body_parts, raise_error=True)
        for bp_cnt, bp_pair in enumerate(body_parts):
            check_valid_tuple(x=bp_pair, source=f'{self.__class__.__name__} bp_pair {bp_cnt}', accepted_lengths=(2,), valid_dtypes=(str,), accepted_values=self.body_parts_lst, raise_error=True)
            check_instance(source=f'{self.__class__.__name__} bp_pair {bp_cnt}', accepted_types=(list, tuple,), instance=bp_pair, raise_error=True)


    def __find_body_part_columns(self):
        self.body_parts_dict = {}
        for bp_pair_cnt, bp_pair in enumerate(self.body_parts):
            animal_name_1 = self.find_animal_name_from_body_part_name(bp_name=bp_pair[0], bp_dict=self.animal_bp_dict)
            animal_name_2 = self.find_animal_name_from_body_part_name(bp_name=bp_pair[1], bp_dict=self.animal_bp_dict)
            self.body_parts_dict[bp_pair_cnt] = {"ANIMAL NAME 1": animal_name_1, "BODY-PART 1": bp_pair[0], "BODY-PART HEADERS 1": [f"{bp_pair[0]}_x", f"{bp_pair[0]}_y", f"{bp_pair[0]}_p"], "ANIMAL NAME 2": animal_name_2, "BODY-PART 2": bp_pair[1], "BODY-PART HEADERS 2": [f"{bp_pair[1]}_x", f"{bp_pair[1]}_y", f"{bp_pair[1]}_p"]}

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.file_paths)
        self.results = pd.DataFrame(columns=["VIDEO", "TIME BIN #", "START TIME", "END TIME", "ANIMAL 1", "BODY-PART 1",  "ANIMAL 2", "BODY-PART 2", "MEASUREMENT", "VALUE"])
        self.__find_body_part_columns()
        for file_cnt, file_path in enumerate(self.file_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(file_path)
            if self.verbose: stdout_information(msg=f"Analysing {video_name}... (Video {file_cnt+1}/{len(self.file_paths)})")
            self.data_df = read_df(file_path=file_path, file_type=self.file_type)
            self.video_info, self.px_per_mm, self.fps = self.read_video_info(video_name=video_name)
            bin_length_frames = int(self.fps * self.time_bin)
            if bin_length_frames == 0:
                raise FrameRangeError(msg=f"The specified time-bin length of {self.time_bin} is TOO SHORT for video {video_name} which has a specified FPS of {self.fps}. This results in time bins that are LESS THAN a single frame.", source=self.__class__.__name__,)
            for k, v in self.body_parts_dict.items():
                check_that_column_exist(df=self.data_df, column_name=v["BODY-PART HEADERS 1"], file_name=file_path)
                check_that_column_exist(df=self.data_df, column_name=v["BODY-PART HEADERS 2"], file_name=file_path)
            for bp_pair_cnt, bp_pair in self.body_parts_dict.items():
                bp_1_data = self.data_df[bp_pair["BODY-PART HEADERS 1"]]
                bp_2_data = self.data_df[bp_pair["BODY-PART HEADERS 2"]]
                if self.threshold > 0.00:
                    idx_1 = list(bp_1_data.index[bp_1_data[bp_pair["BODY-PART HEADERS 1"][-1]] < self.threshold])
                    idx_2 = list(bp_2_data.index[bp_2_data[bp_pair["BODY-PART HEADERS 2"][-1]] < self.threshold])
                    drop_idx = sorted(list(set(idx_1) | set(idx_2)))
                    bp_1_data = bp_1_data.drop(index=drop_idx, errors="ignore").values[:, :2]
                    bp_2_data = bp_2_data.drop(index=drop_idx, errors="ignore").values[:, :2]
                else:
                    bp_1_data, bp_2_data = bp_1_data.values[:, :2], bp_2_data.values[:, :2]
                if len(bp_1_data) < 1 or len(bp_2_data) < 1:
                    NotEnoughDataWarning(msg=f'Cannot compute distances in file {file_path}. No body-part distance comparisons possible at probability (threshold) value {self.threshold}', source=self.__class__.__name__)
                else:
                    distance = FeatureExtractionMixin.keypoint_distances(a=bp_1_data, b=bp_2_data, px_per_mm=self.px_per_mm, in_centimeters=False)
                    distance_lists = [distance[i: i + bin_length_frames] for i in range(0, distance.shape[0], bin_length_frames)]
                    for bin_cnt, distance_list in enumerate(distance_lists):
                        bin_times = find_time_stamp_from_frame_numbers(start_frame=int(bin_length_frames * bin_cnt), end_frame=min(int(bin_length_frames * (bin_cnt + 1)), len(self.data_df)), fps=self.fps)
                        if self.distance_mean:
                            self.results.loc[len(self.results)] = [video_name, bin_cnt, bin_times[0], bin_times[1], bp_pair["ANIMAL NAME 1"], bp_pair["BODY-PART 1"], bp_pair["ANIMAL NAME 2"], bp_pair["BODY-PART 2"], "MEAN DISTANCE (CM)", np.mean(distance_list)]
                        if self.distance_median:
                            self.results.loc[len(self.results)] = [video_name, bin_cnt, bin_times[0], bin_times[1], bp_pair["ANIMAL NAME 1"], bp_pair["BODY-PART 1"], bp_pair["ANIMAL NAME 2"], bp_pair["BODY-PART 2"], "MEDIAN DISTANCE (CM)", np.median(distance)]
                        if self.distance_var:
                            self.results.loc[len(self.results)] = [video_name, bin_cnt, bin_times[0], bin_times[1],  bp_pair["ANIMAL NAME 1"], bp_pair["BODY-PART 1"], bp_pair["ANIMAL NAME 2"], bp_pair["BODY-PART 2"], "VARIANCE DISTANCE (CM)",  np.std(distance)]
            video_timer.stop_timer()
            #if self.verbose:
            #    stdout_information(msg=f'Distance analysis in video {video_name} complete', elapsed_time=video_timer.elapsed_time_str)
        if self.transpose:
            self.results = (self.results.assign(col=lambda x: x["ANIMAL 1"] + "_" + x["BODY-PART 1"] + " versus " +  x["ANIMAL 2"] + "_" + x["BODY-PART 2"] + " " + x["MEASUREMENT"]).pivot(index="VIDEO", columns="col", values="VALUE").reset_index())

    def save(self):
        self.results.set_index("VIDEO").to_csv(self.save_path)
        self.timer.stop_timer()
        if self.verbose: stdout_success(msg=f"Distance log saved in {self.save_path}", elapsed_time=self.timer.elapsed_time_str)
#
#
# if __name__ == "__main__" and not hasattr(sys, 'ps1'):
#     parser = argparse.ArgumentParser(description="Compute movement statistics from pose-estimation data.")
#     parser.add_argument('--config_path', type=str, required=True, help='Path to SimBA project config.')
#     parser.add_argument('--body_parts', type=str, nargs='+', required=True, help='Body-parts to use for movement calculations.')
#     parser.add_argument('--threshold', type=float, default=0.0, help='Confidence threshold for detections (0.0 - 1.0).')
#     args = parser.parse_args()
#     body_parts = list(args.body_parts[0].split(","))
#
#     runner = MovementCalculator(config_path=args.config_path,
#                                 body_parts=body_parts,
#                                 threshold=args.threshold)
#     runner.run()
#     runner.save()


# test = MovementCalculator(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini",
#                           body_parts=['Animal_1 CENTER OF GRAVITY', 'Nose'], #['Simon CENTER OF GRAVITY', 'JJ CENTER OF GRAVITY', 'Animal_1 CENTER OF GRAVITY']
#                           threshold=0.00)
# test.run()
# test.save()


# test = MovementCalculator(config_path=r"C:\troubleshooting\ROI_movement_test\project_folder\project_config.ini",
#                           body_parts=['Animal_1 CENTER OF GRAVITY'], #['Simon CENTER OF GRAVITY', 'JJ CENTER OF GRAVITY', 'Animal_1 CENTER OF GRAVITY']
#                           threshold=0.00)
# test.run()






# test = MovementCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                           body_parts=['Simon CENTER OF GRAVITY'], #['Simon CENTER OF GRAVITY', 'JJ CENTER OF GRAVITY']
#                           threshold=0.00)
# test.run()


# test.save()

# test = MovementCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/project_config.ini',
#                           body_parts=['Animal_1 CENTER OF GRAVITY'], #['Simon CENTER OF GRAVITY', 'JJ CENTER OF GRAVITY']
#                           threshold=0.00)
# test.run()
# test.save()v



# test = DistanceCalculator(config_path=r"E:\troubleshooting\mitra_pbn\mitra_pbn\project_folder\project_config.ini",
#                           body_parts=(('nose', 'center'),),
#                           threshold=0.5,
#                           transpose=True)
# test.run()
# test.save()
