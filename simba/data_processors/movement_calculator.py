__author__ = "Simon Nilsson; sronilsson@gmail.com"
import argparse
import os
import sys
from typing import List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd

from simba.feature_extractors.perimeter_jit import jitted_centroid
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.feature_extraction_supplement_mixin import \
    FeatureExtractionSupplemental
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_float, check_str,
    check_that_column_exist, check_valid_boolean, check_valid_lst,
    check_valid_tuple, check_instance, check_valid_dict, check_if_keys_exist_in_dict)
from simba.utils.errors import InvalidInputError, NoDataError, FrameRangeError
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory, get_fn_ext, read_df, seconds_to_timestamp)


START, END = 'START', 'END'

class MovementCalculator(ConfigReader, FeatureExtractionMixin):
    """
    Compute aggregate movement statistics from pose-estimation data in SimBA project.

    :param Union[str, os.PathLike] config_path: path to SimBA project config file in Configparser format
    :param List[str] body_parts: Body-parts to use for movement calculations OR ``Animal_name CENTER OF GRAVITY``. If ``Animal_name CENTER OF GRAVITY``, then SimBA will approximate animal centroids through convex hull.
    :param float threshold: Filter body-part detection below set threshold (Value 0-1). Default: 0.00
    :param Optional[List[str]] file_paths: Files to calculate movements for. If None, then all files in ``project_folder/csv/outlier_corrected_movement_location`` directory.
    :param Optional[Union[str, os.PathLike]] save_path: Path to save the movement log. If None, saves to ``project_folder/logs/Movement_log_{datetime}.csv``. Default: None
    :param bool distance: If True, calculate distance metrics. Default: True
    :param bool velocity: If True, calculate velocity metrics. Default: True
    :param Optional[Dict[str, dict]] video_time_stamps: Dictionary mapping video file names (without extension) to time windows. Each value is a dict with ``START`` and ``END`` keys (in seconds). Only frames within the time window are analyzed. If None, all frames are used. Default: None.
    :param bool transpose: If True, transpose the results DataFrame. Default: False
    :param bool verbose: If True, print progress messages. Default: True
    :param bool frame_count: If True, include frame count in results. Default: False
    :param bool video_length: If True, include video length in results. Default: False

    .. note::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario2.md#part-4--analyze-machine-results>`__.

    .. image:: _static/img/MovementCalculator.webp
       :width: 400
       :align: center

    :examples:
    >>> body_parts=['Animal_1 CENTER OF GRAVITY']
    >>> movement_processor = MovementCalculator(config_path='project_folder/project_config.ini', body_parts=body_parts)
    >>> movement_processor.run()
    >>> movement_processor.save()
    >>> time_stamps = pd.read_csv('mastersheet.csv')[['VIDEO_FILE_NAME', 'START', 'END']].set_index('VIDEO_FILE_NAME').to_dict(orient='index')
    >>> movement_processor = MovementCalculator(config_path='project_folder/project_config.ini', body_parts=['center'], video_time_stamps=time_stamps)
    >>> movement_processor.run()
    >>> movement_processor.save()
    """
    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 body_parts: Union[List[str], Tuple[str]],
                 threshold: float = 0.00,
                 file_paths: Optional[List[str]] = None,
                 save_path: Optional[Union[str, os.PathLike]] = None,
                 distance: bool = True,
                 velocity: bool = True,
                 video_time_stamps: Optional[Dict[str, tuple]] = None,
                 transpose: bool = False,
                 verbose: bool = True,
                 frame_count: bool = False,
                 video_length: bool = False):

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
            self.save_path = os.path.join(self.logs_path, f"Movement_log_{self.datetime}.csv")
        else:
            check_str(name=f'{self.__class__.__name__} save_path', value=save_path, raise_error=True)
            self.save_path = save_path
        file_names = [get_fn_ext(x)[1] for x in self.file_paths]
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_valid_boolean(value=distance, source=f'{self.__class__.__name__} distance', raise_error=True)
        check_valid_boolean(value=velocity, source=f'{self.__class__.__name__} velocity', raise_error=True)
        check_valid_boolean(value=transpose, source=f'{self.__class__.__name__} transpose', raise_error=True)
        check_valid_boolean(value=verbose, source=f'{self.__class__.__name__} verbose', raise_error=True)
        check_valid_boolean(value=frame_count, source=f'{self.__class__.__name__} frame_count', raise_error=True)
        check_valid_boolean(value=video_length, source=f'{self.__class__.__name__} video_length', raise_error=True)
        if video_time_stamps is not None:
            check_instance(source=f'{self.__class__.__name__} time_stamps', instance=video_time_stamps, accepted_types=(dict,))
            for k, v in video_time_stamps.items():
                if k not in file_names:
                    raise InvalidInputError(msg=f'time_stamps key {k} not found among input files: {file_names}', source=self.__class__.__name__)
                check_valid_dict(x=v, required_keys=(START, END,), valid_values_dtypes=(int, float,), min_value=0, source=f'{self.__class__.__name__} time_stamps')
                if v[START] >= v[END]:
                    raise InvalidInputError(msg=f'time_stamps for {k}: START ({v[START]}) must be less than END ({v[END]})', source=self.__class__.__name__)
        if not distance and not velocity:
            raise InvalidInputError(msg='distance AND velocity are both False. To compute movement metrics, set at least one value to True.', source=self.__class__.__name__)
        self.distance, self.velocity, = distance, velocity
        if isinstance(body_parts, list):
            check_valid_lst(data=body_parts, source=f'{self.__class__.__name__} body_parts', min_len=1, valid_dtypes=(str,), valid_values=self.body_parts_lst)
        elif isinstance(body_parts, tuple):
            check_valid_tuple(x=body_parts, source=f'{self.__class__.__name__} body_parts', minimum_length=1, valid_dtypes=(str,), accepted_values=self.body_parts_lst)
        else:
            raise InvalidInputError(msg='Body-parts has to be a list of tuple of strings', source=f'{self.__class__.__name__} body_parts')
        self.body_parts, self.threshold, self.body_parts, self.transpose, self.verbose = file_paths, threshold, body_parts, transpose, verbose
        self.frame_count, self.video_length, self.time_stamps = frame_count, video_length, video_time_stamps

    def __find_body_part_columns(self):
        self.body_parts_dict, self.bp_list = {}, []
        for bp_cnt, bp_name in enumerate(self.body_parts):
            if not bp_name.endswith("CENTER OF GRAVITY"):
                animal_name = self.find_animal_name_from_body_part_name(bp_name=bp_name, bp_dict=self.animal_bp_dict)
                self.body_parts_dict[bp_cnt] = {"ANIMAL NAME": animal_name, "BODY-PART": bp_name, "BODY-PART HEADERS": [f"{bp_name}_x", f"{bp_name}_y", f"{bp_name}_p"]}
                self.bp_list.extend((self.body_parts_dict[bp_cnt]["BODY-PART HEADERS"]))
            else:
                pass

    def run(self):
        check_all_file_names_are_represented_in_video_log(video_info_df=self.video_info_df, data_paths=self.file_paths)
        self.results = pd.DataFrame(columns=["VIDEO", "ANIMAL", "BODY-PART", "MEASUREMENT", "VALUE"])
        for file_cnt, file_path in enumerate(self.file_paths):
            video_timer = SimbaTimer(start=True)
            self.__find_body_part_columns()
            _, video_name, _ = get_fn_ext(file_path)
            if self.verbose: stdout_information(msg=f"Analysing {video_name}... (Video {file_cnt+1}/{len(self.file_paths)})")
            self.data_df = read_df(file_path=file_path, file_type=self.file_type)
            self.video_info, self.px_per_mm, self.fps = self.read_video_info(video_name=video_name)
            if self.time_stamps is not None and check_if_keys_exist_in_dict(data=self.time_stamps, key=video_name, name=f'{self.__class__.__name__} time_stamps', raise_error=False):
                start_time, end_time = self.time_stamps[video_name][START], self.time_stamps[video_name][END]
                start_frm, end_frm = int(self.time_stamps[video_name][START] * self.fps), int(self.time_stamps[video_name][END] * self.fps)
                if start_frm > len(self.data_df) or end_frm > len(self.data_df):
                    raise FrameRangeError(msg=f'Cannot compute movement between frame {start_frm} (time s: {start_time}) and frame {end_frm} (time s: {end_time}) in video {video_name}. The data only has {len(self.data_df)} frames.', source=self.__class__.__name__)
                self.data_df = self.data_df.loc[start_frm:end_frm].reset_index()
                if self.verbose: stdout_information(msg=f"Slicing video {video_name} between frames {start_frm} and {end_frm}...)")
            # else:
            #     start_frm, end_frm = int(0 * self.fps), int(899 * self.fps)
            #     self.data_df = self.data_df.loc[start_frm:end_frm].reset_index()
            if self.bp_list:
                check_that_column_exist(df=self.data_df, column_name=self.bp_list, file_name=file_path)
                self.data_df = self.data_df[self.bp_list]
                for animal_cnt, animal_data in self.body_parts_dict.items():
                    animal_df = self.data_df[animal_data["BODY-PART HEADERS"]]
                    if self.threshold > 0.00: animal_df = animal_df[animal_df[animal_data["BODY-PART HEADERS"][-1]] >= self.threshold]
                    animal_df = animal_df.iloc[:, 0:2].reset_index(drop=True)
                    distance, velocity = FeatureExtractionSupplemental.distance_and_velocity(x=animal_df.values, fps=self.fps, pixels_per_mm=self.px_per_mm, centimeters=True)
                    if self.distance:
                        self.results.loc[len(self.results)] = [video_name, animal_data["ANIMAL NAME"], animal_data["BODY-PART"], "DISTANCE (CM)", distance]
                    if self.velocity:
                        self.results.loc[len(self.results)] = [video_name, animal_data["ANIMAL NAME"], animal_data["BODY-PART"], "VELOCITY (CM/S)", velocity]
            else:
                for animal in self.body_parts:
                    animal_name = animal.split("CENTER OF GRAVITY")[0].strip()
                    x, y = (self.data_df[self.animal_bp_dict[animal_name]["X_bps"]], self.data_df[self.animal_bp_dict[animal_name]["Y_bps"]])
                    z = pd.concat([x, y], axis=1)[[item for items in zip(x.columns, y.columns) for item in items]]
                    df = pd.DataFrame(jitted_centroid(points=np.reshape(z.values, (len(z / 2), -1, 2)).astype(np.float32)), columns=["X", "Y"])
                    df = self.dataframe_savgol_smoother(df=df, fps=self.fps).astype(np.int32)
                    distance, velocity = FeatureExtractionSupplemental.distance_and_velocity(x=df.values, fps=self.fps, pixels_per_mm=self.px_per_mm, centimeters=True)
                    if self.distance:
                        self.results.loc[len(self.results)] = [video_name, animal_name, "GRAVITY CENTER", "DISTANCE (CM)", distance]
                    if self.velocity:
                        self.results.loc[len(self.results)] = [video_name, animal_name, "GRAVITY CENTER", "VELOCITY (CM/S)", velocity]

            if self.frame_count:
                self.results.loc[len(self.results)] = [video_name, "", "", "VIDEO FRAME COUNT", len(self.data_df)]
            if self.video_length:
                timestamp = seconds_to_timestamp(seconds=int(np.ceil(len(self.data_df) / self.fps)), hh_mm_ss_sss=False)
                self.results.loc[len(self.results)] = [video_name, "", "", "VIDEO LENGTH (HH:MM:SS)", timestamp]

            video_timer.stop_timer()
            if self.verbose:
                stdout_information(msg=f'Movement analysis in video {video_name} complete', elapsed_time=video_timer.elapsed_time_str)

    def save(self):
        if self.transpose:
            self.results = (self.results.assign(col=lambda x: (x["ANIMAL"] + "_" + x["BODY-PART"] + "_" + x["MEASUREMENT"]).str.strip("_")).pivot(index="VIDEO", columns="col", values="VALUE").reset_index())
        self.results.set_index("VIDEO").to_csv(self.save_path)
        self.timer.stop_timer()
        if self.verbose: stdout_success(msg=f"Movement log saved in {self.save_path}", elapsed_time=self.timer.elapsed_time_str)


if __name__ == "__main__" and not hasattr(sys, 'ps1'):
    parser = argparse.ArgumentParser(description="Compute movement statistics from pose-estimation data.")
    parser.add_argument('--config_path', type=str, required=True, help='Path to SimBA project config.')
    parser.add_argument('--body_parts', type=str, nargs='+', required=True, help='Body-parts to use for movement calculations.')
    parser.add_argument('--threshold', type=float, default=0.0, help='Confidence threshold for detections (0.0 - 1.0).')
    args = parser.parse_args()
    body_parts = list(args.body_parts[0].split(","))

    runner = MovementCalculator(config_path=args.config_path,
                                body_parts=body_parts,
                                threshold=args.threshold)
    runner.run()
    runner.save()


# TIME_TIME_PATH = r"F:\troubleshooting\sam\sam\project_folder\DATA MASTERSHEET (1).csv"
# time_stamps = pd.read_csv(TIME_TIME_PATH)[['VIDEO_FILE_NAME', 'START', 'END']].set_index('VIDEO_FILE_NAME').to_dict(orient='index')
# test = MovementCalculator(config_path=r"F:\troubleshooting\sam\sam\project_folder\project_config.ini",
#                           body_parts=['center'],  #['Simon CENTER OF GRAVITY', 'JJ CENTER OF GRAVITY', 'Animal_1 CENTER OF GRAVITY']
#                           threshold=0.00,
#                           frame_count=True,
#                           video_time_stamps=time_stamps,
#                           video_length=True,
#                           transpose=True)
# test.run()
# test.save()


# test = MovementCalculator(config_path=r"E:\troubleshooting\mitra_pbn\mitra_pbn\project_folder\project_config.ini",
#                           body_parts=('center',), #['Simon CENTER OF GRAVITY', 'JJ CENTER OF GRAVITY', 'Animal_1 CENTER OF GRAVITY']
#                           threshold=0.784,
#                           velocity=False)
# test.run()
# test.save()






# test = MovementCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini',
#                           body_parts=['Simon CENTER OF GRAVITY'], #['Simon CENTER OF GRAVITY', 'JJ CENTER OF GRAVITY']
#                           threshold=0.00)
# test.run()


# test.save()

# test = MovementCalculator(config_path='/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/project_config.ini',
#                           body_parts=['Animal_1 CENTER OF GRAVITY'], #['Simon CENTER OF GRAVITY', 'JJ CENTER OF GRAVITY']
#                           threshold=0.00)
# test.run()
# test.save()
