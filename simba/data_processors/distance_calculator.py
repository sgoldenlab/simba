__author__ = "Simon Nilsson; sronilsson@gmail.com"
import os
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_float,
    check_instance, check_str, check_that_column_exist, check_valid_boolean,
    check_valid_lst, check_valid_tuple)
from simba.utils.errors import InvalidInputError, NoDataError
from simba.utils.printing import SimbaTimer, stdout_information, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df)
from simba.utils.warnings import NotEnoughDataWarning


class DistanceCalculator(ConfigReader, FeatureExtractionMixin):
    """
    Compute body-part to body-part distance statistics from SimBA pose data.

    For each video/data file, the class calculates Euclidean distances between the
    specified body-part pairs and reports summary statistics in centimeters.

    :param Union[str, os.PathLike] config_path: Path to the SimBA project config file.
    :param Iterable[Tuple[str, str]] body_parts: Iterable of 2-tuples defining body-part pairs to compare, e.g. ``(("Nose_1", "Nose_2"), ("Tail_base_1", "Tail_base_2"))``.
    :param float threshold: Optional confidence threshold in [0.0, 1.0]. If > 0, frames are excluded for a body-part pair when either body-part has probability < threshold.
    :param Optional[List[str]] file_paths: Optional input data source. Can be: (i) list of CSV paths, (ii) single CSV path, or (iii) directory with CSVs. If None, uses project outlier-corrected files.
    :param Optional[Union[str, os.PathLike]] save_path: Output CSV path. If None, a file is created in the project logs directory.
    :param bool distance_mean: Include mean-distance measurement in the output.
    :param bool distance_median: Include median-distance measurement in the output.
    :param bool distance_var: Include variance-distance measurement in the output.
    :param bool verbose: If True, print progress updates.
    :param bool transpose: If True, save result in wide format (one row per video).

    :example:
    >>> runner = DistanceCalculator(
    ...     config_path=r"C:\\my_project\\project_config.ini",
    ...     body_parts=(("Nose_1", "Nose_2"),),
    ...     threshold=0.0,
    ...     transpose=True
    ... )
    >>> runner.run()
    >>> runner.save()
    """
    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 body_parts: Iterable[Tuple[str, str]],
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
        if not distance_mean and not distance_median and not distance_var:
            raise InvalidInputError(msg='All metrics are un-checked. To compute distance metrics, check at least one output variable.', source=self.__class__.__name__)



        self.distance_mean, self.distance_median, self.distance_var = distance_mean, distance_median, distance_var
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
        self.results = pd.DataFrame(columns=["VIDEO", "ANIMAL 1", "BODY-PART 1",  "ANIMAL 2", "BODY-PART 2", "MEASUREMENT", "VALUE"])
        self.__find_body_part_columns()
        for file_cnt, file_path in enumerate(self.file_paths):
            video_timer = SimbaTimer(start=True)
            _, video_name, _ = get_fn_ext(file_path)
            if self.verbose: stdout_information(msg=f"Analysing {video_name}... (Video {file_cnt+1}/{len(self.file_paths)})")
            self.data_df = read_df(file_path=file_path, file_type=self.file_type)
            self.video_info, self.px_per_mm, self.fps = self.read_video_info(video_name=video_name)
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
                    mean_distance, median_distance, var_distance= np.mean(distance), np.median(distance), np.std(distance)
                    if self.distance_mean:
                        self.results.loc[len(self.results)] = [video_name, bp_pair["ANIMAL NAME 1"], bp_pair["BODY-PART 1"], bp_pair["ANIMAL NAME 2"], bp_pair["BODY-PART 2"], "MEAN DISTANCE (CM)", mean_distance]
                    if self.distance_median:
                        self.results.loc[len(self.results)] = [video_name, bp_pair["ANIMAL NAME 1"], bp_pair["BODY-PART 1"], bp_pair["ANIMAL NAME 2"], bp_pair["BODY-PART 2"], "MEDIAN DISTANCE (CM)", median_distance]
                    if self.distance_var:
                        self.results.loc[len(self.results)] = [video_name, bp_pair["ANIMAL NAME 1"], bp_pair["BODY-PART 1"], bp_pair["ANIMAL NAME 2"], bp_pair["BODY-PART 2"], "VARIANCE DISTANCE (CM)", var_distance]
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
