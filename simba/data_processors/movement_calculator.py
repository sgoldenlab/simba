__author__ = "Simon Nilsson"
import argparse
import os
import sys
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from simba.feature_extractors.perimeter_jit import jitted_centroid
from simba.mixins.config_reader import ConfigReader
from simba.mixins.feature_extraction_mixin import FeatureExtractionMixin
from simba.mixins.feature_extraction_supplement_mixin import \
    FeatureExtractionSupplemental
from simba.utils.checks import (
    check_all_file_names_are_represented_in_video_log, check_float, check_str,
    check_that_column_exist, check_valid_lst)
from simba.utils.errors import NoDataError
from simba.utils.printing import stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext, read_df)


class MovementCalculator(ConfigReader, FeatureExtractionMixin):
    """
    Compute aggregate movement statistics from pose-estimation data in SimBA project.

    :parameters str config_path: path to SimBA project config file in Configparser format
    :param List[str] body_parts: Body-parts to use for movement calculations OR ``Animal_name CENTER OF GRAVITY``. If ``Animal_name CENTER OF GRAVITY``, then SimBA will approximate animal centroids through convex hull.
    :param float threshold: Filter body-part detection below set threshold (Value 0-1). Default: 0.00
    :param List[str] or None file_paths: Files to calucalte movements for. If None, then all files in ``project_folder/csv/outlier_corrected_movement_location`` directory.

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
    """
    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 body_parts: List[str],
                 threshold: float = 0.00,
                 file_paths: Optional[List[str]] = None,
                 save_path: Optional[Union[str, os.PathLike]] = None):

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
        check_float(name=f'{self.__class__.__name__} threshold', value=threshold, min_value=0.0, max_value=1.0)
        check_valid_lst(data=body_parts, source=f'{self.__class__.__name__} file_paths', min_len=1, valid_dtypes=(str,), valid_values=self.body_parts_lst)
        self.body_parts, self.threshold, self.body_parts = file_paths, threshold, body_parts

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
        self.results = pd.DataFrame(columns=["VIDEO", "ANIMAL", "BODY-PART", "MEASURE", "VALUE"])
        for file_cnt, file_path in enumerate(self.file_paths):
            self.__find_body_part_columns()
            _, video_name, _ = get_fn_ext(file_path)
            print(f"Analysing {video_name}... (Video {file_cnt+1}/{len(self.file_paths)})")
            self.data_df = read_df(file_path=file_path, file_type=self.file_type)
            self.video_info, self.px_per_mm, self.fps = self.read_video_info(video_name=video_name)
            if self.bp_list:
                check_that_column_exist(df=self.data_df, column_name=self.bp_list, file_name=file_path)
                self.data_df = self.data_df[self.bp_list]
                for animal_cnt, animal_data in self.body_parts_dict.items():
                    animal_df = self.data_df[animal_data["BODY-PART HEADERS"]]
                    if self.threshold > 0.00:
                        animal_df = animal_df[animal_df[animal_data["BODY-PART HEADERS"][-1]] >= self.threshold]
                    animal_df = animal_df.iloc[:, 0:2].reset_index(drop=True)
                    distance, velocity = FeatureExtractionSupplemental.distance_and_velocity(x=animal_df.values, fps=self.fps, pixels_per_mm=self.px_per_mm, centimeters=True)
                    self.results.loc[len(self.results)] = [video_name, animal_data["ANIMAL NAME"], animal_data["BODY-PART"], "Distance (cm)", distance]
                    self.results.loc[len(self.results)] = [ video_name, animal_data["ANIMAL NAME"], animal_data["BODY-PART"], "Velocity (cm/s)", velocity]
            else:
                for animal in self.body_parts:
                    animal_name = animal.split("CENTER OF GRAVITY")[0].strip()
                    x, y = (self.data_df[self.animal_bp_dict[animal_name]["X_bps"]], self.data_df[self.animal_bp_dict[animal_name]["Y_bps"]])
                    z = pd.concat([x, y], axis=1)[[item for items in zip(x.columns, y.columns) for item in items]]
                    df = pd.DataFrame(jitted_centroid(points=np.reshape(z.values, (len(z / 2), -1, 2)).astype(np.float32)), columns=["X", "Y"])
                    df = self.dataframe_savgol_smoother(df=df, fps=self.fps).astype(int)
                    distance, velocity = FeatureExtractionSupplemental.distance_and_velocity(x=df.values, fps=self.fps, pixels_per_mm=self.px_per_mm, centimeters=True)
                    self.results.loc[len(self.results)] = [video_name, animal_name, "GRAVITY CENTER", "Distance (cm)", distance]
                    self.results.loc[len(self.results)] = [video_name, animal_name, "GRAVITY CENTER", "Velocity (cm/s)", velocity]


    def save(self):
        self.results.set_index("VIDEO").to_csv(self.save_path)
        self.timer.stop_timer()
        stdout_success(msg=f"Movement log saved in {self.save_path}", elapsed_time=self.timer.elapsed_time_str)


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
# test.save()
