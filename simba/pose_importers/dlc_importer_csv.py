__author__ = "Simon Nilsson"

import glob
import os
import shutil
from typing import List, Union

import pandas as pd

from simba.data_processors.interpolation_smoothing import Interpolate, Smooth
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_filepath_list_is_empty, check_int)
from simba.utils.data import smooth_data_gaussian, smooth_data_savitzky_golay
from simba.utils.enums import Methods
from simba.utils.errors import FileExistError, NoFilesFoundError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (get_fn_ext,
                                    get_number_of_header_columns_in_df,
                                    read_config_file,
                                    read_project_path_and_file_type)


def import_dlc_csv(config_path: Union[str, os.PathLike], source: str) -> List[str]:
    """
    Import file or folder of  DLC pose-estimation CSV files to SimBA project.
    Returns list of file paths that has been imported.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter str source: path to file or folder containing DLC pose-estimation CSV files
    :return List[str]: Paths of imported files.

    :example:
    >>> import_dlc_csv(config_path='project_folder/project_config.ini', source='CSV_import/Together_1.csv')
    >>> ['project_folder/csv/input_csv/Together_1.csv']
    """

    config = read_config_file(config_path=config_path)
    project_path, file_type = read_project_path_and_file_type(config=config)
    original_file_name_dir = os.path.join(
        project_path, "csv", "input_csv", "original_filename"
    )
    input_csv_dir = os.path.join(project_path, "csv", "input_csv")
    imported_files = glob.glob(input_csv_dir + "/*." + file_type)
    imported_file_names, imported_file_paths = [], []
    for file_path in imported_files:
        _, video_name, _ = get_fn_ext(filepath=file_path)
        imported_file_names.append(video_name)
    if not os.path.exists(original_file_name_dir):
        os.makedirs(original_file_name_dir)
    if os.path.isdir(source):
        csv_files = glob.glob(source + "/*.csv")
        check_if_filepath_list_is_empty(
            csv_files,
            error_msg=f"SIMBA ERROR: NO .csv files found in {source} directory.",
        )
    else:
        csv_files = [source]

    for file_path in csv_files:
        video_timer = SimbaTimer(start=True)
        check_file_exist_and_readable(file_path=file_path)
        _, video_name, file_ext = get_fn_ext(filepath=file_path)
        if "DLC_" in video_name:
            new_file_name = video_name.split("DLC_")[0] + ".csv"
        elif "DeepCut" in video_name:
            new_file_name = video_name.split("DeepCut")[0] + ".csv"
        else:
            new_file_name = video_name.split(".")[0] + ".csv"
        new_file_name_wo_ext = new_file_name.split(".")[0]
        video_basename = os.path.basename(file_path)
        print(f"Importing {video_name} to SimBA project...")
        if new_file_name_wo_ext in imported_file_names:
            raise FileExistError(
                "SIMBA IMPORT ERROR: {} already exist in project. Remove file from project or rename imported video file name before importing.".format(
                    new_file_name
                )
            )
        shutil.copy(file_path, input_csv_dir)
        shutil.copy(file_path, original_file_name_dir)
        os.rename(
            os.path.join(input_csv_dir, video_basename),
            os.path.join(input_csv_dir, new_file_name),
        )
        df = pd.read_csv(os.path.join(input_csv_dir, new_file_name))
        header_cols = get_number_of_header_columns_in_df(df=df)
        if header_cols == 3:
            df = df.iloc[1:]
        if file_type == "parquet":
            df = pd.read_csv(os.path.join(input_csv_dir, video_basename))
            df = df.apply(pd.to_numeric, errors="coerce")
            df.to_parquet(os.path.join(input_csv_dir, new_file_name))
            os.remove(os.path.join(input_csv_dir, video_basename))
        if file_type == "csv":
            df.to_csv(os.path.join(input_csv_dir, new_file_name), index=False)
        imported_file_paths.append(os.path.join(input_csv_dir, new_file_name))
        video_timer.stop_timer()
        print(
            f"Pose-estimation data for video {video_name} imported to SimBA project (elapsed time: {video_timer.elapsed_time_str}s)..."
        )
    return imported_file_paths


def import_single_dlc_tracking_csv_file(
    config_path: str,
    interpolation_setting: str,
    smoothing_setting: str,
    smoothing_time: int,
    file_path: str,
):
    timer = SimbaTimer(start=True)
    if (smoothing_setting == Methods.GAUSSIAN.value) or (
        smoothing_setting == Methods.SAVITZKY_GOLAY.value
    ):
        check_int(name="SMOOTHING TIME WINDOW", value=smoothing_time, min_value=1)
    check_file_exist_and_readable(file_path=file_path)
    imported_file_paths = import_dlc_csv(config_path=config_path, source=file_path)
    if interpolation_setting != "None":
        _ = Interpolate(
            input_path=imported_file_paths[0],
            config_path=config_path,
            method=interpolation_setting,
            initial_import_multi_index=True,
        )
    if (smoothing_setting == Methods.GAUSSIAN.value) or (
        smoothing_setting == Methods.SAVITZKY_GOLAY.value
    ):
        _ = Smooth(
            config_path=config_path,
            input_path=imported_file_paths[0],
            time_window=smoothing_time,
            smoothing_method=smoothing_setting,
            initial_import_multi_index=True,
        )
    timer.stop_timer()
    stdout_success(
        msg=f"Imported {str(len(imported_file_paths))} pose estimation file(s)",
        elapsed_time=timer.elapsed_time_str,
    )


def import_multiple_dlc_tracking_csv_file(
    config_path: str,
    interpolation_setting: str,
    smoothing_setting: str,
    smoothing_time: int,
    data_dir: str,
):
    timer = SimbaTimer(start=True)
    if (smoothing_setting == Methods.GAUSSIAN.value) or (
        smoothing_setting == Methods.SAVITZKY_GOLAY.value
    ):
        check_int(name="SMOOTHING TIME WINDOW", value=smoothing_time, min_value=1)
    check_if_dir_exists(in_dir=data_dir)
    imported_file_paths = import_dlc_csv(config_path=config_path, source=data_dir)
    if interpolation_setting != "None":
        _ = Interpolate(
            input_path=os.path.dirname(imported_file_paths[0]),
            config_path=config_path,
            method=interpolation_setting,
            initial_import_multi_index=True,
        )
    if (smoothing_setting == Methods.GAUSSIAN.value) or (
        smoothing_setting == Methods.SAVITZKY_GOLAY.value
    ):
        _ = Smooth(
            config_path=config_path,
            input_path=os.path.dirname(imported_file_paths[0]),
            time_window=int(smoothing_time),
            smoothing_method=smoothing_setting,
            initial_import_multi_index=True,
        )
    timer.stop_timer()
    stdout_success(
        msg=f"Imported {str(len(imported_file_paths))} pose estimation file(s)",
        elapsed_time=timer.elapsed_time_str,
    )
