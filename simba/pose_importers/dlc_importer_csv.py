__author__ = "Simon Nilsson"

import os
import shutil
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from simba.data_processors.interpolate import Interpolate
from simba.data_processors.smoothing import Smoothing
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_keys_exist_in_dict, check_instance,
                                check_int, check_str)
from simba.utils.errors import FileExistError, InvalidInputError
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    get_fn_ext,
                                    get_number_of_header_columns_in_df)

DLC_ = 'DLC_'
DeepCut = 'DeepCut'

def import_dlc_csv(config_path: Union[str, os.PathLike], source: str) -> List[str]:
    """
    Import file or folder of  DLC pose-estimation CSV files to SimBA project.
    Returns list of file paths that has been imported.

    :parameter str config_path: path to SimBA project config file in Configparser format
    :parameter str source: path to file or folder containing DLC pose-estimation CSV files
    :return List[str]: Paths to location of imported files.

    :example:
    >>> import_dlc_csv(config_path='project_folder/project_config.ini', source='CSV_import/Together_1.csv')
    >>> ['project_folder/csv/input_csv/Together_1.csv']
    """

    check_file_exist_and_readable(file_path=config_path)
    conf = ConfigReader(config_path=config_path, read_video_info=False)
    original_file_name_dir = os.path.join(conf.input_csv_dir, "original_filename")
    if not os.path.exists(original_file_name_dir): os.makedirs(original_file_name_dir)
    prev_imported_file_paths = find_files_of_filetypes_in_directory(directory=conf.input_csv_dir, extensions=[f'.{conf.file_type}'], raise_warning=False, raise_error=False)
    prev_imported_file_names = [get_fn_ext(x)[1] for x in prev_imported_file_paths]
    if os.path.isdir(source):
        new_data_paths = find_files_of_filetypes_in_directory(directory=source, extensions=['.csv'], raise_warning=False, raise_error=True)
    elif os.path.isfile(source):
        new_data_paths = [source]
    else:
        raise InvalidInputError(msg=f'{source} is not a valid data directory path or file path.', source=import_dlc_csv.__name__)

    imported_file_paths = []
    for file_cnt, file_path in enumerate(new_data_paths):
        video_timer = SimbaTimer(start=True)
        check_file_exist_and_readable(file_path=file_path)
        _, video_name, file_ext = get_fn_ext(filepath=file_path)
        if DLC_ in video_name:
            new_file_name = video_name.split(DLC_)[0] + ".csv"
        elif DeepCut in video_name:
            new_file_name = video_name.split(DeepCut)[0] + ".csv"
        else:
            new_file_name = video_name.split(".")[0] + ".csv"
        new_file_name_wo_ext = new_file_name.split(".")[0]
        video_basename = os.path.basename(file_path)
        print(f"Importing {video_name} to SimBA project...")
        if new_file_name_wo_ext in prev_imported_file_names:
            raise FileExistError(f"SIMBA IMPORT ERROR: {new_file_name} already exist in project in the directory {conf.input_csv_dir}. Remove file from project or rename imported video file name before importing.")
        shutil.copy(file_path, conf.input_csv_dir)
        shutil.copy(file_path, original_file_name_dir)
        os.rename(os.path.join(conf.input_csv_dir, video_basename), os.path.join(conf.input_csv_dir, new_file_name))
        df = pd.read_csv(os.path.join(conf.input_csv_dir, new_file_name))
        header_cols = get_number_of_header_columns_in_df(df=df)
        if header_cols == 3:
            df = df.iloc[1:]
        if conf.file_type == "parquet":
            df = pd.read_csv(os.path.join(conf.input_csv_dir, video_basename))
            df = df.apply(pd.to_numeric, errors="coerce")
            df.to_parquet(os.path.join(conf.input_csv_dir, new_file_name))
            os.remove(os.path.join(conf.input_csv_dir, video_basename))
        if conf.file_type == "csv":
            df.to_csv(os.path.join(conf.input_csv_dir, new_file_name), index=False)
        imported_file_paths.append(os.path.join(conf.input_csv_dir, new_file_name))
        video_timer.stop_timer()
        print(f"Pose-estimation data for video {video_name} imported to SimBA project (elapsed time: {video_timer.elapsed_time_str}s)...")
    return imported_file_paths

def import_dlc_csv_data(config_path: Union[str, os.PathLike],
                        data_path: Union[str, os.PathLike],
                        interpolation_settings: Optional[Dict[str, Any]] = None,
                        smoothing_settings: Optional[Dict[str, Any]] = None) -> None:

    """
    Import multiple DLC CSV tracking files to SimBA project and apply specified interpolation and smoothing
    parameters to the imported data.

    :param Union[str, os.PathLike] config_path: Path to SimBA config file in ConfigParser format.
    :param Union[str, os.PathLike] data_path: Path to directory holding DLC pose-estimation data in CSV format, or path to a single CSV file with DLC pose-estimation data.
    :param Optional[Dict[str, Any]] interpolation_settings: Dictionary holding settings for interpolation.
    :param Optional[Dict[str, Any]] smoothing_settings: Dictionary holding settings for smoothing.
    :return None:

    :example:
    >>> interpolation_settings = {'type': 'body-parts', 'method': 'linear'}
    >>> smoothing_settings = None #{'time_window': 500, 'method': 'savitzky-golay'}
    >>> import_dlc_csv_data(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini', data_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/new_data', interpolation_settings=interpolation_settings, smoothing_settings=smoothing_settings)

    """

    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=config_path)
    if (not os.path.isdir(data_path)) and (not os.path.isfile(data_path)):
        raise InvalidInputError(msg=f'{data_path} is not a valid data directory path or file path.', source=import_dlc_csv.__name__)
    if interpolation_settings != None:
        check_instance(source=f'{import_dlc_csv_data.__name__} interpolation_settings', accepted_types=(dict,), instance=interpolation_settings)
        check_if_keys_exist_in_dict(data=interpolation_settings, key=['type', 'method'])
        check_str(name='type', value=interpolation_settings['type'].lower(), options=['animals', 'body-parts'])
        check_str(name='method', value=interpolation_settings['method'].lower(), options=['nearest', 'linear', 'quadratic'])
    if smoothing_settings != None:
        check_instance(source=f'{import_dlc_csv_data.__name__} smoothing_settings', accepted_types=(dict,), instance=smoothing_settings)
        check_if_keys_exist_in_dict(data=smoothing_settings, key=['time_window', 'method'])
        check_int(name='time_window', value=smoothing_settings['time_window'], min_value=1)
        check_str(name='method', value=smoothing_settings['method'].lower(), options=['savitzky-golay', 'gaussian'])

    imported_file_paths = import_dlc_csv(config_path=config_path, source=data_path)
    if interpolation_settings != None:
        interpolator = Interpolate(config_path=config_path, data_path=imported_file_paths, type=interpolation_settings['type'], method=interpolation_settings['method'], multi_index_df_headers=True, copy_originals=False)
        interpolator.run()
    if smoothing_settings != None:
        smoother = Smoothing(config_path=config_path, data_path=imported_file_paths, time_window=smoothing_settings['time_window'], method=smoothing_settings['method'], multi_index_df_headers=True, copy_originals=False)
        smoother.run()
    timer.stop_timer()
    stdout_success(msg=f"Imported {len(imported_file_paths)} pose estimation file(s) to directory", elapsed_time=timer.elapsed_time_str)
