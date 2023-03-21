__author__ = "Simon Nilsson", "JJ Choong"

import shutil
import os, glob
import pandas as pd
from simba.drop_bp_cords import get_fn_ext
from simba.read_config_unit_tests import (read_config_file,
                                          read_project_path_and_file_type,
                                          check_file_exist_and_readable,
                                          check_if_filepath_list_is_empty,
                                          check_int)
from simba.misc_tools import SimbaTimer, get_number_of_header_columns_in_df
from simba.enums import Methods
from simba.interpolate_pose import Interpolate
from simba.misc_tools import smooth_data_savitzky_golay, smooth_data_gaussian
from simba.utils.errors import NoFilesFoundError, FileExistError


def import_dlc_csv(config_path: str, source: str) -> list:
    """
    Imports file or folder DLC pose-estimation CSV files to SimBA project. Returns list of file paths to the imported files.

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    source: str
        path to file or folder containing DLC pose-estimation CSV files

    :return
    ----------
    list

    Examples
    ----------
    >>> import_dlc_csv(config_path='project_folder/project_config.ini', source='CSV_import/Together_1.csv')
    >>> ['project_folder/csv/input_csv/Together_1.csv']
    """

    config = read_config_file(ini_path=config_path)
    project_path, file_type = read_project_path_and_file_type(config=config)
    original_file_name_dir = os.path.join(project_path, 'csv', 'input_csv', 'original_filename')
    input_csv_dir = os.path.join(project_path, 'csv', 'input_csv')
    imported_files = glob.glob(input_csv_dir + '/*.' + file_type)
    imported_file_names = []
    imported_file_paths = []
    for file_path in imported_files:
        _, video_name, _ = get_fn_ext(filepath=file_path)
        imported_file_names.append(video_name)
    if not os.path.exists(original_file_name_dir): os.makedirs(original_file_name_dir)
    if os.path.isdir(source):
        csv_files = glob.glob(source + '/*.csv')
        check_if_filepath_list_is_empty(csv_files, error_msg=f'SIMBA ERROR: NO .csv files found in {source} directory.')
    else:
        csv_files = [source]

    for file_path in csv_files:
        video_timer = SimbaTimer()
        video_timer.start_timer()
        check_file_exist_and_readable(file_path=file_path)
        _, video_name, file_ext = get_fn_ext(filepath=file_path)
        if 'DLC_' in video_name:
            new_file_name = video_name.split('DLC_')[0] + '.csv'
        elif 'DeepCut' in video_name:
            new_file_name = video_name.split('DeepCut')[0] + '.csv'
        else:
            new_file_name = video_name.split('.')[0] + '.csv'
        new_file_name_wo_ext = new_file_name.split('.')[0]
        video_basename = os.path.basename(file_path)
        print(f'Importing {video_name} to SimBA project...')
        if new_file_name_wo_ext in imported_file_names:
            raise FileExistError('SIMBA IMPORT ERROR: {} already exist in project. Remove file from project or rename imported video file name before importing.'.format(new_file_name))
        shutil.copy(file_path, input_csv_dir)
        shutil.copy(file_path, original_file_name_dir)
        os.rename(os.path.join(input_csv_dir, video_basename), os.path.join(input_csv_dir, new_file_name))
        df = pd.read_csv(os.path.join(input_csv_dir, new_file_name))
        header_cols = get_number_of_header_columns_in_df(df=df)
        if header_cols == 3:
            df = df.iloc[1:]
        if file_type == 'parquet':
            df = pd.read_csv(os.path.join(input_csv_dir, video_basename))
            df = df.apply(pd.to_numeric, errors='coerce')
            df.to_parquet(os.path.join(input_csv_dir, new_file_name))
            os.remove(os.path.join(input_csv_dir, video_basename))
        if file_type == 'csv':
            df.to_csv(os.path.join(input_csv_dir, new_file_name), index=False)
        imported_file_paths.append(os.path.join(input_csv_dir, new_file_name))
        video_timer.stop_timer()
        print(f'Pose-estimation data for video {video_name} imported to SimBA project (elapsed time: {video_timer.elapsed_time_str}s...')
    return imported_file_paths

def import_single_dlc_tracking_csv_file(config_path: str,
                                        interpolation_setting: str,
                                        smoothing_setting: str,
                                        smoothing_time: int,
                                        file_path: str):
    timer = SimbaTimer()
    timer.start_timer()
    if (smoothing_setting == Methods.GAUSSIAN.value) or (smoothing_setting == Methods.SAVITZKY_GOLAY.value):
        check_int(name='SMOOTHING TIME WINDOW', value=smoothing_time, min_value=1)
    if (config_path == 'No file selected') and (file_path == 'No file selected'):
        raise NoFilesFoundError(msg='SIMBA ERROR: Please select a pose-estimation data path.')

    check_file_exist_and_readable(file_path=file_path)
    imported_file_paths = import_dlc_csv(config_path=str(config_path), source=file_path)
    config = read_config_file(ini_path=str(config_path))
    csv_df = pd.read_csv(imported_file_paths[0], index_col=0)
    if interpolation_setting != 'None':
        print(f'Interpolating missing values (Method: {interpolation_setting}) ...')
        interpolate_body_parts = Interpolate(config_path, csv_df)
        interpolate_body_parts.detect_headers()
        interpolate_body_parts.fix_missing_values(interpolation_setting)
        interpolate_body_parts.reorganize_headers()
        interpolate_body_parts.new_df.to_csv(imported_file_paths[0])
    if smoothing_setting == Methods.GAUSSIAN.value:
        print(f'Smoothing data using Gaussian method and {str(smoothing_time)} ms time window ...')
        smooth_data_gaussian(config=config, file_path=imported_file_paths[0], time_window_parameter=int(smoothing_time))
    if smoothing_setting == Methods.SAVITZKY_GOLAY.value:
        print(f'Smoothing data using Savitzky Golay method and {str(smoothing_time)} ms time window ...')
        smooth_data_savitzky_golay(config=config, file_path=imported_file_paths[0], time_window_parameter=int(smoothing_time))
    timer.stop_timer()
    print(f'SIMBA COMPLETE: Imported {str(len(imported_file_paths))} pose estimation file(s) (elapsed time {timer.elapsed_time_str}s)')


def import_multiple_dlc_tracking_csv_file(config_path: str,
                                          interpolation_setting: str,
                                          smoothing_setting: str,
                                          smoothing_time: int,
                                          folder_path: str):
    timer = SimbaTimer()
    timer.start_timer()
    if (smoothing_setting == Methods.GAUSSIAN.value) or (smoothing_setting == Methods.SAVITZKY_GOLAY.value):
        check_int(name='SMOOTHING TIME WINDOW', value=smoothing_time, min_value=1)
    if (config_path== 'No file selected') and (folder_path == 'No folder selected'):
        print('SIMBA ERROR: Please select a pose-estimation data path.')
        raise FileNotFoundError('SIMBA ERROR: Please select a pose-estimation data path.')
    imported_file_paths = import_dlc_csv(config_path=config_path, source=folder_path)
    config = read_config_file(ini_path=config_path)
    if interpolation_setting != 'None':
        print(f'Interpolating missing values (Method: {interpolation_setting}) ...')
        for file_path in imported_file_paths:
            csv_df = pd.read_csv(file_path, index_col=0)
            interpolate_body_parts = Interpolate(config_path, csv_df)
            interpolate_body_parts.detect_headers()
            interpolate_body_parts.fix_missing_values(interpolation_setting)
            interpolate_body_parts.reorganize_headers()
            interpolate_body_parts.new_df.to_csv(file_path)

    if smoothing_setting == Methods.GAUSSIAN.value:
        print(f'Smoothing data using Gaussian method and {str(smoothing_time)} ms time window ...')
        for file_path in imported_file_paths:
            smooth_data_gaussian(config=config, file_path=file_path, time_window_parameter=int(smoothing_time))

    if smoothing_setting == Methods.SAVITZKY_GOLAY.value:
        print(f'Smoothing data using Savitzky Golay method and {str(smoothing_time)} ms time window ...')
        for file_path in imported_file_paths:
            smooth_data_savitzky_golay(config=config, file_path=file_path, time_window_parameter=int(smoothing_time))
    timer.stop_timer()
    print(f'SIMBA COMPLETE: Imported {str(len(imported_file_paths))} pose estimation file(s) (elapsed time {timer.elapsed_time_str}s)')
