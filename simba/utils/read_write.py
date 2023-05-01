__author__ = "Simon Nilsson"

import pandas as pd
import platform
from datetime import datetime
import shutil
import glob, re
import configparser
from configparser import ConfigParser
import pyarrow as pa
import numpy as np
import pickle
import cv2
from pyarrow import csv
import os
from pathlib import Path
import multiprocessing
from simba.utils.printing import SimbaTimer


from simba.utils.errors import (InvalidFileTypeError,
                                MissingProjectConfigEntryError,
                                NotDirectoryError,
                                InvalidInputError,
                                ParametersFileError,
                                InvalidFilepathError,
                                InvalidVideoFileError,
                                DuplicationError,
                                NoFilesFoundError,
                                DataHeaderError,
                                FileExistError)
from simba.utils.warnings import InvalidValueWarning, NoFileFoundWarning, FileExistWarning
from simba.utils.printing import stdout_success
from simba.utils.enums import Formats, Dtypes, ConfigKey
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_filepath_list_is_empty)



PARSE_OPTIONS = csv.ParseOptions(delimiter=',')
READ_OPTIONS = csv.ReadOptions(encoding='utf8')

def read_df(file_path: str,
            file_type: str,
            remove_columns: list or None = None,
            usecols: list or None = None,
            check_multiindex: bool = False):

    """
    Helper function to read single data file into memory.

    Parameters
    ----------
    file_path: str
        Path to data file.
    file_type: str
        Type of data. OPTIONS: 'parquet' or 'csv'.
    idx: int,
        Index column location. Default: 0.
    remove_columns: list or None,
        If list, then remove columns
    usecols: list or None,
        If list, keep columns

    Returns
    -------
    df: pd.DataFrame

    """
    check_file_exist_and_readable(file_path=file_path)
    if file_type == Formats.CSV.value:
        try:
            df = csv.read_csv(file_path, parse_options=PARSE_OPTIONS, read_options=READ_OPTIONS)
            duplicate_headers = list(set([x for x in df.column_names if df.column_names.count(x) > 1]))
            if len(duplicate_headers) > 0:
                new_headers = [duplicate_headers[0] + f'_{x}' for x in range(len(df.column_names))]
                df = df.rename_columns(new_headers)
            df = df.to_pandas().iloc[:, 1:]
            if check_multiindex:
                header_col_cnt = get_number_of_header_columns_in_df(df=df)
                df = df.drop(df.index[list(range(0, header_col_cnt))]).apply(pd.to_numeric).reset_index(drop=True)
            df = df.astype(np.float32)
        except Exception as e:
            print(e, e.args)
            raise InvalidFileTypeError(msg=f'{file_path} is not a valid CSV file')
        if remove_columns:
            df = df[df.columns[~df.columns.isin(remove_columns)]]
        if usecols:
            df = df[df.columns[df.columns.isin(usecols)]]
    elif file_type == Formats.PARQUET.value:
        df = pd.read_parquet(file_path)
        if check_multiindex:
            header_col_cnt = get_number_of_header_columns_in_df(df=df)
            df = df.drop(df.index[list(range(0, header_col_cnt))]).apply(pd.to_numeric).reset_index(drop=True)
        df = df.astype(np.float32)

    elif file_type == Formats.PICKLE.value:
        with open(file_path, 'rb') as fp:
            df = pickle.load(fp)
    else:
        raise InvalidFileTypeError(msg=f'{file_type} is not a valid filetype OPTIONS: [pickle, csv, parquet]')

    return df

def write_df(df: pd.DataFrame,
             file_type: str,
             save_path: str,
             multi_idx_header: bool = False):

    """
    Helper function to save single dataframe from memory.

    Parameters
    ----------
    df: pd.DataFrame
        Pandas dataframe to save to disk.
    file_type: str
        Type of data. OPTIONS: 'parquet' or 'csv'.
    save_path: str,
        Location where to store the data.
    multi_idx_header: bool,
        If the data should be saved as a multi-index header file.

    Returns
    -------
    None
    """

    if file_type == Formats.CSV.value:
        if not multi_idx_header:
            df = df.drop('scorer', axis=1, errors='ignore')
            idx = np.arange(len(df)).astype(str)
            df.insert(0, '', idx)
            df = pa.Table.from_pandas(df=df)
            if '__index_level_0__' in df.columns:
                df = df.drop(['__index_level_0__'])
            csv.write_csv(df, save_path)
        else:
            df = df.drop('scorer', axis=1, errors='ignore')
            df.to_csv(save_path)
    elif file_type == Formats.PARQUET.value:
        df.to_parquet(save_path)
    elif file_type == Formats.PICKLE.value:
        try:
            with open(save_path, 'wb') as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e.args[0])
            raise InvalidFileTypeError(msg='Data could not be saved as a pickle.')
    else:
        raise InvalidFileTypeError(msg=f'{file_type} is not a valid filetype OPTIONS: [csv, pickle, parquet]')

def get_fn_ext(filepath: str):
    """
    Helper to split file path into three components: (i) directory, (ii) file name, and (iii) file extension.
    Parameters
    ----------
    filepath: str
        Path to file.

    Returns
    -------
    dir_name: str
    file_name: str
    file_extension: str
    """
    file_extension = Path(filepath).suffix
    try:
        file_name = os.path.basename(filepath.rsplit(file_extension, 1)[0])
    except ValueError:
        raise InvalidFilepathError(msg='{} is not a valid filepath'.format(filepath))
    dir_name = os.path.dirname(filepath)
    return dir_name, file_name, file_extension

def read_config_entry(config: configparser.ConfigParser,
                      section: str,
                      option: str,
                      data_type: str,
                      default_value=None,
                      options=None):
    try:
        if config.has_option(section, option):
            if data_type == Dtypes.FLOAT.value:
                value = config.getfloat(section, option)
            elif data_type == Dtypes.INT.value:
                value = config.getint(section, option)
            elif data_type == Dtypes.STR.value:
                value = config.get(section, option).strip()
            elif data_type == Dtypes.FOLDER.value:
                value = config.get(section, option).strip()
                if not os.path.isdir(value):
                    raise NotDirectoryError(msg=f'The SimBA config file includes paths to a folder ({value}) that does not exist.')
            if options != None:
                if value not in options:
                    raise InvalidInputError(msg=f'{option} is set to {str(value)} in SimBA, but this is not among the valid options: ({options})')
                else:
                    return value
            return value
        elif default_value != None:
            return default_value
        else:
            raise MissingProjectConfigEntryError(msg=f'SimBA could not find an entry for option {option} under section {section} in the project_config.ini. Please specify the settings in the settings menu.')
    except ValueError:
        if default_value != None:
            return default_value
        else:
            raise MissingProjectConfigEntryError(msg=f'SimBA could not find an entry for option {option} under section {section} in the project_config.ini. Please specify the settings in the settings menu.')


def read_project_path_and_file_type(config: configparser.ConfigParser):
    project_path = read_config_entry(config=config,
                                     section=ConfigKey.GENERAL_SETTINGS.value,
                                     option=ConfigKey.PROJECT_PATH.value,
                                     data_type=ConfigKey.FOLDER_PATH.value)
    file_type = read_config_entry(config=config,
                                  section=ConfigKey.GENERAL_SETTINGS.value,
                                  option=ConfigKey.FILE_TYPE.value,
                                  data_type=Dtypes.STR.value,
                                  default_value=Formats.CSV.value)

    return project_path, file_type


def read_video_info_csv(file_path: str):

    """
    Helper to read the project_folder/logs/video_info.csv of the SimBA project in as a pd.DataFrame
    Parameters
    ----------
    file_path: str
    Returns
    -------
    pd.DataFrame
    """
    check_file_exist_and_readable(file_path=file_path)
    info_df = pd.read_csv(file_path)
    for c in ['Video', 'fps', 'Resolution_width', 'Resolution_height', 'Distance_in_mm', 'pixels/mm']:
        if c not in info_df.columns:
            raise ParametersFileError(
                msg=f'The project "project_folder/logs/video_info.csv" does not not have an anticipated header ({c}). Please re-create the file and make sure each video has a {c} value')
    info_df['Video'] = info_df['Video'].astype(str)
    for c in ['fps', 'Resolution_width', 'Resolution_height', 'Distance_in_mm', 'pixels/mm']:
        try:
            info_df[c] = info_df[c].astype(float)
        except:
            raise ParametersFileError(msg=f'One or more values in the {c} column of the "project_folder/logs/video_info.csv" file could not be interpreted as a numeric value. Please re-create the file and make sure the entries in the {c} column are all numeric.')
    if info_df['fps'].min() <= 1:
        InvalidValueWarning(msg='Videos in your SimBA project have an FPS of 1 or less. Please use videos with more than one frame per second, or correct the inaccurate fps inside the `project_folder/logs/videos_info.csv` file')
    return info_df

def read_config_file(config_path: str):
    """ Helper to read project_config.ini file"""
    config = ConfigParser()
    try:
        config.read(config_path)
    except Exception as e:
        print(e.args)
        raise MissingProjectConfigEntryError(msg=f'{self.config_path} is not a valid project_config file. Please check the project_config.ini path.')
    return config


def get_video_meta_data(video_path: str):
    """
    Helper to read video metadata (fps, resolution, frame cnt etc.) from video file.

    Parameters
    ----------
    video_path: str
        Path to video file.

    Returns
    -------
    vdata: dict
        Python dictionary holding video meta data

    Notes
    ----------

    Examples
    >>> get_video_meta_data('/Users/simon/Desktop/envs/simba_dev/tests/test_data/video_tests/Together_1.avi')
    {'video_name': 'Together_1', 'fps': 30, 'width': 400, 'height': 600, 'frame_count': 300, 'resolution_str': '400 x 600', 'video_length_s': 10}

    """

    video_data = {}
    cap = cv2.VideoCapture(video_path)
    _, video_data['video_name'], _ = get_fn_ext(video_path)
    video_data['fps'] = int(cap.get(cv2.CAP_PROP_FPS))
    video_data['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_data['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_data['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for k, v in video_data.items():
        if v == 0:
            raise InvalidVideoFileError(msg=f'SIMBA WARNING: Video {video_data["video_name"]} has {k} of {str(v)} (full error video path: {video_path}).')
    video_data['resolution_str'] = str(f'{video_data["width"]} x {video_data["height"]}')
    video_data['video_length_s'] = int(video_data['frame_count'] / video_data['fps'])
    return video_data

def remove_a_folder(folder_dir: str):
    """Helper to remove a directory"""
    shutil.rmtree(folder_dir, ignore_errors=True)

def concatenate_videos_in_folder(in_folder: str,
                                 save_path: str,
                                 video_format: str = 'mp4',
                                 remove_splits: bool = True) -> None:
    """
    Helper to  concatenate (temporally) all video files in a folder into a
    single video.

    Parameters
    ----------
    in_folder: str
        Path to folder holding un-concatenated video files.
    save_path: str
        Path to the saved the output file.
    """

    timer = SimbaTimer(start=True)
    files = glob.glob(in_folder + '/*.{}'.format(video_format))
    check_if_filepath_list_is_empty(filepaths=files,
                                    error_msg='SIMBA ERROR: Cannot join videos in {}. The directory contain ZERO files.'.format(
                                        in_folder))
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    temp_txt_path = Path(in_folder, 'files.txt')
    with open(temp_txt_path, 'w') as f:
        for file in files:
            f.write("file '" + str(Path(file)) + "'\n")
    if os.path.exists(save_path): os.remove(save_path)
    returned = os.system(
        'ffmpeg -f concat -safe 0 -i "{}" "{}" -c copy -hide_banner -loglevel info'.format(temp_txt_path, save_path))
    while True:
        if returned != 0:
            pass
        else:
            if remove_splits:
                remove_a_folder(folder_dir=in_folder)
            break
    timer.stop_timer()
    stdout_success(msg='Video concatenated', elapsed_time=timer.elapsed_time_str)


def get_bp_headers(body_parts_lst: list) -> list:
    """
    Helper to create ordered list of all column header fields for SimBA project dataframes.
    """
    bp_headers = []
    for bp in body_parts_lst:
        c1, c2, c3 = (f'{bp}_x', f'{bp}_y', f'{bp}_p')
        bp_headers.extend((c1, c2, c3))
    return bp_headers

def read_video_info(vid_info_df: pd.DataFrame,
                    video_name: str):
    """
    Helper to read the meta data (pixels per mm, resolution, fps etc) from the video_info.csv for a single input file

    Parameters
    ----------
    vid_info_df: pd.DataFrame
        Pandas dataframe representing the content of the project_folder/logs/video_info.csv of the SimBA project.
    video_name: str
        The name of the video without extension to get the meta data for.

    Returns
    -------
    video_settings: pd.DataFrame
    px_per_mm: float
    fps: float
    """

    video_settings = vid_info_df.loc[vid_info_df['Video'] == video_name]
    if len(video_settings) > 1:
        raise DuplicationError(msg=f'SimBA found multiple rows in the project_folder/logs/video_info.csv named {str(video_name)}. Please make sure that each video name is represented ONCE in the video_info.csv')
    elif len(video_settings) < 1:
        raise ParametersFileError(msg=f' SimBA could not find {str(video_name)} in the video_info.csv file. Make sure all videos analyzed are represented in the project_folder/logs/video_info.csv file.')
    else:
        try:
            px_per_mm = float(video_settings['pixels/mm'])
            fps = float(video_settings['fps'])
            return video_settings, px_per_mm, fps
        except TypeError:
            raise ParametersFileError(msg=f'Make sure the videos that are going to be analyzed are represented with APPROPRIATE VALUES inside the project_folder/logs/video_info.csv file in your SimBA project. Could not interpret the fps, pixels per millimeter and/or fps as numerical values for video {video_name}')


def find_all_videos_in_directory(directory: str,
                                 as_dict: bool = False,
                                 video_formats: tuple = ('.avi', '.mp4', '.mov', '.flv', '.m4v')):
    """
    Helper to return list of all video file paths in a directory

    Parameters
    ----------
    directory: str
        Directory to search.
    video_formats: tuple
        Allowed video formats.
    as_dict: bool
        Returns dictionary in format {'Video_1': /MyvideoPath/Video_1.mp4}

    Parameters
    ----------
    list or dict
    """

    video_lst = []
    for i in os.listdir(directory):
        if i.lower().endswith(video_formats):
            video_lst.append(i)
    if not video_lst:
        video_lst.append('No videos found')
        NoFileFoundWarning(msg=f'No videos found in directory ({directory})')

    if video_lst and as_dict:
        video_dict = {}
        for video_name in video_lst:
            video_path = os.path.join(directory, video_name)
            _, name, _ = get_fn_ext(filepath=video_path)
            video_dict[name] = video_path
        return video_dict

    return video_lst


def find_video_of_file(video_dir: str,
                       filename: str):
    """
    Helper to find the video file that represents a data file.

    Parameters
    ----------
    video_dir: str
        Directory holding putative video file
    filename: str
        Data file name, e.g., ``Video_1``.

    Returns
    -------
    return_path: str

    """
    try:
        all_files_in_video_folder = [f for f in next(os.walk(video_dir))[2] if not f[0] == '.']
    except StopIteration:
        raise NoFilesFoundError(msg=f'No files found in the {video_dir} directory')
    all_files_in_video_folder = [os.path.join(video_dir, x) for x in all_files_in_video_folder]
    return_path = None
    for file_path in all_files_in_video_folder:
        _, video_filename, ext = get_fn_ext(file_path)
        if ((video_filename == filename) and ((ext.lower() == '.mp4') or (ext.lower() == '.avi'))):
            return_path = file_path

    if return_path is None:
        NoFileFoundWarning(f'SimBA could not find a video file representing {filename} in the project video directory')
    return return_path


def find_files_of_filetypes_in_directory(directory: str,
                                         extensions: list,
                                         raise_warning: bool = True):
    """
    Helper to find all files in a folder with specified extensions

    Parameters
    ----------
    directory: str
        Directory holding files
    extensions: list
        Accepted file extensions

    Returns
    -------
    files: list

    """
    try:
        all_files_in_folder = [f for f in next(os.walk(directory))[2] if not f[0] == '.']
    except StopIteration:
        raise NoFilesFoundError(msg=f'No files found in the {directory} directory with accepted extensions {str(extensions)}')
    all_files_in_folder = [os.path.join(directory, x) for x in all_files_in_folder]
    accepted_file_paths = []
    for file_path in all_files_in_folder:
        _, file_name, ext = get_fn_ext(file_path)
        if ext.lower() in extensions:
            accepted_file_paths.append(file_path)
    if not accepted_file_paths and raise_warning:
        raise NoFileFoundWarning(msg=f'SimBA could not find any files with accepted extensions {extensions} in the {str(directory)} directory')
    return accepted_file_paths


def convert_parquet_to_csv(directory: str):
    """
    Helper to convert all parquet files in a folder to csv format.

    Parameters
    ----------
    directory: str
        Path to directory holding parquet files
    """

    if not os.path.isdir(directory):
        raise NotDirectoryError(msg='SIMBA ERROR: {} is not a valid directory'.format(directory))
    files_found = glob.glob(directory + '/*.parquet')
    if len(files_found) < 1:
        raise NoFilesFoundError('SIMBA ERROR: No parquet files (with .parquet file ending) found in the {} directory'.format(directory))
    for file_cnt, file_path in enumerate(files_found):
        print('Reading in {} ...'.format(os.path.basename(file_path)))
        df = pd.read_parquet(file_path)
        new_file_path = os.path.join(directory, os.path.basename(file_path).replace('.parquet', '.csv'))
        if 'scorer' in df.columns:
            df = df.set_index('scorer')
        df.to_csv(new_file_path)
        print('Saved {}...'.format(new_file_path))
    stdout_success(msg=f'{str(len(files_found))} parquet files in {directory} converted to csv')


def convert_csv_to_parquet(directory: str):
    """
    Helper to convert all csv files in a folder to parquet format.

    Parameters
    ----------
    directory: str
        Path to directory holding csv files

    Returns
    -------
    None
    """
    if not os.path.isdir(directory):
        raise NotDirectoryError(msg='SIMBA ERROR: {} is not a valid directory'.format(directory))
    files_found = glob.glob(directory + '/*.csv')
    if len(files_found) < 1:
        raise NoFilesFoundError(msg='SIMBA ERROR: No parquet files (with .csv file ending) found in the {} directory'.format(directory))
    print('Converting {} files...'.format(str(len(files_found))))
    for file_cnt, file_path in enumerate(files_found):
        print('Reading in {} ...'.format(os.path.basename(file_path)))
        df = pd.read_csv(file_path)
        new_file_path = os.path.join(directory, os.path.basename(file_path).replace('.csv', '.parquet'))
        df.to_parquet(new_file_path)
        print('Saved {}...'.format(new_file_path))
    stdout_success(msg=f'{str(len(files_found))} csv files in {directory} converted to parquet')


def get_file_name_info_in_directory(directory: str,
                                    file_type: str):
    """
    Helper to return dict of all file path with extension as values as basenames as keys.
    E.g., ['Video_1:, 'C:\project_folder\csv\machine_results\Video_1, ...]

    Parameters
    ----------
    directory: str
        Directory to search.
    video_formats: tuple
        Allowed video formats.

    Parameters
    ----------
    video_lst: list
    """

    results = {}
    file_paths = glob.glob(directory + "/*." + file_type)
    for file_path in file_paths:
        _, file_name, ext = get_fn_ext(file_path)
        results[file_name] = file_path

    return results


def archive_processed_files(config_path: str,
                            archive_name: str):
    """
    Helper to archive files within a SimBA project.

    Parameters
    ----------
    config_path: str
        Path to SimBA project ``project_config.ini``.
    archive_name: str
        Name of archive.
    """

    config = read_config_file(config_path=config_path)
    file_type = read_config_entry(config, ConfigKey.GENERAL_SETTINGS.value, ConfigKey.FILE_TYPE.value, 'str', 'csv')
    project_path = read_config_entry(config, ConfigKey.GENERAL_SETTINGS.value, ConfigKey.PROJECT_PATH.value,
                                     data_type=ConfigKey.FOLDER_PATH.value)
    videos_dir = os.path.join(project_path, 'videos')
    csv_dir = os.path.join(os.path.dirname(config_path), 'csv')
    log_path = os.path.join(project_path, 'logs')
    video_info_path = os.path.join(log_path, 'video_info.csv')
    csv_subdirs, file_lst = [], []
    for content_name in os.listdir(csv_dir):
        if os.path.isdir(os.path.join(csv_dir, content_name)):
            csv_subdirs.append(os.path.join(csv_dir, content_name))

    for subdirectory in csv_subdirs:
        subdirectory_files = [x for x in glob.glob(subdirectory + '/*') if os.path.isfile(x)]
        for file_path in subdirectory_files:
            directory, file_name, ext = get_fn_ext(os.path.join(subdirectory, file_path))
            if ext == '.{}'.format(file_type):
                file_lst.append(os.path.join(subdirectory, file_path))

    if len(file_lst) < 1:
        raise NoFilesFoundError(msg='SIMBA ERROR: No data files located in your project_folder/csv sub-directories in the worflow file format {}'.format(file_type))

    for file_path in file_lst:
        file_folder = os.path.dirname(file_path)
        save_directory = os.path.join(file_folder, archive_name)
        save_file_path = os.path.join(save_directory, os.path.basename(file_path))
        if not os.path.exists(save_directory): os.mkdir(save_directory)
        print('Moving file {}...'.format(file_path))
        shutil.move(file_path, save_file_path)

    log_archive_path = os.path.join(log_path, archive_name)
    if not os.path.exists(log_archive_path): os.mkdir(log_archive_path)
    if os.path.isfile(video_info_path):
        save_file_path = os.path.join(log_archive_path, 'video_info.csv')
        print('Moving file {}...'.format(video_info_path))
        shutil.move(video_info_path, save_file_path)

    videos_file_paths = [f for f in glob.glob(videos_dir) if os.path.isfile(f)]
    video_archive_path = os.path.join(videos_dir, archive_name)
    if not os.path.exists(video_archive_path): os.mkdir(video_archive_path)
    for video_file in videos_file_paths:
        save_video_path = os.path.join(video_archive_path, os.path.basename(video_file))
        shutil.move(video_file, save_video_path)
    stdout_success(msg='Archiving completed')

def str_2_bool(input_str: str):
  return input_str.lower() in ("yes", "true", "1")



def tabulate_clf_info(clf_path: str):
    """
    Helper to print the hyper-parameters and creation date of a pickled classifier.

    Parameters
    ----------
    clf_path: str
        Path to classifier

    Returns
    -------
    None

    """

    _, clf_name, _ = get_fn_ext(clf_path)
    check_file_exist_and_readable(file_path=clf_path)
    try:
        clf_obj = pickle.load(open(clf_path, 'rb'))
    except:
        raise InvalidFilepathError(msg=f'The {clf_path} file is not a pickle file')
    try:
        clf_features_no = clf_obj.n_features_
        clf_criterion = clf_obj.criterion
        clf_estimators = clf_obj.n_estimators
        clf_min_samples_leaf = clf_obj.min_samples_split
        clf_n_jobs = clf_obj.n_jobs
        clf_verbose = clf_obj.verbose
        if clf_verbose == 1: clf_verbose = True
        if clf_verbose == 0: clf_verbose = False
    except:
        raise InvalidFilepathError(msg=f'The {clf_path} file is not an scikit-learn RF classifier')
    creation_time = 'Unknown'
    try:
        if platform.system() == 'Windows':
            creation_time = os.path.getctime(clf_path)
        elif platform.system() == 'Darwin':
            creation_time = os.stat(clf_path)
            creation_time = creation_time.st_birthtime
    except AttributeError:
        pass
    if creation_time != 'Unknown':
        creation_time = str(datetime.utcfromtimestamp(creation_time).strftime('%Y-%m-%d %H:%M:%S'))

    print(str(clf_name), "CLASSIFIER INFORMATION")
    for (name, val) in zip(
            ['NUMBER OF FEATURES', 'NUMBER OF TREES', 'CLASSIFIER CRITERION', 'CLASSIFIER_MIN_SAMPLE_LEAF',
             'CLASSIFIER_N_JOBS', 'CLASSIFIER VERBOSE SETTING', 'CLASSIFIER PATH', 'CLASSIFIER CREATION TIME'],
            [clf_features_no, clf_estimators, clf_criterion, clf_min_samples_leaf,
             clf_n_jobs, clf_verbose, clf_path, str(creation_time)]):
        print(name + ': ' + str(val))

def get_all_clf_names(config: configparser.ConfigParser,
                      target_cnt: int):
    """
    Helper to get all classifier names in a SimBA project.

    Parameters
    ----------
    config:  configparser.ConfigParser
        Parsed SimBA project_config.ini
    target_cnt
        Count of models

    Returns
    -------
    model_names: list
    """

    model_names = []
    for i in range(target_cnt):
        entry_name = 'target_name_{}'.format(str(i+1))
        model_names.append(read_config_entry(config, ConfigKey.SML_SETTINGS.value, entry_name, data_type=Dtypes.STR.value))
    return model_names

def read_meta_file(meta_file_path):
    return pd.read_csv(meta_file_path, index_col=False).to_dict(orient='records')[0]


def read_simba_meta_files(folder_path: str):
    file_paths = find_files_of_filetypes_in_directory(directory=folder_path, extensions=['.csv'])
    meta_file_lst = []
    for i in file_paths:
        if i.__contains__("meta"):
            meta_file_lst.append(os.path.join(folder_path, i))
    if len(meta_file_lst) == 0:
        NoFileFoundWarning(msg=f'The training meta-files folder in your project ({folder_path}) does not have any meta files inside it (no files in this folder has the "meta" substring in the filename)')
    return meta_file_lst


def find_core_cnt():
    """
    Helper to find the local cpu count and half of the cpu counts.

    Returns
    -------
    cpu_cnt: int
    cpu_cnt_to_use: int
    """
    cpu_cnt = multiprocessing.cpu_count()
    cpu_cnt_to_use = int(cpu_cnt / 4)
    if cpu_cnt_to_use < 1:
        cpu_cnt_to_use = 1
    return cpu_cnt, cpu_cnt_to_use

def get_number_of_header_columns_in_df(df: pd.DataFrame):
    """ Returns the number of non-numerical header rows in dataframe """
    for i in range(len(df)):
        try:
            temp = df.iloc[i:].apply(pd.to_numeric).reset_index(drop=True)
            return i
        except ValueError:
            pass
    raise DataHeaderError(msg='Could find the count of header columns in dataframe')

def get_memory_usage_of_df(df: pd.DataFrame) -> dict:
    results = {}
    results['bytes'] = df.memory_usage(index=True).sum()
    results['megabytes'] = round(results['bytes'] / 1000000, 6)
    results['gigabytes'] = round(results['bytes'] / 1000000000, 6)
    return results



def copy_single_video_to_project(simba_ini_path: str,
                                 source_path: str,
                                 allowed_video_formats: tuple = ('avi', 'mp4')) -> None:
    """ Helper to import single video file to SimBA project

    Parameters
    ----------
    simba_ini_path: str
        path to SimBA project config file in Configparser format
    source_path: str
        Path to video file.
    allowed_video_formats: tuple
        Allowed video formats. DEFAULT: avi or mp4
    """

    timer = SimbaTimer(start=True)
    _, file_name, file_ext = get_fn_ext(source_path)
    check_file_exist_and_readable(file_path=source_path)
    print('Copying video {} file...'.format(file_name))
    if file_ext[1:].lower().strip() not in allowed_video_formats:
        raise InvalidFileTypeError(msg='SimBA works best with avi and mp4 video-files. Or please convert your videos to mp4 or avi to continue before importing it.')
    new_filename = os.path.join(file_name + file_ext)
    destination = os.path.join(os.path.dirname(simba_ini_path), 'videos', new_filename)
    if os.path.isfile(destination):
        raise FileExistError(msg=f'{file_name} already exist in SimBA project. To import, delete this video file before importing the new video file with the same name.')
    else:
        timer.stop_timer()
        shutil.copy(source_path, destination)
        stdout_success(msg=f'Video {file_name} imported to SimBA project (project_folder/videos directory', elapsed_time=timer.elapsed_time_str)



def copy_multiple_videos_to_project(config_path: str,
                                    source: str,
                                    file_type: str,
                                    allowed_video_formats: tuple = ('avi', 'mp4')) -> None:
    """ Helper to import directory of videos to SimBA project

    Parameters
    ----------
    config_path: str
        path to SimBA project config file in Configparser format
    source_path: str
        Path to video file directory
    file_type: str
        Video format of imported videos (i.e.,.: mp4 or avi)
    allowed_video_formats: tuple
        Allowed video formats. DEFAULT: avi or mp4

    """

    if file_type.lower().strip() not in allowed_video_formats:
        raise InvalidFileTypeError(msg='SimBA only works with avi and mp4 videofiles (Please enter mp4 or avi in entrybox). Or convert your videos to mp4 or avi to continue.')
    video_path_lst = find_all_videos_in_directory(directory=source, video_formats=(file_type))
    video_path_lst = [os.path.join(source, x) for x in video_path_lst]
    if len(video_path_lst) == 0:
        raise NoFilesFoundError(msg='SIMBA ERROR: No videos found in {} directory in file-type {}'.format(source, file_type))
    destination_dir = os.path.join(os.path.dirname(config_path), 'videos')
    for file_cnt, file_path in enumerate(video_path_lst):
        timer = SimbaTimer()
        timer.start_timer()
        dir_name, filebasename, file_extension = get_fn_ext(file_path)
        file_extension = file_extension.lower()
        newFileName = os.path.join(filebasename + file_extension)
        dest1 = os.path.join(destination_dir, newFileName)
        if os.path.isfile(dest1):
            FileExistWarning(msg=f'{filebasename} already exist in SimBA project. Skipping video...')
        else:
            shutil.copy(file_path, dest1)
            timer.stop_timer()
            print('{} copied to project (Video {}/{}, elapsed timer {}s)...'.format(filebasename, str(file_cnt + 1),
                                                                                    str(len(video_path_lst)),
                                                                                    timer.elapsed_time_str))
    stdout_success(msg=f'{str(len(video_path_lst))} videos copied to project.')

def find_all_videos_in_project(videos_dir: str):
    """ Returns filenames of .avi and .mp4 files in a directory"""
    video_paths = []
    file_paths_in_folder = [f for f in next(os.walk(videos_dir))[2] if not f[0] == '.']
    file_paths_in_folder = [os.path.join(videos_dir, f) for f in file_paths_in_folder]
    for file_cnt, file_path in enumerate(file_paths_in_folder):
        try:
            _, file_name, file_ext = get_fn_ext(file_path)
        except ValueError:
            raise InvalidFilepathError(msg='{} is not a valid filepath'.format(file_path))
        if (file_ext.lower() == '.mp4') or (file_ext.lower() == '.avi'):
            video_paths.append(file_path)
    if len(video_paths) == 0:
        raise NoFilesFoundError(msg=f'No videos in mp4 or avi format found imported to SimBA project in the {videos_dir} directory')
    else:
        return video_paths
