__author__ = "Simon Nilsson"

import configparser
import glob
import multiprocessing
import os
import pickle
import platform
import re
import shutil
import subprocess
import threading
import webbrowser
from configparser import ConfigParser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import pkg_resources
import pyarrow as pa
from pyarrow import csv
from shapely.geometry import (LineString, MultiLineString, MultiPolygon, Point,
                              Polygon)

from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_filepath_list_is_empty,
                                check_if_string_value_is_valid_video_timestamp,
                                check_instance, check_int,
                                check_nvidea_gpu_available, check_valid_lst)
from simba.utils.enums import ConfigKey, Dtypes, Formats, Keys, Options
from simba.utils.errors import (DataHeaderError, DuplicationError,
                                FeatureNumberMismatchError,
                                FFMPEGCodecGPUError, FileExistError,
                                FrameRangeError, IntegerError,
                                InvalidFilepathError, InvalidFileTypeError,
                                InvalidInputError, InvalidVideoFileError,
                                MissingProjectConfigEntryError, NoDataError,
                                NoFilesFoundError, NotDirectoryError,
                                ParametersFileError, PermissionError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.warnings import (FileExistWarning, InvalidValueWarning,
                                  NoDataFoundWarning, NoFileFoundWarning)

# from simba.utils.keyboard_listener import KeyboardListener


PARSE_OPTIONS = csv.ParseOptions(delimiter=",")
READ_OPTIONS = csv.ReadOptions(encoding="utf8")


def read_df(
    file_path: Union[str, os.PathLike],
    file_type: Union[str, os.PathLike],
    has_index: Optional[bool] = True,
    remove_columns: Optional[List[str]] = None,
    usecols: Optional[List[str]] = None,
    anipose_data: Optional[bool] = False,
    check_multiindex: Optional[bool] = False,
    multi_index_headers_to_keep: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read single tabular data file or pickle

    .. note::
       For improved runtime, defaults to :external:py:meth:`pyarrow.csv.write_cs` if file type is ``csv``.

    :parameter str file_path: Path to data file
    :parameter str file_type: Type of data. OPTIONS: 'parquet', 'csv', 'pickle'.
    :parameter Optional[bool]: If the input file has an initial index column. Default: True.
    :parameter Optional[List[str]] remove_columns: If not None, then remove columns in lits.
    :parameter Optional[List[str]] usecols: If not None, then keep columns in list.
    :parameter bool check_multiindex: check file is multi-index headers. Default: False.
    :parameter int multi_index_headers_to_keep: If reading multi-index file, and we want to keep one of the dropped multi-index levels as the header in the output file, specify the index of the multiindex hader as int.
    :return pd.DataFrame

    :example:
    >>> read_df(file_path='project_folder/csv/input_csv/Video_1.csv', file_type='csv', check_multiindex=True)
    """
    check_file_exist_and_readable(file_path=file_path)
    if file_type == Formats.CSV.value:
        try:
            df = csv.read_csv(
                file_path, parse_options=PARSE_OPTIONS, read_options=READ_OPTIONS
            )
            duplicate_headers = list(
                set([x for x in df.column_names if df.column_names.count(x) > 1])
            )
            if len(duplicate_headers) > 0:
                new_headers = [
                    duplicate_headers[0] + f"_{x}" for x in range(len(df.column_names))
                ]
                df = df.rename_columns(new_headers)
            if anipose_data:
                df = df.to_pandas()
                has_index = True
            else:
                df = df.to_pandas().iloc[:, 1:]
            if check_multiindex:
                header_col_cnt = get_number_of_header_columns_in_df(df=df)
                if multi_index_headers_to_keep is not None:
                    if multi_index_headers_to_keep not in list(
                        range(0, header_col_cnt)
                    ):
                        raise InvalidInputError(
                            msg=f"The selected multi-header index column {multi_index_headers_to_keep} does not exist in the multi-index header columns: {list(range(0, header_col_cnt))}",
                            source=read_df.__name__,
                        )
                    else:
                        new_header = list(
                            df.iloc[multi_index_headers_to_keep, :].values
                        )
                        new_header_xy = []
                        for header in list(set(new_header)):
                            new_header_xy.append(f"{header}_x")
                            new_header_xy.append(f"{header}_y"), new_header_xy.append(
                                f"{header}_likelihood"
                            )
                        df = df.drop(df.index[list(range(0, header_col_cnt))]).apply(
                            pd.to_numeric
                        )
                        df.columns = new_header_xy
                else:
                    df = df.drop(df.index[list(range(0, header_col_cnt))]).apply(
                        pd.to_numeric
                    )
            if not has_index:
                df = df.reset_index()
            else:
                df = df.reset_index(drop=True)
            df = df.astype(np.float32)

        except Exception as e:
            print(e, e.args)
            raise InvalidFileTypeError(
                msg=f"{file_path} is not a valid CSV file", source=read_df.__name__
            )
        if remove_columns:
            df = df[df.columns[~df.columns.isin(remove_columns)]]
        if usecols:
            df = df[df.columns[df.columns.isin(usecols)]]
    elif file_type == Formats.PARQUET.value:
        df = pd.read_parquet(file_path)
        if check_multiindex:
            header_col_cnt = get_number_of_header_columns_in_df(df=df)
            df = (
                df.drop(df.index[list(range(0, header_col_cnt))])
                .apply(pd.to_numeric)
                .reset_index(drop=True)
            )
        df = df.astype(np.float32)

    elif file_type == Formats.PICKLE.value:
        with open(file_path, "rb") as fp:
            df = pickle.load(fp)
    else:
        raise InvalidFileTypeError(
            msg=f"{file_type} is not a valid filetype OPTIONS: [pickle, csv, parquet]",
            source=read_df.__name__,
        )

    return df


def write_df(
    df: pd.DataFrame,
    file_type: str,
    save_path: Union[str, os.PathLike],
    multi_idx_header: bool = False,
) -> None:
    """
    Write single tabular data file.

    .. note::
       For improved runtime, defaults to ``pyarrow.csv`` if file_type == ``csv``.

    :parameter pd.DataFrame df: Pandas dataframe to save to disk.
    :parameter str file_type: Type of data. OPTIONS: ``parquet``, ``csv``,  ``pickle``.
    :parameter str save_path: Location where to store the data.
    :parameter bool check_multiindex: check if input file is multi-index headers. Default: False.

    :example:
    >>> write_df(df=df, file_type='csv', save_path='project_folder/csv/input_csv/Video_1.csv')
    """

    if file_type == Formats.CSV.value:
        if not multi_idx_header:
            df = df.drop("scorer", axis=1, errors="ignore")
            idx = np.arange(len(df)).astype(str)
            df.insert(0, "", idx)
            df = pa.Table.from_pandas(df=df)
            if "__index_level_0__" in df.column_names:
                df = df.remove_column(df.column_names.index("__index_level_0__"))
            csv.write_csv(df, save_path)
        else:
            try:
                df = df.drop("scorer", axis=1, errors="ignore")
            except TypeError:
                pass
            df.to_csv(save_path)
    elif file_type == Formats.PARQUET.value:
        df.to_parquet(save_path)
    elif file_type == Formats.PICKLE.value:
        try:
            with open(save_path, "wb") as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e.args[0])
            raise InvalidFileTypeError(
                msg="Data could not be saved as a pickle.", source=write_df.__name__
            )
    else:
        raise InvalidFileTypeError(
            msg=f"{file_type} is not a valid filetype OPTIONS: [csv, pickle, parquet]",
            source=write_df.__name__,
        )


def get_fn_ext(filepath: Union[os.PathLike, str]) -> (str, str, str):
    """
    Split file path into three components: (i) directory, (ii) file name, and (iii) file extension.

    :parameter str filepath: Path to file.
    :return str: File directory name
    :return str: File name
    :return str: File extension

    :example:
    >>> get_fn_ext(filepath='C:/My_videos/MyVideo.mp4')
    >>> ('My_videos', 'MyVideo', '.mp4')
    """
    file_extension = Path(filepath).suffix
    try:
        file_name = os.path.basename(filepath.rsplit(file_extension, 1)[0])
    except ValueError:
        raise InvalidFilepathError(msg=f"{filepath} is not a valid filepath", source=get_fn_ext.__name__)
    dir_name = os.path.dirname(filepath)
    return dir_name, file_name, file_extension


def read_config_entry(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    data_type: str,
    default_value: Optional[Any] = None,
    options: Optional[List] = None,
) -> Union[float, int, str]:
    """
    Helper to read entry in SimBA project_config.ini parsed by configparser.ConfigParser.

    :param configparser.ConfigParser config: Parsed SimBA project_config.ini. Use :meth:`simba.utils.read_config_file` to parse file.
    :param str section: Section name of entry to parse.
    :param str option: Option name of entry to parse.
    :param str data_type: Type of data to parse. E.g., `str`, `int`, `float`.
    :param Optional[Any] default_value: If no matching entry can be found in the project_config.ini, use this as default.
    :param Optional[List] or None options: List of valid options. If not None, checks that the returned entry value exists in this list.
    :return Any

    :example:
    >>> read_config_entry(config='project_folder/project_config.ini', section='General settings', option='project_name', data_type='str')
    >>> 'two_animals_14_bps'
    """

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
                    raise NotDirectoryError(
                        msg=f"The SimBA config file includes paths to a folder ({value}) that does not exist.",
                        source=read_config_entry.__name__,
                    )
            if options != None:
                if value not in options:
                    raise InvalidInputError(
                        msg=f"{option} is set to {str(value)} in SimBA, but this is not among the valid options: ({options})",
                        source=read_config_entry.__name__,
                    )
                else:
                    return value
            return value

        elif default_value != None:
            return default_value
        else:
            raise MissingProjectConfigEntryError(
                msg=f"SimBA could not find an entry for option {option} under section {section} in the project_config.ini. Please specify the settings in the settings menu and make sure the path to your project config is correct",
                source=read_config_entry.__name__,
            )
    except ValueError:
        if default_value != None:
            return default_value
        else:
            raise MissingProjectConfigEntryError(
                msg=f"SimBA could not find an entry for option {option} under section {section} in the project_config.ini. Please specify the settings in the settings menu.",
                source=read_config_entry.__name__,
            )


def read_project_path_and_file_type(config: configparser.ConfigParser) -> (str, str):
    """
    Helper to read the path and file type of the SimBA project from the project_config.ini.

    :param configparser.ConfigParser config: parsed SimBA config in configparser.ConfigParser format
    :return str: The path of the project ``project_folder``.
    :return str: The set file type of the project (i.e., ``csv`` or ``parquet``).
    """

    project_path = read_config_entry(
        config=config,
        section=ConfigKey.GENERAL_SETTINGS.value,
        option=ConfigKey.PROJECT_PATH.value,
        data_type=ConfigKey.FOLDER_PATH.value,
    ).strip()
    file_type = read_config_entry(
        config=config,
        section=ConfigKey.GENERAL_SETTINGS.value,
        option=ConfigKey.FILE_TYPE.value,
        data_type=Dtypes.STR.value,
        default_value=Formats.CSV.value,
    ).strip()
    if not os.path.isdir(project_path):
        raise NotDirectoryError(
            msg=f"The project config file {config} has project path {project_path} that does not exist",
            source=read_project_path_and_file_type.__name__,
        )

    return project_path, file_type


def read_video_info_csv(file_path: Union[str, os.PathLike]) -> pd.DataFrame:
    """
    Read the project_folder/logs/video_info.csv of the SimBA project as a pd.DataFrame

    :parameter str file_path: Path to the SimBA project ``project_folder/logs/video_info.csv`` file
    :return pd.DataFrame
    :raise ParametersFileError: Invalid format of ``project_folder/logs/video_info.csv``.
    :raise InvalidValueWarning: Some videos are registered with FPS >= 1.

    :example:
    >>> read_video_info_csv(file_path='project_folder/logs/video_info.csv')
    """

    check_file_exist_and_readable(file_path=file_path)
    info_df = pd.read_csv(file_path)
    for c in [
        "Video",
        "fps",
        "Resolution_width",
        "Resolution_height",
        "Distance_in_mm",
        "pixels/mm",
    ]:
        if c not in info_df.columns:
            raise ParametersFileError(
                msg=f'The project "project_folder/logs/video_info.csv" does not not have an anticipated header ({c}). Please re-create the file and make sure each video has a {c} value',
                source=read_video_info_csv.__name__,
            )
    info_df["Video"] = info_df["Video"].astype(str)
    for c in [
        "fps",
        "Resolution_width",
        "Resolution_height",
        "Distance_in_mm",
        "pixels/mm",
    ]:
        try:
            info_df[c] = info_df[c].astype(float)
        except:
            raise ParametersFileError(
                msg=f'One or more values in the {c} column of the "project_folder/logs/video_info.csv" file could not be interpreted as a numeric value. Please re-create the file and make sure the entries in the {c} column are all numeric.',
                source=read_video_info_csv.__name__,
            )
    if info_df["fps"].min() <= 1:
        InvalidValueWarning(
            msg="Videos in your SimBA project have an FPS of 1 or less. Please use videos with more than one frame per second, or correct the inaccurate fps inside the `project_folder/logs/videos_info.csv` file",
            source=read_video_info_csv.__name__,
        )
    return info_df


def read_config_file(config_path: Union[str, os.PathLike]) -> configparser.ConfigParser:
    """
    Helper to parse SimBA project project_config.ini file

    :parameter str config_path: Path to project_config.ini file
    :return configparser.ConfigParser: parsed project_config.ini file
    :raise MissingProjectConfigEntryError: Invalid file format.

    :example:
    >>> read_config_file(config_path='project_folder/project_config.ini')
    """

    config = ConfigParser()
    try:
        config.read(config_path)
    except Exception as e:
        print(e.args)
        raise MissingProjectConfigEntryError(
            msg=f"{config_path} is not a valid project_config file. Please check the project_config.ini path.",
            source=read_config_entry.__name__,
        )
    return config


def get_video_meta_data(
    video_path: Union[str, os.PathLike], fps_as_int: bool = True
) -> dict:
    """
    Read video metadata (fps, resolution, frame cnt etc.) from video file (e.g., mp4).

    :parameter str video_path: Path to a video file.
    :parameter bool fps_as_int: If True, force video fps to int through floor rounding, else float. Default = True.
    :return dict: Video file meta data.

    :example:
    >>> get_video_meta_data('test_data/video_tests/Video_1.avi')
    {'video_name': 'Video_1', 'fps': 30, 'width': 400, 'height': 600, 'frame_count': 300, 'resolution_str': '400 x 600', 'video_length_s': 10}
    """

    video_data = {}
    cap = cv2.VideoCapture(video_path)
    _, video_data["video_name"], _ = get_fn_ext(video_path)
    video_data["fps"] = cap.get(cv2.CAP_PROP_FPS)
    if fps_as_int:
        video_data["fps"] = int(video_data["fps"])
    video_data["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_data["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_data["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for k, v in video_data.items():
        if v == 0:
            raise InvalidVideoFileError(
                msg=f'Video {video_data["video_name"]} either does not exist or has {k} of {str(v)} (full error video path: {video_path}).',
                source=get_video_meta_data.__name__,
            )
    video_data["resolution_str"] = str(
        f'{video_data["width"]} x {video_data["height"]}'
    )
    video_data["video_length_s"] = int(video_data["frame_count"] / video_data["fps"])
    return video_data


def remove_a_folder(folder_dir: Union[str, os.PathLike]) -> None:
    """Helper to remove a directory"""
    shutil.rmtree(folder_dir, ignore_errors=True)


def concatenate_videos_in_folder(
    in_folder: Union[str, os.PathLike],
    save_path: Union[str, os.PathLike],
    file_paths: Optional[List[Union[str, os.PathLike]]] = None,
    video_format: Optional[str] = "mp4",
    substring: Optional[str] = None,
    remove_splits: Optional[bool] = True,
    gpu: Optional[bool] = False,
    verbose: Optional[bool] = True,
) -> None:
    """
    Concatenate (temporally) all video files in a folder into a single video.

    .. important::
       Input video parts will be joined in alphanumeric order, should ideally have to have sequential numerical ordered file names, e.g., ``1.mp4``, ``2.mp4``....

    .. note::
       If substring and file_paths are both not None, then file_paths with be sliced and only file paths with substring will be retained.

    :parameter Union[str, os.PathLike] in_folder: Path to folder holding un-concatenated video files.
    :parameter Union[str, os.PathLike] save_path: Path to the saved the output file. Note: If the path exist, it will be overwritten
    :parameter Optional[List[Union[str, os.PathLike]]] file_paths: If not None, then the files that should be joined. If None, then all files. Default None.
    :parameter Optional[str] video_format: The format of the video clips that should be concatenated. Default: mp4.
    :parameter Optional[str] substring: If a string, then only videos in in_folder with a filename that contains substring will be joined. If None, then all are joined. Default: None.
    :parameter Optional[str] video_format: Format of the input video files in ``in_folder``. Default: ``mp4``.
    :parameter Optional[bool] remove_splits: If true, the input splits in the ``in_folder`` will be removed following concatenation. Default: True.
    """

    if not check_nvidea_gpu_available() and gpu:
        raise FFMPEGCodecGPUError(
            msg="No FFMpeg GPU codec found.",
            source=concatenate_videos_in_folder.__name__,
        )
    timer = SimbaTimer(start=True)
    if file_paths is None:
        files = glob.glob(in_folder + "/*.{}".format(video_format))
    else:
        for file_path in file_paths:
            check_file_exist_and_readable(file_path=file_path)
        files = file_paths
    check_if_filepath_list_is_empty(
        filepaths=files,
        error_msg=f"SIMBA ERROR: Cannot join videos in directory {in_folder}. The directory contain ZERO files in format {video_format}",
    )
    if substring is not None:
        sliced_paths = []
        for file_path in files:
            if substring in get_fn_ext(filepath=file_path)[1]:
                sliced_paths.append(file_path)
        check_if_filepath_list_is_empty(
            filepaths=sliced_paths,
            error_msg=f"SIMBA ERROR: Cannot join videos in directory {in_folder}. The directory contain ZERO files in format {video_format} with substring {substring}",
        )
        files = sliced_paths
    files.sort(key=lambda f: int(re.sub("\D", "", f)))
    temp_txt_path = Path(in_folder, "files.txt")
    if os.path.isfile(temp_txt_path):
        os.remove(temp_txt_path)
    with open(temp_txt_path, "w") as f:
        for file in files:
            f.write("file '" + str(Path(file)) + "'\n")

    if os.path.exists(save_path):
        os.remove(save_path)
    if check_nvidea_gpu_available() and gpu:
        returned = os.system(f'ffmpeg -hwaccel auto -c:v h264_cuvid -f concat -safe 0 -i "{temp_txt_path}" -c copy -hide_banner -loglevel info "{save_path}"')
    else:
        returned = os.system(f'ffmpeg -f concat -safe 0 -i "{temp_txt_path}" "{save_path}" -c copy -hide_banner -loglevel info')
    while True:
        if returned != 0:
            pass
        else:
            if remove_splits:
                remove_a_folder(folder_dir=Path(in_folder))
            break
    timer.stop_timer()
    stdout_success(
        msg="Video concatenated",
        elapsed_time=timer.elapsed_time_str,
        source=concatenate_videos_in_folder.__name__,
    )


def get_bp_headers(body_parts_lst: List[str]) -> list:
    """
    Helper to create ordered list of all column header fields from body-part names for SimBA project dataframes.

    :parameter List[str] body_parts_lst: Body-part names in the SimBA prject
    :return List[str]: Body-part headers

    :examaple:
    >>> get_bp_headers(body_parts_lst=['Nose'])
    >>> ['Nose_x', 'Nose_y', 'Nose_p']
    """

    bp_headers = []
    for bp in body_parts_lst:
        c1, c2, c3 = (f"{bp}_x", f"{bp}_y", f"{bp}_p")
        bp_headers.extend((c1, c2, c3))
    return bp_headers


def read_video_info(vid_info_df: pd.DataFrame,
                    video_name: str,
                    raise_error: Optional[bool] = True) -> Tuple[pd.DataFrame, float, float]:
    """
    Helper to read the metadata (pixels per mm, resolution, fps etc) from the video_info.csv for a single input file/video

    :parameter pd.DataFrame vid_info_df: Parsed ``project_folder/logs/video_info.csv`` file. This file can be parsed by :meth:`simba.utils.read_write.read_video_info_csv`.
    :parameter str video_name: Name of the video as represented in the ``Video`` column of the ``project_folder/logs/video_info.csv`` file.
    :parameter Optional[bool] raise_error: If True, raises error if the video cannot be found in the ``vid_info_df`` file. If False, returns None if the video cannot be found.
    :returns Tuple[pd.DataFrame, float, float]: One row DataFrame representing the video in the ``project_folder/logs/video_info.csv`` file, the frame rate of the video, and the the pixels per millimeter of the video

    :example:
    >>> video_info_df = read_video_info_csv(file_path='project_folder/logs/video_info.csv')
    >>> read_video_info(vid_info_df=vid_info_df, video_name='Together_1')
    """

    video_settings = vid_info_df.loc[vid_info_df["Video"] == video_name]
    if len(video_settings) > 1:
        raise DuplicationError(msg=f"SimBA found multiple rows in the project_folder/logs/video_info.csv named {str(video_name)}. Please make sure that each video name is represented ONCE in the video_info.csv", source=read_video_info.__name__)
    elif len(video_settings) < 1:
        if raise_error:
            raise ParametersFileError(msg=f" SimBA could not find {str(video_name)} in the video_info.csv file. Make sure all videos analyzed are represented in the project_folder/logs/video_info.csv file.", source=read_video_info.__name__)
        else:
            return None
    else:
        try:
            px_per_mm = float(video_settings["pixels/mm"])
            fps = float(video_settings["fps"])
            return video_settings, px_per_mm, fps
        except TypeError:
            raise ParametersFileError(msg=f"Make sure the videos that are going to be analyzed are represented with APPROPRIATE VALUES inside the project_folder/logs/video_info.csv file in your SimBA project. Could not interpret the fps, pixels per millimeter and/or fps as numerical values for video {video_name}", source=read_video_info.__name__)


def find_all_videos_in_directory(
    directory: Union[str, os.PathLike],
    as_dict: Optional[bool] = False,
    raise_error: bool = False,
    video_formats: Optional[Tuple[str]] = (".avi", ".mp4", ".mov", ".flv", ".m4v", '.webm'),
) -> Union[dict, list]:
    """
    Get all video file paths within a directory

    :param str directory: Directory to search for video files.
    :param bool as_dict: If True, returns dictionary with the video name as key and file path as value.
    :param bool raise_error: If True, raise error if no videos are found. Else, NoFileFoundWarning.
    :param Tuple[str] video_formats: Acceptable video formats. Default: '.avi', '.mp4', '.mov', '.flv', '.m4v'.
    :return List[str] or Dict[str, str]
    :raises NoFilesFoundError: If ``raise_error`` and ``directory`` has no files in formats ``video_formats``.

    :examples:
    >>> find_all_videos_in_directory(directory='project_folder/videos')
    """

    video_lst = []
    for i in os.listdir(directory):
        if i.lower().endswith(video_formats):
            video_lst.append(i)
    if not video_lst:
        if raise_error:
            raise NoFilesFoundError(
                f"No videos found in directory {directory} in formats {video_formats}."
            )
        else:
            video_lst.append("No videos found")
            NoFileFoundWarning(
                msg=f"No videos found in directory ({directory})",
                source=find_all_videos_in_directory.__name__,
            )

    if video_lst and as_dict:
        video_dict = {}
        for video_name in video_lst:
            video_path = os.path.join(directory, video_name)
            _, name, _ = get_fn_ext(filepath=video_path)
            video_dict[name] = video_path
        return video_dict

    return video_lst


def read_frm_of_video(video_path: Union[str, os.PathLike, cv2.VideoCapture],
                      frame_index: int = 0,
                      opacity: Optional[float] = None,
                      size: Optional[Tuple[int, int]] = None,
                      greyscale: Optional[bool] = False,
                      bw: Optional[bool] = False,
                      clahe: Optional[bool] = False) -> np.ndarray:

    """
    Reads single image from video file.

    :param Union[str, os.PathLike] video_path: Path to video file, or cv2.VideoCapture object.
    :param int frame_index: The frame of video to return. Default: 1.
    :param Optional[int] opacity: Value between 0 and 100 or None. If float, returns image with opacity. 100 fully opaque. 0.0 fully transparant.
    :param Optional[Tuple[int, int]] size: If tuple, resizes the image to size. Else, returns original image size.
    :param Optional[bool] greyscale: If true, returns the greyscale image. Default False.
    :param Optional[bool] clahe: If true, returns clahe enhanced image. Default False.
    :return np.ndarray: Image as numpy array.

    :example:
    >>> img = read_frm_of_video(video_path='/Users/simon/Desktop/envs/platea_featurizer/data/video/3D_Mouse_5-choice_MouseTouchBasic_s9_a4_grayscale.mp4', clahe=True)
    >>> cv2.imshow('img', img)
    >>> cv2.waitKey(5000)
    """

    check_instance(
        source=read_frm_of_video.__name__,
        instance=video_path,
        accepted_types=(str, cv2.VideoCapture),
    )
    if type(video_path) == str:
        check_file_exist_and_readable(file_path=video_path)
        video_meta_data = get_video_meta_data(video_path=video_path)
    else:
        video_meta_data = {"frame_count": int(video_path.get(cv2.CAP_PROP_FRAME_COUNT))}
    if (frame_index > video_meta_data["frame_count"]) or (frame_index < 0):
        raise FrameRangeError(
            msg=f'Frame {frame_index} is out of range: The video {video_path} contains {video_meta_data["frame_count"]} frames.',
            source=read_frm_of_video.__name__,
        )
    if type(video_path) == str:
        capture = cv2.VideoCapture(video_path)
    else:
        capture = video_path
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, img = capture.read()
    if ret:
        if opacity:
            opacity = float(opacity / 100)
            check_float(
                name="Opacity",
                value=opacity,
                min_value=0.00,
                max_value=1.00,
                raise_error=True,
            )
            opacity = 1 - opacity
            h, w, clr = img.shape[:3]
            opacity_image = np.ones((h, w, clr), dtype=np.uint8) * int(255 * opacity)
            img = cv2.addWeighted(
                img.astype(np.uint8),
                1 - opacity,
                opacity_image.astype(np.uint8),
                opacity,
                0,
            )
        if size:
            img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        if greyscale:
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if clahe:
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16)).apply(img)

    else:
        NoDataFoundWarning(
            msg=f"Frame {frame_index} for video {video_path} could not be read."
        )
    return img


def find_video_of_file(video_dir: Union[str, os.PathLike],
                       filename: str,
                       raise_error: Optional[bool] = False,
                       warning: Optional[bool] = True) -> Union[str, os.PathLike]:
    """
    Helper to find the video file with the SimBA project that represents a known data file path.

    :param str video_dir: Directory holding putative video file.
    :param str filename: Data file name, e.g., ``Video_1``.
    :param Optional[bool] raise_error: If True, raise error if no file can be found. If False, returns None if no file can be found. Default: False
    :param Optional[bool] warning: If True, print warning if no file can be found. If False, no warning is printed if file cannot be found. Default: False
    :return str: Video path.

    :examples:
    >>> find_video_of_file(video_dir='project_folder/videos', filename='Together_1')
    >>> 'project_folder/videos/Together_1.avi'

    """
    try:
        all_files_in_video_folder = [f for f in next(os.walk(video_dir))[2] if not f[0] == "."]
    except StopIteration:
        if raise_error:
            raise NoFilesFoundError(msg=f"No files found in the {video_dir} directory", source=find_video_of_file.__name__)
        elif warning:
            NoFileFoundWarning(msg=f"SimBA could not find a video file representing {filename} in the project video directory {video_dir}", source=find_video_of_file.__name__)
        return None

    all_files_in_video_folder = [os.path.join(video_dir, x) for x in all_files_in_video_folder]
    return_path = None
    for file_path in all_files_in_video_folder:
        _, video_filename, ext = get_fn_ext(file_path)
        if (video_filename == filename) and (ext.lower() in Options.ALL_VIDEO_FORMAT_OPTIONS.value):
            return_path = file_path

    if return_path is None and raise_error:
        raise NoFilesFoundError(msg=f"SimBA could not find a video file representing {filename} in the project video directory {video_dir}", source=find_video_of_file.__name__)
    elif return_path is None and warning:
        NoFileFoundWarning(msg=f"SimBA could not find a video file representing {filename} in the project video directory {video_dir}", source=find_video_of_file.__name__)
    return return_path


def find_files_of_filetypes_in_directory(directory: str,
                                         extensions: list,
                                         raise_warning: Optional[bool] = True,
                                         raise_error: Optional[bool] = False) -> List[str]:
    """
    Find all files in a directory of specified extensions/types.

    :param str directory: Directory holding files.
    :param List[str] extensions: Accepted file extensions.
    :param bool raise_warning: If True, raise error if no files are found.

    :return List[str]: All files in ``directory`` with extensions.

    :example:
    >>> find_files_of_filetypes_in_directory(directory='project_folder/videos', extensions=['mp4', 'avi', 'png'], raise_warning=False)
    """

    try:
        all_files_in_folder = [
            f for f in next(os.walk(directory))[2] if not f[0] == "."
        ]
    except StopIteration:
        if raise_warning:
            raise NoFilesFoundError(
                msg=f"No files found in the {directory} directory with accepted extensions {str(extensions)}",
                source=find_files_of_filetypes_in_directory.__name__,
            )
        else:
            all_files_in_folder = []
            pass
    all_files_in_folder = [os.path.join(directory, x) for x in all_files_in_folder]
    accepted_file_paths = []
    for file_path in all_files_in_folder:
        _, file_name, ext = get_fn_ext(file_path)
        if ext.lower() in extensions:
            accepted_file_paths.append(file_path)
    if not accepted_file_paths and raise_warning:
        NoFileFoundWarning(
            msg=f"SimBA could not find any files with accepted extensions {extensions} in the {directory} directory",
            source=find_files_of_filetypes_in_directory.__name__,
        )
    if not accepted_file_paths and raise_error:
        raise NoDataError(
            msg=f"SimBA could not find any files with accepted extensions {extensions} in the {directory} directory",
            source=find_files_of_filetypes_in_directory.__name__,
        )
    return accepted_file_paths


def convert_parquet_to_csv(directory: str) -> None:
    """
    Convert all parquet files in a directory to csv format.

    :param str directory: Path to directory holding parquet files
    :raise NoFilesFoundError: The directory has no ``parquet`` files.

    :examples:
    >>> convert_parquet_to_csv(directory='project_folder/csv/input_csv')
    """

    if not os.path.isdir(directory):
        raise NotDirectoryError(
            msg="SIMBA ERROR: {} is not a valid directory".format(directory),
            source=convert_parquet_to_csv.__name__,
        )
    files_found = glob.glob(directory + "/*.parquet")
    if len(files_found) < 1:
        raise NoFilesFoundError(
            "SIMBA ERROR: No parquet files (with .parquet file ending) found in the {} directory".format(
                directory
            ),
            source=convert_parquet_to_csv.__name__,
        )
    for file_cnt, file_path in enumerate(files_found):
        print("Reading in {} ...".format(os.path.basename(file_path)))
        df = pd.read_parquet(file_path)
        new_file_path = os.path.join(
            directory, os.path.basename(file_path).replace(".parquet", ".csv")
        )
        if "scorer" in df.columns:
            df = df.set_index("scorer")
        df.to_csv(new_file_path)
        print("Saved {}...".format(new_file_path))
    stdout_success(
        msg=f"{str(len(files_found))} parquet files in {directory} converted to csv",
        source=convert_parquet_to_csv.__name__,
    )


def convert_csv_to_parquet(directory: Union[str, os.PathLike]) -> None:
    """
    Convert all csv files in a folder to parquet format.

    :param str directory: Path to directory holding csv files.
    :raise NoFilesFoundError: The directory has no ``csv`` files.

    :examples:
    >>> convert_parquet_to_csv(directory='project_folder/csv/input_csv')
    """
    if not os.path.isdir(directory):
        raise NotDirectoryError(
            msg="SIMBA ERROR: {} is not a valid directory".format(directory),
            source=convert_csv_to_parquet.__name__,
        )
    files_found = glob.glob(directory + "/*.csv")
    if len(files_found) < 1:
        raise NoFilesFoundError(
            msg="SIMBA ERROR: No parquet files (with .csv file ending) found in the {} directory".format(
                directory
            ),
            source=convert_csv_to_parquet.__name__,
        )
    print("Converting {} files...".format(str(len(files_found))))
    for file_cnt, file_path in enumerate(files_found):
        print("Reading in {} ...".format(os.path.basename(file_path)))
        df = pd.read_csv(file_path)
        new_file_path = os.path.join(
            directory, os.path.basename(file_path).replace(".csv", ".parquet")
        )
        df.to_parquet(new_file_path)
        print("Saved {}...".format(new_file_path))
    stdout_success(
        msg=f"{str(len(files_found))} csv files in {directory} converted to parquet",
        source=convert_csv_to_parquet.__name__,
    )


def get_file_name_info_in_directory(
    directory: Union[str, os.PathLike], file_type: str
) -> Dict[str, str]:
    """
    Get dict of all file paths in a directory with specified extension as values and file base names as keys.

    :param str directory: Directory containing files.
    :param str file_type: File-type in ``directory`` of interest
    :return dict: All found files as values and file base names as keys.

    :example:
    >>> get_file_name_info_in_directory(directory='C:\project_folder\csv\machine_results', file_type='csv')
    >>> {'Video_1': 'C:\project_folder\csv\machine_results\Video_1'}
    """

    results = {}
    file_paths = glob.glob(directory + "/*." + file_type)
    for file_path in file_paths:
        _, file_name, ext = get_fn_ext(file_path)
        results[file_name] = file_path

    return results


def archive_processed_files(
    config_path: Union[str, os.PathLike], archive_name: str
) -> None:
    """
    Archive files within a SimBA project.

    :param str config_path: Path to SimBA project ``project_config.ini``.
    :param str archive_name: Name of archive.

    .. seealso::
       `Tutorial <https://github.com/sgoldenlab/simba/blob/master/docs/Scenario4_new.md>`_

    :example:
    >>> archive_processed_files(config_path='project_folder/project_config.ini', archive_name='my_archive')
    """

    config = read_config_file(config_path=config_path)
    file_type = read_config_entry(
        config,
        ConfigKey.GENERAL_SETTINGS.value,
        ConfigKey.FILE_TYPE.value,
        "str",
        "csv",
    )
    project_path = read_config_entry(
        config,
        ConfigKey.GENERAL_SETTINGS.value,
        ConfigKey.PROJECT_PATH.value,
        data_type=ConfigKey.FOLDER_PATH.value,
    )
    videos_dir = os.path.join(project_path, "videos")
    csv_dir = os.path.join(os.path.dirname(config_path), "csv")
    log_path = os.path.join(project_path, "logs")
    video_info_path = os.path.join(log_path, "video_info.csv")
    csv_subdirs, file_lst = [], []
    for content_name in os.listdir(csv_dir):
        if os.path.isdir(os.path.join(csv_dir, content_name)):
            csv_subdirs.append(os.path.join(csv_dir, content_name))

    for subdirectory in csv_subdirs:
        subdirectory_files = [
            x for x in glob.glob(subdirectory + "/*") if os.path.isfile(x)
        ]
        for file_path in subdirectory_files:
            directory, file_name, ext = get_fn_ext(
                os.path.join(subdirectory, file_path)
            )
            if ext == ".{}".format(file_type):
                file_lst.append(os.path.join(subdirectory, file_path))

    if len(file_lst) < 1:
        raise NoFilesFoundError(
            msg="SIMBA ERROR: No data files located in your project_folder/csv sub-directories in the worflow file format {}".format(
                file_type
            ),
            source=archive_processed_files.__name__,
        )

    for file_path in file_lst:
        file_folder = os.path.dirname(file_path)
        save_directory = os.path.join(file_folder, archive_name)
        save_file_path = os.path.join(save_directory, os.path.basename(file_path))
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
        print("Moving file {}...".format(file_path))
        shutil.move(file_path, save_file_path)

    log_archive_path = os.path.join(log_path, archive_name)
    if not os.path.exists(log_archive_path):
        os.mkdir(log_archive_path)
    if os.path.isfile(video_info_path):
        save_file_path = os.path.join(log_archive_path, "video_info.csv")
        print("Moving file {}...".format(video_info_path))
        shutil.move(video_info_path, save_file_path)

    videos_file_paths = [f for f in glob.glob(videos_dir) if os.path.isfile(f)]
    video_archive_path = os.path.join(videos_dir, archive_name)
    if not os.path.exists(video_archive_path):
        os.mkdir(video_archive_path)
    for video_file in videos_file_paths:
        save_video_path = os.path.join(video_archive_path, os.path.basename(video_file))
        shutil.move(video_file, save_video_path)
    stdout_success(msg="Archiving completed", source=archive_processed_files.__name__)


def str_2_bool(input_str: str) -> bool:
    """
    Helper to convert string representation of bool to bool.

    :example:
    >>> str_2_bool(input_str='yes')
    >>> True
    """
    return input_str.lower() in ("yes", "true", "1")


def tabulate_clf_info(clf_path: Union[str, os.PathLike]) -> None:
    """
    Print the hyperparameters and creation date of a pickled classifier.

    :param str clf_path: Path to classifier
    :raise InvalidFilepathError: The file is not a pickle or not a scikit-learn RF classifier.
    """

    _, clf_name, _ = get_fn_ext(clf_path)
    check_file_exist_and_readable(file_path=clf_path)
    try:
        clf_obj = pickle.load(open(clf_path, "rb"))
    except:
        raise InvalidFilepathError(
            msg=f"The {clf_path} file is not a pickle file",
            source=tabulate_clf_info.__name__,
        )
    try:
        clf_features_no = clf_obj.n_features_
        clf_criterion = clf_obj.criterion
        clf_estimators = clf_obj.n_estimators
        clf_min_samples_leaf = clf_obj.min_samples_split
        clf_n_jobs = clf_obj.n_jobs
        clf_verbose = clf_obj.verbose
        if clf_verbose == 1:
            clf_verbose = True
        if clf_verbose == 0:
            clf_verbose = False
    except:
        raise InvalidFilepathError(
            msg=f"The {clf_path} file is not an scikit-learn RF classifier",
            source=tabulate_clf_info.__name__,
        )
    creation_time = "Unknown"
    try:
        if platform.system() == "Windows":
            creation_time = os.path.getctime(clf_path)
        elif platform.system() == "Darwin":
            creation_time = os.stat(clf_path)
            creation_time = creation_time.st_birthtime
    except AttributeError:
        pass
    if creation_time != "Unknown":
        creation_time = str(
            datetime.utcfromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")
        )

    print(str(clf_name), "CLASSIFIER INFORMATION")
    for name, val in zip(
        [
            "NUMBER OF FEATURES",
            "NUMBER OF TREES",
            "CLASSIFIER CRITERION",
            "CLASSIFIER_MIN_SAMPLE_LEAF",
            "CLASSIFIER_N_JOBS",
            "CLASSIFIER VERBOSE SETTING",
            "CLASSIFIER PATH",
            "CLASSIFIER CREATION TIME",
        ],
        [
            clf_features_no,
            clf_estimators,
            clf_criterion,
            clf_min_samples_leaf,
            clf_n_jobs,
            clf_verbose,
            clf_path,
            str(creation_time),
        ],
    ):
        print(name + ": " + str(val))


def get_all_clf_names(config: configparser.ConfigParser, target_cnt: int) -> List[str]:
    """
    Get all classifier names in a SimBA project.

    :param configparser.ConfigParser config: Parsed SimBA project_config.ini
    :param int target_cnt: Count of models in SimBA project
    :return List[str]: Classifier model names

    :example:
    >>> get_all_clf_names(config=config, target_cnt=2)
    >>> ['Attack', 'Sniffing']
    """

    model_names = []
    for i in range(target_cnt):
        entry_name = f"target_name_{i + 1}"
        model_names.append(read_config_entry(config, ConfigKey.SML_SETTINGS.value, entry_name, data_type=Dtypes.STR.value))
    return model_names


def read_meta_file(meta_file_path: Union[str, os.PathLike]) -> dict:
    """
    Read in single SimBA modelconfig meta file CSV to python dictionary.

    :param str meta_file_path: Path to SimBA config meta file
    :return dict: Dictionary holding model parameters.

    :example:
    >>> read_meta_file('project_folder/configs/Attack_meta_0.csv')
    >>> {'Classifier_name': 'Attack', 'RF_n_estimators': 2000, 'RF_max_features': 'sqrt', 'RF_criterion': 'gini', ...}
    """
    check_file_exist_and_readable(file_path=meta_file_path)
    return pd.read_csv(meta_file_path, index_col=False).to_dict(orient="records")[0]


def read_simba_meta_files(folder_path: str, raise_error: bool = False) -> List[str]:
    """
    Read in paths of SimBA model config files directory (`project_folder/configs'). Consider files that have `meta` suffix only.

    :param str folder_path: directory with SimBA model config meta files
    :param bool raise_error: If True, raise error if no files are found with ``meta`` suffix. Else, print warning. Default: False.
    :return List[str]: List of paths to  SimBA model config meta files.

    :example:
    >>> read_simba_meta_files(folder_path='/project_folder/configs')
    >>> ['project_folder/configs/Attack_meta_1.csv', 'project_folder/configs/Attack_meta_0.csv']
    """

    file_paths = find_files_of_filetypes_in_directory(
        directory=folder_path, extensions=[".csv"]
    )
    meta_file_lst = []
    for i in file_paths:
        if i.__contains__("meta"):
            meta_file_lst.append(os.path.join(folder_path, i))
    if len(meta_file_lst) == 0 and not raise_error:
        NoFileFoundWarning(
            msg=f'The training meta-files folder in your project ({folder_path}) does not have any meta files inside it (no files in this folder has the "meta" substring in the filename)',
            source=read_simba_meta_files.__name__,
        )
    elif len(meta_file_lst) == 0 and raise_error:
        raise NoFilesFoundError(
            msg=f'The training meta-files folder in your project ({folder_path}) does not have any meta files inside it (no files in this folder has the "meta" substring in the filename)',
            source=read_simba_meta_files.__name__,
        )

    return meta_file_lst


def find_core_cnt() -> Tuple[int, int]:
    """
    Find the local cpu count and quarter of the cpu counts.

    :return int: The local cpu count
    :return int: The local cpu count // 4

    :example:
    >>> find_core_cnt()
    >>> (8, 2)

    """
    cpu_cnt = multiprocessing.cpu_count()
    cpu_cnt_to_use = int(cpu_cnt / 4)
    if cpu_cnt_to_use < 1:
        cpu_cnt_to_use = 1
    return cpu_cnt, cpu_cnt_to_use


def get_number_of_header_columns_in_df(df: pd.DataFrame) -> int:
    """
    Returns the count of non-numerical header rows in dataframe. E.g., can be helpful to determine if dataframe is multi-index columns.

    :param pd.DataFrame df: Dataframe to check the count of non-numerical header rows for.

    :example:
    >>> get_number_of_header_columns_in_df(df='project_folder/csv/input_csv/Video_1.csv')
    >>> 3
    """
    for i in range(len(df)):
        try:
            temp = df.iloc[i:].apply(pd.to_numeric).reset_index(drop=True)
            return i
        except ValueError:
            pass
    raise DataHeaderError(
        msg="Could find the count of header columns in dataframe, all rows appear non-numeric",
        source=get_number_of_header_columns_in_df.__name__,
    )


def get_memory_usage_of_df(df: pd.DataFrame) -> Dict[str, float]:
    """
    Get the RAM memory usage of a dataframe.

    :param pd.DataFrame df: Parsed dataframe
    :return dict: The memory usage of the dataframe in bytes, mb, and gb.

    :example:
    >>> df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    >>> {'bytes': 3328, 'megabytes': 0.003328, 'gigabytes': 3e-06}
    """

    results = {}
    results["bytes"] = df.memory_usage(index=True).sum()
    results["megabytes"] = round(results["bytes"] / 1000000, 6)
    results["gigabytes"] = round(results["bytes"] / 1000000000, 6)
    return results


def copy_single_video_to_project(
    simba_ini_path: Union[str, os.PathLike],
    source_path: Union[str, os.PathLike],
    symlink: bool = False,
    allowed_video_formats: Optional[Tuple[str]] = ("avi", "mp4"),
    overwrite: Optional[bool] = False,
) -> None:
    """
    Import single video file to SimBA project

    :param Union[str, os.PathLike] simba_ini_path: path to SimBA project config file in Configparser format
    :param Union[str, os.PathLike] source_path: Path to video file outside SimBA project.
    :param Optional[bool] symlink: If True, creates soft copy rather than hard copy. Default: False.
    :param Optional[Tuple[str]] allowed_video_formats: Allowed video formats. DEFAULT: avi or mp4
    :param Optional[bool] overwrite: If True, overwrites existing video if it exists in SimBA project. Else, raise FileExistError.
    """

    timer = SimbaTimer(start=True)
    _, file_name, file_ext = get_fn_ext(source_path)
    check_file_exist_and_readable(file_path=source_path)
    print("Copying video {} file...".format(file_name))
    if file_ext[1:].lower().strip() not in allowed_video_formats:
        raise InvalidFileTypeError(
            msg="SimBA works best with avi and mp4 video-files. Or please convert your videos to mp4 or avi to continue before importing it.",
            source=copy_single_video_to_project.__name__,
        )
    new_filename = os.path.join(file_name + file_ext)
    destination = os.path.join(os.path.dirname(simba_ini_path), "videos", new_filename)
    if os.path.isfile(destination) and not overwrite:
        raise FileExistError(
            msg=f"{file_name} already exist in SimBA project. To import, delete this video file before importing the new video file with the same name.",
            source=copy_single_video_to_project.__name__,
        )
    else:
        if not symlink:
            shutil.copy(source_path, destination)
        else:
            try:
                if os.path.isfile(destination):
                    os.remove(destination)
                os.symlink(source_path, destination)
            except OSError as e:
                raise PermissionError(
                    msg="Symbolic link privilege not held. Try running SimBA in terminal opened in admin mode",
                    source=copy_single_video_to_project.__name__,
                )
        timer.stop_timer()
        if not symlink:
            stdout_success(
                msg=f"Video {file_name} imported to SimBA project (project_folder/videos directory",
                elapsed_time=timer.elapsed_time_str,
                source=copy_single_video_to_project.__name__,
            )
        else:
            stdout_success(
                msg=f"Video {file_name}  SYMLINK imported to SimBA project (project_folder/videos directory",
                elapsed_time=timer.elapsed_time_str,
                source=copy_single_video_to_project.__name__,
            )


def copy_multiple_videos_to_project(
    config_path: Union[str, os.PathLike],
    source: Union[str, os.PathLike],
    file_type: str,
    symlink: Optional[bool] = False,
    allowed_video_formats: Optional[Tuple[str]] = ("avi", "mp4"),
) -> None:
    """
    Import directory of videos to SimBA project.

    :param Union[str, os.PathLike] simba_ini_path: path to SimBA project config file in Configparser format
    :param Union[str, os.PathLike] source_path: Path to directory with video files outside SimBA project.
    :param str file_type: Video format of imported videos (i.e.,: mp4 or avi)
    :param Optional[bool] symlink: If True, creates soft copies rather than hard copies. Default: False.
    :param Optional[Tuple[str]] allowed_video_formats: Allowed video formats. DEFAULT: avi or mp4
    """

    multiple_video_timer = SimbaTimer(start=True)
    if file_type.lower().strip() not in allowed_video_formats:
        raise InvalidFileTypeError(
            msg="SimBA only works with avi and mp4 video files (Please enter mp4 or avi in entrybox). Or convert your videos to mp4 or avi to continue.",
            source=copy_multiple_videos_to_project.__name__,
        )
    video_path_lst = find_all_videos_in_directory(
        directory=source, video_formats=(file_type), raise_error=True
    )
    video_path_lst = [os.path.join(source, x) for x in video_path_lst]
    if len(video_path_lst) == 0:
        raise NoFilesFoundError(
            msg=f"SIMBA ERROR: No videos found in {source} directory of file-type {file_type}",
            source=copy_multiple_videos_to_project.__name__,
        )
    destination_dir = os.path.join(os.path.dirname(config_path), "videos")
    for file_cnt, file_path in enumerate(video_path_lst):
        timer = SimbaTimer(start=True)
        dir_name, filebasename, file_extension = get_fn_ext(file_path)
        file_extension = file_extension.lower()
        newFileName = os.path.join(filebasename + file_extension)
        dest1 = os.path.join(destination_dir, newFileName)
        if os.path.isfile(dest1):
            FileExistWarning(
                msg=f"{filebasename} already exist in SimBA project. Skipping video...",
                source=copy_multiple_videos_to_project.__name__,
            )
        else:
            if not symlink:
                shutil.copy(file_path, dest1)
            else:
                try:
                    os.symlink(file_path, dest1)
                except OSError:
                    raise PermissionError(msg="Symbolic link privilege not held. Try running SimBA in terminal opened in admin mode")
            timer.stop_timer()
            if not symlink:
                print(
                    f"{filebasename} copied to project (Video {file_cnt + 1}/{len(video_path_lst)}, elapsed timer {timer.elapsed_time_str}s)..."
                )
            else:
                print(
                    f"{filebasename} copied to project (SYMLINK) (Video {file_cnt + 1}/{len(video_path_lst)}, elapsed timer {timer.elapsed_time_str}s)..."
                )

    multiple_video_timer.stop_timer()
    stdout_success(
        msg=f"{len(video_path_lst)} videos copied to project.",
        elapsed_time=multiple_video_timer.elapsed_time_str,
        source=copy_multiple_videos_to_project.__name__,
    )


def find_all_videos_in_project(
    videos_dir: Union[str, os.PathLike], basename: Optional[bool] = False
) -> List[str]:
    """
    Get filenames of .avi and .mp4 files within a directory

    :param str videos_dir: Directory holding video files.
    :param bool basename: If true returns basenames, else file paths.

    :example:
    >>> find_all_videos_in_project(videos_dir='project_folder/videos')
    >>> ['project_folder/videos/Together_2.avi', 'project_folder/videos/Together_3.avi', 'project_folder/videos/Together_1.avi']

    """
    video_paths = []
    file_paths_in_folder = [f for f in next(os.walk(videos_dir))[2] if not f[0] == "."]
    file_paths_in_folder = [os.path.join(videos_dir, f) for f in file_paths_in_folder]
    for file_cnt, file_path in enumerate(file_paths_in_folder):
        try:
            _, file_name, file_ext = get_fn_ext(file_path)
        except ValueError:
            raise InvalidFilepathError(
                msg=f"{file_path} is not a valid filepath",
                source=find_all_videos_in_project.__name__,
            )
        if (file_ext.lower() == ".mp4") or (file_ext.lower() == ".avi"):
            if not basename:
                video_paths.append(file_path)
            else:
                video_paths.append(file_name)
    if len(video_paths) == 0:
        raise NoFilesFoundError(
            msg=f"No videos in mp4 or avi format found imported to SimBA project in the {videos_dir} directory",
            source=find_all_videos_in_project.__name__,
        )
    else:
        return video_paths


def check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp: str, video_path: Union[str, os.PathLike]) -> None:
    """
    Helper to check that a timestamp in HH:MM:SS format is a valid timestamp in a video file.

    :param str timestamp: Timestamp in HH:MM:SS format.
    :param str video_path: Path to a video file.
    :raises FrameRangeError: If timestamp is not in the video file. E.g., timestamp 00:01:00 will raise FrameRangeError if the video is 59s long.

    :example:
    >>> check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp='01:00:05', video_path='/Users/simon/Desktop/video_tests/Together_1.avi')
    >>> "FrameRangeError: The timestamp '01:00:05' does not occur in video Together_1.avi, the video has length 10s"
    """

    check_file_exist_and_readable(file_path=video_path)
    check_if_string_value_is_valid_video_timestamp(value=timestamp, name="Timestamp")
    video_meta_data = get_video_meta_data(video_path=video_path)
    h, m, s = timestamp.split(":")
    time_stamp_in_seconds = int(h) * 3600 + int(m) * 60 + int(s)
    if not video_meta_data["video_length_s"] >= time_stamp_in_seconds:
        video_length_str = timedelta(seconds=video_meta_data["video_length_s"])
        raise FrameRangeError(
            msg=f'The timestamp {timestamp} does not occur in video {video_meta_data["video_name"]}, the video has length {video_length_str}',
            source=check_if_hhmmss_timestamp_is_valid_part_of_video.__name__,
        )

def timestamp_to_seconds(timestamp: str) -> int:
    """
    Returns the number of seconds into the video given a timestamp in HH:MM:SS format.
    :raises FrameRangeError: If timestamp is not a valid format.

    :example:
    >>> timestamp_to_seconds(timestamp='00:00:05')
    >>> 5
    """

    check_if_string_value_is_valid_video_timestamp(value=timestamp, name="Timestamp")
    h, m, s = timestamp.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)




def find_time_stamp_from_frame_numbers(
    start_frame: int, end_frame: int, fps: float
) -> List[str]:
    """
    Given start and end frame numbers and frames per second (fps), return a list of formatted time stamps
    corresponding to the frame range start and end time.

    :param int start_frame: The starting frame index.
    :param int end_frame: The ending frame index.
    :param float fps: Frames per second.
    :return List[str]: A list of time stamps in the format 'HH:MM:SS:MS'.

    :example:
    >>> find_time_stamp_from_frame_numbers(start_frame=11, end_frame=20, fps=3.4)
    >>> ['00:00:03:235', '00:00:05:882']
    """

    def get_time(frame_index, fps):
        total_seconds = frame_index / fps
        milliseconds = int((total_seconds % 1) * 1000)
        total_seconds = int(total_seconds)
        seconds = total_seconds % 60
        total_seconds //= 60
        minutes = total_seconds % 60
        hours = total_seconds // 60
        return "{:02d}:{:02d}:{:02d}.{:03d}".format(
            hours, minutes, seconds, milliseconds
        )

    check_int(name="start_frame", value=start_frame, min_value=0)
    check_int(name="end_frame", value=end_frame, min_value=0)
    check_float(name="FPS", value=fps, min_value=1)
    if start_frame > end_frame:
        raise FrameRangeError(
            msg=f"Start frame ({start_frame}) cannot be before end frame ({end_frame})"
        )

    return [get_time(start_frame, fps), get_time(end_frame, fps)]


def read_roi_data(
    roi_path: Union[str, os.PathLike]
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    Method to read in ROI definitions from SimBA project
    """
    check_file_exist_and_readable(file_path=roi_path)
    try:
        rectangles_df = pd.read_hdf(roi_path, key=Keys.ROI_RECTANGLES.value).dropna(
            how="any"
        )
        circles_df = pd.read_hdf(roi_path, key=Keys.ROI_CIRCLES.value).dropna(how="any")
        polygon_df = pd.read_hdf(roi_path, key=Keys.ROI_POLYGONS.value)
    except:
        raise InvalidFileTypeError(
            msg=f"{roi_path} is not a valid SimBA ROI definitions file",
            source=read_roi_data.__name__,
        )
    if "Center_XCenter_Y" in polygon_df.columns:
        polygon_df = polygon_df.drop(["Center_XCenter_Y"], axis=1)
    polygon_df = polygon_df.dropna(how="any")

    return rectangles_df, circles_df, polygon_df


def create_directory(path: Union[str, os.PathLike]):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass


def find_max_vertices_coordinates(
    shapes: List[Union[Polygon, LineString, MultiPolygon, Point]],
    buffer: Optional[int] = None,
) -> Tuple[int, int]:
    """
    Find the maximum x and y coordinates among the vertices of a list of Shapely geometries.

    Can be useful for plotting puposes, to dtermine the rquired size of the canvas to fit all geometries.

    :param List[Union[Polygon, LineString, MultiPolygon, Point]] shapes: A list of Shapely geometries including Polygons, LineStrings, MultiPolygons, and Points.
    :param Optional[int] buffer: If int, adds to maximum x and y.
    :returns Tuple[int, int]: A tuple containing the maximum x and y coordinates found among the vertices.

    :example:
    >>> polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    >>> line = LineString([(1, 1), (2, 2), (3, 1), (4, 0)])
    >>> multi_polygon = MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])])
    >>> point = Point(3, 4)
    >>> find_max_vertices_coordinates([polygon, line, multi_polygon, point])
    >>> (4, 4)
    """

    for shape in shapes:
        check_instance(
            source=find_max_vertices_coordinates.__name__,
            instance=shape,
            accepted_types=(Polygon, LineString, MultiPolygon, Point, MultiLineString),
        )

    max_x, max_y = -np.inf, -np.inf
    for shape in shapes:
        if isinstance(shape, Polygon):
            for vertex in shape.exterior.coords:
                max_x, max_y = max(max_x, vertex[0]), max(max_y, vertex[1])

        if isinstance(shape, MultiPolygon):
            for polygon in shape.geoms:
                for vertex in polygon.exterior.coords:
                    max_x, max_y = max(max_x, vertex[0]), max(max_y, vertex[1])

        if isinstance(shape, LineString):
            for vertex in shape.coords:
                max_x, max_y = max(max_x, vertex[0]), max(max_y, vertex[1])

        if isinstance(shape, Point):
            max_x, max_y = max(max_x, shape.coords[0][0]), max(
                max_y, shape.coords[0][1]
            )

        if isinstance(shape, MultiLineString):
            for line in shape.geoms:
                for vertex in line.coords:
                    max_x, max_y = max(max_x, vertex[0]), max(max_y, vertex[1])

    if buffer:
        check_int(name="Buffer", value=buffer, min_value=1)
        max_x += buffer
        max_y += buffer

    return int(max_x), int(max_y)

def clean_sleap_file_name(filename: str) -> str:
    """
    Clean a SLEAP input filename by removing '.analysis' suffix, the video number, and project name prefix, to match orginal video name.

     .. note::
       Modified from `vtsai881 <https://github.com/vtsai881>`_.

    :param str filename: The original filename to be cleaned to match video name.
    :returns str: The cleaned filename.

    :example:
    >>> clean_sleap_file_name("projectname.v00x.00x_videoname.analysis.csv")
    >>> 'videoname.csv'
    >>> clean_sleap_file_name("projectname.v00x.00x_videoname.analysis.h5")
    >>> 'videoname.h5'
    """

    if (".analysis" in filename.lower()) and ("_" in filename) and (filename.count('.') >= 3):
        filename_parts = filename.split('.')
        video_num_name = filename_parts[2]
        if '_' in video_num_name:
            return video_num_name.split('_', 1)[1]
        else:
            return filename
    else:
        return filename


def clean_sleap_filenames_in_directory(dir: Union[str, os.PathLike]) -> None:
    """
    Clean up SLEAP input filenames in the specified directory by removing a prefix
    and a suffix, and renaming the files to match the names of the original video files.

    .. note::
       Modified from `vtsai881 <https://github.com/vtsai881>`_.

    :param Union[str, os.PathLike] dir: The directory path where the SLEAP CSV or H5 files are located.

    :example:
    >>> clean_sleap_filenames_in_directory(dir='/Users/simon/Desktop/envs/troubleshooting/Hornet_SLEAP/import/')
    """

    SLEAP_CSV_SUBSTR = ".analysis"
    check_if_dir_exists(in_dir=dir)
    for file_path in glob.glob(
        dir + f"/*.{Formats.CSV.value}" + f"/*.{Formats.H5.value}"
    ):
        file_name = os.path.basename(p=file_path)
        if (SLEAP_CSV_SUBSTR in file_name) and ("_" in file_name):
            new_name = os.path.join(
                dir,
                file_name.replace(file_name.split("_")[0] + "_", "").replace(
                    SLEAP_CSV_SUBSTR, ""
                ),
            )
            os.rename(file_path, new_name)
        else:
            pass


def copy_files_in_directory(
    in_dir: Union[str, os.PathLike],
    out_dir: Union[str, os.PathLike],
    raise_error: bool = True,
    filetype: Optional[str] = None,
) -> None:
    """
    Copy files from the specified input directory to the output directory.

    :param Union[str, os.PathLike] in_dir: The input directory from which files will be copied.
    :param Union[str, os.PathLike] out_dir: The output directory where files will be copied to.
    :param bool raise_error: If True, raise an error if no files are found in the input directory. Default is True.
    :param Optional[str] filetype: If specified, only copy files with the given file extension. Default is None, meaning all files will be copied.

    :example:
    >>> copy_files_in_directory('/input_dir', '/output_dir', raise_error=True, filetype='txt')
    """

    check_if_dir_exists(in_dir=in_dir)
    if not os.listdir(out_dir):
        os.makedirs(out_dir)
    if filetype is not None:
        file_paths = glob.glob(in_dir + f"/*.{filetype}")
    else:
        file_paths = glob.glob(in_dir + f"/*.")
    if len(file_paths) == 0 and raise_error:
        raise NoFilesFoundError(
            msg=f"No files found in {in_dir}", source=copy_files_in_directory.__name__
        )
    elif len(file_paths) == 0:
        pass
    else:
        for file_path in file_paths:
            shutil.copy(file_path, out_dir)


def remove_files(
    file_paths: List[Union[str, os.PathLike]], raise_error: Optional[bool] = False
) -> None:
    """
    Delete (remove) the files specified within a list of filepaths.

    :param Union[str, os.PathLike] file_paths: A list of file paths to be removed.
    :param Optional[bool] raise_error: If True, raise exceptions for errors during file deletion. Else, pass. Defaults to False.

    :examples:
    >>> file_paths = ['/path/to/file1.txt', '/path/to/file2.txt']
    >>> remove_files(file_paths, raise_error=True)
    """

    for file_path in file_paths:
        if not os.path.isfile(file_path) and raise_error:
            raise NoFilesFoundError(
                msg=f"Cannot delete {file_path}. File does not exist",
                source=remove_files.__name__,
            )
        elif not os.path.isfile(file_path):
            pass
        else:
            try:
                os.remove(file_path)
            except:
                if raise_error:
                    raise PermissionError(
                        msg=f"Cannot read {file_path}. Is the file open in an alternative app?",
                        source=remove_files.__name__,
                    )
                else:
                    pass


def web_callback(url: str) -> None:
    try:
        result = urlparse(url)
        webbrowser.open_new(url)
        # return all([result.scheme, result.netloc])
    except ValueError:
        raise InvalidInputError(msg="Invalid URL: {url}", source=web_callback.__name__)


def get_pkg_version(pkg: str):
    """
    Helper to get the version of a package in the current python environment.

    :example:
    >>> get_pkg_version(pkg='simba-uw-tf-dev')
    >>> 1.82.7
    >>> get_pkg_version(pkg='bla-bla')
    >>> None
    """
    try:
        return pkg_resources.get_distribution(pkg).version
    except pkg_resources.DistributionNotFound:
        return None


def write_pickle(data: Dict[str, Any], save_path: Union[str, os.PathLike]) -> None:
    """
    Write a single object as pickle.

    :param str data_path: Pickled file path.
    :param str save_path: Location of saved pickle.

    :example:
    >>> write_pickle(data=my_model, save_path='/test/unsupervised/cluster_models/My_model.pickle')
    """

    check_if_dir_exists(in_dir=os.path.dirname(save_path))
    try:
        with open(save_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(e.args[0])
        raise InvalidFileTypeError(
            msg="Data could not be saved as a pickle.", source=write_pickle.__name__
        )


def read_pickle(
    data_path: Union[str, os.PathLike], verbose: Optional[bool] = False
) -> dict:
    """
    Read a single or directory of pickled objects. If directory, returns dict with numerical sequential integer keys for
    each object.

    :param str data_path: Pickled file path, or directory of pickled files.
    :param Optional[bool] verbose: If True, prints progress. Default False.
    :returns dict

    :example:
    >>> data = read_pickle(data_path='/test/unsupervised/cluster_models')
    """
    if os.path.isdir(data_path):
        if verbose:
            print(f"Reading in data directory {data_path}...")
        data = {}
        files_found = glob.glob(data_path + f"/*.{Formats.PICKLE.value}")
        if len(files_found) == 0:
            raise NoFilesFoundError(
                msg=f"SIMBA ERROR: Zero pickle files found in {data_path}.",
                source=read_pickle.__name__,
            )
        for file_cnt, file_path in enumerate(files_found):
            if verbose:
                _, file_name, _ = get_fn_ext(filepath=file_path)
                print(f"Reading in data file {file_name}...")
            with open(file_path, "rb") as f:
                try:
                    data[file_cnt] = pickle.load(f)
                except Exception as e:
                    print(e.args)
                    raise InvalidFileTypeError(
                        msg=f"Could not decompress file {file_path} - invalid pickle",
                        source=read_pickle.__name__,
                    )
    elif os.path.isfile(data_path):
        if verbose:
            _, file_name, _ = get_fn_ext(filepath=data_path)
            print(f"Reading in data file {file_name}...")
        with open(data_path, "rb") as f:
            try:
                data = pickle.load(f)
            except Exception as e:
                print(e.args)
                raise InvalidFileTypeError(
                    msg=f"Could not decompress file {data_path} - invalid pickle",
                    source=read_pickle.__name__,
                )
    else:
        raise InvalidFilepathError(
            msg=f"The path {data_path} is neither a valid file or directory path",
            source=read_pickle.__name__,
        )

    return data


def drop_df_fields(
    data: pd.DataFrame, fields: List[str], raise_error: Optional[bool] = False
) -> pd.DataFrame:
    """
    Drops specified fields in dataframe.

    :param pd.DataFrame: Data in pandas format.
    :param  List[str] fields: Columns to drop.
    :return pd.DataFrame
    """

    check_instance(
        source=drop_df_fields.__name__, instance=data, accepted_types=(pd.DataFrame,)
    )
    check_valid_lst(
        data=fields,
        source=drop_df_fields.__name__,
        valid_dtypes=(str,),
        min_len=1,
        raise_error=raise_error,
    )
    if raise_error:
        return data.drop(columns=fields, errors="raise")
    else:
        return data.drop(columns=fields, errors="ignore")


def get_unique_values_in_iterable(
    data: Iterable,
    name: Optional[str] = "",
    min: Optional[int] = 1,
    max: Optional[int] = None,
) -> int:
    """
    Helper to get and check the number of unique variables in iterable. E.g., check the number of unique identified clusters.

    :param np.ndarray data: 1D iterable.
    :param Optional[str] name: Arbitrary name of iterable for informative error messaging.
    :param Optional[int] min: Optional minimum number of unique variables. Default 1.
    :param Optional[int] max: Optional maximum number of unique variables. Default None.
    """
    check_instance(
        source=get_unique_values_in_iterable.__name__,
        instance=data,
        accepted_types=(
            np.ndarray,
            list,
            tuple,
        ),
    )
    check_instance(
        source=get_unique_values_in_iterable.__name__,
        instance=name,
        accepted_types=(str,),
    )

    if not all(
        isinstance(item, (int, float, str, np.int64, np.int32, np.float32, np.float64))
        for item in data
    ):
        dtypes = [type(i) for i in data if i not in (int, float, str)]
        raise InvalidInputError(
            msg=f"Data {name} contains invalid dtypes {dtypes}. Accepted dtypes: int, float, str",
            source=get_unique_values_in_iterable.__name__,
        )
    if isinstance(data, (list, tuple)):
        data = np.array(data)
    cnt = np.unique(data).shape[0]
    if min is not None:
        check_int(name=name, value=min, min_value=1)
        if cnt < min:
            raise IntegerError(
                msg=f"{name} has {cnt} unique observations, but {min} unique observations is required for the operation.",
                source=get_unique_values_in_iterable.__name__,
            )
    if max is not None:
        check_int(name=name, value=max, min_value=1)
        if cnt > max:
            raise IntegerError(
                msg=f"{name} has {cnt} unique observations, but no more than {max} unique observations is allowed for the operation.",
                source=get_unique_values_in_iterable.__name__,
            )
    return cnt


def copy_files_to_directory(file_paths: List[Union[str, os.PathLike]],
                            dir: Union[str, os.PathLike],
                            verbose: Optional[bool] = True,
                            integer_save_names: Optional[bool] = False) -> List[Union[str, os.PathLike]]:
    """
    Copy a list of files to a specified directory.

    :param List[Union[str, os.PathLike]] file_paths: List of paths to the files to be copied.
    :param Union[str, os.PathLike] dir: Path to the directory where files will be copied.
    :param Optional[bool] verbose: If True, prints progress information. Default True.
    :param Optional[bool] integer_save_names: If True, saves files with integer names. E.g., file one in ``file_paths`` will be saved as dir/0.
    :return List[Union[str, os.PathLike]]: List of paths to the copied files
    """

    check_valid_lst(data=file_paths, source=copy_files_to_directory.__name__, min_len=1)
    _ = [check_file_exist_and_readable(x) for x in file_paths]
    check_if_dir_exists(in_dir=dir, source=copy_files_to_directory)
    destinations = []
    for cnt, file_path in enumerate(file_paths):
        if verbose:
            print(
                f"Copying file {os.path.basename(file_path)} ({cnt+1}/{len(file_paths)})..."
            )
        if not integer_save_names:
            destination = os.path.join(dir, os.path.basename(file_path))
        else:
            _, file_name, ext = get_fn_ext(filepath=file_path)
            destination = os.path.join(dir, f"{cnt}{ext}")
        try:
            if os.path.isfile(destination):
                os.remove(destination)
        except Exception as e:
            print(e.args)
            raise PermissionError(
                msg=f"Not allowed to overwrite file {destination}. Try running SimBA in terminal opened in admin mode or delete existing file before copying.",
                source=copy_files_to_directory.__name__,
            )
        destinations.append(destination)
        shutil.copy(file_path, destination)
    return destinations


def seconds_to_timestamp(seconds: int) -> str:
    """
    Convert an integer number representing seconds to a HH:MM:SS format.
    """
    check_int(name=f"{seconds_to_timestamp.__name__} seconds", value=seconds, min_value=0)
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = int(seconds % 60)
    return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


def read_data_paths(path: Union[str, os.PathLike, None],
                    default: List[Union[str, os.PathLike]],
                    default_name: Optional[str] = "",
                    file_type: Optional[str] = "csv") -> List[str]:
    """
    Helper to flexibly read in a set of file-paths.

    :param Union[str, os.PathLike] path: None or path to a file or a folder or list of paths to files.
    :param List[Union[str, os.PathLike]] default: If ``path`` is None. Use this passed list of file paths.
    :param Optional[str] default_name: A readable name representing the ``default`` for interpretable error msgs. Defaults to empty string.
    :param Optional[str] file_type: If path is a directory, read in all files in directory with this file extension. Default: ``csv``.
    :return List[str]: List of file paths.
    """

    if path is None:
        if len(default) == 0:
            raise NoFilesFoundError(msg=f"No files in format found in {default_name}", source=read_data_paths.__name__)
        else:
            for i in default:
                check_file_exist_and_readable(file_path=i)
            data_paths = default
    elif isinstance(path, str):
        if os.path.isfile(path):
            check_file_exist_and_readable(file_path=path)
            data_paths = [path]
        elif os.path.isdir(path):
            data_paths = find_files_of_filetypes_in_directory(directory=path, extensions=[f".{file_type}"], raise_error=True)
            if len(data_paths) == 0:
                raise NoFilesFoundError(msg=f"No files in format {file_type} found in {default_name}", source=read_data_paths.__name__)
        else:
            raise NoFilesFoundError(msg=f"{path} is not a valid path string (it's nether a file or a folder)", source=read_data_paths.__name__)
    elif isinstance(path, (list, tuple)):
        check_valid_lst(data=path, source=f"{read_data_paths.__name__} path", valid_dtypes=(str,), min_len=1)
        data_paths = []
        for i in path:
            check_file_exist_and_readable(file_path=i)
            data_paths.append(i)
    else:
        raise NoFilesFoundError(msg=f"{type(path)} is not a valid type for path", source=read_data_paths.__name__)
    return data_paths
