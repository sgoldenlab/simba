__author__ = "Simon Nilsson"

import base64
import configparser
import functools
import glob
import io
import itertools
import json
import math
import multiprocessing
import os
import pickle
import platform
import re
import shutil
import subprocess
import webbrowser
from ast import literal_eval
from configparser import ConfigParser
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import h5py
from PIL import Image

try:
    from typing import Literal
except:
    from typing_extensions import Literal

from urllib import request
from urllib.parse import urlparse

import cv2
import numpy as np
import pandas as pd
import pkg_resources
import pyarrow as pa
from numba import njit, prange
from pyarrow import csv
from shapely.geometry import (LineString, MultiLineString, MultiPolygon, Point,
                              Polygon)

import simba
from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_filepath_list_is_empty,
                                check_if_keys_exist_in_dict,
                                check_if_string_value_is_valid_video_timestamp,
                                check_if_valid_img, check_if_valid_rgb_tuple,
                                check_instance, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_valid_array, check_valid_boolean,
                                check_valid_dataframe, check_valid_lst,
                                check_valid_url, is_video_color)
from simba.utils.enums import (ENV_VARS, OS, ConfigKey, Defaults, Dtypes,
                               Formats, Keys, Links, Options, Paths)
from simba.utils.errors import (DataHeaderError, DuplicationError,
                                FFMPEGCodecGPUError, FFMPEGNotFoundError,
                                FileExistError, FrameRangeError, IntegerError,
                                InvalidFilepathError, InvalidFileTypeError,
                                InvalidInputError, InvalidVideoFileError,
                                MissingProjectConfigEntryError, NoDataError,
                                NoFilesFoundError, NotDirectoryError,
                                ParametersFileError, PermissionError,
                                SimBAPAckageVersionError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.warnings import (
    FileExistWarning, FrameRangeWarning, GPUToolsWarning, InvalidValueWarning,
    NoFileFoundWarning, ThirdPartyAnnotationsInvalidFileFormatWarning)

SIMBA_DIR = os.path.dirname(simba.__file__)

PARSE_OPTIONS = csv.ParseOptions(delimiter=",")
READ_OPTIONS = csv.ReadOptions(encoding="utf8")


def read_df(file_path: Union[str, os.PathLike],
            file_type: Union[str, os.PathLike] = 'csv',
            has_index: Optional[bool] = True,
            remove_columns: Optional[List[str]] = None,
            usecols: Optional[List[str]] = None,
            anipose_data: Optional[bool] = False,
            check_multiindex: Optional[bool] = False,
            multi_index_headers_to_keep: Optional[int] = None) -> Union[pd.DataFrame, dict]:
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
    :return: Table data in pd.DataFrame format.
    :rtype: pd.DataFrame

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
            raise InvalidFileTypeError(msg=f"{file_path} is not a valid CSV file", source=read_df.__name__)
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


def write_df(df: pd.DataFrame,
             file_type: str,
             save_path: Union[str, os.PathLike],
             multi_idx_header: bool = False) -> None:
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
            try:
                csv.write_csv(df, save_path)
            except Exception as e:
                print(e.args)
                raise PermissionError(msg=f'Could not save file at {save_path}. Is the file being used by a different process?', source=write_df.__name__)
        else:
            try:
                df = df.drop("scorer", axis=1, errors="ignore")
            except TypeError:
                pass
            try:
                df.to_csv(save_path)
            except Exception as e:
                print(e.args)
                raise PermissionError(msg=f'Could not save file at {save_path}. Is the file being used by a different process?', source=write_df.__name__)
    elif file_type == Formats.PARQUET.value:
        df.to_parquet(save_path)
    elif file_type == Formats.PICKLE.value:
        try:
            with open(save_path, "wb") as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e.args[0])
            raise InvalidFileTypeError(msg="Data could not be saved as a pickle.", source=write_df.__name__)
    else:
        raise InvalidFileTypeError(msg=f"{file_type} is not a valid filetype OPTIONS: [csv, pickle, parquet]", source=write_df.__name__)


def get_fn_ext(filepath: Union[os.PathLike, str],
               raise_error: bool = True) -> Union[Tuple[str, str, str], Tuple[None, None, None]]:
    """
    Split file path into three components: (i) directory, (ii) file name, and (iii) file extension.

    :parameter str filepath: Path to file.
    :returns: 3-part tuple with file directory name, file name (w/o extension), and file extension.
    :rtype: Tuple[str, str, str]

    :example:
    >>> get_fn_ext(filepath='C:/My_videos/MyVideo.mp4')
    >>> ('My_videos', 'MyVideo', '.mp4')
    """

    check_instance(source=f'{get_fn_ext} filepath', accepted_types=(str, os.PathLike), instance=filepath)
    file_extension = Path(filepath).suffix
    try:
        file_name = os.path.basename(filepath.rsplit(file_extension, 1)[0])
    except ValueError:
        if raise_error:
            raise InvalidFilepathError(msg=f"{filepath} is not a valid filepath", source=get_fn_ext.__name__)
        else:
            return None, None, None
    dir_name = os.path.dirname(filepath)
    return dir_name, file_name, file_extension


def read_config_entry(config: configparser.ConfigParser,
                      section: str,
                      option: str,
                      data_type: str,
                      default_value: Optional[Any] = None,
                      options: Optional[List] = None) -> Union[float, int, str]:
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
            raise MissingProjectConfigEntryError(msg=f"SimBA could not find an entry for option {option} under section {section} in the project_config.ini. Please specify the settings in the settings menu and make sure the path to your project config is correct", source=read_config_entry.__name__)
    except ValueError as e:
        print(e.args)
        if default_value != None:
            return default_value
        else:
            raise MissingProjectConfigEntryError(
                msg=f"SimBA could not find an entry for option {option} under section {section} in the project_config.ini. Please specify the settings in the settings menu.",
                source=read_config_entry.__name__,
            )


def read_project_path_and_file_type(config: configparser.ConfigParser) -> Tuple[str, str]:
    """
    Helper to read the path and file type of the SimBA project from the project_config.ini.

    :param configparser.ConfigParser config: parsed SimBA config in configparser.ConfigParser format
    :returns: The path of the project ``project_folder`` and  the set file type of the project (i.e., ``csv`` or ``parquet``) as two-part tuple.
    :rtype: Tuple[str, str]
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


def bgr_to_rgb_tuple(value: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """ convert bgr tuple to rgb tuple"""
    check_if_valid_rgb_tuple(data=value)
    return (value[2], value[1], value[0])

def read_video_info_csv(file_path: Union[str, os.PathLike]) -> pd.DataFrame:
    """
    Helper to read the project_folder/logs/video_info.csv of the SimBA project in as a pd.DataFrame

    :param Union[str, os.PathLike] file_path: Path to the project_folder/logs/video_info.csv file.
    :return: Dataframe representation of the file.
    :rtype: pd.DataFrame
    """

    EXPECTED_COLS = ["Video", "fps", "Resolution_width", "Resolution_height", "Distance_in_mm", "pixels/mm"]
    EXPECTED_FLOAT_COLS = ["fps", "Resolution_width", "Resolution_height", "Distance_in_mm", "pixels/mm"]

    if not os.path.isfile(file_path):
        raise NoFilesFoundError(msg=f"Could not find the video_info.csv table in your SimBA project. Create it using the [Video parameters] tab. SimBA expects the file at location {file_path}. See SimBA documentation for more info: https://t.ly/OtY79", source=read_video_info_csv.__name__)
    info_df = pd.read_csv(file_path)
    for c in EXPECTED_COLS:
        if c not in info_df.columns:
            raise ParametersFileError(msg=f'The file {file_path} does not not have an anticipated header ({c}). Please re-create the file and make sure each video has a {c} header column name', source=read_video_info_csv.__name__)
    info_df["Video"] = info_df["Video"].astype(str)
    for c in EXPECTED_FLOAT_COLS:
        col_vals = list(info_df[c])
        validity = check_valid_lst(data=col_vals, source='', valid_dtypes=Formats.NUMERIC_DTYPES.value, raise_error=False)
        if not validity:
            raise ParametersFileError(msg=f'One or more values in the {c} column of the {file_path} file could not be interpreted as a numeric value. Please check or re-create the file and make sure the entries in the {c} column are all numeric.', source=read_video_info_csv.__name__)
        else:
            pass
    if info_df["fps"].min() <= 1:
        videos_w_low_fps = ', '.join(list(info_df[info_df['fps'] <= 1]['Video']))
        InvalidValueWarning(
            msg=f"Video(s) in your SimBA project have an FPS of 1 or less. This includes video(s) {videos_w_low_fps}. It is recommended to use videos with more than one frame per second. If inaccurate, correct the FPS values inside the {file_path} file",
            source=read_video_info_csv.__name__)
    if info_df["pixels/mm"].min() == 0:
        videos_w_low_conversion_factor = ', '.join(list(info_df[info_df['pixels/mm'] == 0]['Video']))
        InvalidValueWarning(msg=f"Video(s) in your SimBA project have an pixel/mm conversion factor of 0. This includes video(s) {videos_w_low_conversion_factor}. Correct the pixel/mm conversion factor values inside the {file_path} file", source=read_video_info_csv.__name__)

    return info_df


def read_config_file(config_path: Union[str, os.PathLike]) -> configparser.ConfigParser:
    """
    Helper to parse SimBA project project_config.ini file

    :parameter Union[str, os.PathLike] config_path: Path to project_config.ini file
    :return: parsed project_config.ini file
    :rtype: configparser.ConfigParser
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


def get_video_meta_data(video_path: Union[str, os.PathLike, cv2.VideoCapture],
                        fps_as_int: bool = True,
                        raise_error: bool = True) -> Union[Dict[str, Any], None]:
    """
    Read video metadata (fps, resolution, frame cnt etc.) from video file (e.g., mp4).

    .. seealso::
       To use FFmpeg instead of OpenCV, see :func:`simba.utils.read_write.get_video_info_ffmpeg`.

    :param str video_path: Path to a video file.
    :param bool fps_as_int: If True, force video fps to int through floor rounding, else float. Default = True.
    :param bool raise_error: If True, raises an error if data cannot be read. If False, returns None. Default True.
    :return: The video metadata in dict format with parameter (e.g., ``fps``)  as keys.
    :rtype: Dict[str, Any].

    :example:
    >>> get_video_meta_data('test_data/video_tests/Video_1.avi')
    {'video_name': 'Video_1', 'fps': 30, 'width': 400, 'height': 600, 'frame_count': 300, 'resolution_str': '400 x 600', 'video_length_s': 10}
    """

    video_data = {}
    if isinstance(video_path, str):
        check_file_exist_and_readable(file_path=video_path)
        cap = cv2.VideoCapture(video_path)
        _, video_data["video_name"], _ = get_fn_ext(video_path)
    elif isinstance(video_path, cv2.VideoCapture):
        cap = video_path
        video_data["video_name"] = ''
    else:
        if raise_error:
            raise InvalidInputError(msg=f'video_path is neither a file path or a cv2.VideoCapture: {type(video_path)}', source=get_video_meta_data.__name__)
        else:
            return None
    video_data["fps"] = cap.get(cv2.CAP_PROP_FPS)
    if fps_as_int:
        video_data["fps"] = int(video_data["fps"])
    video_data["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_data["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_data["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if cap.get(cv2.CAP_PROP_CHANNEL) == 3:
        video_data["color_format"] = 'rgb'
    else:
        video_data["color_format"] = 'grey'
    for k, v in video_data.items():
        if v == 0:
            if raise_error:
                raise InvalidVideoFileError(msg=f'Video {video_data["video_name"]} either does not exist or has {k} of {str(v)} (full error video path: {video_path}).', source=get_video_meta_data.__name__)
            else:
                return None
    video_data["resolution_str"] = str(f'{video_data["width"]} x {video_data["height"]}')
    video_data["video_length_s"] = int(video_data["frame_count"] / video_data["fps"])
    return video_data


def get_video_info_ffmpeg(video_path: Union[str, os.PathLike]) -> Dict[str, Any]:
    """
    Extracts metadata information from a video file using FFmpeg's ffprobe.

    .. note::
       FFMpeg based metadata extraction seems preferable over OpenCV with data in .h264 format.

    .. seealso::
       To use OpenCV instead of FFmpeg, see :func:`simba.utils.read_write.get_video_meta_data`

    :param Union[str, os.PathLike] video_path: The file path to the video for which metadata is to be extracted.
    :return: A dictionary containing video metadata:
    :rtype: Dict[str, Any]
    """
    if not check_ffmpeg_available(raise_error=False):
        raise FFMPEGNotFoundError(msg=f'Cannot get video meta data from video using FFMPEG: FFMPEG not found on computer.', source=get_video_info_ffmpeg.__name__)
    check_file_exist_and_readable(file_path=video_path)
    video_name = get_fn_ext(filepath=video_path)[1]
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames,duration,pix_fmt", "-of", "json", video_path]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    data = json.loads(result.stdout)
    try:
        stream = data['streams'][0]
        width = int(stream['width'])
        height = int(stream['height'])
        num, denom = map(int, stream['r_frame_rate'].split('/'))
        fps = num / denom
        frame_count = int(stream.get('nb_read_frames', 0))
        duration = float(data.get('format', {}).get('duration', 0))
        if duration == 0 and frame_count and fps:
            duration = frame_count / fps
        pix_fmt = stream.get('pix_fmt', '')
        resolution_str = str(f'{width} x {height}')

        if 'gray' in pix_fmt: color_format = 'grey'
        else: color_format = 'rgb'
        return {"video_name": video_name,
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration_sec": duration,
                "color_format": color_format,
                'resolution_str': resolution_str}

    except (KeyError, IndexError, ValueError) as e:
        print(e.args)
        raise InvalidVideoFileError(msg=f'Cannot use FFMPEG to extract video meta data for video {video_name}, try OpenCV?', source=get_video_info_ffmpeg.__name__)

def remove_a_folder(folder_dir: Union[str, os.PathLike], ignore_errors: Optional[bool] = True) -> None:
    """Helper to remove a directory"""
    check_if_dir_exists(in_dir=folder_dir, source=remove_a_folder.__name__)
    try:
        shutil.rmtree(folder_dir, ignore_errors=ignore_errors)
    except Exception as e:
        raise PermissionError(msg=f'Could not delete directory: {folder_dir}. is the directory or its content beeing used by anothe process?', source=remove_a_folder.__name__)


def concatenate_videos_in_folder(in_folder: Union[str, os.PathLike, bytes],
                                 save_path: Union[str, os.PathLike],
                                 file_paths: Optional[List[Union[str, os.PathLike]]] = None,
                                 video_format: Optional[str] = "mp4",
                                 substring: Optional[str] = None,
                                 remove_splits: Optional[bool] = True,
                                 gpu: Optional[bool] = False,
                                 fps: Optional[Union[int, str]] = None,
                                 verbose: bool = True) -> None:
    """
    Concatenate (temporally) all video files in a folder into a single video.

    .. important::
       Input video parts will be joined in alphanumeric order, should ideally have to have sequential numerical ordered file names, e.g., ``1.mp4``, ``2.mp4``....

    .. note::
       If substring and file_paths are both not None, then file_paths with be sliced and only file paths with substring will be retained.

    :param Union[str, os.PathLike] in_folder: Path to folder holding un-concatenated video files.
    :param Union[str, os.PathLike] save_path: Path to the saved the output file. Note: If the path exist, it will be overwritten
    :param Optional[List[Union[str, os.PathLike]]] file_paths: If not None, then the files that should be joined. If None, then all files. Default None.
    :param Optional[str] video_format: The format of the video clips that should be concatenated. Default: mp4.
    :param Optional[str] substring: If a string, then only videos in in_folder with a filename that contains substring will be joined. If None, then all are joined. Default: None.
    :param Optional[str] video_format: Format of the input video files in ``in_folder``. Default: ``mp4``.
    :param Optional[bool] remove_splits: If true, the input splits in the ``in_folder`` will be removed following concatenation. Default: True.
    :rtype: None
    """

    if not check_nvidea_gpu_available() and gpu:
        raise FFMPEGCodecGPUError(msg="No FFMpeg GPU codec found.", source=concatenate_videos_in_folder.__name__)
    timer = SimbaTimer(start=True)
    if file_paths is None:
        files = glob.glob(in_folder + "/*.{}".format(video_format))
    else:
        for file_path in file_paths:
            check_file_exist_and_readable(file_path=file_path)
        files = file_paths
    check_if_filepath_list_is_empty(filepaths=files, error_msg=f"SIMBA ERROR: Cannot join videos in directory {in_folder}. The directory contain ZERO files in format {video_format}")
    if substring is not None:
        sliced_paths = []
        for file_path in files:
            if substring in get_fn_ext(filepath=file_path)[1]:
                sliced_paths.append(file_path)
        check_if_filepath_list_is_empty(filepaths=sliced_paths, error_msg=f"SIMBA ERROR: Cannot join videos in directory {in_folder}. The directory contain ZERO files in format {video_format} with substring {substring}")
        files = sliced_paths
    files.sort(key=lambda f: int(re.sub("\D", "", f)))
    temp_txt_path = Path(in_folder, "files.txt")
    if os.path.isfile(temp_txt_path):
        os.remove(temp_txt_path)
    with open(temp_txt_path, "w") as f:
        for file in files:
            f.write("file '" + str(Path(file)) + "'\n")

    out_fps = None
    if fps is not None:
        check_int(name='fps', value=fps, min_value=0)
        int_fps = int(fps)
        if isinstance(fps, str):
            if int_fps > len(files):
                raise InvalidInputError(msg=f'If FPS is a string it represents the video index ({fps}) which is more than the number of videos in the input directory ({len(files)})', source=concatenate_videos_in_folder.__name__)
            out_fps = float(get_video_meta_data(video_path=files[int_fps])['fps'])
        elif isinstance(fps, (int, float)):
            out_fps = fps
        else:
            raise InvalidInputError(msg=f'FPS of the output video has to be None, or a string index, or a float, or an integer',source=concatenate_videos_in_folder.__name__)

    if check_nvidea_gpu_available() and gpu:
        if fps is None:
            returned = os.system(f"ffmpeg -f concat -safe 0 -i \"{temp_txt_path}\" -c:v h264_nvenc -pix_fmt yuv420p -c:a copy -hide_banner -loglevel info \"{save_path}\" -y")
            if returned != 0:
                GPUToolsWarning(msg="GPU temporal concatenation failed, falling back to CPU temporal concatenation...", source=concatenate_videos_in_folder.__name__)
                returned = os.system(f'ffmpeg -f concat -safe 0 -i "{temp_txt_path}" "{save_path}" -c copy -hide_banner -loglevel info -y')
        else:
            returned = os.system(f"ffmpeg -f concat -safe 0 -i \"{temp_txt_path}\" -r {out_fps} -c:v h264_nvenc -pix_fmt yuv420p -c:a copy -hide_banner -loglevel info \"{save_path}\" -y")
            if returned != 0:
                GPUToolsWarning(msg="GPU temporal concatenation failed, falling back to CPU temporal concatenation...", source=concatenate_videos_in_folder.__name__)
                returned = os.system(f'ffmpeg -f concat -safe 0 -i "{temp_txt_path}" -r {out_fps} -c:v libx264 -crf 28 -preset veryfast -threads 12 -c:a copy -movflags +faststart -hide_banner -loglevel info "{save_path}" -y')
            #returned = os.system(f'ffmpeg -hwaccel auto -c:v h264_cuvid -f concat -safe 0 -i "{temp_txt_path}" -r {out_fps} -c:v h264_nvenc -c:a copy -hide_banner -loglevel info "{save_path}" -y')
            #returned = os.system(f'ffmpeg -hwaccel cuda -hwaccel_output_format cuda -c:v h264_cuvid -f concat -safe 0 -i "{temp_txt_path}" -vf scale_cuda=1280:720,format=nv12 -r {out_fps} -c:v h264_nvenc -c:a copy -hide_banner -loglevel info "{save_path}" -y')
    else:
        if fps is None:
            returned = os.system(f'ffmpeg -f concat -safe 0 -i "{temp_txt_path}" "{save_path}" -c copy -hide_banner -loglevel info -y')
        else:
            #returned = os.system(f'ffmpeg -f concat -safe 0 -i "{temp_txt_path}" -r {out_fps} -c:v libx264 -c:a copy -movflags +faststart -hide_banner -loglevel info "{save_path}" -y')
            returned = os.system(f'ffmpeg -f concat -safe 0 -i "{temp_txt_path}" -r {out_fps} -c:v libx264 -crf 28 -preset veryfast -threads 12 -c:a copy -movflags +faststart -hide_banner -loglevel info "{save_path}" -y')
            #returned = os.system(f'ffmpeg -f concat -safe 0 -i "{temp_txt_path}" -r {out_fps} -c:v libx264 -c:a copy -hide_banner -loglevel info "{save_path}" -y')
    while True:
        if returned != 0:
            pass
        else:
            if remove_splits:
                remove_a_folder(folder_dir=Path(in_folder))
            break
    timer.stop_timer()
    if verbose:
        stdout_success(msg="Video concatenated", elapsed_time=timer.elapsed_time_str, source=concatenate_videos_in_folder.__name__)


def get_bp_headers(body_parts_lst: List[str]) -> list:
    """
    Helper to create ordered list of all column header fields from body-part names for SimBA project dataframes.

    :parameter List[str] body_parts_lst: Body-part names in the SimBA prject
    :return: Body-part headers
    :rtype: List[str]

    :examaple:
    >>> get_bp_headers(body_parts_lst=['Nose'])
    >>> ['Nose_x', 'Nose_y', 'Nose_p']
    """

    bp_headers = []
    for bp in body_parts_lst:
        c1, c2, c3 = (f"{bp}_x", f"{bp}_y", f"{bp}_p")
        bp_headers.extend((c1, c2, c3))
    return bp_headers


def read_video_info(video_name: str,
                    video_info_df: Union[pd.DataFrame, None] = None,
                    vid_info_df: Union[pd.DataFrame, None] = None,
                    raise_error: Optional[bool] = True) -> Union[Tuple[pd.DataFrame, float, float], Tuple[None, None, None]]:

    """
    Helper to read the metadata (pixels per mm, resolution, fps etc) from the video_info.csv for a single input file/video

    :param pd.DataFrame vid_info_df: Parsed ``project_folder/logs/video_info.csv`` file. This file can be parsed by :func:`simba.utils.read_write.read_video_info_csv`.
    :param pd.DataFrame video_info_df: Alias for ``vid_info_df``. If both are provided, the ``vid_info_df`` is used.
    :param str video_name: Name of the video as represented in the ``Video`` column of the ``project_folder/logs/video_info.csv`` file.
    :param Optional[bool] raise_error: If True, raises error if the video cannot be found in the ``vid_info_df`` file. If False, returns None if the video cannot be found.
    :returns: 3-part tuple: One row DataFrame representing the video in the ``project_folder/logs/video_info.csv`` file, the frame rate of the video, and the the pixels per millimeter of the video
    :rtype: Union[Tuple[pd.DataFrame, float, float], Tuple[None, None, None]]

    :example:
    >>> video_info_df = read_video_info_csv(file_path='project_folder/logs/video_info.csv')
    >>> read_video_info(vid_info_df=video_info_df, video_name='Together_1')
    """


    FPS = 'fps'
    PXELS_PER_MM = "pixels/mm"
    VIDEO = "Video"
    REQUIRED_FIELDS = [PXELS_PER_MM, FPS, VIDEO]

    if isinstance(vid_info_df, pd.DataFrame):
        check_valid_dataframe(df=vid_info_df, source=f'{read_video_info.__name__} vid_info_df', required_fields=REQUIRED_FIELDS)
        video_info_df = deepcopy(vid_info_df)
    elif isinstance(video_info_df, pd.DataFrame):
        check_valid_dataframe(df=video_info_df, source=f'{read_video_info.__name__} video_info_df', required_fields=REQUIRED_FIELDS)
    else:
        raise InvalidInputError(msg='Both provide a valid dataframe as EITHER vid_info_df or the alias video_info_df.', source=read_video_info.__name__)

    check_str(name=f'{read_video_info.__name__} video_name', value=video_name, allow_blank=False)
    check_valid_boolean(value=[raise_error], source=f'{read_video_info.__name__} raise_error')
    video_settings = video_info_df.loc[video_info_df[VIDEO] == video_name]
    if len(video_settings) > 1:
        raise DuplicationError(msg=f"SimBA found multiple rows in `project_folder/logs/video_info.csv` for videos named {video_name}. Please make sure that each video name is represented ONCE in the file", source='')
    elif len(video_settings) < 1:
        if raise_error:
            raise ParametersFileError(msg=f"SimBA could not find {video_name} in the `project_folder/logs/video_info.csv` file. Make sure all videos analyzed are represented in the file.", source='')
        else:
            return (None, None, None)
    else:
        px_per_mm = video_settings[PXELS_PER_MM].values[0]
        fps = video_settings[FPS].values[0]
        if math.isnan(px_per_mm):
            raise ParametersFileError(msg=f'Pixels per millimeter for video {video_name} in the `project_folder/logs/video_info.csv` file is not a valid number. Please correct it to proceed.')
        if math.isnan(fps):
            raise ParametersFileError(msg=f'The FPS for video {video_name} in the `project_folder/logs/video_info.csv` file is not a valid number. Please correct it to proceed.')
        check_float(name=f'pixels per millimeter video {video_name}', value=px_per_mm)
        check_float(name=f'fps video {video_name}', value=fps)
        px_per_mm, fps = float(px_per_mm), float(fps)
        if px_per_mm <= 0:
            InvalidValueWarning(msg=f"Video {video_name} has a pixel per millimeter conversion factor of 0 or less. Correct the pixel/mm conversion factor values inside the `project_folder/logs/video_info.csv` file", source='')
        if fps <= 1:
            InvalidValueWarning(msg=f"Video {video_name} an FPS of 1 or less.  It is recommended to use videos with more than one frame per second. If inaccurate, correct the FPS values inside the `project_folder/logs/video_info.csv` file", source='')
        return video_settings, px_per_mm, fps

def find_all_videos_in_directory(directory: Union[str, os.PathLike],
                                 as_dict: Optional[bool] = False,
                                 raise_error: bool = False,
                                 video_formats: Optional[Tuple[str]] = (".avi", ".mp4", ".mov", ".flv", ".m4v", '.webm'),) -> Union[dict, list]:
    """
    Get all video file paths within a provided directory

    :param str directory: Directory to search for video files.
    :param bool as_dict: If True, returns dictionary with the video name as key and file path as value.
    :param bool raise_error: If True, raise error if no videos are found. Else, NoFileFoundWarning.
    :param Tuple[str] video_formats: Acceptable video formats. Default: '.avi', '.mp4', '.mov', '.flv', '.m4v'.
    :return Either a list or dictionary of all available video files in the ``directory``.
    :rtype: Union[dict, list]

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
                      frame_index: Optional[int] = 0,
                      opacity: Optional[float] = None,
                      size: Optional[Tuple[int, int]] = None,
                      greyscale: Optional[bool] = False,
                      black_and_white: Optional[bool] = False,
                      clahe: Optional[bool] = False,
                      use_ffmpeg: Optional[bool] = False) -> np.ndarray:

    """
    Reads single image from video file.

    .. seealso::
       To read a batch of images with GPU acceleration, see :func:`simba.utils.read_write.read_img_batch_from_video_gpu`.
       To read a batch of videos using multicore CPU acceleration, see :func:`simba.utils.read_write.read_img_batch_from_video`.
       To read frames batches asynchrnously, see :func:`simba.video_processors.async_frame_reader.AsyncVideoFrameReader`.

    :param Union[str, os.PathLike] video_path: Path to video file, or cv2.VideoCapture object.
    :param int frame_index: The frame of video to return. Default: 1. Note, if frame index -1 is passed, the last frame of the video is read in.
    :param Optional[int] opacity: Value between 0 and 100 or None. If float value, returns image with opacity. 100 fully opaque. 0.0 fully transparant.
    :param Optional[Tuple[int, int]] size: If tuple, resizes the image to size. Else, returns original image size.
    :param Optional[bool] greyscale: If true, returns the greyscale image. Default False.
    :param Optional[bool] black_and_white: If true, returns black and white image at threshold 127. Default False.
    :param Optional[bool] clahe: If true, returns clahe enhanced image. Default False.
    :return: Image as numpy array.
    :rtype: np.ndarray

    :example:
    >>> img = read_frm_of_video(video_path='/Users/simon/Desktop/envs/platea_featurizer/data/video/3D_Mouse_5-choice_MouseTouchBasic_s9_a4_grayscale.mp4', clahe=True)
    >>> cv2.imshow('img', img)
    >>> cv2.waitKey(5000)
    """

    check_instance(source=read_frm_of_video.__name__, instance=video_path, accepted_types=(str, cv2.VideoCapture))
    if use_ffmpeg:
        if not isinstance(video_path, str):
            raise InvalidInputError(msg='If using FFmpeg for video meta data extraction, pass data path rather than cv2.VideoCapture', source=read_frm_of_video.__name__)
    if type(video_path) == str:
        check_file_exist_and_readable(file_path=video_path)
        if not use_ffmpeg:
            video_meta_data = get_video_meta_data(video_path=video_path)
        else:
            video_meta_data = get_video_info_ffmpeg(video_path=video_path)
    else:
        video_meta_data = {"frame_count": int(video_path.get(cv2.CAP_PROP_FRAME_COUNT)),
                           "fps": video_path.get(cv2.CAP_PROP_FPS),
                           'width': int(video_path.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           'height': int(video_path.get(cv2.CAP_PROP_FRAME_HEIGHT))}

    check_int(name='frame_index', value=frame_index, min_value=-1)
    if frame_index == -1:
        frame_index = video_meta_data["frame_count"] - 1
    if (frame_index > video_meta_data["frame_count"]) or (frame_index < 0):
        raise FrameRangeError(msg=f'Frame {frame_index} is out of range: The video {video_path} contains {video_meta_data["frame_count"]} frames.', source=read_frm_of_video.__name__)
    if not use_ffmpeg:
        if type(video_path) == str:
            capture = cv2.VideoCapture(video_path)
        else:
            capture = video_path
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, img = capture.read()
        if not ret:
            raise FrameRangeError(msg=f"Frame {frame_index} for video {video_path} could not be read.")
    else:
        if not isinstance(video_path, str):
            raise NoDataError(msg='When using FFMpeg, pass video path', source=read_frm_of_video.__name__)
        is_color = is_video_color(video=video_path)
        timestamp = frame_index / video_meta_data['fps']
        if is_color:
            cmd = f"ffmpeg -hwaccel cuda -ss {timestamp:.10f} -i {video_path} -vframes 1 -f rawvideo -pix_fmt bgr24 -v error -"
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            img = np.frombuffer(result.stdout, np.uint8).reshape((video_meta_data["height"], video_meta_data["width"], 3))
        else:
            cmd = f"ffmpeg -hwaccel cuda -ss {timestamp:.10f} -i {video_path} -vframes 1 -f rawvideo -pix_fmt gray -v error -"
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            img = np.frombuffer(result.stdout, np.uint8).reshape((video_meta_data["height"], video_meta_data["width"]))
    if opacity:
        opacity = float(opacity / 100)
        check_float(name="Opacity", value=opacity, min_value=0.00, max_value=1.00, raise_error=True)
        opacity = 1 - opacity
        h, w, clr = img.shape[:3]
        opacity_image = np.ones((h, w, clr), dtype=np.uint8) * int(255 * opacity)
        img = cv2.addWeighted( img.astype(np.uint8), 1 - opacity, opacity_image.astype(np.uint8), opacity, 0)
    if size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    if greyscale or black_and_white:
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if black_and_white:
        img = np.where(img > 127, 255, 0).astype(np.uint8)
    if clahe:
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16)).apply(img)

    return img


def read_img(img_path: Union[str, os.PathLike],
             greyscale: bool = False,
             clahe: bool = False,
             opacity: Optional[float] = None) -> np.ndarray:

    file_ext = get_fn_ext(filepath=img_path)[2].lower()
    if file_ext not in Options.ALL_IMAGE_FORMAT_OPTIONS.value:
        raise InvalidFilepathError(
            msg=f'The image path {img_path} does not have a valid image extension. Got: {file_ext}. Valid: {Options.ALL_IMAGE_FORMAT_OPTIONS.value}',
            source=read_img.__name__)
    check_file_exist_and_readable(file_path=img_path)
    img = cv2.imread(filename=img_path)
    if opacity is not None:
        opacity = float(opacity / 100)
        check_float(name="Opacity", value=opacity, min_value=0.00, max_value=1.00, raise_error=True)
        opacity = 1 - opacity
        h, w, clr = img.shape[:3]
        opacity_image = np.ones((h, w, clr), dtype=np.uint8) * int(255 * opacity)
        img = cv2.addWeighted(img.astype(np.uint8), 1 - opacity, opacity_image.astype(np.uint8), opacity, 0)
    if greyscale:
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if clahe:
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16)).apply(img)

    return img.astype(np.uint8)


def find_video_of_file(video_dir: Union[str, os.PathLike],
                       filename: str,
                       raise_error: Optional[bool] = False,
                       warning: Optional[bool] = True) -> Union[str, os.PathLike, None]:
    """
    Helper to find the video file with the SimBA project that represents a known data file path.

    :param str video_dir: Directory holding putative video file.
    :param str filename: Data file name, e.g., ``Video_1``.
    :param Optional[bool] raise_error: If True, raise error if no file can be found. If False, returns None if no file can be found. Default: False
    :param Optional[bool] warning: If True, print warning if no file can be found. If False, no warning is printed if file cannot be found. Default: False
    :return: Video file path.
    :rtype: Union[str, os.PathLike]

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


def find_files_of_filetypes_in_directory(directory: Union[str, os.PathLike],
                                         extensions: List[str],
                                         raise_warning: bool = True,
                                         as_dict: bool = False,
                                         raise_error: bool = False) -> Union[List[str], Dict[str, str]]:
    """
    Find all files in a directory of specified extensions/types.

    :param str directory: Directory holding files.
    :param List[str] extensions: Accepted file extensions.
    :param bool raise_warning: If True, raise warning if no files are found. Default True.
    :param bool raise_error: If True, raise error if no files are found. Default False.
    :param bool as_dict: If True, returns a dictionary with all filenames as keys and filepaths as values. If False, then a list of all filepaths. Default False.
    :return: All files in ``directory`` with the specified extension(s).
    :rtype: Union[List[str], Dict[str, str]]

    :example:
    >>> find_files_of_filetypes_in_directory(directory='project_folder/videos', extensions=['mp4', 'avi', 'png'], raise_warning=False)
    """

    if not os.path.isdir(directory):
        if raise_warning:
            NoFileFoundWarning(msg=f'{directory} is not a valid directory', source=find_files_of_filetypes_in_directory.__name__)
            return []
        if raise_error:
            raise NoFilesFoundError(msg=f'{directory} is not a valid directory', source=find_files_of_filetypes_in_directory.__name__)
    try:
        all_files_in_folder = [f for f in next(os.walk(directory))[2] if not f[0] == "."]
    except StopIteration:
        if raise_warning:
            raise NoFilesFoundError(msg=f"No files found in the {directory} directory with accepted extensions {str(extensions)}", source=find_files_of_filetypes_in_directory.__name__)
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
        NoFileFoundWarning(msg=f"SimBA could not find any files with accepted extensions {extensions} in the {directory} directory", source=find_files_of_filetypes_in_directory.__name__)
    if not accepted_file_paths and raise_error:
        raise NoDataError(msg=f"SimBA could not find any files with accepted extensions {extensions} in the {directory} directory", source=find_files_of_filetypes_in_directory.__name__)
    if as_dict:
        out = {}
        for file_path in accepted_file_paths:
            _, file_name, _ = get_fn_ext(file_path)
            out[file_name] = file_path
        return out
    else:
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
        new_file_path = os.path.join(directory, os.path.basename(file_path).replace(".parquet", ".csv"))
        if "scorer" in df.columns:
            df = df.set_index("scorer")
        df.to_csv(new_file_path)
        print("Saved {}...".format(new_file_path))
    stdout_success(msg=f"{str(len(files_found))} parquet files in {directory} converted to csv", source=convert_parquet_to_csv.__name__)


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


def get_file_name_info_in_directory(directory: Union[str, os.PathLike], file_type: str) -> Dict[str, str]:
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


def archive_processed_files(config_path: Union[str, os.PathLike], archive_name: str) -> None:
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
    if isinstance(input_str, bool):
        return input_str
    else:
        check_str(name='input_str', value=input_str)
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
        raise InvalidFilepathError(msg=f"The {clf_path} file is not a pickle file", source=tabulate_clf_info.__name__)
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
    :return: Classifier model names
    :rtype: List[str]

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
    :return: List of paths to SimBA model config meta files.
    :rtype: List[str]

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
    :return: Dict holding the memory usage of the dataframe in bytes, mb, and gb.
    :rtype: Dict[str, float]

    :example:
    >>> df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    >>> {'bytes': 3328, 'megabytes': 0.003328, 'gigabytes': 3e-06}
    """
    if not isinstance(df, pd.DataFrame):
        raise InvalidInputError(msg='df has to be a pandas dataframe', source=get_memory_usage_of_df.__name__)
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


def find_all_videos_in_project(videos_dir: Union[str, os.PathLike],
                               basename: Optional[bool] = False,
                               raise_error: bool = True) -> List[str]:
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
        if raise_error:
            raise NoFilesFoundError(msg=f"No videos in mp4 or avi format found imported to SimBA project in the {videos_dir} directory", source=find_all_videos_in_project.__name__)
        else:
            return []
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

    :param str timestamp: Timestamp in HH:MM:SS format
    :returns: The timestamps as seconds.
    :rtype: int
    :raises FrameRangeError: If timestamp is not a valid format.

    :example:
    >>> timestamp_to_seconds(timestamp='00:00:05')
    >>> 5
    """

    check_if_string_value_is_valid_video_timestamp(value=timestamp, name="Timestamp")
    h, m, s = timestamp.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)




def find_time_stamp_from_frame_numbers(start_frame: int, end_frame: int, fps: float) -> List[str]:
    """
    Given start and end frame numbers and frames per second (fps), return a list of formatted time stamps
    corresponding to the frame range start and end time.

    :param int start_frame: The starting frame index.
    :param int end_frame: The ending frame index.
    :param float fps: Frames per second.
    :return: A list of time stamps in the format 'HH:MM:SS:MS'.
    :rtype: List[str]

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


def read_roi_data(roi_path: Union[str, os.PathLike]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Method to read in ROI definitions from SimBA project.

    :param Union[str, os.PathLike] roi_path: path to `ROI_definitions.h5` on disk.
    :return: 3-part Tuple of dataframes representing rectangles, circles, polygons.
    :rtype: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]

    """
    check_file_exist_and_readable(file_path=roi_path)
    try:
        rectangles_df = pd.read_hdf(roi_path, key=Keys.ROI_RECTANGLES.value).dropna(how="any")
        circles_df = pd.read_hdf(roi_path, key=Keys.ROI_CIRCLES.value).dropna(how="any")
        polygon_df = pd.read_hdf(roi_path, key=Keys.ROI_POLYGONS.value)
    except (pickle.UnpicklingError, ValueError) as e:
        if "unsupported pickle protocol" in str(e).lower():
            print(e.args)
            raise SimBAPAckageVersionError(msg=f'Pickle VERSION ERROR. The ROIs where likely drawn using a SimBA installation that was built in a **Python** version different from the current Python version (e.g., 3.10 vs 3.6). You currently have Python version {OS.PYTHON_VER.value}. You may have drawn the ROIs in SimBA using Python version >3.8 and now you have Python version <3.8. To fix, use the same Python version as you used when drawing the ROIs')
        else:
            print(e.args)
            raise InvalidFileTypeError(msg=f"{roi_path} is not a valid SimBA ROI definitions file. See above for more detailed error cause.", source=read_roi_data.__name__)
    except Exception as e:
        print(e.args)
        raise InvalidFileTypeError(msg=f"{roi_path} is not a valid SimBA ROI definitions file. See above for more detailed error cause.", source=read_roi_data.__name__)

    if "Center_XCenter_Y" in polygon_df.columns:
        polygon_df = polygon_df.drop(["Center_XCenter_Y"], axis=1)
    if 'Center_X' not in rectangles_df.columns:
        if len(rectangles_df) > 0:
            rectangles_df['Center_X'] = rectangles_df['topLeftX'] + round(rectangles_df['width']/2)
        else:
            rectangles_df['Center_X'] = pd.Series(dtype='int')
    if 'Center_Y' not in rectangles_df.columns:
        if len(rectangles_df) > 0:
            rectangles_df['Center_Y'] = rectangles_df['topLeftY'] + round(rectangles_df['height']/2)
        else:
            rectangles_df['Center_Y'] = pd.Series(dtype='int')
    #circles_df['Video'] = circles_df['Video'].replace('Trial     1_dSLR1_sample_A1_na', '501_MA142_Gi_Saline_0513')
    return rectangles_df, circles_df, polygon_df

#read_roi_data(roi_path=r"C:\troubleshooting\mitra\project_folder\logs\measures\ROI_definitions.h5")



def create_directory(paths: Union[str, os.PathLike, bytes, List[str], Tuple[str]], overwrite: bool = False) -> None:

    """
    Create one or multiple directories.

    :param Union[str, os.PathLike, bytes, List[str], Tuple[str]] paths: A single path or a list/tuple of paths to create. Each path must be a non-empty string.
    :param overwrite: If True and the directory already exists, it will be deleted and recreated. If False, the existing directory will be preserved.
    :return: None
    """

    if isinstance(paths, (list, tuple)):
        for i in paths:
            check_str(name=f'{create_directory.__name__} paths', value=i, allow_blank=False, raise_error=True)
    else:
        check_str(name=f'{create_directory.__name__} paths', value=paths, allow_blank=False, raise_error=True)
        paths = [paths]
    for path in paths:
        path = os.path.abspath(path)
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except Exception as e:
                raise PermissionError(f'SimBA is not allowed to create the directory {path} ({e})')
        elif overwrite:
            try:
                remove_a_folder(folder_dir=path)
                os.makedirs(path)
            except Exception as e:
                raise PermissionError(f'SimBA is not allowed to overwrite the directory {path} ({e})')


def find_max_vertices_coordinates(shapes: List[Union[Polygon, LineString, MultiPolygon, Point]], buffer: Optional[int] = None) -> Tuple[int, int]:
    """
    Find the maximum x and y coordinates among the vertices of a list of geometries.

    Can be useful for plotting puposes, to dtermine the rquired size of the canvas to fit all geometries.

    :param List[Union[Polygon, LineString, MultiPolygon, Point]] shapes: A list of Shapely geometries including Polygons, LineStrings, MultiPolygons, and Points.
    :param Optional[int] buffer: If int, adds to maximum x and y.
    :returns: A two-part tuple containing the maximum x and y coordinates found among the vertices.
    :rtype: Tuple[int, int]

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


def clean_superanimal_topview_filename(file_name: str):

    SUPERANIMAL_TOPVIEW = "_superanimal_topviewmouse_"
    if (SUPERANIMAL_TOPVIEW in file_name.lower()):
        filename_parts = file_name.split(SUPERANIMAL_TOPVIEW)
        if len(filename_parts) >= 2:
            return filename_parts[0]
        else:
            return file_name
    else:
        return file_name


def read_dlc_superanimal_h5(path: Union[str, os.PathLike], col_names: List[str]) -> pd.DataFrame:
    EXPECTED_KEYS = ['df_with_missing']
    DF_W_MISSING = 'df_with_missing'
    NESTED_EXPECTED_KEYS = ['_i_table', 'table']
    check_file_exist_and_readable(file_path=path)
    check_valid_lst(data=col_names, source=f'{read_dlc_superanimal_h5.__name__} col_names', valid_dtypes=(str,), min_len=1)
    try:
        pose_data = h5py.File(path, "r")
    except Exception as e:
        raise InvalidInputError(msg=f'The DLC file {path} could not be read as a valid H5 file: {e.args}', source=read_dlc_superanimal_h5.__name__)
    pose_keys = list(pose_data.keys())
    missing_keys = [x for x in EXPECTED_KEYS if x not in pose_keys]
    if len(missing_keys) > 0:
        raise InvalidInputError(msg=f'The file {path} does not contain the expected key(s) {missing_keys}. Is it a valid superanimal DLC h5 file?', source=read_dlc_superanimal_h5.__name__)
    missing_keys = [x for x in NESTED_EXPECTED_KEYS if x not in pose_data[DF_W_MISSING]]
    if len(missing_keys) > 0:
        raise InvalidInputError(msg=f'The file {path} does not contain the expected key(s) {missing_keys}. Is it a valid superanimal DLC h5 file?', source=read_dlc_superanimal_h5.__name__)
    data = pose_data[DF_W_MISSING]['table'][...]
    data = pd.DataFrame([item[-1] for item in data])
    if len(data.columns) < len(col_names):
        raise InvalidInputError(msg=f'The file {path} does contains {len(data.columns)} columns. With your current project, SimBA expects this file to contain at least {len(col_names)} columns', source=read_dlc_superanimal_h5.__name__)
    data = data.loc[:, :len(col_names)-1]
    data.columns = col_names
    return data


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


def copy_files_in_directory(in_dir: Union[str, os.PathLike],
                            out_dir: Union[str, os.PathLike],
                            raise_error: bool = True,
                            filetype: Optional[str] = None) -> None:
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
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if filetype is not None:
        file_paths = glob.glob(in_dir + f"/*.{filetype}")
    else:
        file_paths = glob.glob(in_dir + f"/*.")
    if len(file_paths) == 0 and raise_error:
        raise NoFilesFoundError(msg=f"No files found in {in_dir}", source=copy_files_in_directory.__name__)
    elif len(file_paths) == 0:
        pass
    else:
        for file_path in file_paths:
            shutil.copy(file_path, out_dir)


def remove_multiple_folders(folders: List[Union[os.PathLike, str]], raise_error: Optional[bool] = False) -> None:

    """
    Helper to remove multiple directories.
    :param folders List[os.PathLike]: List of directory paths.
    :param bool raise_error: If True, raise ``NotDirectoryError`` error of folder does not exist. if False, then pass. Default False.
    :raises NotDirectoryError: If ``raise_error`` and directory does not exist.

    :example:
    >>> remove_multiple_folders(folders= ['gerbil/gerbil_data/featurized_data/temp'])
    """

    folders = [x for x in folders if x is not None]
    for folder_path in folders:
        if raise_error and not os.path.isdir(folder_path):
            raise NotDirectoryError(msg=f"Cannot delete directory {folder_path}. The directory does not exist.", source=remove_multiple_folders.__name__)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path, ignore_errors=True)
        else:
            pass




def remove_files(file_paths: List[Union[str, os.PathLike]], raise_error: Optional[bool] = False) -> None:
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
            raise NoFilesFoundError(msg=f"Cannot delete {file_path}. File does not exist", source=remove_files.__name__)
        elif not os.path.isfile(file_path):
            pass
        else:
            try:
                os.remove(file_path)
            except:
                if raise_error:
                    raise PermissionError(msg=f"Cannot read {file_path}. Is the file open in an alternative app?", source=remove_files.__name__)
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

def fetch_pip_data(pip_url: str = Links.SIMBA_PIP_URL.value) -> Union[Tuple[Dict[str, Any], str], Tuple[None, None]]:
    """ Helper to fetch the pypi data associated with a package """
    if check_valid_url(url=pip_url):
        try:
            opener = request.build_opener(request.HTTPHandler(), request.HTTPSHandler())
            with opener.open(pip_url, timeout=2) as response:
                if response.status == 200:
                    encoding = response.info().get_content_charset("utf-8")
                    data = response.read().decode(encoding)
                    json_data = json.loads(data)
                    latest_release = json_data.get("info", {}).get("version", "")
                    return json_data, latest_release
        except Exception as e:
            #print(e.args)
            return None, None
    else:
        return None, None


def write_pickle(data: Dict[Any, Any], save_path: Union[str, os.PathLike]) -> None:
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


def read_pickle(data_path: Union[str, os.PathLike], verbose: Optional[bool] = False) -> Dict[Any, Any]:
    """
    Read a single or directory of pickled objects. If directory, returns dict with numerical sequential integer keys for
    each object.

    :param str data_path: Pickled file path, or directory of pickled files.
    :param Optional[bool] verbose: If True, prints progress. Default False.
    :returns: Dictionary representation of the pickle.
    :rtype: Dict[Any, Any]

    :example:
    >>> data = read_pickle(data_path='/test/unsupervised/cluster_models')
    """
    data = None
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
        raise InvalidFilepathError(msg=f"The path {data_path} is neither a valid file or directory path", source=read_pickle.__name__)

    return data


def drop_df_fields(data: pd.DataFrame, fields: List[str], raise_error: Optional[bool] = False) -> pd.DataFrame:
    """
    Drops specified fields in dataframe.

    :param pd.DataFrame: Data in pandas format.
    :param  List[str] fields: Columns to drop.
    :return pd.DataFrame
    """

    check_instance( source=drop_df_fields.__name__, instance=data, accepted_types=(pd.DataFrame,))
    check_valid_lst(data=fields, source=drop_df_fields.__name__, valid_dtypes=(str,), min_len=1, raise_error=raise_error)
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


def copy_files_to_directory(file_paths: Union[List[Union[str, os.PathLike]], Union[str, os.PathLike]],
                            dir: Union[str, os.PathLike],
                            verbose: Optional[bool] = True,
                            integer_save_names: Optional[bool] = False) -> List[Union[str, os.PathLike]]:
    """
    Copy a list of files to a specified directory.

    :param List[Union[str, os.PathLike]] file_paths: List of paths to the files to be copied, or a single filepath string.
    :param Union[str, os.PathLike] dir: Path to the directory where files will be copied.
    :param Optional[bool] verbose: If True, prints progress information. Default True.
    :param Optional[bool] integer_save_names: If True, saves files with integer names. E.g., file one in ``file_paths`` will be saved as dir/0.
    :return List[Union[str, os.PathLike]]: List of paths to the copied files
    """

    check_instance(source=f'{copy_files_to_directory.__name__} file_paths', instance=file_paths, accepted_types=(str, list, os.PathLike,))
    if isinstance(file_paths, (str,)):
        file_paths = [file_paths]
    check_valid_lst(data=file_paths, source=f'{copy_files_to_directory.__name__} file_paths', min_len=1, valid_dtypes=(str, np.str_,))
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


def seconds_to_timestamp(seconds: Union[int, float, List[Union[int, float]]]) -> Union[str, List[str]]:
    """
    Convert an integer number representing seconds, or a list of integers representing seconds, to a HH:MM:SS format.
    """
    if isinstance(seconds, (int, float)):
        check_float(name=f"{seconds_to_timestamp.__name__} seconds", value=seconds, min_value=0)
        data = [seconds]
    elif isinstance(seconds, list):
        check_valid_lst(data=seconds, source=f"{seconds_to_timestamp.__name__} seconds", valid_dtypes=(int, float), min_len=1)
        data = seconds
    else:
        raise InvalidInputError(msg=f'Got {type(seconds)} for seconds. Only list or float or integer accepted.', source=seconds_to_timestamp.__name__)
    results = []
    for i in data:
        hours = int(i / 3600)
        minutes = int((i % 3600) / 60)
        seconds = int(i % 60)
        results.append("{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds))
    return results[0] if len(data) == 1 else results


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

@njit("(uint8[:, :, :, :],)", fastmath=True, parallel=True)
def img_stack_to_greyscale(imgs: np.ndarray):
    """
    Jitted conversion of a 4D stack of color images (RGB format) to grayscale.

    .. image:: _static/img/img_stack_to_greyscale.png
       :width: 600
       :align: center

    :parameter np.ndarray imgs: A 4D array representing color images. It should have the shape (num_images, height, width, 3) where the last dimension represents the color channels (R, G, B).
    :returns np.ndarray: A 3D array containing the grayscale versions of the input images. The shape of the output array is (num_images, height, width).

    :example:
    >>> imgs = ImageMixin().read_img_batch_from_video( video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos/Together_1.avi', start_frm=0, end_frm=100)
    >>> imgs = np.stack(list(imgs.values()))
    >>> imgs_gray = ImageMixin.img_stack_to_greyscale(imgs=imgs)
    """
    results = np.full((imgs.shape[0], imgs.shape[1], imgs.shape[2]), np.nan).astype(np.uint8)
    for i in prange(imgs.shape[0]):
        vals = (0.07 * imgs[i][:, :, 2] + 0.72 * imgs[i][:, :, 1] + 0.21 * imgs[i][:, :, 0])
        results[i] = vals.astype(np.uint8)
    return results


@njit("(uint8[:, :, :, :],)", fastmath=True, parallel=True)
def img_stack_to_bw(imgs: np.ndarray):
    """
    Jitted conversion of a 4D stack of color images (RGB format) to black and white.

    .. image:: _static/img/img_stack_to_greyscale.png
       :width: 600
       :align: center

    :parameter np.ndarray imgs: A 4D array representing color images. It should have the shape (num_images, height, width, 3) where the last dimension represents the color channels (R, G, B).
    :returns np.ndarray: A 3D array containing the black and white versions of the input images. The shape of the output array is (num_images, height, width).

    :example:
    >>> imgs = ImageMixin().read_img_batch_from_video( video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos/Together_1.avi', start_frm=0, end_frm=100)
    >>> imgs = np.stack(list(imgs.values()))
    >>> imgs_gray = ImageMixin.img_stack_to_greyscale(imgs=imgs)
    """
    results = np.full((imgs.shape[0], imgs.shape[1], imgs.shape[2]), np.nan).astype(np.uint8)
    for i in prange(imgs.shape[0]):
        vals = (0.07 * imgs[i][:, :, 2] + 0.72 * imgs[i][:, :, 1] + 0.21 * imgs[i][:, :, 0])
        results[i] = np.where(vals > 127, 255, 0).astype(np.uint8)
    return results

@njit(fastmath=True)
def img_to_bw(img: np.ndarray) -> np.ndarray:
    """
    Jitted conversion of a single image (grayscale or RGB) to black and white.

    :param img: A 2D grayscale image (H, W) or 3D RGB image (H, W, 3), dtype uint8.
    :return: A 2D binary black and white image with values 0 or 255.
    """
    if img.ndim == 2:
        h, w = img.shape
        result = np.empty((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                result[i, j] = 255 if img[i, j] > 127 else 0
    else:
        h, w, _ = img.shape
        result = np.empty((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                val = 0.07 * img[i, j, 2] + 0.72 * img[i, j, 1] + 0.21 * img[i, j, 0]
                result[i, j] = 255 if val > 127 else 0
    return result

def read_img_batch_from_video_gpu(video_path: Union[str, os.PathLike],
                                  start_frm: Optional[int] = None,
                                  end_frm: Optional[int] = None,
                                  verbose: bool = False,
                                  greyscale: bool = False,
                                  black_and_white: bool = False,
                                  out_format: Literal['dict', 'array'] = 'dict') -> Union[Dict[int, np.ndarray], np.ndarray]:

    """
    Reads a batch of frames from a video file using GPU acceleration.

    This function uses FFmpeg with CUDA acceleration to read frames from a specified range in a video file. It supports both RGB and greyscale video formats. Frames are returned as a dictionary where the keys are
    frame indices and the values are NumPy arrays representing the image data.

    .. note::
       When black-and-white videos are saved as MP4, there can be some small errors in pixel values during compression. A video with only (0, 255) pixel values therefore gets other pixel values, around 0 and 255, when read in again.
       If you expect that the video you are reading in is black and white, set ``black_and_white`` to True to round any of these wonly value sto 0 and 255.

    .. seealso::
       For CPU multicore acceleration, see :func:`simba.mixins.image_mixin.ImageMixin.read_img_batch_from_video`

    :param video_path: Path to the video file. Can be a string or an os.PathLike object.
    :param start_frm: The starting frame index to read. If None, starts from the beginning of the video.
    :param end_frm: The ending frame index to read. If None, reads until the end of the video.
    :param verbose: If True, prints progress information to the console.
    :param greyscale: If True, returns the images in greyscale. Default False.
    :param black_and_white: If True, returns the images in black and white. Default False.
    :return: A dictionary where keys are frame indices (integers) and values are NumPy arrays containing the image data of each frame.
    """

    check_file_exist_and_readable(file_path=video_path)
    video_meta_data = get_video_meta_data(video_path=video_path, fps_as_int=False)
    if start_frm is not None:
        check_int(name=read_img_batch_from_video_gpu.__name__, value=start_frm, min_value=0, max_value=video_meta_data["frame_count"])
    else:
        start_frm = 0
    if end_frm is not None:
        check_int(name=read_img_batch_from_video_gpu.__name__,value=end_frm, min_value=0,max_value=video_meta_data["frame_count"])
        end_frm = end_frm + 1
    else:
        end_frm = video_meta_data["frame_count"] + 1
    if end_frm < start_frm:
        raise FrameRangeError(msg=f'The end frame ({end_frm}) has to be after of the same as the start frame ({start_frm})', source=read_img_batch_from_video_gpu.__name__)

    start_time, end_time = start_frm / video_meta_data["fps"], end_frm / video_meta_data["fps"]
    duration = end_time - start_time
    frame_width = video_meta_data['width']
    frame_height = video_meta_data['height']
    frame_size = frame_width * frame_height * 3
    color_format = 'bgr24'
    ffmpeg_cmd = ['ffmpeg',
                  '-hwaccel', 'cuda',
                  '-ss', f'{start_time:.10f}',
                  '-i', video_path,
                  '-t', f'{duration:.10f}',
                  '-f', 'rawvideo',
                  '-pix_fmt', f'{color_format}',
                  '-']

    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if out_format == 'dict':
        frames = {}
    else:
        frames = np.zeros((end_frm-start_frm, frame_height, frame_width, 3), dtype=np.uint8)
    frm_cnt = deepcopy(start_frm)
    video_name = get_fn_ext(filepath=video_path)[1]
    iteration_frm_cnt = 0
    while frm_cnt < end_frm:
        if verbose:
            print(f'Reading frame {frm_cnt+1} / {end_frm} ... ({video_name})')
        raw_frame = ffmpeg_process.stdout.read(frame_size)
        if len(raw_frame) == 0:
            break
        if color_format == 'bgr24':
            img = np.frombuffer(raw_frame, dtype=np.uint8).reshape((frame_height, frame_width, 3))
        else:
            img = np.frombuffer(raw_frame, dtype=np.uint8).reshape((frame_height, frame_width))
        if out_format == 'dict':
            frames[frm_cnt] = img
        else:
            frames[iteration_frm_cnt] = img
        frm_cnt += 1
        iteration_frm_cnt += 1
    if greyscale or black_and_white:
        if out_format == 'dict':
            greyscale_imgs = img_stack_to_greyscale(imgs=np.stack(list(frames.values()), axis=0)).astype(np.uint8)
            for cnt, i in enumerate(range(start_frm, end_frm)):
                frames[i] = greyscale_imgs[cnt]
            del greyscale_imgs
        else:
            frames = img_stack_to_greyscale(imgs=frames).astype(np.uint8)

    if black_and_white:
        binary_frms = {}
        if out_format == 'dict':
            for frm_id, frm in frames.items():
                binary_frms[frm_id] = np.where(frm > 127, 255, 0).astype(np.uint8)
        else:
            for frm_id in range(frames.shape[0]):
                binary_frms[frm_id] = np.where(frames[frm_id] > 127, 255, 0).astype(np.uint8)
        frames = binary_frms

    return frames


def find_largest_blob_location(imgs: Dict[int, np.ndarray],
                               verbose: bool = False,
                               video_name: Optional[str] = None,
                               inclusion_zone: Optional[Union[Polygon, MultiPolygon,]] = None) -> Dict[int, np.ndarray]:
    """
    Helper to find the largest connected component in binary image. E.g., Use to find a "blob" (i.e., animal) within a background subtracted image.

    :param Dict[int, np.ndarray] imgs: Dictionary of images where the key is the frame id and the value is an image in np.ndarray format.
    :param bool verbose: If True, prints progress. Default: False.
    :param video_name video_name: The name of the video being processed for interpretable progress msg if ``verbose``.
    :param Optional[np.ndarray] inclusion_zones: If not None, then 2D numpy array of ROI / shape vertices. If not None, the largest blob will be searched for only in the ROI.
    :return: Dictionary where the key is the frame id and the value is a 2D array with x and y coordinates.
    :rtype: Dict[int, np.ndarray]
    """

    check_valid_boolean(value=[verbose], source=f'{find_largest_blob_location.__name__} verbose', raise_error=True)
    if inclusion_zone is not None:
        check_instance(source=f'{find_largest_blob_location.__name__} inclusion_zone', instance=inclusion_zone, accepted_types=(MultiPolygon, Polygon,), raise_error=True)

    results, prior_window = {}, None
    for frm_idx, img in imgs.items():
        if verbose:
            if video_name is None:
                print(f'Finding blob in image {frm_idx}...')
            else:
                print(f'Finding blob in image {frm_idx} (Video {video_name})...')
        try:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
            if (inclusion_zone is not None) and num_labels != 1:
                centroid_points = [Point(xy) for xy in centroids]
                centroid_idx = [inclusion_zone.contains(Point(xy)) for xy in centroid_points]
                centroids, stats, labels = centroids[centroid_idx], stats[centroid_idx], labels[centroid_idx]
            if (num_labels == 1) or (centroids.shape[0] == 0):
                results[frm_idx] = np.array([0, 0]).astype(np.int32)
            else:
                largest_blob_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                results[frm_idx] = centroids[largest_blob_label].astype(np.int32)
        except Exception as e:
            print(e.args)
            results[frm_idx] = np.array([np.nan, np.nan])
    return results


def bento_file_reader(file_path: Union[str, os.PathLike],
                      fps: Optional[float] = None,
                      orient: Optional[Literal['index', 'columns']] = 'index',
                      save_path: Optional[Union[str, os.PathLike]] = None,
                      raise_error: Optional[bool] = False,
                      log_setting: Optional[bool] = False) -> Union[None, Dict[str, pd.DataFrame]]:

    """
    Reads a BENTO annotation file and processes it into a dictionary of DataFrames, each representing a classified behavior.
    Optionally, the results can be saved to a specified path.

    The function handles both frame-based and second-based annotations, converting the latter to frame-based
    annotations if the frames-per-second (FPS) is provided or can be inferred from the file.

    :param Union[str, os.PathLike] file_path: Path to the BENTO annotation file.
    :param Optional[float] fps: Frames per second (FPS) for converting second-based annotations to frames. If not provided, the function  will attempt to infer FPS from the file. If FPS is required and cannot be inferred, an error is raised.
    :param Optional[Union[str, os.PathLike]] save_path: Path to save the processed results as a pickle file. If None, results are returned instead of saved.
    :return: A dictionary where the keys are classifier names and the values are DataFrames with 'START' and 'STOP'  columns representing the start and stop frames of each behavior.
    :rtype: Dict[str, pd.DataFrame]

    :example:
    >>> bento_file_reader(file_path=r"C:\troubleshooting\bento_test\bento_files\20240812_crumpling3.annot")
    """

    def _orient_columns_melt(df: pd.DataFrame) -> pd.DataFrame:
        df = df[['START', 'STOP']].astype(np.int32).reset_index()
        df = df.melt(id_vars='index', var_name=None).drop('index', axis=1)
        df["BEHAVIOR"] = clf_name
        df.columns = ["EVENT", "FRAME", 'BEHAVIOR']
        return df.sort_values(by='FRAME', ascending=True)[['BEHAVIOR', "EVENT", "FRAME"]].reset_index(drop=True)

    check_file_exist_and_readable(file_path=file_path)
    check_str(name=f'{bento_file_reader.__name__} orient', value=orient, options=('index', 'columns'))
    if fps is not None:
        check_int(name=f'{bento_file_reader.__name__} fps', value=fps, min_value=1)
    _, video_name, _ = get_fn_ext(filepath=file_path)
    try:
        df = pd.read_csv(file_path, index_col=False, low_memory=False, header=None, encoding='utf-8').astype(str)
    except:
        df = pd.read_csv(file_path, index_col=False, low_memory=False, header=None, encoding='ascii').astype(str)
    idx = df[0].str.contains(pat='>', regex=True)
    idx = list(idx.index[idx])
    results = {}
    if len(idx) == 0:
        if raise_error:
            raise NoDataError(f"{file_path} is not a valid BENTO file. See the docs for expected file format.", source=bento_file_reader.__name__)
        else:
            ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BENTO", file_path=file_path, source=bento_file_reader.__name__, log_status=log_setting)
            return results
    idx.append(len(df))
    idx_mod = [0] + idx + [max(idx) + 1]
    clf_dfs = [df.iloc[idx_mod[n]:idx_mod[n + 1]] for n in range(len(idx_mod) - 1)][1:-1]
    for clf_idx in range(len(clf_dfs)):
        clf_df = clf_dfs[clf_idx].reset_index(drop=True)
        clf_name = clf_df.iloc[0, 0][1:]
        clf_df = clf_df.iloc[2:, 0].reset_index(drop=True)
        out_clf_df = clf_df.str.split('\t', expand=True)
        if len(out_clf_df.columns) > 3:
            if raise_error:
                raise InvalidFileTypeError(msg=f'SimBA found {len(out_clf_df.columns)} columns for file {file_path} and classifier {clf_name} when trying to split the data by tabs.')
            else:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BENTO", file_path=file_path, source=bento_file_reader.__name__, log_status=log_setting)
                return results
        numeric_check = list(out_clf_df.apply(lambda s: pd.to_numeric(s, errors='coerce').notnull().all()))
        if False in numeric_check:
            if raise_error:
                raise InvalidInputError(msg=f'SimBA found values in the annotation data for behavior {clf_name} in file {file_path} that could not be interpreted as numeric values (seconds or frame numbers)')
            else:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BENTO", file_path=file_path, source=bento_file_reader.__name__, log_status=log_setting)
                return results
        out_clf_df.columns = ['START', 'STOP', 'DURATION']
        out_clf_df = out_clf_df.astype(np.float32)
        int_check = np.array_equal(out_clf_df, out_clf_df.astype(int))
        if int_check:
            if orient == 'index':
                results[clf_name] = out_clf_df[['START', 'STOP']].astype(np.int32)
            else:
                results[clf_name] = _orient_columns_melt(df=out_clf_df)

        else:
            if fps is None:
                try:
                    fps_idx = df[0].str.contains(pat='Annotation framerate', regex=True)
                    fps_str = df.iloc[list(fps_idx.index[fps_idx])][0].values[0]
                    fps = float(fps_str.split(':')[1])
                except:
                    raise FrameRangeError(f'The annotations are in seconds and FPS was not passed. FPS could also not be read from the BENTO file', source=bento_file_reader.__name__)
            out_clf_df["START"] = out_clf_df["START"].astype(float) * fps
            out_clf_df["STOP"] = out_clf_df["STOP"].astype(float) * fps
            if orient == 'index':
                results[clf_name] = out_clf_df[['START', 'STOP']].astype(np.int32)
            else:
                results[clf_name] = _orient_columns_melt(df=out_clf_df)
    if save_path is None:
        return results
    else:
        write_pickle(data=results, save_path=save_path)

def _is_new_boris_version(pd_df: pd.DataFrame):
    """
    Check the format of a boris annotation file.

    In the new version, additional column names are present, while
    others have slightly different name. Here, we check for the presence
    of a column name present only in the newer version.

    :return: True if newer version
    """
    return "Media file name" in list(pd_df.columns)

def _find_cap_insensitive_name(target: str, values: List[str]) -> Union[None, str]:
    check_str(name=f'{_find_cap_insensitive_name.__name__} target', value=target)
    check_valid_lst(data=values, source=f'{_find_cap_insensitive_name.__name__} values', valid_dtypes=(str,), min_len=1)
    target_lower, values_lower = target.lower(), [x.lower() for x in values]
    if target_lower not in values_lower:
        return None
    else:
        return values[values_lower.index(target_lower)]
def read_boris_file(file_path: Union[str, os.PathLike],
                    fps: Optional[Union[int, float]] = None,
                    orient: Optional[Literal['index', 'columns']] = 'index',
                    save_path: Optional[Union[str, os.PathLike]] = None,
                    raise_error: Optional[bool] = False,
                    log_setting: Optional[bool] = False) -> Union[None, Dict[str, Dict[str, pd.DataFrame]]]:
    """
    Reads a BORIS behavioral annotation file, processes the data, and optionally saves the results to a file.

    :param Union[str, os.PathLike] file_path: The path to the BORIS file to be read. The file should be a CSV containing behavioral annotations.
    :param Optional[Union[int, float]] fps: Frames per second (FPS) to convert time annotations into frame numbers. If not provided, it will be extracted from the BORIS file if available.
    :param Optional[Literal['index', 'columns']] orient: Determines the orientation of the results. 'index' will organize data with start and stop times as indices, while 'columns' will store data in columns.
    :param Optional[Union[str, os.PathLike] save_path: The path where the processed results should be saved as a pickle file. If not provided, the results will be returned instead.
    :param Optional[bool] raise_error: Whether to raise errors if the file format or content is invalid. If False, warnings will be logged instead of raising exceptions.
    :param Optional[bool] log_setting: Whether to log warnings and errors.  This is relevant when `raise_error` is set to False.
    :return: If `save_path` is None, returns a dictionary where keys are behaviors and values are dataframes containing start and stop frames for each behavior. If `save_path` is provided, the results are saved and nothing is returned.
    """

    MEDIA_FILE_NAME = "Media file name"
    BEHAVIOR_TYPE = 'Behavior type'
    OBSERVATION_ID = "Observation id"
    TIME = "Time"
    FPS = 'FPS'
    EVENT = 'EVENT'
    BEHAVIOR = "Behavior"
    START = 'START'
    FRAME = 'FRAME'
    STOP = 'STOP'
    STATUS = "Status"
    FRAME_INDEX = 'Image index'
    MEDIA_FILE_PATH = "Media file path"

    check_file_exist_and_readable(file_path=file_path)
    if fps is not None:
        check_int(name=f'{read_boris_file.__name__} fps', min_value=1, value=fps)
    check_str(name=f'{read_boris_file.__name__} orient', value=orient, options=('index', 'columns'))
    if save_path is not None:
        check_if_dir_exists(in_dir=os.path.dirname(save_path))
    boris_df = pd.read_csv(file_path)
    if not _is_new_boris_version(boris_df):
        expected_headers = [TIME, MEDIA_FILE_PATH, BEHAVIOR, STATUS]
        if not OBSERVATION_ID in boris_df.columns:
            if raise_error:
                raise InvalidFileTypeError(msg=f'{file_path} is not a valid BORIS file', source=read_boris_file.__name__)
            else:

                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path, source=read_boris_file.__name__, log_status=log_setting)
                return {}
        start_idx = boris_df[boris_df[OBSERVATION_ID] == TIME].index.values
        if len(start_idx) != 1:
            if raise_error:
                raise InvalidFileTypeError(msg=f'{file_path} is not a valid BORIS file', source=read_boris_file.__name__)
            else:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path, source=read_boris_file.__name__, log_status=log_setting)
                return {}
        df = pd.read_csv(file_path, skiprows=range(0, int(start_idx + 1)))
    else:
        MEDIA_FILE_PATH, STATUS = MEDIA_FILE_NAME, BEHAVIOR_TYPE
        expected_headers = [TIME, MEDIA_FILE_PATH, BEHAVIOR, STATUS]
        df = pd.read_csv(file_path)
    check_valid_dataframe(df=df, source=f'{read_boris_file.__name__} {file_path}', required_fields=expected_headers)
    df = df.dropna(how='all').reset_index(drop=True)
    numeric_check = pd.to_numeric(df[TIME], errors='coerce').notnull().all()
    if not numeric_check:
        if raise_error:
            raise InvalidInputError(msg=f'SimBA found TIME DATA annotation in file {file_path} that could not be interpreted as numeric values (seconds or frame numbers)')
        else:
            ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path, source=read_boris_file.__name__, log_status=log_setting)
            return {}
    df[TIME] = df[TIME].astype(np.float32)
    media_file_names_in_file = df[MEDIA_FILE_PATH].unique()
    FRAME_INDEX = _find_cap_insensitive_name(target=FRAME_INDEX, values=list(df.columns))
    if fps is None and FRAME_INDEX is None:
        FPS = _find_cap_insensitive_name(target=FPS, values=list(df.columns))
        if not FPS in df.columns:
            if raise_error:
                raise FrameRangeError(f'The annotations are in seconds and FPS was not passed. FPS could also not be read from the BORIS file', source=read_boris_file.__name__)
            else:
                FrameRangeWarning(msg=f'The annotations are in seconds and FPS was not passed. FPS could also not be read from the BORIS file', source=read_boris_file.__name__)
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path, source=read_boris_file.__name__, log_status=log_setting)
                return {}
        if len(media_file_names_in_file) == 1:
            fps = df[FPS].iloc[0]
            check_float(name='fps', value=fps, min_value=10e-6, raise_error=True)
            fps = [float(fps)]
        else:
            print(media_file_names_in_file)
            fps_lst = df[FPS].iloc[0].split(';')
            fps = []
            for fps_value in fps_lst:
                check_float(name='fps', value=fps_value, min_value=10e-6, raise_error=True)
                fps.append(float(fps_value))
    if FRAME_INDEX is not None:
        expected_headers.append(FRAME_INDEX)
    df = df[expected_headers]
    results = {}
    for video_cnt, video_file_name in enumerate(media_file_names_in_file):
        video_name = get_fn_ext(filepath=video_file_name)[1]
        results[video_name] = {}
        video_df = df[df[MEDIA_FILE_PATH] == video_file_name].reset_index(drop=True)
        if FRAME_INDEX is None:
            video_df['FRAME'] = (video_df[TIME] * fps[video_cnt]).astype(int)
        else:
            video_df['FRAME'] = video_df[FRAME_INDEX]
        video_df = video_df.drop([TIME, MEDIA_FILE_PATH], axis=1)
        video_df = video_df.rename(columns={BEHAVIOR: 'BEHAVIOR', STATUS: EVENT})
        for clf in video_df['BEHAVIOR'].unique():
            video_clf_df = video_df[video_df['BEHAVIOR'] == clf].reset_index(drop=True)
            if orient == 'index':
                start_clf, stop_clf = video_clf_df[video_clf_df[EVENT] == START].reset_index(drop=True), video_clf_df[video_clf_df[EVENT] == STOP].reset_index(drop=True)
                start_clf = start_clf.rename(columns={FRAME: START}).drop([EVENT, 'BEHAVIOR'], axis=1)
                stop_clf = stop_clf.rename(columns={FRAME: STOP}).drop([EVENT], axis=1)
                if len(start_clf) != len(stop_clf):
                    if raise_error:
                        raise FrameRangeError(f'In file {file_path}, the number of start events ({len(start_clf)}) and stop events ({len(stop_clf)}) for behavior {clf} and video {video_name} is not equal', source=read_boris_file.__name__)
                    else:
                        FrameRangeWarning( msg=f'In file {file_path}, the number of start events ({len(start_clf)}) and stop events ({len(stop_clf)}) for behavior {clf} and video {video_name} is not equal', source=read_boris_file.__name__)
                        continue
                video_clf_df = pd.concat([start_clf, stop_clf], axis=1)[['BEHAVIOR', START, STOP]]
            results[video_name][clf] = video_clf_df
    if save_path is None:
        return results
    else:
        write_pickle(data=results, save_path=save_path)

# files = find_files_of_filetypes_in_directory(directory=r"C:\Users\sroni\Downloads\boris_files\boris_files", extensions=['.csv'])
# for file in files:
#     read_boris_file(file_path=file)

def img_stack_to_video(x: np.ndarray,
                       save_path: Union[str, os.PathLike],
                       fps: float,
                       gpu: Optional[bool] = False,
                       bitrate: Optional[int] = 5000) -> None:

    """
    Converts a NumPy image stack to a video file, with optional GPU acceleration and configurable bitrate.

    :param np.ndarray x: A NumPy array representing the image stack. The array should have shape (N, H, W) for greyscale or (N, H, W, 3) for RGB images, where N is the number of frames, H is the height, and W is the width.
    :param Union[str, os.PathLike] save_path: Path to the output video file where the video will be saved.
    :param float fps: Frames per second for the output video. Should be a positive floating-point number.
    :param Optional[bool] gpu: Whether to use GPU acceleration for encoding. If True, the video encoding will use NVIDIA's NVENC encoder. Defaults to False.
    :param Optional[int] bitrate: Bitrate for the video encoding in kilobits per second (kbps). Should be an integer between 1000 and 35000. Defaults to 5000.
    :return: None
    """

    check_if_dir_exists(in_dir=os.path.dirname(save_path), source=img_stack_to_video.__name__)
    check_valid_array(data=x, source=img_stack_to_video.__name__, accepted_ndims=(3, 4))
    check_float(name=f'{img_stack_to_video.__name__} fps', value=fps, min_value=10e-6)
    check_valid_boolean(value=gpu, source=img_stack_to_video.__name__)
    check_int(name=f'{img_stack_to_video.__name__} bitrate', value=bitrate, min_value=1000, max_value=35000)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError('No GPU found but GPU flag is True')
    is_color = (x.ndim == 4 and x.shape[3] == 3)
    N, H, W = x.shape[:3]
    pix_fmt = 'gray' if not is_color else 'rgb24'
    timer = SimbaTimer(start=True)
    vcodec = 'mpeg4'
    if gpu:
        vcodec = 'h264_nvenc'

    cmd = [
        'ffmpeg',
        '-loglevel', 'error',
        '-stats',
        '-hide_banner',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-s', f'{W}x{H}',
        '-pix_fmt', pix_fmt,
        '-r', str(fps),
        '-i', '-',
        '-an',
        '-vcodec', f'{vcodec}',
        '-b:v', f'{bitrate}k',
        save_path
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for frame in x:
        if frame.dtype != np.uint8:
            frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8)
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()
    timer.stop_timer()
    stdout_success(msg=f'Video complete. Saved at {save_path}', elapsed_time=timer.elapsed_time_str)


def _b64_to_arr(img_b64) -> np.ndarray:
    """
    Helper to convert byte string (e.g., from labelme, to image in numpy format
    """
    f = io.BytesIO()
    f.write(base64.b64decode(img_b64))
    img_arr = np.array(Image.open(f))
    return img_arr


def labelme_to_dlc(labelme_dir: Union[str, os.PathLike],
                   scorer: Optional[str] = 'SN',
                   save_dir: Optional[Union[str, os.PathLike]] = None) -> None:
    """
    Convert labels from labelme format to DLC format.

    :param Union[str, os.PathLike] labelme_dir: Directory with labelme json files.
    :param Optional[str] scorer: Name of the scorer (anticipated by DLC as header)
    :param Optional[Union[str, os.PathLike]] save_dir: Directory where to save the DLC annotations. If None, then same directory as labelme_dir with `_dlc_annotations` suffix.
    :return: None

    :example:
    >>> labelme_dir = r'D:\ts_annotations'
    >>> labelme_to_dlc(labelme_dir=labelme_dir)
    """

    check_if_dir_exists(in_dir=labelme_dir)
    annotation_paths = find_files_of_filetypes_in_directory(directory=labelme_dir, extensions=['.json'],
                                                            raise_error=True)
    results_dict = {}
    images = {}
    for annot_path in annotation_paths:
        with open(annot_path) as f:
            annot_data = json.load(f)
        check_if_keys_exist_in_dict(data=annot_data, key=['shapes', 'imageData', 'imagePath'], name=annot_path)
        img_name = os.path.basename(annot_data['imagePath'])
        images[img_name] = _b64_to_arr(annot_data['imageData'])
        for bp_data in annot_data['shapes']:
            check_if_keys_exist_in_dict(data=bp_data, key=['label', 'points'], name=annot_path)
            point_x, point_y = bp_data['points'][0][0], bp_data['points'][0][1]
            lbl = bp_data['label']
            id = os.path.join('labeled-data', os.path.basename(labelme_dir), img_name)
            if id not in results_dict.keys():
                results_dict[id] = {f'{lbl}': {'x': point_x, 'y': point_y}}
            else:
                results_dict[id].update({f'{lbl}': {'x': point_x, 'y': point_y}})

    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(labelme_dir), os.path.basename(labelme_dir) + '_dlc_annotations')
        if not os.path.isdir(save_dir): os.makedirs(save_dir)

    bp_names = set()
    for img, bp in results_dict.items(): bp_names.update(set(bp.keys()))
    col_names = list(itertools.product(*[[scorer], bp_names, ['x', 'y']]))
    columns = pd.MultiIndex.from_tuples(col_names)
    results = pd.DataFrame(columns=columns)
    results.columns.names = ['scorer', 'bodyparts', 'coords']
    for img, bp_data in results_dict.items():
        for bp_name, bp_cords in bp_data.items():
            results.at[img, (scorer, bp_name, 'x')] = bp_cords['x']
            results.at[img, (scorer, bp_name, 'y')] = bp_cords['y']

    for img_name, img in images.items():
        img_save_path = os.path.join(save_dir, img_name)
        cv2.imwrite(img_save_path, img)
    save_path = os.path.join(save_dir, f'CollectedData_{scorer}.csv')
    results.to_csv(save_path)


def read_shap_feature_categories_csv() -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """ Helper to read feature names and their categories used for binning and visualizing shapely values"""
    feature_categories_csv_path = os.path.join(SIMBA_DIR, Paths.SIMBA_SHAP_CATEGORIES_PATH.value)
    check_file_exist_and_readable(file_path=feature_categories_csv_path)
    feature_categories_df = pd.read_csv(feature_categories_csv_path, header=[0, 1])
    x_names = [list(x) for x in list(feature_categories_df.values)]
    x_names = [item for sublist in x_names for item in sublist]
    x_names = [x for x in x_names if not x is np.nan]
    x_cat_names, time_bin_names = set(list(feature_categories_df.columns.levels[0])), set(list(feature_categories_df.columns.levels[1]))
    return (feature_categories_df, x_names, x_cat_names, time_bin_names)


def read_shap_img_paths():
    """ Helper to read in the images used to create the SHAP visualization"""
    shap_img_path = os.path.join(SIMBA_DIR, Paths.SIMBA_SHAP_IMG_PATH.value)
    check_if_dir_exists(in_dir=shap_img_path)
    scale_img_paths = {"baseline_scale": os.path.join(shap_img_path, "baseline_scale.jpg"),
                       "small_arrow": os.path.join(shap_img_path, "down_arrow.jpg"),
                       "side_scale": os.path.join(shap_img_path, "side_scale.jpg"),
                       "color_bar": os.path.join(shap_img_path, "color_bar.jpg")}
    for k, v in scale_img_paths.items(): check_file_exist_and_readable(file_path=v)
    category_img_paths = {"Animal distances": os.path.join(shap_img_path, "animal_distances.jpg"),
                          "Intruder movement": os.path.join(shap_img_path, "intruder_movement.jpg"),
                          "Resident+intruder movement": os.path.join(shap_img_path, "resident_intruder_movement.jpg"),
                          "Resident movement": os.path.join(shap_img_path, "resident_movement.jpg"),
                          "Intruder shape": os.path.join(shap_img_path, "intruder_shape.jpg"),
                          "Resident+intruder shape": os.path.join(shap_img_path, "resident_intruder_shape.jpg"),
                          "Resident shape": os.path.join(shap_img_path, "resident_shape.jpg")}
    for k, v in category_img_paths.items(): check_file_exist_and_readable(file_path=v)

    return scale_img_paths, category_img_paths

def get_memory_usage_array(x: np.ndarray) -> Dict[str, float]:
    """
    Calculates the memory usage of a NumPy array in bytes, megabytes, and gigabytes.

    :param x: A NumPy array for which memory usage will be calculated. It should be a valid NumPy array with a defined size and dtype.
    :return: A dictionary with memory usage information, containing the following keys: - "bytes": Memory usage in bytes. - "megabytes": Memory usage in megabytes. - "gigabytes": Memory usage in gigabytes.
    """

    check_valid_array(data=x, source=get_memory_usage_array.__name__)
    results = {}
    mb = int(x.size * x.itemsize / (1024 ** 2))
    results["bytes"] = int(mb * 1000)
    results["megabytes"] = mb
    results["gigabytes"] = int(mb / 1000)
    return results

def read_json(x: Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]], encoding: str = 'utf-8') -> dict:
    """
    Reads one or multiple JSON files from disk and returns their contents as a dictionary.

    :param Union[Union[str, os.PathLike], List[Union[str, os.PathLike]]] x: A path or list of paths to JSON files on disk.
    :return: A dictionary with JSON data. If multiple files are provided, keys are derived from filenames.
    :rtype: dict
    """
    try:
        if isinstance(x, (str, os.PathLike)):
            check_file_exist_and_readable(x)
            with open(x, 'r', encoding=encoding) as file:
                results = json.load(file)
        elif isinstance(x, (list, tuple,)):
            results = {}
            for file_path in x:
                check_file_exist_and_readable(file_path)
                data_name = get_fn_ext(filepath=file_path)[1]
                results[data_name] = json.load(x)
        else:
            raise InvalidInputError(msg='x is not a valid iterable of paths or a valid path', source=read_json.__name__)

        return  results

    except Exception as e:
        raise InvalidFileTypeError(f"Unexpected error reading json: {e}", source=read_json.__name__)



def save_json(data: dict, filepath: Union[str, os.PathLike], encoding: str = 'utf-8') -> None:
    """
    Saves a dictionary as a JSON file to the specified filepath.

    :param dict data: Dictionary containing data to save.
    :param Union[str, os.PathLike] filepath: Path where the JSON file should be saved.
    """
    try:
        with open(filepath, 'w', encoding=encoding) as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        raise IOError(f"Error saving JSON file to {filepath}: {e}")


def df_to_xlsx_sheet(xlsx_path: Union[str, os.PathLike],
                     df: pd.DataFrame,
                     sheet_name: str,
                     create_file: bool = True) -> None:

    """
    Append a dataframe as a new sheet in an MS Excel file.

    :param Union[str, os.PathLike] xlsx_path: Path to an existing MS Excel file on disk.
    :param pd.DataFrame df: A dataframe to save as a sheet in the MS Excel file.
    :param str sheet_name: Name of the sheet to save the dataframe under.
    """

    check_valid_boolean(value=[create_file], source=df_to_xlsx_sheet.__name__, raise_error=True)
    if not os.path.isfile(xlsx_path):
        if not create_file:
            raise NoFilesFoundError(msg=f'{xlsx_path} is not a valid file path')
        else:
            create_empty_xlsx_file(xlsx_path=xlsx_path)
    check_valid_dataframe(df=df, source=df_to_xlsx_sheet.__name__)
    check_str(name=f'{df_to_xlsx_sheet} sheet_name', value=sheet_name, allow_blank=False)
    excel_file = pd.ExcelFile(xlsx_path)
    if sheet_name in excel_file.sheet_names:
        raise DuplicationError(msg=f'Sheet name {sheet_name} already exist in file {xlsx_path} with sheetnames: {excel_file.sheet_names}', source=df_to_xlsx_sheet.__name__)
    with pd.ExcelWriter(xlsx_path, mode='a') as writer:
        df.to_excel(writer, sheet_name=sheet_name)

def create_empty_xlsx_file(xlsx_path: Union[str, os.PathLike]):
    """
    Create an empty MS Excel file.
    :param Union[str, os.PathLike] xlsx_path: Path where to save MS Excel file on disk.
    """
    check_if_dir_exists(in_dir=os.path.dirname(xlsx_path))
    pd.DataFrame().to_excel(xlsx_path, index=False)

def get_desktop_path(raise_error: bool = False):
    """ Get the path to the user desktop directory """
    desktop_path_option_1 = os.path.join(os.path.expanduser("~"), "Desktop")
    if not os.path.isdir(desktop_path_option_1):
        desktop_path_option_2 = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop")
        if os.path.isdir(desktop_path_option_2):
            return desktop_path_option_2
        else:
            if raise_error:
                raise InvalidFilepathError(msg=f'{desktop_path_option_1} OR {desktop_path_option_2} are not valid directories')
            else:
                return None
    else:
        return desktop_path_option_1

def get_downloads_path(raise_error: bool = False):
    """ Get the path to the user downloads directory """
    downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
    if not os.path.isdir(downloads_path):
        if raise_error:
            raise InvalidFilepathError(msg=f'{downloads_path} is not a valid directory')
        else:
            return None
    else:
        return downloads_path


def _read_img_batch_from_video_helper(frm_idx: np.ndarray, video_path: Union[str, os.PathLike], greyscale: bool, verbose: bool, black_and_white: bool, clahe: bool):
    """Multiprocess helper used by read_img_batch_from_video to read in images from video file."""
    start_idx, end_frm, current_frm = frm_idx[0], frm_idx[-1] + 1, frm_idx[0]
    results = {}
    video_meta_data = get_video_meta_data(video_path=video_path)
    cap = cv2.VideoCapture(video_path)
    cap.set(1, current_frm)
    while current_frm < end_frm:
        if verbose:
            print(f'Reading frame {current_frm}/{video_meta_data["frame_count"]} ({video_meta_data["video_name"]})...')
        img = cap.read()[1]
        if img is not None:
            if greyscale or black_and_white or clahe:
                if len(img.shape) != 2:
                    img = (0.07 * img[:, :, 2] + 0.72 * img[:, :, 1] + 0.21 * img[:, :, 0]).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            if black_and_white:
                img = np.where(img > 127, 255, 0).astype(np.uint8)
            if clahe:
                img = img_array_to_clahe(img=img)
        else:
            if greyscale or black_and_white:
                img = np.full(shape=(video_meta_data['height'], video_meta_data['width']), fill_value=0, dtype=np.uint8)
            else:
                img = np.full(shape=(video_meta_data['height'], video_meta_data['width'], 3), fill_value=0, dtype=np.uint8)
        results[current_frm] = img
        current_frm += 1
    return results

def read_img_batch_from_video(video_path: Union[str, os.PathLike],
                              start_frm: Optional[int] = None,
                              end_frm: Optional[int] = None,
                              greyscale: bool = False,
                              black_and_white: bool = False,
                              clahe: bool = False,
                              core_cnt: int = -1,
                              verbose: bool = False) -> Dict[int, np.ndarray]:
    """
    Read a batch of frames from a video file. This method reads frames from a specified range of frames within a video file using multiprocessing.
    .. seealso::
       For GPU acceleration, see :func:`simba.utils.read_write.read_img_batch_from_video_gpu`

    .. note::
      When black-and-white videos are saved as MP4, there can be some small errors in pixel values during compression. A video with only (0, 255) pixel values therefore gets other pixel values, around 0 and 255, when read in again.
      If you expect that the video you are reading in is black and white, set ``black_and_white`` to True to round any of these wonly value sto 0 and 255.

    :param Union[str, os.PathLike] video_path: Path to the video file.
    :param int start_frm: Starting frame index.
    :param int end_frm: Ending frame index.
    :param Optionalint] core_cnt: Number of CPU cores to use for parallel processing. Default is -1, indicating using all available cores.
    :param Optional[bool] greyscale: If True, reads the images as greyscale. If False, then as original color scale. Default: False.
    :param bool black_and_white: If True, returns the images in black and white. Default False.
    :param bool clahe: If True, returns clahe enhanced images.
    :returns: A dictionary containing frame indices as keys and corresponding frame arrays as values.
    :rtype: Dict[int, np.ndarray]

    :example:
    >>> read_img_batch_from_video(video_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/videos/Together_1.avi', start_frm=0, end_frm=50)
    """

    if platform.system() == "Darwin":
        if not multiprocessing.get_start_method(allow_none=True):
            multiprocessing.set_start_method("fork", force=True)
    check_file_exist_and_readable(file_path=video_path)
    video_meta_data = get_video_meta_data(video_path=video_path)
    if start_frm is not None:
        check_int(name=read_img_batch_from_video.__name__,value=start_frm, min_value=0,max_value=video_meta_data["frame_count"])
    else:
        start_frm = 0
    if end_frm is not None:
        check_int(name=read_img_batch_from_video.__name__, value=end_frm, min_value=start_frm+1, max_value=video_meta_data["frame_count"])
    else:
        end_frm = video_meta_data["frame_count"] -1
    check_int(name=read_img_batch_from_video.__name__, value=core_cnt, min_value=-1)
    check_valid_boolean(value=[greyscale, black_and_white], source=f'{read_img_batch_from_video.__name__} greyscale black_and_white')
    check_valid_boolean(value=clahe, source=f'{read_img_batch_from_video.__name__} clahe')
    if core_cnt < 0:
        core_cnt = multiprocessing.cpu_count()
    if end_frm <= start_frm:
        FrameRangeError(msg=f"Start frame ({start_frm}) has to be before end frame ({end_frm})", source=read_img_batch_from_video.__name__)
    frm_lst = np.array_split(np.arange(start_frm, end_frm + 1), core_cnt)
    results = {}
    with multiprocessing.Pool(core_cnt, maxtasksperchild=Defaults.LARGE_MAX_TASK_PER_CHILD.value) as pool:
        constants = functools.partial(_read_img_batch_from_video_helper,
                                      video_path=video_path,
                                      greyscale=greyscale,
                                      black_and_white=black_and_white,
                                      clahe=clahe,
                                      verbose=verbose)
        for cnt, result in enumerate(pool.imap(constants, frm_lst, chunksize=1)):
            results.update(result)
    pool.join()
    pool.terminate()

    return results



def read_df_array(df: pd.DataFrame, column: str):
    """
    Convert string representations of 2D arrays in a DataFrame column to actual numpy arrays.

    :param pd.DataFrame df: The DataFrame containing the column.
    :param str column: The name of the column with string representations of 2D arrays.
    :returns: A list of numpy arrays, each corresponding to an entry in the specified column.
    :rtype: List[np.ndarray]
    """

    def _col_to_arrays(s):
        # Remove newline characters and normalize spaces
        s = ' '.join(s.split())

        # Fix missing commas between sublists: "[1 2][3 4]" → "[1, 2], [3, 4]"
        s = s.replace('][', '], [')

        # Replace spaces inside brackets with commas: "[1 2]" → "[1, 2]"
        s = s.replace('[ ', '[').replace(' ]', ']').replace(' ', ', ')

        try:
            return np.array(literal_eval(s))  # Convert string to actual NumPy array
        except (SyntaxError, ValueError):
            raise ValueError(f"Invalid array format in column '{column}': {s}")

    df[column] = df[column].apply(_col_to_arrays)  # Convert in-place
    return df

def read_sleap_csv(file_path: Union[str, os.PathLike]) -> Tuple[pd.DataFrame, list, list]:
    """
    Reads and validates a SLEAP-exported CSV file containing tracking data.

    :param  Union[str, os.PathLike] file_path: Path to the SLEAP CSV file.
    :returns: Tuple with (i) The validated and cleaned DataFrame, (ii) A list of unique body part names, (iii) A flattened list of coordinate column names for each body part (e.g., ['nose.x', 'nose.y', ...]).
    :rtype: Tuple[pd.DataFrame, list, list]
    """
    REQUIRED_COLUMNS = ['track', 'frame_idx', 'instance.score']
    check_file_exist_and_readable(file_path=file_path)
    data_df = pd.read_csv(file_path)
    check_valid_dataframe(df=data_df, source=read_sleap_csv.__name__, required_fields=REQUIRED_COLUMNS)
    check_valid_dataframe(df=data_df.drop(REQUIRED_COLUMNS[0], axis=1), source=read_sleap_csv.__name__, valid_dtypes=Formats.NUMERIC_DTYPES.value)
    data_df[REQUIRED_COLUMNS[0]] = data_df[REQUIRED_COLUMNS[0]].astype(str).str.replace(r"[^\d.]+", "", regex=True).astype(int)
    headers = list(data_df.drop(REQUIRED_COLUMNS, axis=1).columns)
    bp_names = np.unique([x.split('.', 1)[0] for x in headers])
    bp_headers = [(f'{x}.x', f'{x}.y') for x in bp_names]

    return data_df, bp_names, [i for t in bp_headers for i in t]




def recursive_file_search(directory: Union[str, os.PathLike],
                          extensions: Union[str, List[str]],
                          case_sensitive: bool = False,
                          substrings: Optional[Union[str, List[str]]] = None,
                          skip_substrings: Optional[Union[str, List[str]]] = None,
                          raise_error: bool = True,
                          as_dict: bool = False) -> Union[List[str], Dict[str, str]]:
    """
    Recursively search for files in a directory and all subdirectories that:
    - Contain any of the given substrings in their filename
    - Have one of the specified file extensions

    :param directory: Directory to start the search from.
    :param substrings: A substring or list of substrings to match in filenames. If None, all files with the specified extensions will be returned.
    :param substrings: A substring or list of substrings to match. If filename contains this substring, it will be removed. If None, all files with the specified extensions will be returned.
    :param extensions: A file extension or list of allowed extensions (with or without dot).
    :param case_sensitive: If True, substring match is case-sensitive. Default False.
    :param raise_error: If True, raise an error if no matches are found.
    :param as_dict: If True, return a dictionary where rge file names ar ekeys and filepaths ar the values.
    :return: List of matching file paths.
    """

    check_if_dir_exists(in_dir=directory)
    if substrings is not None:
        if isinstance(substrings, str):
            substrings = [substrings]
        check_valid_lst(data=substrings, valid_dtypes=(str,), min_len=1, raise_error=True)

    if skip_substrings is not None:
        if isinstance(skip_substrings, str):
            skip_substrings = [skip_substrings]
        check_valid_lst(data=skip_substrings, valid_dtypes=(str,), min_len=1, raise_error=True)

    if isinstance(extensions, str): extensions = [extensions]
    if isinstance(extensions, (tuple,)): extensions = list(extensions)
    check_valid_lst(data=extensions, valid_dtypes=(str,), min_len=1, raise_error=True, source=f'{recursive_file_search.__name__} extensions')
    check_valid_boolean(value=case_sensitive, source=f'{recursive_file_search.__name__} case_sensitive', raise_error=True)
    check_valid_boolean(value=raise_error, source=f'{recursive_file_search.__name__} raise_error', raise_error=True)
    check_valid_boolean(value=as_dict, source=f'{recursive_file_search.__name__} as_dict', raise_error=True)

    extensions = [ext.lower().lstrip('.') for ext in extensions]
    if not case_sensitive and substrings is not None:
        substrings = [s.lower() for s in substrings]

    results = []
    for root, _, files in os.walk(directory):
        for f in files:
            _, name, ext = get_fn_ext(filepath=f)
            ext = ext.lstrip('.').lower()
            if ext in extensions:
                if substrings is not None:
                    match_substr = any(s in f if case_sensitive else s in f.lower() for s in substrings)
                else:
                    match_substr = True
                if skip_substrings is not None:
                    skip_match_substr = any(s in f if case_sensitive else s in f.lower() for s in skip_substrings)
                else:
                    skip_match_substr = False
                if ext in extensions and match_substr and not skip_match_substr:
                    results.append(os.path.join(root, f))

    if not results and raise_error:
        raise NoFilesFoundError(msg=f'No files with extensions {extensions} and substrings {substrings} found in {directory}', source=recursive_file_search.__name__)

    if as_dict:
        results = {get_fn_ext(filepath=x)[1]: x for x in results}

    return results


def read_sleap_h5(file_path: Union[str, os.PathLike]) -> pd.DataFrame:
    """
     Helper to read in SLEAP H5 file in format expected by SimBA
     """

    EXPECTED_KEYS = ["tracks", "point_scores", "node_names", "track_names"]
    check_file_exist_and_readable(file_path=file_path)
    with h5py.File(file_path, "r") as f:
        missing_keys = [x for x in EXPECTED_KEYS if not x in list(f.keys())]
        if missing_keys:
            raise InvalidFileTypeError(msg=f'{file_path} is not a valid SLEAP H5 file. Missing expected keys: {missing_keys}')
        tracks = f["tracks"][:].T
        point_scores = f["point_scores"][:].T

    csv_rows = []
    n_frames, n_nodes, _, n_tracks = tracks.shape
    for frame_ind in range(n_frames):
        csv_row = []
        for track_ind in range(n_tracks):
            for node_ind in range(n_nodes):
                for xyp in range(3):
                    if xyp == 0 or xyp == 1:
                        data = tracks[frame_ind, node_ind, xyp, track_ind]
                    else:
                        data = point_scores[frame_ind, node_ind, track_ind]
                    csv_row.append(f"{data:.3f}")
        csv_rows.append(" ".join(csv_row))
    csv_rows = "\n".join(csv_rows)
    data_df = pd.read_csv(io.StringIO(csv_rows), delim_whitespace=True, header=None).fillna(0)
    return data_df

def img_array_to_clahe(img: np.ndarray,
                       clip_limit: int = 2,
                       tile_grid_size: Tuple[int, int] = (16, 16)) -> np.ndarray:
    check_if_valid_img(data=img, source=img_array_to_clahe.__name__, raise_error=True)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size).apply(img)


def read_sys_env():
    env = {}
    env[ENV_VARS.PRINT_EMOJIS.value] = str_2_bool(os.getenv(ENV_VARS.PRINT_EMOJIS.value, "True"))
    env[ENV_VARS.UNSUPERVISED_INTERFACE.value] = str_2_bool(os.getenv(ENV_VARS.UNSUPERVISED_INTERFACE.value, "False"))
    env[ENV_VARS.NUMBA_PRECOMPILE.value] = str_2_bool(os.getenv(ENV_VARS.NUMBA_PRECOMPILE.value, "False"))
    env[ENV_VARS.CUML.value] = str_2_bool(os.getenv(ENV_VARS.CUML.value, "False"))
    return env


def get_recent_projects_paths(max: int = 10) -> List[str]:
    file_path = os.path.join(SIMBA_DIR, Paths.RECENT_PROJECTS_PATHS.value)
    if not os.path.isfile(file_path):
        return []
    try:
        with open(file_path, "r") as file:
            project_paths = [line.strip() for line in file if line.strip()]
            project_paths = list(set(project_paths))
            return [x for x in project_paths if os.path.isfile(x)][:max]
    except:
        return []


def write_to_recent_project_paths(config_path: Union[str, os.PathLike]):
    file_path = os.path.join(SIMBA_DIR, Paths.RECENT_PROJECTS_PATHS.value)
    existing_paths = get_recent_projects_paths()
    if os.path.isfile(config_path) and (config_path not in existing_paths):
        try:
            with open(file_path, "r") as f:
                existing_content = f.read()
            with open(file_path, "w") as f:
                f.write(config_path + "\n" + existing_content)
        except:
            pass
    else:
        pass


def read_facemap_h5(file_path: Union[str, os.PathLike]) -> pd.DataFrame:
    """
    Convert FaceMap pose-estimation data to pandas Dataframe format.

    .. seealso::
       See FaceMap GitHub repository for expected H5 file format: https://github.com/MouseLand/facemap

    :param Union[str, os.PathLike] file_path: Path to facemap data file in .h5 format.
    :return: FaceMap pose-estimation data in DataFrame format.
    :rtype: pd.DataFrame
    """

    BODYPARTS = ["eye(back)", "eye(bottom)", "eye(front)", "eye(top)", "lowerlip", "mouth", "nose(bottom)", "nose(r)", "nose(tip)", "nose(top)", "nosebridge", "paw", "whisker(I)", "whisker(III)", "whisker(II)"]
    FACEMAP = 'Facemap'
    COORD_KEYS = ['x', 'y', 'likelihood']

    check_file_exist_and_readable(file_path=file_path, raise_error=True)
    pose_data = h5py.File(file_path, "r")
    pose_keys = list(pose_data.keys())
    if not FACEMAP in pose_keys:
        raise InvalidInputError(msg=f'The file {file_path} does not contain the key {FACEMAP}', source=read_facemap_h5.__name__)
    pose_data = pose_data[FACEMAP]
    missing_bp_keys = [x for x in BODYPARTS if x not in pose_data.keys()]
    if len(missing_bp_keys) > 0:
        raise InvalidInputError(msg=f'The file {file_path} are missing the expected body-part keys: {missing_bp_keys}', source=read_facemap_h5.__name__)

    results = pd.DataFrame()
    for bodypart in BODYPARTS:
        bp_data = pose_data[bodypart]
        missing_bp_cord_keys = [x for x in COORD_KEYS if x not in bp_data.keys()]
        if len(missing_bp_cord_keys) > 0:
            raise InvalidInputError(msg=f'The body-part {bodypart} in file {file_path} are missing the expected data keys: {missing_bp_cord_keys}', source=read_facemap_h5.__name__)
        bp_x = pose_data[bodypart][COORD_KEYS[0]][:]
        bp_y = pose_data[bodypart][COORD_KEYS[1]][:]
        bp_p = pose_data[bodypart][COORD_KEYS[2]][:]
        check_valid_array(data=bp_x, source=f'{file_path} {bodypart} x', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value)
        check_valid_array(data=bp_y, source=f'{file_path} {bodypart} y', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[bp_x.shape[0]])
        check_valid_array(data=bp_p, source=f'{file_path} {bodypart} p', accepted_ndims=(1,), accepted_dtypes=Formats.NUMERIC_DTYPES.value, accepted_axis_0_shape=[bp_x.shape[0]], min_value=0, max_value=1.0)
        results[f'{bodypart}_x'] = bp_x.astype(np.int32)
        results[f'{bodypart}_y'] = bp_y.astype(np.int32)
        results[f'{bodypart}_p'] = bp_p.astype(np.float32)
    return results





#concatenate_videos_in_folder(in_folder=r'C:\troubleshooting\RAT_NOR\project_folder\frames\output\path_plots\03152021_NOB_IOT_8', save_path=r"C:\troubleshooting\RAT_NOR\project_folder\frames\output\path_plots\new.mp4", remove_splits=False)