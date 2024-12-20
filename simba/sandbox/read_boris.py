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
import webbrowser
from configparser import ConfigParser
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from typing import Literal
except:
    from typing_extensions import Literal

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

from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_filepath_list_is_empty,
                                check_if_string_value_is_valid_video_timestamp,
                                check_instance, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_valid_array, check_valid_boolean,
                                check_valid_dataframe, check_valid_lst)
from simba.utils.enums import ConfigKey, Dtypes, Formats, Keys, Options
from simba.utils.errors import (DataHeaderError, DuplicationError,
                                FFMPEGCodecGPUError, FileExistError,
                                FrameRangeError, IntegerError,
                                InvalidFilepathError, InvalidFileTypeError,
                                InvalidInputError, InvalidVideoFileError,
                                MissingProjectConfigEntryError, NoDataError,
                                NoFilesFoundError, NotDirectoryError,
                                ParametersFileError, PermissionError)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import get_fn_ext, write_pickle
from simba.utils.warnings import (
    FileExistWarning, FrameRangeWarning, InvalidValueWarning,
    NoDataFoundWarning, NoFileFoundWarning,
    ThirdPartyAnnotationsInvalidFileFormatWarning)

# from simba.utils.keyboard_listener import KeyboardListener


PARSE_OPTIONS = csv.ParseOptions(delimiter=",")
READ_OPTIONS = csv.ReadOptions(encoding="utf8")

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
                raise InvalidFileTypeError(msg=f'{file_path} is not a valid BORIS file',
                                           source=read_boris_file.__name__)
            else:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path,
                                                              source=read_boris_file.__name__, log_status=log_setting)
                return {}
        start_idx = boris_df[boris_df[OBSERVATION_ID] == TIME].index.values
        if len(start_idx) != 1:
            if raise_error:
                raise InvalidFileTypeError(msg=f'{file_path} is not a valid BORIS file',
                                           source=read_boris_file.__name__)
            else:
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path,
                                                              source=read_boris_file.__name__, log_status=log_setting)
                return {}
        df = pd.read_csv(file_path, skiprows=range(0, int(start_idx + 1)))
    else:
        MEDIA_FILE_PATH, STATUS = MEDIA_FILE_NAME, BEHAVIOR_TYPE
        expected_headers = [TIME, MEDIA_FILE_PATH, BEHAVIOR, STATUS]
        df = pd.read_csv(file_path)
    check_valid_dataframe(df=df, source=f'{read_boris_file.__name__} {file_path}', required_fields=expected_headers)
    numeric_check = pd.to_numeric(df[TIME], errors='coerce').notnull().all()
    if not numeric_check:
        if raise_error:
            raise InvalidInputError(
                msg=f'SimBA found TIME DATA annotation in file {file_path} that could not be interpreted as numeric values (seconds or frame numbers)')
        else:
            ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path, source=read_boris_file.__name__, log_status=log_setting)
            return {}
    df[TIME] = df[TIME].astype(np.float32)
    media_file_names_in_file = df[MEDIA_FILE_PATH].unique()
    FRAME_INDEX = _find_cap_insensitive_name(target=FRAME_INDEX, values=list(df.columns))
    if fps is None:
        FPS = _find_cap_insensitive_name(target=FPS, values=list(df.columns))
        if not FPS in df.columns:
            if raise_error:
                raise FrameRangeError(
                    f'The annotations are in seconds and FPS was not passed. FPS could also not be read from the BORIS file',
                    source=read_boris_file.__name__)
            else:
                FrameRangeWarning(
                    msg=f'The annotations are in seconds and FPS was not passed. FPS could also not be read from the BORIS file',
                    source=read_boris_file.__name__)
                ThirdPartyAnnotationsInvalidFileFormatWarning(annotation_app="BORIS", file_path=file_path, source=read_boris_file.__name__, log_status=log_setting)
                return {}
        if len(media_file_names_in_file) == 1:
            fps = df[FPS].iloc[0]
            check_float(name='fps', value=fps, min_value=10e-6, raise_error=True)
            fps = [float(fps)]
        else:
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
        video_fps = fps[video_cnt]
        video_df = df[df[MEDIA_FILE_PATH] == video_file_name].reset_index(drop=True)
        if FRAME_INDEX is None:
            video_df['FRAME'] = (video_df[TIME] * video_fps).astype(int)
        else:
            video_df['FRAME'] = video_df[FRAME_INDEX]
        video_df = video_df.drop([TIME, MEDIA_FILE_PATH], axis=1)
        video_df = video_df.rename(columns={BEHAVIOR: 'BEHAVIOR', STATUS: EVENT})
        for clf in video_df['BEHAVIOR'].unique():
            video_clf_df = video_df[video_df['BEHAVIOR'] == clf].reset_index(drop=True)
            if orient == 'index':
                start_clf, stop_clf = video_clf_df[video_clf_df[EVENT] == START].reset_index(drop=True), video_clf_df[
                    video_clf_df[EVENT] == STOP].reset_index(drop=True)
                start_clf = start_clf.rename(columns={FRAME: START}).drop([EVENT, 'BEHAVIOR'], axis=1)
                stop_clf = stop_clf.rename(columns={FRAME: STOP}).drop([EVENT], axis=1)
                if len(start_clf) != len(stop_clf):
                    if raise_error:
                        raise FrameRangeError(
                            f'In file {file_path}, the number of start events ({len(start_clf)}) and stop events ({len(stop_clf)}) for behavior {clf} and video {video_name} is not equal',
                            source=read_boris_file.__name__)
                    else:
                        FrameRangeWarning(
                            msg=f'In file {file_path}, the number of start events ({len(start_clf)}) and stop events ({len(stop_clf)}) for behavior {clf} and video {video_name} is not equal',
                            source=read_boris_file.__name__)
                        return results
                video_clf_df = pd.concat([start_clf, stop_clf], axis=1)[['BEHAVIOR', START, STOP]]
            results[video_name][clf] = video_clf_df
    if save_path is None:
        return results
    else:
        write_pickle(data=results, save_path=save_path)

read_boris_file(file_path=r"C:\troubleshooting\boris_test\project_folder\boris_files\tabular.trial.csv")
