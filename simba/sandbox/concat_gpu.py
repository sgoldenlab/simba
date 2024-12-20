__author__ = "Simon Nilsson"

import base64
import configparser
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
from configparser import ConfigParser
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from PIL import Image

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
                                check_if_keys_exist_in_dict,
                                check_if_string_value_is_valid_video_timestamp,
                                check_if_valid_rgb_tuple, check_instance,
                                check_int, check_nvidea_gpu_available,
                                check_str, check_valid_array,
                                check_valid_boolean, check_valid_dataframe,
                                check_valid_lst, is_video_color)
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
from simba.utils.read_write import (get_fn_ext, get_video_meta_data,
                                    remove_a_folder)
from simba.utils.warnings import (
    FileExistWarning, FrameRangeWarning, InvalidValueWarning,
    NoDataFoundWarning, NoFileFoundWarning,
    ThirdPartyAnnotationsInvalidFileFormatWarning)


def concatenate_videos_in_folder(in_folder: Union[str, os.PathLike],
                                 save_path: Union[str, os.PathLike],
                                 file_paths: Optional[List[Union[str, os.PathLike]]] = None,
                                 video_format: Optional[str] = "mp4",
                                 substring: Optional[str] = None,
                                 remove_splits: Optional[bool] = True,
                                 gpu: Optional[bool] = False,
                                 fps: Optional[Union[int, str]] = None) -> None:
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
            #returned = os.system(f'ffmpeg -hwaccel auto -c:v h264_cuvid -f concat -safe 0 -i "{temp_txt_path}" -c:v h264_nvenc -c:a copy -hide_banner -loglevel info "{save_path}" -y')
            #returned = os.system(f"ffmpeg -hwaccel cuvid -hwaccel_device 0 -c:v h264_cuvid -f concat -safe 0 -i \"{temp_txt_path}\" -c:v h264_nvenc -c:a copy -hide_banner -loglevel info \"{save_path}\" -y")
        else:
            returned = os.system(f"ffmpeg -f concat -safe 0 -i \"{temp_txt_path}\" -r {out_fps} -c:v h264_nvenc -pix_fmt yuv420p -c:a copy -hide_banner -loglevel info \"{save_path}\" -y")
            #returned = os.system(f'ffmpeg -hwaccel auto -c:v h264_cuvid -f concat -safe 0 -i "{temp_txt_path}" -r {out_fps} -c:v h264_nvenc -c:a copy -hide_banner -loglevel info "{save_path}" -y')
            #returned = os.system(f'ffmpeg -hwaccel cuda -hwaccel_output_format cuda -c:v h264_cuvid -f concat -safe 0 -i "{temp_txt_path}" -vf scale_cuda=1280:720,format=nv12 -r {out_fps} -c:v h264_nvenc -c:a copy -hide_banner -loglevel info "{save_path}" -y')
    else:
        if fps is None:
            returned = os.system(f'ffmpeg -f concat -safe 0 -i "{temp_txt_path}" "{save_path}" -c copy -hide_banner -loglevel info -y')
        else:
            returned = os.system(f'ffmpeg -f concat -safe 0 -i "{temp_txt_path}" -r {out_fps} -c:v libx264 -c:a copy -hide_banner -loglevel info "{save_path}" -y')
    while True:
        if returned != 0:
            pass
        else:
            if remove_splits:
                remove_a_folder(folder_dir=Path(in_folder))
            break
    timer.stop_timer()
    stdout_success(msg="Video concatenated", elapsed_time=timer.elapsed_time_str, source=concatenate_videos_in_folder.__name__)



concatenate_videos_in_folder(in_folder=r'C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos\temp - Copy (2)', save_path=r'C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos\501_MA142_Gi_Saline_0513_rotated_gpu.mp4', remove_splits=False, gpu=True, fps=50)