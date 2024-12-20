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

import ffmpeg
from PIL import Image

from simba.utils.read_write import get_video_meta_data

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
from simba.utils.warnings import (
    FileExistWarning, FrameRangeWarning, InvalidValueWarning,
    NoDataFoundWarning, NoFileFoundWarning,
    ThirdPartyAnnotationsInvalidFileFormatWarning)


def read_frm_of_video(video_path: Union[str, os.PathLike, cv2.VideoCapture],
                      frame_index: Optional[int] = 0,
                      opacity: Optional[float] = None,
                      size: Optional[Tuple[int, int]] = None,
                      greyscale: Optional[bool] = False,
                      clahe: Optional[bool] = False,
                      use_ffmpeg: Optional[bool] = False) -> np.ndarray:

    """
    Reads single image from video file.

    :param Union[str, os.PathLike] video_path: Path to video file, or cv2.VideoCapture object.
    :param int frame_index: The frame of video to return. Default: 1. Note, if frame index -1 is passed, the last frame of the video is read in.
    :param Optional[int] opacity: Value between 0 and 100 or None. If float value, returns image with opacity. 100 fully opaque. 0.0 fully transparant.
    :param Optional[Tuple[int, int]] size: If tuple, resizes the image to size. Else, returns original image size.
    :param Optional[bool] greyscale: If true, returns the greyscale image. Default False.
    :param Optional[bool] clahe: If true, returns clahe enhanced image. Default False.
    :return: Image as numpy array.
    :rtype: np.ndarray

    :example:
    >>> img = read_frm_of_video(video_path='/Users/simon/Desktop/envs/platea_featurizer/data/video/3D_Mouse_5-choice_MouseTouchBasic_s9_a4_grayscale.mp4', clahe=True)
    >>> cv2.imshow('img', img)
    >>> cv2.waitKey(5000)
    """

    check_instance(source=read_frm_of_video.__name__, instance=video_path, accepted_types=(str, cv2.VideoCapture))
    if type(video_path) == str:
        check_file_exist_and_readable(file_path=video_path)
        video_meta_data = get_video_meta_data(video_path=video_path)
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
    if greyscale:
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if clahe:
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.createCLAHE(clipLimit=2, tileGridSize=(16, 16)).apply(img)

    return img


import time

ffmpeg_times = []
opencv_times = []
for i in range(10):
    start = time.time()
    img = read_frm_of_video(video_path=r"C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos\502_MA141_Gi_Saline_0513.mp4", use_ffmpeg=True)
    ffmpeg_times.append(time.time() - start)
for i in range(10):
    start = time.time()
    img = read_frm_of_video(video_path=r"C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos\502_MA141_Gi_Saline_0513.mp4", use_ffmpeg=False)
    opencv_times.append(time.time() - start)
print(np.mean(ffmpeg_times), np.mean(opencv_times))