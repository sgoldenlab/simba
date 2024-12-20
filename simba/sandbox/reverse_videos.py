__author__ = "Simon Nilsson"


import functools
import glob
import multiprocessing
import os
import platform
import shutil
import subprocess
import time
from copy import deepcopy
from datetime import datetime
from tkinter import *
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageTk
from shapely.geometry import Polygon

try:
    from typing import Literal
except:
    from typing_extensions import Literal

import simba
from simba.mixins.config_reader import ConfigReader
from simba.mixins.image_mixin import ImageMixin
from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable, check_float,
                                check_if_dir_exists,
                                check_if_filepath_list_is_empty,
                                check_if_string_value_is_valid_video_timestamp,
                                check_instance, check_int,
                                check_nvidea_gpu_available, check_str,
                                check_that_hhmmss_start_is_before_end,
                                check_valid_lst, check_valid_tuple)
from simba.utils.data import find_frame_numbers_from_time_stamp
from simba.utils.enums import OS, ConfigKey, Formats, Options, Paths
from simba.utils.errors import (CountError, DirectoryExistError,
                                FFMPEGCodecGPUError, FFMPEGNotFoundError,
                                FileExistError, FrameRangeError,
                                InvalidFileTypeError, InvalidInputError,
                                InvalidVideoFileError, NoDataError,
                                NoFilesFoundError, NotDirectoryError)
from simba.utils.lookups import (get_ffmpeg_crossfade_methods, get_fonts,
                                 percent_to_crf_lookup, percent_to_qv_lk)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (
    check_if_hhmmss_timestamp_is_valid_part_of_video,
    concatenate_videos_in_folder, find_all_videos_in_directory, find_core_cnt,
    find_files_of_filetypes_in_directory, get_fn_ext, get_video_meta_data,
    read_config_entry, read_config_file, read_frm_of_video)
from simba.utils.warnings import (FileExistWarning, InValidUserInputWarning,
                                  SameInputAndOutputWarning)
from simba.video_processors.extract_frames import video_to_frames
from simba.video_processors.roi_selector import ROISelector
from simba.video_processors.roi_selector_circle import ROISelectorCircle
from simba.video_processors.roi_selector_polygon import ROISelectorPolygon

MAX_FRM_SIZE = 1080, 650

def reverse_videos(path: Union[str, os.PathLike],
                   save_dir: Optional[Union[str, os.PathLike]] = None,
                   quality: Optional[int] = 60) -> None:
    """
    Reverses one or more video files located at the specified path and saves the reversed videos in the specified
    directory.

    .. video:: _static/img/reverse_videos.webm
       :width: 800
       :loop:

    :param Union[str, os.PathLike] path: Path to the video file or directory containing video files to be reversed.
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the reversed videos. If not provided, reversed videos will be saved in a subdirectory named 'reversed_<timestamp>' in the same directory as the input file(s).
    :param Optional[int] quality: Output video quality expressed as a percentage. Default is 60. Values range from 1 (low quality, high compression) to 100 (high quality, low compression).
    :return: None

    :example:
    >>> reverse_videos(path='/Users/simon/Desktop/envs/simba/troubleshooting/open_field_below/project_folder/videos/reverse/TheVideoName_video_name_2_frame_no.mp4')
    """

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    check_instance(source=f'{reverse_videos.__name__} path', instance=path, accepted_types=(str,))
    check_int(name=f'{reverse_videos.__name__} quality', value=quality)
    datetime_ = datetime.now().strftime("%Y%m%d%H%M%S")
    crf_lk = percent_to_crf_lookup()
    crf = crf_lk[str(quality)]
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir, source=reverse_videos.__name__)
    if os.path.isfile(path):
        file_paths = [path]
        if save_dir is None:
            save_dir = os.path.join(os.path.dirname(path), f'reversed_{datetime_}')
            os.makedirs(save_dir)
    elif os.path.isdir(path):
        file_paths = find_files_of_filetypes_in_directory(directory=path, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        if save_dir is None:
            save_dir = os.path.join(path, f'mp4_{datetime_}')
            os.makedirs(save_dir)
    else:
        raise InvalidInputError(msg=f'Path is not a valid file or directory path.', source=reverse_videos.__name__)
    for file_cnt, file_path in enumerate(file_paths):
        _, video_name, ext = get_fn_ext(filepath=file_path)
        print(f'Reversing video {video_name} (Video {file_cnt+1}/{len(file_paths)})...')
        _ = get_video_meta_data(video_path=file_path)
        out_path = os.path.join(save_dir, f'{video_name}{ext}')
        cmd = f'ffmpeg -i "{file_path}" -vf reverse -af areverse -c:v libx264 -crf {crf} "{out_path}" -loglevel error -stats -hide_banner -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(file_paths)} video(s) reversed and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=reverse_videos.__name__,)

