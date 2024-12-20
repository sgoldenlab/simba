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


def extract_frames_single_video(file_path: Union[str, os.PathLike],
                                save_dir: Optional[Union[str, os.PathLike]]) -> None:
    """
    Extract all frames for a single video.

    .. note::
       Image frames are saved as PNG files named with integers in order of appearance, i.e., ``0.png, 1.png ...``

    :parameter Union[str, os.PathLike] file_path: Path to video file.
    :parameter Optional[Union[str, os.PathLike]] save_dir: Optional directory where to save the frames. If ``save_dir`` is not passed,
    results are stored within a subdirectory in the same directory as the input file.

    :example:
    >>> _ = extract_frames_single_video(file_path='project_folder/videos/Video_1.mp4')
    >>> extract_frames_single_video(file_path='/Users/simon/Desktop/imgs_4/test.mp4', save_dir='/Users/simon/Desktop/imgs_4/frames')
    """

    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    _ = get_video_meta_data(file_path)
    dir_name, file_name, ext = get_fn_ext(filepath=file_path)
    if save_dir is None:
        save_dir = os.path.join(dir_name, file_name)
        if not os.path.exists(save_dir): os.makedirs(save_dir)
    else:
        check_if_dir_exists(in_dir=save_dir, source=extract_frames_single_video.__name__, create_if_not_exist=True)
    print(f"Processing video {file_name}...")
    video_to_frames(file_path, save_dir, overwrite=True, every=1, chunk_size=1000)
    timer.stop_timer()
    stdout_success(msg=f"Video {file_name} converted to images in {dir_name} directory!", elapsed_time=timer.elapsed_time_str, source=extract_frames_single_video.__name__)


