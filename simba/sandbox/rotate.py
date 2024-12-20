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
__author__ = "Simon Nilsson"

import glob
import os
import subprocess
import sys
import threading
from copy import deepcopy
from datetime import datetime
from tkinter import *
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageTk

import simba
from simba.labelling.extract_labelled_frames import AnnotationFrameExtractor
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.frame_mergerer_ffmpeg import FrameMergererFFmpeg
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon,
                                        CreateToolTip, DropDownMenu, Entry_Box,
                                        FileSelect, FolderSelect)
from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_filepath_list_is_empty,
                                check_if_string_value_is_valid_video_timestamp,
                                check_int, check_nvidea_gpu_available,
                                check_str,
                                check_that_hhmmss_start_is_before_end)
from simba.utils.data import convert_roi_definitions
from simba.utils.enums import Dtypes, Formats, Keys, Links, Options, Paths
from simba.utils.errors import (CountError, DuplicationError, FrameRangeError,
                                InvalidInputError, MixedMosaicError,
                                NoChoosenClassifierError, NoFilesFoundError,
                                NotDirectoryError)
from simba.utils.lookups import get_color_dict, get_fonts
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (
    check_if_hhmmss_timestamp_is_valid_part_of_video,
    concatenate_videos_in_folder, find_all_videos_in_directory,
    find_files_of_filetypes_in_directory, get_fn_ext, get_video_meta_data,
    seconds_to_timestamp, str_2_bool)
from simba.video_processors.brightness_contrast_ui import \
    brightness_contrast_ui
from simba.video_processors.clahe_ui import interactive_clahe_ui
from simba.video_processors.extract_seqframes import extract_seq_frames
from simba.video_processors.multi_cropper import MultiCropper
from simba.video_processors.px_to_mm import get_coordinates_nilsson
from simba.video_processors.video_processing import (
    VideoRotator, batch_convert_video_format, batch_create_frames,
    batch_video_to_greyscale, change_fps_of_multiple_videos, change_img_format,
    change_single_video_fps, clahe_enhance_video, clip_video_in_range,
    clip_videos_by_frame_ids, convert_to_avi, convert_to_bmp, convert_to_jpeg,
    convert_to_mov, convert_to_mp4, convert_to_png, convert_to_tiff,
    convert_to_webm, convert_to_webp,
    convert_video_powerpoint_compatible_format, copy_img_folder,
    crop_multiple_videos, crop_multiple_videos_circles,
    crop_multiple_videos_polygons, crop_single_video, crop_single_video_circle,
    crop_single_video_polygon, downsample_video, extract_frame_range,
    extract_frames_single_video, frames_to_movie, gif_creator,
    multi_split_video, remove_beginning_of_video, resize_videos_by_height,
    resize_videos_by_width, roi_blurbox, superimpose_elapsed_time,
    superimpose_frame_count, superimpose_freetext, superimpose_overlay_video,
    superimpose_video_names, superimpose_video_progressbar,
    video_bg_subtraction_mp, video_bg_subtraction, video_concatenator,
    video_to_greyscale, watermark_video, rotate_video, flip_videos)

sys.setrecursionlimit(10**7)
#
#
#
#
def rotate_video(video_path: Union[str, os.PathLike],
                 degrees: int,
                 gpu: Optional[bool] = False,
                 quality: Optional[int] = 60,
                 save_dir: Optional[Union[str, os.PathLike]] = None):

    """
    Rotate a video or a directory of videos by a specified number of degrees.

    :param Union[str, os.PathLike] video_path: Path to the input video file or directory containing video files.
    :param int degrees: Number of degrees (between 1 and 359, inclusive) to rotate the video clockwise.
    :param Optional[bool] gpu: If True, attempt to use GPU acceleration for rotation (default is False).
    :param Optional[int] quality: Quality of the output video, an integer between 1 and 100 (default is 60).
    :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the rotated video(s). If None, the directory of the input video(s) will be used.
    :return: None.

    :example:
    >>> rotate_video(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/reptile/rot_test.mp4', degrees=180)
    """

    check_ffmpeg_available(raise_error=True)
    timer = SimbaTimer(start=True)
    check_int(name=f'{rotate_video.__name__} font_size', value=degrees, min_value=1, max_value=359)
    check_int(name=f'{rotate_video.__name__} quality', value=quality, min_value=1, max_value=100)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)", source=rotate_video.__name__)
    crf_lk = percent_to_crf_lookup()
    crf = crf_lk[str(quality)]
    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=rotate_video.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, ext = get_fn_ext(video_path)
        print(f'Rotating video {video_name} {degrees} degrees (Video {file_cnt + 1}/{len(video_paths)})...')
        save_path = os.path.join(save_dir, f'{video_name}_rotated{ext}')
        if gpu:
            cmd = f'ffmpeg -hwaccel auto -i "{video_path}" -vf "hwupload_cuda,rotate={degrees}*(PI/180),format=nv12|cuda" -c:v h264_nvenc "{save_path}" -loglevel error -stats -y'
        else:
            cmd = f'ffmpeg -i "{video_path}" -vf "rotate={degrees}*(PI/180)" -c:v libx264 -crf {crf} "{save_path}" -loglevel error -stats -y'
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"{len(video_paths)} video(s) ratated {degrees} and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=rotate_video.__name__,)



# class RotateVideoSetDegreesPopUp(PopUpMixin):
#     def __init__(self):
#         PopUpMixin.__init__(self, title="ROTATE VIDEOS")
#         settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
#         self.degrees_dropdown = DropDownMenu(settings_frm, "CLOCKWISE DEGREES:", list(range(1, 360, 1)), labelwidth=25)
#         self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY (%):", list(range(10, 110, 10)), labelwidth=25)
#         self.quality_dropdown.setChoices(60)
#         self.degrees_dropdown.setChoices('90')
#         self.degrees_dropdown.grid(row=0, column=0, sticky=NW)
#
#         settings_frm.grid(row=0, column=0, sticky="NW")
#         self.degrees_dropdown.grid(row=0, column=0, sticky="NW")
#         self.quality_dropdown.grid(row=1, column=0, sticky="NW")
#
#         single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE TEXT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
#         self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
#         single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))
#
#         single_video_frm.grid(row=1, column=0, sticky="NW")
#         self.selected_video.grid(row=0, column=0, sticky="NW")
#         single_video_run.grid(row=1, column=0, sticky="NW")
#
#         multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE TEXT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
#         self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
#         multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", command=lambda: self.run(multiple=True))
#
#         multiple_videos_frm.grid(row=2, column=0, sticky="NW")
#         self.selected_video_dir.grid(row=0, column=0, sticky="NW")
#         multiple_videos_run.grid(row=1, column=0, sticky="NW")
#         self.main_frm.mainloop()
#
#
#     def run(self, multiple: bool):
#         degrees = int(self.degrees_dropdown.getChoices())
#         quality = int(self.quality_dropdown.getChoices())
#         if not multiple:
#             data_path = self.selected_video.file_path
#             check_file_exist_and_readable(file_path=data_path)
#         else:
#             data_path = self.selected_video_dir.folder_path
#             check_if_dir_exists(in_dir=data_path)
#
#         threading.Thread(target=rotate_video(video_path=data_path,
#                                              degrees=degrees,
#                                              quality=quality)).start()

#RotateVideoSetDegreesPopUp()


#
# def flip_videos(video_path: Union[str, os.PathLike],
#                 horizontal_flip: Optional[bool] = False,
#                 vertical_flip: Optional[bool] = False,
#                 quality: Optional[int] = 60,
#                 save_dir: Optional[Union[str, os.PathLike]] = None):
#     """
#     Flip a video or directory of videos horizontally, vertically, or both, and save them to the specified directory.
#
#     .. video:: _static/img/overlay_video_progressbar.webm
#        :width: 900
#        :loop:
#
#     :param Union[str, os.PathLike] video_path: Path to the input video file or directory containing video files.
#     :param Optional[bool] horizontal_flip: If True, flip the video(s) horizontally (default is False).
#     :param Optional[bool] vertical_flip: If True, flip the video(s) vertically (default is False).
#     :param Optional[int] quality: Quality of the output video, an integer between 1 and 100 (default is 60).
#     :param Optional[Union[str, os.PathLike]] save_dir: Directory to save the flipped video(s). If None, the directory of the input video(s) will be used.
#     :return: None.
#     """
#
#
#     check_ffmpeg_available(raise_error=True)
#     timer = SimbaTimer(start=True)
#     check_int(name=f'{rotate_video.__name__} quality', value=quality, min_value=1, max_value=100)
#     if not horizontal_flip and not vertical_flip: raise InvalidInputError(msg='Flip videos vertically and/or horizontally. Got both as False', source=flip_videos.__name__)
#     crf_lk = percent_to_crf_lookup()
#     crf = crf_lk[str(quality)]
#     if os.path.isfile(video_path):
#         video_paths = [video_path]
#     elif os.path.isdir(video_path):
#         video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
#     else:
#         raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=flip_videos.__name__)
#     if save_dir is not None:
#         check_if_dir_exists(in_dir=save_dir)
#     else:
#         save_dir = os.path.dirname(video_paths[0])
#     for file_cnt, video_path in enumerate(video_paths):
#         _, video_name, ext = get_fn_ext(video_path)
#         print(f'Flipping video {video_name} (Video {file_cnt + 1}/{len(video_paths)})...')
#         save_path = os.path.join(save_dir, f'{video_name}_flipped{ext}')
#         if vertical_flip and not horizontal_flip:
#             cmd = f'ffmpeg -i "{video_path}" -vf "vflip" -c:v libx264 -crf {crf} "{save_path}" -loglevel error -stats -y'
#         elif horizontal_flip and not vertical_flip:
#             cmd = f'ffmpeg -i "{video_path}" -vf "hflip" -c:v libx264 -crf {crf} "{save_path}" -loglevel error -stats -y'
#         else:
#             cmd = f'ffmpeg -i "{video_path}" -vf "hflip,vflip" -c:v libx264 -crf {crf} "{save_path}" -loglevel error -stats -y'
#         subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
#     timer.stop_timer()
#     stdout_success(msg=f"{len(video_paths)} video(s) flipped and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=flip_videos.__name__,)
#

class FlipVideosPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="FLIP VIDEOS")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.horizontal_dropdown = DropDownMenu(settings_frm, "HORIZONTAL FLIP:", ['TRUE', 'FALSE'], labelwidth=25)
        self.vertical_dropdown = DropDownMenu(settings_frm, "VERTICAL FLIP:", ['TRUE', 'FALSE'], labelwidth=25)
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY (%):", list(range(10, 110, 10)), labelwidth=25)

        self.horizontal_dropdown.setChoices('FALSE')
        self.vertical_dropdown.setChoices('FALSE')
        self.quality_dropdown.setChoices(60)

        settings_frm.grid(row=0, column=0, sticky="NW")
        self.vertical_dropdown.grid(row=0, column=0, sticky="NW")
        self.horizontal_dropdown.grid(row=1, column=0, sticky="NW")
        self.quality_dropdown.grid(row=2, column=0, sticky="NW")

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE TEXT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE TEXT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        self.main_frm.mainloop()


    def run(self, multiple: bool):
        vertical_flip = str_2_bool(self.vertical_dropdown.getChoices())
        horizontal_flip = str_2_bool(self.horizontal_dropdown.getChoices())
        if not vertical_flip and not horizontal_flip:
            raise InvalidInputError(msg='Flip videos vertically and/or horizontally. Got both as False', source=self.__class__.__name__)
        quality = int(self.quality_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=flip_videos(video_path=data_path,
                                            vertical_flip=vertical_flip,
                                            horizontal_flip=horizontal_flip,
                                            quality=quality)).start()

FlipVideosPopUp()








#flip_videos(vertical_flip=True, horizontal_flip=True, video_path=f'/Users/simon/Desktop/envs/simba/troubleshooting/reptile/flip_test/flip_1.mp4')






