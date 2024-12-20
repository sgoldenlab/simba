import os
from typing import Union, Optional
import subprocess
from simba.utils.read_write import get_fn_ext, get_video_meta_data

import threading
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
    read_config_entry, read_config_file, read_frm_of_video, timestamp_to_seconds)
from simba.utils.warnings import (FileExistWarning, InValidUserInputWarning,
                                  SameInputAndOutputWarning, FrameRangeWarning, ResolutionWarning)
from simba.video_processors.extract_frames import video_to_frames
from simba.video_processors.roi_selector import ROISelector
from simba.video_processors.roi_selector_circle import ROISelectorCircle
from simba.video_processors.roi_selector_polygon import ROISelectorPolygon

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
                                NotDirectoryError, ResolutionError)
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
    convert_to_webm, convert_to_webp, crossfade_two_videos,
    convert_video_powerpoint_compatible_format, copy_img_folder,
    crop_multiple_videos, crop_multiple_videos_circles,
    crop_multiple_videos_polygons, crop_single_video, crop_single_video_circle,
    crop_single_video_polygon, downsample_video, extract_frame_range,
    extract_frames_single_video, frames_to_movie, gif_creator,
    multi_split_video, remove_beginning_of_video, resize_videos_by_height,
    resize_videos_by_width, roi_blurbox, superimpose_elapsed_time,
    superimpose_frame_count, superimpose_freetext, superimpose_overlay_video,
    superimpose_video_names, superimpose_video_progressbar,
    video_bg_subtraction_mp, video_bg_subtraction,
    video_to_greyscale, watermark_video, rotate_video, flip_videos, create_average_frm)
#
class CrossfadeVideosPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CROSS-FADE VIDEOS")
        crossfade_methods = get_ffmpeg_crossfade_methods()
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="NUMBER OF VIDEOS TO JOIN", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.video_path_1 = FileSelect(settings_frm, f"VIDEO PATH 1:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        self.video_path_2 = FileSelect(settings_frm, f"VIDEO PATH 2:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY:", list(range(10, 110, 10)), labelwidth=25)
        self.out_format_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO FORMAT:", Options.ALL_VIDEO_FORMAT_OPTIONS.value, labelwidth=25)
        self.fade_method_dropdown = DropDownMenu(settings_frm, "CROSS-FADE METHOD:", crossfade_methods, labelwidth=25)
        self.duration_dropdown = DropDownMenu(settings_frm, "CROSS-FADE DURATION:", list(range(2, 22, 2)), labelwidth=25)
        self.offset_dropdown = DropDownMenu(settings_frm, "CROSS-FADE OFFSET:", list(range(1, 1001, 1)), labelwidth=25)

        self.quality_dropdown.setChoices(60)
        self.out_format_dropdown.setChoices('.mp4')
        self.fade_method_dropdown.setChoices('fade')
        self.duration_dropdown.setChoices(6)
        self.offset_dropdown.setChoices(10)
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.video_path_1.grid(row=0, column=0, sticky=NW)
        self.video_path_2.grid(row=1, column=0, sticky=NW)
        self.quality_dropdown.grid(row=2, column=0, sticky=NW)
        self.out_format_dropdown.grid(row=3, column=0, sticky=NW)
        self.fade_method_dropdown.grid(row=4, column=0, sticky=NW)
        self.duration_dropdown.grid(row=5, column=0, sticky=NW)
        self.offset_dropdown.grid(row=6, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        video_1_path, video_2_path = self.video_path_1.file_path, self.video_path_2.file_path
        quality = int(self.quality_dropdown.getChoices())
        format = self.out_format_dropdown.getChoices()[1:]
        fade_method = self.fade_method_dropdown.getChoices()
        offset = int(self.offset_dropdown.getChoices())
        duration = int(self.duration_dropdown.getChoices())
        for video_path in [video_1_path, video_2_path]:
            video_meta_data = get_video_meta_data(video_path=video_path)
            if video_meta_data['video_length_s'] < duration:
                raise FrameRangeError(msg=f'Video {video_meta_data["video_name"]} is shorter {video_meta_data["video_length_s"]} than the crossfade duration {duration}.', source=self.__class__.__name__)
            if video_meta_data['video_length_s'] < offset:
                raise FrameRangeError(msg=f'Video {video_meta_data["video_name"]} is shorter {video_meta_data["video_length_s"]} than the crossfade offset {offset}.',source=self.__class__.__name__)
        threading.Thread(crossfade_two_videos(video_path_1=video_1_path,
                                              video_path_2=video_2_path,
                                              crossfade_duration=duration,
                                              crossfade_method=fade_method,
                                              crossfade_offset=offset,
                                              out_format=format,
                                              quality=quality)).start()


#CrossfadeVideosPopUp()




#crossfade_multiple_videos