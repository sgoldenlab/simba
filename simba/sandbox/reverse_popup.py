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
    convert_to_webm, convert_to_webp, reverse_videos,
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
    video_to_greyscale, watermark_video, rotate_video, flip_videos, upsample_fps)

sys.setrecursionlimit(10**7)


class ReverseVideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="REVERSE VIDEOS")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.MP4_CODEC_LK = {'HEVC (H.265)': 'libx265', 'H.264 (AVC)': 'libx264', 'VP9': 'vp9'}
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY:", list(range(10, 110, 10)), labelwidth=25)
        self.quality_dropdown.setChoices(60)
        self.codec_dropdown = DropDownMenu(settings_frm, "COMPRESSION CODEC:", list(self.MP4_CODEC_LK.keys()), labelwidth=25)
        self.codec_dropdown.setChoices('HEVC (H.265)')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.quality_dropdown.grid(row=0, column=0, sticky=NW)
        self.codec_dropdown.grid(row=1, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - REVERSE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - REVERSE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        #self.main_frm.mainloop()

    def run(self, multiple: bool):
        target_quality = int(self.quality_dropdown.getChoices())
        codec = self.MP4_CODEC_LK[self.codec_dropdown.getChoices()]
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        reverse_videos(path=data_path, quality=target_quality, codec=codec)
        #
        # threading.Thread(target=reverse_videos
        #                                         quality=target_quality)).start()


ReverseVideoPopUp()





