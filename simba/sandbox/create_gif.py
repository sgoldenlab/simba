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
    read_config_entry, read_config_file, read_frm_of_video)
from simba.utils.warnings import (FileExistWarning, InValidUserInputWarning,
                                  SameInputAndOutputWarning, FrameRangeWarning)
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
    extract_frames_single_video, frames_to_movie,
    multi_split_video, remove_beginning_of_video, resize_videos_by_height,
    resize_videos_by_width, roi_blurbox, superimpose_elapsed_time,
    superimpose_frame_count, superimpose_freetext, superimpose_overlay_video,
    superimpose_video_names, superimpose_video_progressbar,
    video_bg_subtraction_mp, video_bg_subtraction, video_concatenator,
    video_to_greyscale, watermark_video, rotate_video, flip_videos, create_average_frm)


def gif_creator(file_path: Union[str, os.PathLike],
                start_time: int,
                duration: int,
                width: Optional[int] = None,
                quality: Optional[int] = 100,
                fps: Optional[int] = 15,
                gpu: Optional[bool] = False) -> None:
    """
    Create a sample gif from a video file. The result is stored in the same directory as the
    input file with the ``.gif`` file-ending.

    .. note::
       The height is auto-computed to retain aspect ratio

    :param Union[str, os.PathLike] file_path: Path to video file.
    :param int start_time: Start time of the gif in relation to the video in seconds.
    :param int duration: Duration of the gif.
    :param int width: Width of the gif. If None, then retains the width and height of the input video.
    :param int fps: Width of the gif.
    :param Optional[bool] gpu: If True, use NVIDEA GPU codecs. Default False.

    :example:
    >>> _ = gif_creator(file_path='project_folder/videos/Video_1.avi', start_time=5, duration=10, width=600)
    """

    check_ffmpeg_available(raise_error=True)
    if gpu and not check_nvidea_gpu_available():
        raise FFMPEGCodecGPUError(msg="NVIDEA GPU not available (as evaluated by nvidea-smi returning None", source=gif_creator.__name__)
    timer = SimbaTimer(start=True)
    check_file_exist_and_readable(file_path=file_path)
    check_int(name="Start time", value=start_time, min_value=0)
    check_int(name="Duration", value=duration, min_value=1)
    video_meta_data = get_video_meta_data(file_path)
    if width is None:
        width = video_meta_data['width']
    check_int(name="Width", value=width, min_value=2)
    check_int(name="FPS", value=fps, min_value=1)
    check_int(name="QUALITY", value=quality, min_value=1, max_value=100)
    quality = int((quality - 1) / (100 - 1) * (256 - 1) + 1)
    if width % 2 != 0: width -= 1
    if quality == 1: quality += 1
    if (int(start_time) + int(duration)) > video_meta_data["video_length_s"]:
        raise FrameRangeError(msg=f'The end of the gif (start time: {start_time} + duration: {duration}) is longer than the {file_path} video: {video_meta_data["video_length_s"]}s', source=gif_creator.__name__)
    dir, file_name, ext = get_fn_ext(filepath=file_path)
    save_name = os.path.join(dir, f"{file_name}.gif")
    if gpu:
        command = f'ffmpeg -hwaccel auto -c:v h264_cuvid -ss {start_time} -i "{file_path}" -to {duration} -vf "fps=10,scale={width}:-1" -c:v gif -pix_fmt rgb24 -y "{save_name}" -y'
    else:
        command = f'ffmpeg -ss {start_time} -t {duration} -i "{file_path}" -filter_complex "[0:v] fps={fps},scale=w={width}:h=-1:flags=lanczos,split [a][b];[a] palettegen=stats_mode=single:max_colors={quality} [p];[b][p] paletteuse=dither=bayer:bayer_scale=3" "{save_name}" -loglevel error -stats -hide_banner -y'
    print("Creating gif sample... ")
    subprocess.call(command, shell=True, stdout=subprocess.PIPE)
    timer.stop_timer()
    stdout_success(msg=f"SIMBA COMPLETE: Video converted! {save_name} generated!", elapsed_time=timer.elapsed_time_str, source=gif_creator.__name__)

#gif_creator(file_path='/Users/simon/Desktop/avg_frm_test/F1 HAB.mp4', start_time=0, duration=5, width=None, quality=100)

#
#
class CreateGIFPopUP(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CREATE GIF FROM VIDEO", size=(600, 400))
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(settings_frm, "VIDEO PATH: ", title="Select a video file", file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], lblwidth=40)
        self.start_time_entry_box = Entry_Box(settings_frm, "START TIME (s):", "40", validation="numeric")
        self.duration_entry_box = Entry_Box(settings_frm, "DURATION (s): ", "40", validation="numeric")
        resolution_widths = Options.RESOLUTION_OPTIONS_2.value
        self.resolution_dropdown = DropDownMenu(settings_frm, "GIF WIDTH (ASPECT RATIO RETAINED):", resolution_widths, "40")
        self.quality_dropdown = DropDownMenu(settings_frm, "GIF QUALITY (%):", list(range(1, 101, 1)), "40")
        fps_lst = list(range(1, 101, 1))
        fps_lst.insert(0, 'AUTO')
        self.fps_dropdown = DropDownMenu(settings_frm, "GIF FPS:", fps_lst, "40")
        self.gpu_var = BooleanVar()
        gpu_cb = Checkbutton(settings_frm, text="USE GPU (decreased runtime)", variable=self.gpu_var)
        self.quality_dropdown.setChoices(100)
        self.resolution_dropdown.setChoices('AUTO')
        self.fps_dropdown.setChoices('AUTO')
        settings_frm.grid(row=0, sticky=NW)
        self.selected_video.grid(row=0, sticky=NW, pady=5)
        self.start_time_entry_box.grid(row=1, sticky=NW)
        self.duration_entry_box.grid(row=2, sticky=NW)
        self.resolution_dropdown.grid(row=3, sticky=NW)
        self.quality_dropdown.grid(row=4, sticky=NW)
        self.fps_dropdown.grid(row=5, sticky=NW)
        gpu_cb.grid(row=6, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        video_path = self.selected_video.file_path
        width = self.resolution_dropdown.getChoices()
        start_time = self.start_time_entry_box.entry_get
        duration = self.duration_entry_box.entry_get
        fps = self.fps_dropdown.getChoices()
        quality = int(self.quality_dropdown.getChoices())
        gpu = self.gpu_var.get()
        check_ffmpeg_available()
        if gpu: check_nvidea_gpu_available()
        check_file_exist_and_readable(file_path=video_path)
        video_meta_data = get_video_meta_data(video_path=video_path)
        if width == 'AUTO': width = video_meta_data['width']
        else: width = int(width)
        if fps == 'AUTO': fps = int(video_meta_data['fps'])
        else: fps = int(fps)
        if fps > int(video_meta_data['fps']):
            FrameRangeWarning(msg=f'The chosen FPS ({fps}) is higher than the video FPS ({video_meta_data["fps"]}). The video FPS will be used', source=self.__class__.__name__)
            fps = int(video_meta_data['fps'])
        max_duration = video_meta_data['video_length_s'] - int(start_time)
        check_int(name='start_time', value=start_time, max_value=video_meta_data['video_length_s'], min_value=0)
        check_int(name='duration', value=duration, max_value=max_duration, min_value=1)

        threading.Thread(target=gif_creator(file_path=video_path,
                                            start_time=int(start_time),
                                            duration=int(duration),
                                            width=width,
                                            gpu=gpu,
                                            fps=fps,
                                            quality=int(quality))).start()

#CreateGIFPopUP()


class CalculatePixelsPerMMInVideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CALCULATE PIXELS PER MILLIMETER IN VIDEO", size=(550, 550))
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.video_path = FileSelect(settings_frm, "Select a video file: ",  title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], lblwidth=30)
        self.known_distance = Entry_Box(settings_frm, "Known real-life metric distance (mm): ", "30", validation="numeric")
        run_btn = Button(settings_frm, text="GET PIXELS PER MILLIMETER", command=lambda: self.run())
        settings_frm.grid(row=0, column=0, pady=10, sticky=NW)
        self.video_path.grid(row=0, column=0, pady=10, sticky=NW)
        self.known_distance.grid(row=1, column=0, pady=10, sticky=NW)
        run_btn.grid(row=2, column=0, pady=10, sticky=NW)
        self.main_frm.mainloop()

    def run(self):
        check_file_exist_and_readable(file_path=self.video_path.file_path)
        check_int(name="Distance", value=self.known_distance.entry_get, min_value=1)
        _ = get_video_meta_data(video_path=self.video_path.file_path)
        mm_cnt = get_coordinates_nilsson(self.video_path.file_path, self.known_distance.entry_get)
        print(f"ONE (1) PIXEL REPRESENTS {round(mm_cnt, 4)} MILLIMETERS IN VIDEO {os.path.basename(self.video_path.file_path)}.")

CalculatePixelsPerMMInVideoPopUp()




