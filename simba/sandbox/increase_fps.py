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
    extract_frames_single_video, frames_to_movie, gif_creator,
    multi_split_video, remove_beginning_of_video, resize_videos_by_height,
    resize_videos_by_width, roi_blurbox, superimpose_elapsed_time,
    superimpose_frame_count, superimpose_freetext, superimpose_overlay_video,
    superimpose_video_names, superimpose_video_progressbar,
    video_bg_subtraction_mp, video_bg_subtraction, video_concatenator,
    video_to_greyscale, watermark_video, rotate_video, flip_videos)


def upsample_fps(video_path: Union[str, os.PathLike],
                 fps: int,
                 quality: Optional[int] = 60,
                 save_dir: Optional[Union[str, os.PathLike]] = None) -> None:
    """
    Upsample the frame rate of a video or all videos in a directory to a specified fps with a given quality.

    .. note::
       Uses ``ffmpeg minterpolate``

    .. warning::
       Long run-times for higher resolution videos.

    :param Union[str, os.PathLike] video_path: The path to the input video file or directory containing video files.
    :param int fps: The target frame rate to upsample the video(s) to.
    :param Optional[int] quality: The quality level of the output video(s), represented as a percentage (1-100). Lower values indicate higher quality. Default is 60.
    :param Optional[Union[str, os.PathLike]] save_dir: The directory to save the upsampled video(s). If None, the videos will be saved in the same directory as the input video(s).
    :return: None. The function saves the upsampled video(s) to the specified directory.

    :example:
    >>> upsample_fps(video_path='/Users/simon/Desktop/Box2_IF19_7_20211109T173625_4_851_873_1_cropped.mp4', fps=100, quality=100)
    """

    timer = SimbaTimer(start=True)
    check_ffmpeg_available(raise_error=True)
    check_int(name=f'{upsample_fps.__name__} quality', value=quality, min_value=1, max_value=100)
    check_float(name=f'{upsample_fps.__name__} fps', value=quality, min_value=10e-16)
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
        input_video_meta = get_video_meta_data(video_path=video_path)
        if input_video_meta['fps'] >= fps:
            FrameRangeWarning(msg=f"The FPS of the input video named {video_name} ({input_video_meta['fps']}) is the same or larger than the upscaled target FPS ({fps})", source=upsample_fps.__name__)
        print(f'Up-sampling video {video_name} from {input_video_meta["fps"]} to {fps} FPS (Video {file_cnt + 1}/{len(video_paths)})...')
        save_path = os.path.join(save_dir, f'{video_name}_upsampled{ext}')
        cmd = f"""ffmpeg -i "{video_path}" -filter:v "minterpolate='fps={fps}:scd=none:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:me=epzs:search_param=16:vsbmc=1'" -c:v libx264 -crf {crf} "{save_path}" -loglevel error -stats -y"""
        subprocess.call(cmd, shell=True)
    timer.stop_timer()
    stdout_success(msg=f"{len(video_paths)} video(s) upsampled to {fps} and saved in {save_dir} directory.", elapsed_time=timer.elapsed_time_str, source=upsample_fps.__name__, )


class UpsampleVideospopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="UPSAMPLE VIDEOS USING INTERPOLATION (WARNING: LONG RUN-TIMES)")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.fps_dropdown = DropDownMenu(settings_frm, "NEW FRAME-RATE (FPS):", list(range(1, 500)), labelwidth=25)
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY (%):", list(range(10, 110, 10)), labelwidth=25)

        self.fps_dropdown.setChoices(60)
        self.quality_dropdown.setChoices(60)
        settings_frm.grid(row=0, column=0, sticky="NW")
        self.fps_dropdown.grid(row=0, column=0, sticky="NW")
        self.quality_dropdown.grid(row=1, column=0, sticky="NW")

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - UP-SAMPLE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS -  UP-SAMPLE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        self.main_frm.mainloop()

    def run(self, multiple: bool):
        target_fps = int(self.fps_dropdown.getChoices())
        target_quality = int(self.quality_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=upsample_fps(video_path=data_path,
                                             fps=target_fps,
                                             quality=target_quality)).start()


# get_video_meta_data(video_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/videos/interpolation_test/results.mp4')
# get_video_meta_data('/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/videos/interpolation_test/Together_1.mp4')
#
#
UpsampleVideospopUp()
#upsample_fps(video_path='/Users/simon/Desktop/Box2_IF19_7_20211109T173625_4_851_873_1_cropped.mp4', fps=100, quality=100)
