from typing import Union, Optional
import os
import subprocess
from simba.utils.read_write import get_fn_ext, find_all_videos_in_directory, get_video_meta_data
from simba.video_processors.roi_selector import ROISelector
from simba.utils.checks import check_ffmpeg_available, check_float, check_if_dir_exists, check_file_exist_and_readable
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.errors import InvalidInputError
import threading

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
    video_to_greyscale, watermark_video, rotate_video, flip_videos, upsample_fps, reverse_videos)

sys.setrecursionlimit(10**7)

def video_to_bw(video_path: Union[str, os.PathLike],
                threshold: Optional[float] = 0.5,
                save_dir: Optional[Union[str, os.PathLike]] = None) -> None:
    """
    Convert video to black and white using passed threshold.

    .. video:: _static/img/video_to_bw.webm
       :width: 800
       :autoplay:
       :loop:

    :param Union[str, os.PathLike] video_path: Path to the video
    :param Optional[float] threshold: Value between 0 and 1. Lower values gives more white and vice versa.
    :return: None.

    :example:
    >>> video_to_bw(video_path='/Users/simon/Downloads/1_LH_clipped_cropped_eq_20240515135926.mp4', threshold=0.02)
    """

    check_float(name=f'{video_to_bw.__name__} threshold', value=threshold, min_value=0, max_value=1.0)
    if os.path.isfile(video_path):
        video_paths = [video_path]
    elif os.path.isdir(video_path):
        video_paths = list(find_all_videos_in_directory(directory=video_path, as_dict=True, raise_error=True).values())
    else:
        raise InvalidInputError(msg=f'{video_path} is not a valid file path or a valid directory path', source=video_to_bw.__name__)
    if save_dir is not None:
        check_if_dir_exists(in_dir=save_dir)
    else:
        save_dir = os.path.dirname(video_paths[0])
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, ext = get_fn_ext(video_path)
    threshold = int(255 * threshold)
    check_ffmpeg_available(raise_error=True)
    timer = SimbaTimer(start=True)
    for file_cnt, video_path in enumerate(video_paths):
        _, video_name, ext = get_fn_ext(video_path)
        print(f'Converting video {video_name} to black and white (Video {file_cnt + 1}/{len(video_paths)})...')
        _ = get_video_meta_data(video_path=video_path)
        _, video_name, ext = get_fn_ext(video_path)
        save_path = os.path.join(save_dir, f'{video_name}_bw{ext}')
        cmd = f"ffmpeg -i '{video_path}' -vf \"format=gray,geq=lum_expr='if(lt(lum(X,Y),{threshold}),0,255)'\" -pix_fmt yuv420p '{save_path}' -loglevel error -stats -hide_banner -y"
        subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
        timer.stop_timer()
        stdout_success(msg=f'Video {video_name} converted to black and white.', elapsed_time=timer.elapsed_time_str)



class Convert2BlackWhitePopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="SUPER-IMPOSE TEXT ON VIDEOS")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        threshold = [round(x, 2) for x in list(np.arange(0.01, 1.01, 0.01))]
        self.threshold_dropdown = DropDownMenu(settings_frm, "BLACK THRESHOLD:", threshold, labelwidth=25)
        self.threshold_dropdown.setChoices(0.5)

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.threshold_dropdown.grid(row=0, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - CONVERT TO BLACK & WHITE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - CONVERT TO BLACK & WHITE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        self.main_frm.mainloop()

    def run(self, multiple: bool):
        threshold = float(self.threshold_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=video_to_bw(video_path=data_path,
                                            threshold=threshold)).start()




#Convert2BlackWhitePopUp()



#video_to_bw(video_path='/Users/simon/Desktop/Screen Recording 2024-05-08 at 10.57.59 AM_upsampled_time_superimposed.mov', threshold=0.5)
