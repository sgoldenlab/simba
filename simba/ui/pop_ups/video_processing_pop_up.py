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
from simba.ui.px_to_mm_ui import GetPixelsPerMillimeterInterface
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon,
                                        CreateToolTip, DropDownMenu, Entry_Box,
                                        FileSelect, FolderSelect, SimbaButton,
                                        SimbaCheckbox, SimBADropDown,
                                        SimBALabel)
from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_if_string_value_is_valid_video_timestamp,
                                check_int, check_nvidea_gpu_available,
                                check_str,
                                check_that_hhmmss_start_is_before_end)
from simba.utils.data import convert_roi_definitions
from simba.utils.enums import Dtypes, Formats, Keys, Links, Options, Paths
from simba.utils.errors import (CountError, DuplicationError,
                                FFMPEGCodecGPUError, FrameRangeError,
                                InvalidInputError, MixedMosaicError,
                                NoChoosenClassifierError, NoDataError,
                                NoFilesFoundError, NotDirectoryError,
                                ResolutionError)
from simba.utils.lookups import (get_color_dict, get_ffmpeg_crossfade_methods,
                                 get_fonts, percent_to_crf_lookup)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (
    check_if_hhmmss_timestamp_is_valid_part_of_video,
    concatenate_videos_in_folder, find_all_videos_in_directory, find_core_cnt,
    find_files_of_filetypes_in_directory, find_video_of_file, get_fn_ext,
    get_video_meta_data, seconds_to_timestamp, str_2_bool,
    timestamp_to_seconds)
from simba.utils.warnings import FrameRangeWarning
from simba.video_processors.brightness_contrast_ui import BrightnessContrastUI
from simba.video_processors.clahe_ui import interactive_clahe_ui
from simba.video_processors.extract_seqframes import extract_seq_frames
from simba.video_processors.multi_cropper import MultiCropper
from simba.video_processors.video_processing import (
    VideoRotator, batch_convert_video_format, batch_create_frames,
    batch_video_to_greyscale, change_fps_of_multiple_videos, change_img_format,
    change_single_video_fps, clahe_enhance_video, clahe_enhance_video_mp,
    clip_video_in_range, clip_videos_by_frame_ids, convert_to_avi,
    convert_to_bmp, convert_to_jpeg, convert_to_mov, convert_to_mp4,
    convert_to_png, convert_to_tiff, convert_to_webm, convert_to_webp,
    convert_video_powerpoint_compatible_format, copy_img_folder,
    create_average_frm, crop_multiple_videos, crop_multiple_videos_polygons,
    crop_single_video, crop_single_video_circle, crop_single_video_polygon,
    crossfade_two_videos, downsample_video, extract_frame_range,
    extract_frames_single_video, flip_videos, frames_to_movie, gif_creator,
    multi_split_video, remove_beginning_of_video, resize_videos_by_height,
    resize_videos_by_width, reverse_videos, roi_blurbox, rotate_video,
    superimpose_elapsed_time, superimpose_frame_count, superimpose_freetext,
    superimpose_overlay_video, superimpose_video_names,
    superimpose_video_progressbar, temporal_concatenation, upsample_fps,
    video_bg_subtraction, video_bg_subtraction_mp, video_concatenator,
    video_to_bw, video_to_greyscale, watermark_video)

sys.setrecursionlimit(10**7)

class CLAHEPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CLAHE VIDEO CONVERSION")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings', icon_link=Links.VIDEO_TOOLS.value)
        lbl = SimBALabel(parent=settings_frm, txt='For more control over CLAHE conversion, try "Interactively CLAHE enhance videos" \n in SimBA Tools->Remove color from videos.', font=Formats.FONT_REGULAR_ITALICS.value)
        self.core_cnt_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=list(range(1, find_core_cnt()[0]+1)), label='CORE COUNT:', label_width=25, dropdown_width=20, value=1)
        self.gpu_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label='USE GPU:', label_width=25, dropdown_width=20, value='FALSE')
        if not check_nvidea_gpu_available():
            self.gpu_dropdown.disable()

        settings_frm.grid(row=0, column=0, sticky=NW)
        lbl.grid(row=0, column=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=1, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=2, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - Contrast Limited Adaptive Histogram Equalization", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], lblwidth=25)
        run_single_video_btn = SimbaButton(parent=single_video_frm, txt="Apply CLAHE on VIDEO", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run_single_video, width=160)
        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOs - Contrast Limited Adaptive Histogram Equalization", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", lblwidth=25)

        run_multiple_btn = SimbaButton(parent=multiple_videos_frm, txt="Apply CLAHE on DIRECTORY", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run_directory, width=160)
        single_video_frm.grid(row=1, column=0, sticky=NW)
        self.selected_video.grid(row=0, column=0, sticky=NW)
        run_single_video_btn.grid(row=1, column=0, sticky=NW)

        multiple_videos_frm.grid(row=2, column=0, sticky=NW)
        self.selected_dir.grid(row=0, column=0, sticky=NW)
        run_multiple_btn.grid(row=1, column=0, sticky=NW)

    def _get_settings(self):
        self.core_cnt = int(self.core_cnt_dropdown.get_value())
        self.gpu = str_2_bool(self.gpu_dropdown.get_value())

    def run_single_video(self):
        selected_video = self.selected_video.file_path
        check_file_exist_and_readable(file_path=selected_video)
        self._get_settings()
        print(f'Applying CLAHE conversion on video {selected_video}...')
        if self.core_cnt == 1:
            threading.Thread(target=clahe_enhance_video(file_path=selected_video)).start()
        else:
            threading.Thread(target=clahe_enhance_video_mp(file_path=selected_video)).start()

    def run_directory(self):
        timer = SimbaTimer(start=True)
        video_dir = self.selected_dir.folder_path
        check_if_dir_exists(in_dir=video_dir, source=self.__class__.__name__)
        self.video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        self._get_settings()
        print(f'Applying CLAHE conversion on {len(self.video_paths)} videos...')
        for file_path in self.video_paths:
            if self.core_cnt == 1:
                threading.Thread(target=clahe_enhance_video(file_path=file_path)).start()
            else:
                clahe_enhance_video_mp(file_path=file_path, gpu=self.gpu)
        timer.stop_timer()
        stdout_success(msg=f'CLAHE enhanced {len(self.video_paths)} video(s)', elapsed_time=timer.elapsed_time_str)

class CropVideoPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CROP VIDEO(S)")
        crop_video_lbl_frm = LabelFrame( self.main_frm, text="CROP SINGLE VIDEO", font=Formats.FONT_HEADER.value)
        selected_video = FileSelect(crop_video_lbl_frm, "VIDEO PATH: ", title="Select a video file", lblwidth=20, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        use_gpu_cb_single, use_gpu_var_single = SimbaCheckbox(parent=crop_video_lbl_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')
        button_crop_video_single = SimbaButton(parent=crop_video_lbl_frm, txt="CROP SINGLE VIDEO", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=crop_single_video, cmd_kwargs={'file_path': lambda: selected_video.file_path, 'gpu': lambda: use_gpu_var_single.get()})


        crop_video_lbl_frm_multiple = LabelFrame(self.main_frm, text="CROP MULTIPLE VIDEOS", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        instructions_1 = Label(crop_video_lbl_frm_multiple, text="The crop coordinates you draw in the first video,\n will be applied on all videos in directory.", font=Formats.FONT_REGULAR.value)
        instructions_2 = Label(crop_video_lbl_frm_multiple, text="To draw crop coordinates on each individual video,\n instead use SimBA batch processing.", font=Formats.FONT_REGULAR.value)
        input_folder = FolderSelect(crop_video_lbl_frm_multiple, "VIDEO DIRECTORY:", title="Select Folder with videos", lblwidth=20)
        output_folder = FolderSelect(crop_video_lbl_frm_multiple,"OUTPUT DIRECTORY:",title="Select a folder for your output videos",lblwidth=20)
        use_gpu_cb_multiple, use_gpu_var_multiple = SimbaCheckbox(parent=crop_video_lbl_frm_multiple, txt="Use GPU (reduced runtime)", txt_img='gpu_2')
        button_crop_video_multiple = SimbaButton(parent=crop_video_lbl_frm_multiple, txt="CROP VIDEO DIRECTORY", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=crop_multiple_videos, cmd_kwargs={'directory_path': lambda:input_folder.folder_path, 'output_path': lambda:output_folder.folder_path,  'gpu': use_gpu_var_multiple.get()})

        crop_video_lbl_frm.grid(row=0, sticky=NW)
        selected_video.grid(row=0, sticky=NW)
        use_gpu_cb_single.grid(row=1, column=0, sticky=NW)
        button_crop_video_single.grid(row=2, sticky=NW)

        crop_video_lbl_frm_multiple.grid(row=1, sticky=NW)
        instructions_1.grid(row=0, sticky=NW)
        instructions_2.grid(row=1, sticky=NW)
        input_folder.grid(row=2, sticky=NW)
        output_folder.grid(row=3, sticky=NW)
        use_gpu_cb_multiple.grid(row=4, sticky=NW)
        button_crop_video_multiple.grid(row=5, sticky=NW)
        self.main_frm.mainloop()

#_ = CropVideoPopUp()

class ClipVideoPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CLIP VIDEO", icon='clip')
        selected_video_frm = CreateLabelFrameWithIcon( parent=self.main_frm, header="Video path", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        selected_video = FileSelect(selected_video_frm, "FILE PATH: ", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        selected_video.grid(row=0, column=0, sticky="NW")
        use_gpu_frm = LabelFrame(self.main_frm, text="GPU", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        gpu_available = NORMAL if check_nvidea_gpu_available() else DISABLED
        self.use_gpu_cb, self.use_gpu_var = SimbaCheckbox(parent=use_gpu_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2', state=gpu_available)
        self.use_gpu_cb.grid(row=0, column=0, sticky=NW)
        method_1_frm = LabelFrame(self.main_frm, text="Method 1", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        label_set_time_1 = Label(method_1_frm, text="Please enter the time frame in HH:MM:SS format", font=Formats.FONT_REGULAR.value)
        start_time = Entry_Box(method_1_frm, "Start at (HH:MM:SS):", "22")
        end_time = Entry_Box(method_1_frm, "End at (HH:MM:SS):", "22")
        CreateToolTip(method_1_frm, "Method 1 will retrieve the specified time input. (eg: input of Start at: 00:00:00, End at: 00:01:00, will create a new video from the chosen video from the very start till it reaches the first minute of the video)")
        method_2_frm = LabelFrame(self.main_frm, text="Method 2", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        method_2_time = Entry_Box(method_2_frm, "Seconds:", "22", validation="numeric")
        label_method_2 = Label(method_2_frm, text="Method 2 will retrieve from the end of the video (e.g.,: an input of 3 seconds will get rid of the first 3 seconds of the video).", font=Formats.FONT_REGULAR.value)
        button_cutvideo_method_1 = SimbaButton(parent=method_1_frm, txt="CUT VIDEO", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=clip_video_in_range, cmd_kwargs={'file_path': lambda: selected_video.file_path, 'start_time': lambda:start_time.entry_get, 'end_time': lambda:end_time.entry_get, 'gpu': lambda: self.use_gpu_var.get()})
        button_cutvideo_method_2 = SimbaButton(parent=method_2_frm, txt="CUT VIDEO", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=remove_beginning_of_video, cmd_kwargs={'file_path': lambda:selected_video.file_path, 'time': lambda:method_2_time.entry_get, 'gpu': lambda:self.use_gpu_var.get()})
        selected_video_frm.grid(row=0, sticky=NW)
        use_gpu_frm.grid(row=1, column=0, sticky=NW)
        method_1_frm.grid(row=2, sticky=NW, pady=5)
        label_set_time_1.grid(row=0, sticky=NW)
        start_time.grid(row=1, sticky=NW)
        end_time.grid(row=2, sticky=NW)
        button_cutvideo_method_1.grid(row=3, sticky=NW)
        method_2_frm.grid(row=3, sticky=NW, pady=5)
        label_method_2.grid(row=0, sticky=NW)
        method_2_time.grid(row=2, sticky=NW)
        button_cutvideo_method_2.grid(row=3, sticky=NW)

#ClipVideoPopUp()

class GreyscaleSingleVideoPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="GREYSCALE VIDEO")
        video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="GREYSCALE VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect( video_frm, "VIDEO FILE PATH", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        self.use_gpu_var_video = BooleanVar(value=False)
        use_gpu_video_cb, self.use_gpu_var_video = SimbaCheckbox(parent=video_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')
        run_video_btn = SimbaButton(parent=video_frm, txt="RUN ON SINGLE VIDEO", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run_video)
        dir_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="GREYSCALE VIDEO DIRECTORY", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.dir_selected = FolderSelect(dir_frm, "VIDEO DIRECTORY PATH", title="Select folder with videos", lblwidth=25)

        use_gpu_dir_cb, self.use_gpu_var_dir = SimbaCheckbox(parent=dir_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')
        run_dir_btn = SimbaButton(parent=dir_frm, txt="RUN ON DIRECTORY OF VIDEOS", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run_dir)

        video_frm.grid(row=0, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        use_gpu_video_cb.grid(row=1, column=0, sticky="NW")
        run_video_btn.grid(row=2, column=0, sticky="NW")

        dir_frm.grid(row=1, column=0, sticky="NW")
        self.dir_selected.grid(row=0, column=0, sticky="NW")
        use_gpu_dir_cb.grid(row=1, column=0, sticky="NW")
        run_dir_btn.grid(row=2, column=0, sticky="NW")


    def run_video(self):
        check_file_exist_and_readable(file_path=self.selected_video.file_path)
        video_to_greyscale(file_path=self.selected_video.file_path, gpu=self.use_gpu_var_video.get())

    def run_dir(self):
        check_if_dir_exists(in_dir=self.dir_selected.folder_path)
        batch_video_to_greyscale(path=self.dir_selected.folder_path, gpu=self.use_gpu_var_dir.get())

class SuperImposeFrameCountPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="SUPERIMPOSE FRAME COUNT")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        color_dict = list(get_color_dict().keys())
        font_dict = get_fonts()

        gpu_available = NORMAL if check_nvidea_gpu_available() else DISABLED
        use_gpu_cb, self.use_gpu_var = SimbaCheckbox(parent=settings_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2', state=gpu_available)

        self.font_size_dropdown = SimBADropDown(parent=settings_frm, dropdown_width=35, dropdown_options=list(range(1, 101, 2)), label="FONT SIZE:", label_width=25, value=20)
        self.font_color_dropdown = SimBADropDown(parent=settings_frm, dropdown_width=35, dropdown_options=color_dict, label="FONT COLOR:", label_width=25, value='Black')
        self.font_bg_color_dropdown = SimBADropDown(parent=settings_frm, dropdown_width=35, dropdown_options=color_dict, label="FONT BACKGROUND COLOR:", label_width=25, value='White')
        self.font_dropdown = SimBADropDown(parent=settings_frm, dropdown_width=35, dropdown_options=list(font_dict.keys()), label="FONT:", label_width=25, value='Arial')

        settings_frm.grid(row=0, column=0, sticky="NW")
        self.font_size_dropdown.grid(row=0, column=0, sticky="NW")
        self.font_color_dropdown.grid(row=1, column=0, sticky="NW")
        self.font_bg_color_dropdown.grid(row=2, column=0, sticky="NW")
        self.font_dropdown.grid(row=3, column=0, sticky="NW")
        use_gpu_cb.grid(row=4, column=0, sticky="NW")

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE FRAME COUNT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])

        single_video_run = SimbaButton(parent=single_video_frm, txt="RUN - SINGLE VIDEO", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run_single_video)

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE FRAME COUNT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = SimbaButton(parent=multiple_videos_frm, txt="RUN - MULTIPLE VIDEOS", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run_multiple_videos)

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        self.main_frm.mainloop()

    def run_single_video(self):
        video_path = self.selected_video.file_path
        check_file_exist_and_readable(file_path=self.selected_video.file_path)
        self.video_paths = [video_path]
        self.apply()

    def run_multiple_videos(self):
        video_dir = self.selected_video_dir.folder_path
        check_if_dir_exists(in_dir=video_dir, source=self.__class__.__name__)
        self.video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        self.apply()

    def apply(self):
        check_ffmpeg_available(raise_error=True)
        timer = SimbaTimer(start=True)
        use_gpu = self.use_gpu_var.get()
        font = self.font_dropdown.getChoices()
        font_clr = self.font_color_dropdown.getChoices()
        font_bg_clr = self.font_bg_color_dropdown.getChoices()
        font_size = int(self.font_size_dropdown.getChoices())
        for file_cnt, file_path in enumerate(self.video_paths):
            check_file_exist_and_readable(file_path=file_path)
            superimpose_frame_count(file_path=file_path,
                                    gpu=use_gpu,
                                    fontsize=font_size,
                                    font_color=font_clr,
                                    font=font,
                                    bg_color=font_bg_clr)
        timer.stop_timer()
        stdout_success(msg=f'Frame counts superimposed on {len(self.video_paths)} video(s)', elapsed_time=timer.elapsed_time_str)

#SuperImposeFrameCountPopUp()

class MultiShortenPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CLIP VIDEO INTO MULTIPLE VIDEOS", size=(800, 200))
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm,header="Split videos into different parts",icon_name=Keys.DOCUMENTATION.value,icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(settings_frm, "Video path", title="Select a video file", lblwidth=10, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        self.clip_cnt = Entry_Box(settings_frm, "# of clips", "10", validation="numeric")
        use_gpu_cb, self.use_gpu_var = SimbaCheckbox(parent=settings_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')
        confirm_settings_btn = SimbaButton(parent=settings_frm, txt="CONFIRM", img='tick', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.show_start_stop)
        settings_frm.grid(row=0, sticky=NW)
        self.selected_video.grid(row=1, sticky=NW, columnspan=2)
        self.clip_cnt.grid(row=2, sticky=NW)
        confirm_settings_btn.grid(row=2, column=1, sticky=W)
        use_gpu_cb.grid(row=3, column=0, sticky=W)
        instructions = Label(settings_frm, text="Enter clip start and stop times in HH:MM:SS format", fg="navy", font=Formats.FONT_REGULAR.value)
        instructions.grid(row=4, column=0)

        batch_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="Batch change time", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.batch_start_entry = Entry_Box(batch_frm, "START:", "10")
        self.batch_start_entry.entry_set("00:00:00")
        self.batch_end_entry = Entry_Box(batch_frm, "END", "10")
        self.batch_end_entry.entry_set("00:00:00")


        batch_start_apply = SimbaButton(parent=batch_frm, txt="APPLY", font=Formats.FONT_REGULAR.value, cmd=self.batch_change, cmd_kwargs={'value': "start"})
        batch_end_apply = SimbaButton(parent=batch_frm, txt="APPLY", font=Formats.FONT_REGULAR.value, cmd=self.batch_change, cmd_kwargs={'value': "end"})

        batch_frm.grid(row=0, column=1, sticky=NW)
        self.batch_start_entry.grid(row=0, column=0, sticky=NW)
        batch_start_apply.grid(row=0, column=1, sticky=NW)
        self.batch_end_entry.grid(row=1, column=0, sticky=NW)
        batch_end_apply.grid(row=1, column=1, sticky=NW)

        self.main_frm.mainloop()

    def show_start_stop(self):
        check_int(name="Number of clips", value=self.clip_cnt.entry_get)
        if hasattr(self, "table"):
            self.table.destroy()
        self.table = LabelFrame(self.main_frm)
        self.table.grid(row=2, column=0, sticky=NW)
        Label(self.table, text="Clip #", font=Formats.FONT_REGULAR.value).grid(row=0, column=0)
        Label(self.table, text="Start Time", font=Formats.FONT_REGULAR.value).grid(row=0, column=1, sticky=NW)
        Label(self.table, text="Stop Time", font=Formats.FONT_REGULAR.value).grid(row=0, column=2, sticky=NW)
        self.clip_names, self.start_times, self.end_times = [], [], []
        for i in range(int(self.clip_cnt.entry_get)):
            Label(self.table, text="Clip " + str(i + 1)).grid(row=i + 2, sticky=W)
            self.start_times.append(Entry(self.table))
            self.start_times[i].grid(row=i + 2, column=1, sticky=NW)
            self.end_times.append(Entry(self.table))
            self.end_times[i].grid(row=i + 2, column=2, sticky=NW)
        run_button = Button(
            self.table,
            text="Clip video",
            command=lambda: self.run_clipping(),
            fg="navy",
            font=Formats.FONT_REGULAR.value,
        )
        run_button.grid(row=int(self.clip_cnt.entry_get) + 2, column=2, sticky=W)

    def batch_change(self, value: str):
        if not hasattr(self, "table"):
            raise CountError(
                msg="Select the number of video clippings first",
                source=self.__class__.__name__,
            )
        for start_time_entry, end_time_entry in zip(self.start_times, self.end_times):
            if value == "start":
                start_time_entry.delete(0, END)
                start_time_entry.insert(0, self.batch_start_entry.entry_get)
            else:
                end_time_entry.delete(0, END)
                end_time_entry.insert(0, self.batch_end_entry.entry_get)

    def run_clipping(self):
        start_times, end_times = [], []
        check_file_exist_and_readable(self.selected_video.file_path)
        for start_time, end_time in zip(self.start_times, self.end_times):
            start_times.append(start_time.get())
            end_times.append(end_time.get())
        multi_split_video(
            file_path=self.selected_video.file_path,
            start_times=start_times,
            end_times=end_times,
            gpu=self.use_gpu_var.get(),
        )


#_ = MultiShortenPopUp()


class ChangeImageFormatPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CHANGE IMAGE FORMAT")

        self.input_folder_selected = FolderSelect(self.main_frm, "Image directory", title="Select folder with images:")
        set_input_format_frm = LabelFrame(self.main_frm, text="Original image format", font=Formats.FONT_HEADER.value, padx=15, pady=5)
        set_output_format_frm = LabelFrame(self.main_frm, text="Output image format", font=Formats.FONT_HEADER.value, padx=15, pady=5)

        self.input_file_type, self.out_file_type = StringVar(), StringVar()
        input_png_rb = Radiobutton(set_input_format_frm, text=".png", variable=self.input_file_type, value="png", font=Formats.FONT_REGULAR.value)
        input_jpeg_rb = Radiobutton(
            set_input_format_frm,
            text=".jpg",
            variable=self.input_file_type,
            value="jpg",
            font=Formats.FONT_REGULAR.value,
        )
        input_bmp_rb = Radiobutton(
            set_input_format_frm,
            text=".bmp",
            variable=self.input_file_type,
            value="bmp",
            font=Formats.FONT_REGULAR.value,
        )
        output_png_rb = Radiobutton(
            set_output_format_frm, text=".png", font=Formats.FONT_REGULAR.value, variable=self.out_file_type, value="png"
        )
        output_jpeg_rb = Radiobutton(
            set_output_format_frm, text=".jpg", font=Formats.FONT_REGULAR.value, variable=self.out_file_type, value="jpg"
        )
        output_bmp_rb = Radiobutton(
            set_output_format_frm, text=".bmp", font=Formats.FONT_REGULAR.value, variable=self.out_file_type, value="bmp"
        )
        run_btn = Button(
            self.main_frm,
            text="Convert image file format",
            font=Formats.FONT_REGULAR.value,
            command=lambda: self.run_img_conversion(),
        )
        self.input_folder_selected.grid(row=0, column=0)
        set_input_format_frm.grid(row=1, column=0, pady=5)
        set_output_format_frm.grid(row=2, column=0, pady=5)
        input_png_rb.grid(row=0, column=0)
        input_jpeg_rb.grid(row=1, column=0)
        input_bmp_rb.grid(row=2, column=0)
        output_png_rb.grid(row=0, column=0)
        output_jpeg_rb.grid(row=1, column=0)
        output_bmp_rb.grid(row=2, column=0)
        run_btn.grid(row=3, pady=5)

    def run_img_conversion(self):
        if len(os.listdir(self.input_folder_selected.folder_path)) == 0:
            raise NoFilesFoundError(
                msg="SIMBA ERROR: The input folder {} contains ZERO files.".format(
                    self.input_folder_selected.folder_path
                ),
                source=self.__class__.__name__,
            )
        change_img_format(
            directory=self.input_folder_selected.folder_path,
            file_type_in=self.input_file_type.get(),
            file_type_out=self.out_file_type.get(),
        )


class ConvertVideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CONVERT VIDEO FORMAT", size=(200, 200))
        convert_multiple_videos_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="Convert multiple videos",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        video_dir = FolderSelect(
            convert_multiple_videos_frm,
            "Video directory",
            title="Select folder with videos",
            lblwidth=15,
        )
        video_format_options = ["mp4", "avi", "mov", "flv", "m4v"]
        original_format_dropdown = DropDownMenu(
            convert_multiple_videos_frm,
            "Input format",
            video_format_options,
            labelwidth=15,
        )
        output_format_dropdown = DropDownMenu(
            convert_multiple_videos_frm,
            "Input format",
            video_format_options,
            labelwidth=15,
        )
        original_format_dropdown.setChoices("avi")
        output_format_dropdown.setChoices("mp4")
        gpu_multiple_var = BooleanVar(value=False)
        gpu_multiple_cb = Checkbutton(
            convert_multiple_videos_frm,
            text="Use GPU (reduced runtime)",
            font=Formats.FONT_REGULAR.value,
            variable=gpu_multiple_var,
        )
        convert_multiple_btn = Button(
            convert_multiple_videos_frm,
            text="Convert multiple videos",
            font=Formats.FONT_REGULAR.value,
            command=lambda: batch_convert_video_format(
                directory=video_dir.folder_path,
                input_format=original_format_dropdown.getChoices(),
                output_format=output_format_dropdown.getChoices(),
                gpu=gpu_multiple_var.get(),
            ),
        )
        convert_single_video_frm = LabelFrame(
            self.main_frm,
            text="Convert single video",
            font=Formats.FONT_HEADER.value,
            padx=5,
            pady=5,
        )
        self.selected_video = FileSelect(
            convert_single_video_frm,
            "Video path",
            title="Select a video file",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        self.output_format = StringVar()
        checkbox_v1 = Radiobutton(
            convert_single_video_frm,
            text="Convert to .mp4",
            font=Formats.FONT_REGULAR.value,
            variable=self.output_format,
            value="mp4",
        )
        checkbox_v2 = Radiobutton(
            convert_single_video_frm,
            text="Convert mp4 into PowerPoint supported format",
            font=Formats.FONT_REGULAR.value,
            variable=self.output_format,
            value="pptx",
        )
        self.gpu_single_var = BooleanVar(value=False)
        gpu_single_cb = Checkbutton(
            convert_single_video_frm,
            text="Use GPU (reduced runtime)",
            font=Formats.FONT_REGULAR.value,
            variable=self.gpu_single_var,
        )
        convert_single_btn = Button(
            convert_single_video_frm,
            text="Convert video format",
            font=Formats.FONT_REGULAR.value,
            command=lambda: self.convert_single(),
        )

        convert_multiple_videos_frm.grid(row=0, sticky=NW)
        video_dir.grid(row=0, sticky=NW)
        original_format_dropdown.grid(row=1, sticky=NW)
        output_format_dropdown.grid(row=2, sticky=NW)
        gpu_multiple_cb.grid(row=3, sticky=NW)
        convert_multiple_btn.grid(row=4, pady=10, sticky=NW)
        convert_single_video_frm.grid(row=1, sticky=NW)
        self.selected_video.grid(row=0, sticky=NW)
        checkbox_v1.grid(row=1, column=0, sticky=NW)
        checkbox_v2.grid(row=2, column=0, sticky=NW)
        gpu_single_cb.grid(row=3, column=0, sticky=NW)
        convert_single_btn.grid(row=4, column=0, pady=10, sticky=NW)

    def convert_single(self):
        if self.output_format.get() == "mp4":
            convert_to_mp4(
                file_path=self.selected_video.file_path, gpu=self.gpu_single_var.get()
            )
        if self.output_format.get() == "pptx":
            convert_video_powerpoint_compatible_format(
                file_path=self.selected_video.file_path, gpu=self.gpu_single_var.get()
            )


class ExtractSpecificFramesPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="EXTRACT DEFINED FRAME RANGE FROM VIDEO", icon='frames')
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm,header="SETTINGS", icon_name=Keys.DOCUMENTATION.value,icon_link=Links.VIDEO_TOOLS.value)
        self.video_file_selected = FileSelect(self.settings_frm , "VIDEO PATH:", title="Select a video file", file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], lblwidth=40)
        self.save_dir = FolderSelect(self.settings_frm, "SAVE DIRECTORY: ", lblwidth=40, tooltip_txt='Optional directory where to save the images. \n If None, images are saved in a folder with the suffix `_frames` \n within the same directory as the video file')

        self.format_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['jpeg', 'png', 'webp'], label="FORMAT: ", label_width=40, dropdown_width=25, value='png')
        self.grey_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GREYSCALE: ", label_width=40, dropdown_width=25, value='FALSE')
        self.clahe_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label="CLAHE: ", label_width=40, dropdown_width=25, value='FALSE')
        self.include_fn_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label="INCLUDE VIDEO NAME IN IMAGE NAME: ", label_width=40, dropdown_width=25, value='FALSE')

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.video_file_selected.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)
        self.format_dropdown.grid(row=2, column=0, sticky=NW)
        self.grey_dropdown.grid(row=3, column=0, sticky=NW)
        self.clahe_dropdown.grid(row=4, column=0, sticky=NW)
        self.include_fn_dropdown.grid(row=5, column=0, sticky=NW)

        select_frames_frm = LabelFrame(self.main_frm, text="FRAME RANGE TO BE EXTRACTED", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        self.start_frm = Entry_Box(select_frames_frm, "START FRAME NUMBER:", "40", validation='numeric')
        self.end_frm = Entry_Box(select_frames_frm, "END FRAME NUMBER:", "40", validation='numeric')
        run_btn = SimbaButton(parent=select_frames_frm, txt="RUN", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.start_frm_extraction)

        select_frames_frm.grid(row=1, column=0, sticky=NW)
        self.start_frm.grid(row=2, column=0, sticky=NW)
        self.end_frm.grid(row=3, column=0, sticky=NW)
        run_btn.grid(row=4, pady=5, sticky=NW)
        self.main_frm.mainloop()

    def start_frm_extraction(self):
        start_frame = self.start_frm.entry_get
        end_frame = self.end_frm.entry_get
        video_path = self.video_file_selected.file_path
        check_file_exist_and_readable(file_path=video_path)
        check_int(name="Start frame", value=start_frame)
        check_int(name="End frame", value=end_frame)
        if int(end_frame) < int(start_frame):
            raise FrameRangeError(msg=f"SIMBA ERROR: The end frame ({end_frame}) cannot come before the start frame ({start_frame})", source=self.__class__.__name__)
        video_meta_data = get_video_meta_data(video_path=self.video_file_selected.file_path)
        if int(start_frame) > video_meta_data["frame_count"]:
            raise FrameRangeError(msg=f"SIMBA ERROR: The start frame ({start_frame}) is larger than the number of frames in the video ({video_meta_data['frame_count']})", source=self.__class__.__name__)
        if int(end_frame) > video_meta_data["frame_count"]:
            raise FrameRangeError(msg=f"SIMBA ERROR: The end frame ({end_frame}) is larger than the number of frames in the video ({video_meta_data['frame_count']})", source=self.__class__.__name__)
        grey = str_2_bool(self.grey_dropdown.get_value())
        clahe = str_2_bool(self.clahe_dropdown.get_value())
        include_fn = str_2_bool(self.include_fn_dropdown.get_value())
        save_dir = self.save_dir.folder_path
        save_dir = save_dir if os.path.isdir(save_dir) else None
        img_format = self.format_dropdown.get_value()


        extract_frame_range(file_path=video_path,
                            start_frame=int(start_frame),
                            end_frame=int(end_frame),
                            save_dir=save_dir,
                            greyscale=grey,
                            clahe=clahe,
                            img_format=img_format,
                            include_fn=include_fn)


#ExtractSpecificFramesPopUp()

class ExtractAllFramesPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="EXTRACT ALL FRAMES", icon='frames')
        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm,header="SINGLE VIDEO",icon_name=Keys.DOCUMENTATION.value,icon_link=Links.VIDEO_TOOLS.value)
        video_path = FileSelect( single_video_frm, "VIDEO PATH:", title="Select a video file", file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_btn = SimbaButton(parent=single_video_frm, txt="Extract Frames (Single video)", img='rocket', font=Formats.FONT_REGULAR.value, cmd=extract_frames_single_video, cmd_kwargs={'file_path': lambda:video_path.file_path, 'save_dir': None})
        multiple_videos_frm = CreateLabelFrameWithIcon( parent=self.main_frm, header="MULTIPLE VIDEOS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)

        folder_path = FolderSelect(multiple_videos_frm, "DIRECTORY PATH:", title=" Select video folder")
        multiple_video_btn = SimbaButton(parent=multiple_videos_frm, txt="Extract Frames (Multiple videos)", img='rocket', font=Formats.FONT_REGULAR.value, cmd=batch_create_frames, cmd_kwargs={'directory': lambda:folder_path.folder_path})
        single_video_frm.grid(row=0, sticky=NW, pady=10)
        video_path.grid(row=0, sticky=NW)
        single_video_btn.grid(row=1, sticky=W, pady=10)
        multiple_videos_frm.grid(row=1, sticky=W, pady=10)
        folder_path.grid(row=0, sticky=W)
        multiple_video_btn.grid(row=1, sticky=W, pady=10)


class MultiCropPopUp(PopUpMixin):
    def __init__(self):

        PopUpMixin.__init__(self, title="MULTI-CROP", size=(500, 300), icon='crop')
        self.input_folder = FolderSelect(self.main_frm, "INPUT VIDEO FOLDER: ", lblwidth=25)
        self.output_folder = FolderSelect(self.main_frm, "OUTPUT FOLDER: ", lblwidth=25)
        video_options = Options.ALL_VIDEO_FORMAT_OPTIONS_2.value
        self.video_type_dropdown = SimBADropDown(parent=self.main_frm, dropdown_options=video_options, label="INPUT VIDEO FORMAT: ", label_width=25, dropdown_width=25, value='mp4')
        self.crop_cnt_dropdown = SimBADropDown(parent=self.main_frm, dropdown_options=list(range(2, 31)), label="CROPS PER VIDEO: ", label_width=25, dropdown_width=25, value=2)
        quality_options = list(percent_to_crf_lookup().keys())
        self.quality_dropdown = SimBADropDown(parent=self.main_frm, dropdown_options=quality_options, label="CROP OUTPUT QUALITY: ", label_width=25, dropdown_width=25, value=60)
        self.use_gpu_cb, self.use_gpu_var = SimbaCheckbox(parent=self.main_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')
        self.create_run_frm(run_function=self.run)
        self.input_folder.grid(row=0, sticky=NW)
        self.output_folder.grid(row=1, sticky=NW)
        self.video_type_dropdown.grid(row=2, sticky=NW)
        self.crop_cnt_dropdown.grid(row=3, sticky=NW)
        self.quality_dropdown.grid(row=4, sticky=NW)
        self.use_gpu_cb.grid(row=5, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        check_if_dir_exists(in_dir=self.input_folder.folder_path)
        check_if_dir_exists(in_dir=self.output_folder.folder_path)

        cropper = MultiCropper(file_type=self.video_type_dropdown.getChoices(),
                               input_folder=self.input_folder.folder_path,
                               output_folder=self.output_folder.folder_path,
                               crop_cnt=int(self.crop_cnt_dropdown.getChoices()),
                               gpu=self.use_gpu_var.get(),
                               quality=int(self.quality_dropdown.getChoices()))
        cropper.run()



class ChangeFpsSingleVideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CHANGE FRAME RATE: SINGLE VIDEO", size=(500, 300), icon='fps')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name='settings')
        self.video_path = FileSelect(settings_frm, "VIDEO PATH:", title="Select a video file", lblwidth=15, file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        self.new_fps_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=list(range(1, 101, 1)), label='NEW FPS: ', label_width=15, dropdown_width=20, value=15)
        gpu_cb, self.gpu_var = SimbaCheckbox(parent=settings_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.video_path.grid(row=0, sticky=NW)
        self.new_fps_dropdown.grid(row=1, sticky=NW)
        gpu_cb.grid(row=2, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        video_path = self.video_path.file_path
        video_meta_data = get_video_meta_data(video_path=video_path)
        _, video_name, _ = get_fn_ext(filepath=video_path)
        new_fps = int(self.new_fps_dropdown.getChoices())
        gpu = self.gpu_var.get()
        if video_meta_data['fps'] <= new_fps:
            FrameRangeWarning(msg=f'For video {video_name}, the new FPS ({new_fps}) is higher or the same as the original FPS ({video_meta_data["fps"]})', source=self.__class__.__name__)
        threading.Thread(change_single_video_fps(file_path=video_path, fps=new_fps, gpu=gpu)).start()

class ChangeFpsMultipleVideosPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CHANGE FRAME RATE: MULTIPLE VIDEO", size=(500, 300), icon='fps')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name='settings')
        self.directory_path = FolderSelect(settings_frm, "VIDEO DIRECTORY PATH:", title="Select folder with videos: ", lblwidth="25")

        self.new_fps_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=list(range(1, 101, 1)), label='NEW FPS: ', label_width=25, dropdown_width=20, value=15)
        gpu_cb, self.gpu_var = SimbaCheckbox(parent=settings_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.directory_path.grid(row=0, sticky=NW)
        self.new_fps_dropdown.grid(row=1, sticky=NW)
        gpu_cb.grid(row=2, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        video_dir = self.directory_path.folder_path
        check_if_dir_exists(in_dir=video_dir)
        _ = find_all_videos_in_directory(directory=video_dir, raise_error=True)
        new_fps = int(self.new_fps_dropdown.getChoices())
        gpu = self.gpu_var.get()
        threading.Thread(change_fps_of_multiple_videos(path=video_dir, fps=new_fps, gpu=gpu)).start()

class ExtractSEQFramesPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(
            self, title="EXTRACT ALL FRAMES FROM SEQ FILE", size=(200, 200)
        )
        video_path = FileSelect(
            self.main_frm,
            "Video Path",
            title="Select a video file: ",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        run_btn = Button(
            self.main_frm,
            text="Extract All Frames",
            font=Formats.FONT_REGULAR.value,
            command=lambda: extract_seq_frames(video_path.file_path),
        )
        video_path.grid(row=0)
        run_btn.grid(row=1)


class MergeFrames2VideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="MERGE IMAGE DIRECTORY INTO VIDEO", icon='video')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.folder_path = FolderSelect(settings_frm, "IMAGE DIRECTORY: ", title="Select directory with frames: ", lblwidth=25)
        self.video_fps_dropdown = DropDownMenu(settings_frm, "VIDEO FPS:", list(range(1, 101, 1)), labelwidth=25)
        self.video_quality_dropdown = DropDownMenu(settings_frm, "VIDEO QUALITY:", list(range(10, 110, 10)), labelwidth=25)
        self.video_format_dropdown = DropDownMenu(settings_frm, "VIDEO FORMAT:", ['mp4', 'avi', 'webm'], labelwidth=25)
        gpu_cb, self.gpu_var = SimbaCheckbox(parent=settings_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')
        self.video_quality_dropdown.setChoices(60)
        self.video_format_dropdown.setChoices('mp4')
        self.video_fps_dropdown.setChoices(30)

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.folder_path.grid(row=0, column=0, sticky=NW)
        self.video_fps_dropdown.grid(row=1, column=0, sticky=NW)
        self.video_quality_dropdown.grid(row=2, column=0, sticky=NW)
        self.video_format_dropdown.grid(row=3, column=0, sticky=NW)
        gpu_cb.grid(row=4, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()


    def run(self):
        directory = self.folder_path.folder_path
        _ = find_files_of_filetypes_in_directory(directory=directory, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, raise_error=True, raise_warning=False)
        fps = int(self.video_fps_dropdown.getChoices())
        quality = int(self.video_quality_dropdown.getChoices())
        gpu = self.gpu_var.get()
        format = self.video_format_dropdown.getChoices()
        threading.Thread(frames_to_movie(directory=self.folder_path.folder_path,
                                         fps=fps,
                                         quality=quality,
                                         out_format=format,
                                         gpu=gpu)).start()


class CreateGIFPopUP(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CREATE GIF FROM VIDEO", size=(600, 400), icon='gif')
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
        gpu_cb, self.gpu_var = SimbaCheckbox(parent=settings_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')

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

class CalculatePixelsPerMMInVideoPopUp(PopUpMixin):

    """
    .. video:: _static/img/GetPixelsPerMillimeterInterface.webm
       :width: 800
       :autoplay:
       :loop:
    """
    def __init__(self):
        PopUpMixin.__init__(self, title="CALCULATE PIXELS PER MILLIMETER IN VIDEO", size=(550, 550), icon='distance')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.video_path = FileSelect(settings_frm, "Select a video file: ",  title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], lblwidth=30)
        self.known_distance = Entry_Box(settings_frm, "Known real-life metric distance (mm): ", "30", validation="numeric")
        run_btn = SimbaButton(parent=settings_frm, txt="GET PIXELS PER MILLIMETER", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.run)
        settings_frm.grid(row=0, column=0, pady=10, sticky=NW)
        self.video_path.grid(row=0, column=0, pady=10, sticky=NW)
        self.known_distance.grid(row=1, column=0, pady=10, sticky=NW)
        run_btn.grid(row=2, column=0, pady=10, sticky=NW)
        self.main_frm.mainloop()

    def run(self):
        check_file_exist_and_readable(file_path=self.video_path.file_path)
        check_int(name="Distance", value=self.known_distance.entry_get, min_value=1)
        _ = get_video_meta_data(video_path=self.video_path.file_path)
        px_per_mm_interface = GetPixelsPerMillimeterInterface(video_path=self.video_path.file_path, known_metric_mm=float(self.known_distance.entry_get))
        px_per_mm_interface.run()
        print(f"ONE (1) PIXEL REPRESENTS {round(px_per_mm_interface.ppm, 4)} MILLIMETERS IN VIDEO {os.path.basename(self.video_path.file_path)}.")

class ConcatenatingVideosPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CONCATENATE TWO VIDEOS", size=(600, 300), icon='concat_videos')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.video_path_1 = FileSelect(settings_frm, "FIRST VIDEO PATH: ", title="Select a video file", lblwidth=35, file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        self.video_path_2 = FileSelect(settings_frm, "SECOND VIDEO PATH: ", title="Select a video file", lblwidth=35, file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        resolutions = ["VIDEO 1", "VIDEO 2", 240, 320, 480, 640, 720, 800, 960, 1120, 1080, 1980]
        self.resolution_dropdown = DropDownMenu(settings_frm, "RESOLUTION (ASPECT RATIO RETAINED):", resolutions, "35")
        self.resolution_dropdown.setChoices(resolutions[0])
        self.horizontal = BooleanVar(value=False)
        horizontal_radio_btn = Radiobutton(settings_frm, text="HORIZONTAL concatenation", font=Formats.FONT_REGULAR.value, variable=self.horizontal, value=True)
        self.gpu_dropdown = DropDownMenu(settings_frm, "USE GPU (REDUCED RUN-TIME):", ['TRUE', 'FALSE'], "35")
        vertical_radio_btn = Radiobutton(settings_frm, text="VERTICAL concatenation", font=Formats.FONT_REGULAR.value, variable=self.horizontal, value=False)
        self.gpu_dropdown.setChoices('FALSE')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.video_path_1.grid(row=0, column=0, sticky=NW)
        self.video_path_2.grid(row=1, column=0, sticky=NW)
        self.resolution_dropdown.grid(row=2, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=3, column=0, sticky=NW)
        horizontal_radio_btn.grid(row=4, column=0, sticky=NW)
        vertical_radio_btn.grid(row=5, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        video_1_path = self.video_path_1.file_path
        video_2_path = self.video_path_2.file_path
        resolution = self.resolution_dropdown.getChoices()
        int_res, _ = check_int(name='resolution', value=resolution, raise_error=False)
        if int_res:
            resolution = int(resolution)
        horizontal_bool = self.horizontal.get()
        gpu_bool = str_2_bool(self.gpu_dropdown.getChoices())
        check_file_exist_and_readable(file_path=video_1_path)
        check_file_exist_and_readable(file_path=video_2_path)
        video_1_meta = get_video_meta_data(video_path=video_1_path)
        video_2_meta = get_video_meta_data(video_path=video_2_path)
        if horizontal_bool and not int_res:
            if not video_1_meta['height'] == video_2_meta['height']:
                raise ResolutionError(f'For HORIZONTAL concatenation, the videos has to be the same height. Got Video 1 height: {video_1_meta["height"]}, Video 2 height: {video_2_meta["height"]}. Select a specific resolution or convert the video resolution heights of the two videos first to be the same.', source=self.__class__.__name__)
        elif not horizontal_bool and not int_res:
            if not video_1_meta['width'] == video_2_meta['width']:
                raise ResolutionError(f'For VERTICAL concatenation, the videos has to be the same width. Got Video 1 width: {video_1_meta["width"]}, Video 2 width: {video_2_meta["width"]}. Select a specific resolution or convert the video resolution widths of the two videos first to be the same.', source=self.__class__.__name__)
        threading.Thread(video_concatenator(video_one_path=video_1_path, video_two_path=video_2_path, resolution=resolution, horizontal=horizontal_bool, gpu=gpu_bool)).start()

class ConcatenatorPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Optional[Union[str, os.PathLike]] = None):
        PopUpMixin.__init__(self, title="MERGE (CONCATENATE) VIDEOS", icon='concat_videos')
        self.config_path = config_path
        self.select_video_cnt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEOS #", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.CONCAT_VIDEOS.value)
        self.select_video_cnt_dropdown = DropDownMenu( self.select_video_cnt_frm, "VIDEOS #", list(range(2, 21)), "15")
        self.select_video_cnt_dropdown.setChoices(2)

        self.select_video_cnt_btn  = SimbaButton(parent=self.select_video_cnt_frm, txt="APPLY", img='tick', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.populate_table)
        self.select_video_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.select_video_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.select_video_cnt_btn.grid(row=0, column=1, sticky=NW)
        self.main_frm.mainloop()

    def populate_table(self):
        if hasattr(self, "video_table_frm"):
            self.video_table_frm.destroy()
            self.join_type_frm.destroy()
        self.video_table_frm = LabelFrame(
            self.main_frm,
            text="VIDEO PATHS",
            pady=5,
            padx=5,
            font=Formats.FONT_HEADER.value,
            fg="black",
        )
        self.video_table_frm.grid(row=1, sticky=NW)
        self.join_type_frm = LabelFrame(
            self.main_frm,
            text="JOIN TYPE",
            pady=5,
            padx=5,
            font=Formats.FONT_HEADER.value,
            fg="black",
        )
        self.join_type_frm.grid(row=2, sticky=NW)
        self.videos_dict = {}
        for cnt in range(int(self.select_video_cnt_dropdown.getChoices())):
            self.videos_dict[cnt] = FileSelect(
                self.video_table_frm,
                "Video {}: ".format(str(cnt + 1)),
                title="Select a video file",
                file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
            )
            self.videos_dict[cnt].grid(row=cnt, column=0, sticky=NW)

        self.join_type_var = StringVar()
        self.icons_dict = {}
        simba_dir = os.path.dirname(simba.__file__)
        icon_assets_dir = os.path.join(simba_dir, Paths.ICON_ASSETS.value)
        concat_icon_dir = os.path.join(icon_assets_dir, "concat_icons")
        for file_cnt, file_path in enumerate(glob.glob(concat_icon_dir + "/*.png")):
            _, file_name, _ = get_fn_ext(file_path)
            self.icons_dict[file_name] = {}
            self.icons_dict[file_name]["img"] = ImageTk.PhotoImage(
                Image.open(file_path)
            )
            self.icons_dict[file_name]["btn"] = Radiobutton(
                self.join_type_frm,
                text=file_name,
                variable=self.join_type_var,
                value=file_name,
            )
            self.icons_dict[file_name]["btn"].config(
                image=self.icons_dict[file_name]["img"]
            )
            self.icons_dict[file_name]["btn"].image = self.icons_dict[file_name]["img"]
            self.icons_dict[file_name]["btn"].grid(row=0, column=file_cnt, sticky=NW)
        self.join_type_var.set(value="mosaic")
        self.resolution_frm = LabelFrame(
            self.main_frm,
            text="RESOLUTION",
            pady=5,
            padx=5,
            font=Formats.FONT_HEADER.value,
            fg="black",
        )
        self.resolution_width = DropDownMenu(
            self.resolution_frm, "Width", ["480", "640", "1280", "1920", "2560"], "15"
        )
        self.resolution_width.setChoices("480")
        self.resolution_height = DropDownMenu(
            self.resolution_frm, "Height", ["480", "640", "1280", "1920", "2560"], "15"
        )
        self.resolution_height.setChoices("640")
        self.gpu_frm = LabelFrame(
            self.main_frm,
            text="GPU",
            pady=5,
            padx=5,
            font=Formats.FONT_HEADER.value,
            fg="black",
        )

        use_gpu_cb, self.use_gpu_var = SimbaCheckbox(parent=self.gpu_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')


        use_gpu_cb.grid(row=0, column=0, sticky="NW")
        self.resolution_frm.grid(row=3, column=0, sticky=NW)
        self.gpu_frm.grid(row=4, column=0, sticky="NW")
        self.resolution_width.grid(row=0, column=0, sticky=NW)
        self.resolution_height.grid(row=1, column=0, sticky=NW)

        run_btn = Button(self.main_frm, text="RUN", command=lambda: self.run())
        run_btn.grid(row=5, column=0, sticky=NW)

    def run(self):
        file_paths = []
        for cnt, (video_name, video_data) in enumerate(self.videos_dict.items()):
            _ = get_video_meta_data(video_path=video_data.file_path)
            file_paths.append(video_data.file_path)

        if (len(file_paths) < 3) & (self.join_type_var.get() == "mixed_mosaic"):
            raise MixedMosaicError(
                msg="If using the mixed mosaic join type, please tick check-boxes for at least three video types.",
                source=self.__class__.__name__,
            )
        if (len(file_paths) < 3) & (self.join_type_var.get() == "mosaic"):
            self.join_type_var.set(value="vertical")

        video_merger = FrameMergererFFmpeg(
            config_path=self.config_path,
            video_paths=file_paths,
            video_height=int(self.resolution_height.getChoices()),
            video_width=int(self.resolution_width.getChoices()),
            concat_type=self.join_type_var.get(),
            gpu=self.use_gpu_var.get(),
        )

        threading.Thread(target=video_merger.run())


#ConcatenatorPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
#ConcatenatorPopUp(config_path=None)



class VideoRotatorPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="ROTATE VIDEOS")
        self.save_dir_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SAVE LOCATION", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.save_dir = FolderSelect(self.save_dir_frm, "Save directory:", lblwidth=20)

        self.setting_frm = LabelFrame(self.main_frm, text="SETTINGS", font=Formats.FONT_HEADER.value)

        use_gpu_cb, self.use_gpu_var = SimbaCheckbox(parent=self.setting_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')
        use_ffmpeg_cb, self.use_ffmpeg_var = SimbaCheckbox(parent=self.setting_frm, txt="Use GPU (reduced runtime)")
        use_gpu_cb.grid(row=0, column=0, sticky=NW)
        use_ffmpeg_cb.grid(row=1, column=0, sticky=NW)

        self.rotate_dir_frm = LabelFrame(self.main_frm, text="ROTATE VIDEOS IN DIRECTORY", font=Formats.FONT_HEADER.value)
        self.input_dir = FolderSelect(self.rotate_dir_frm, "Video directory:", lblwidth=20)

        self.run_dir = SimbaButton(parent=self.rotate_dir_frm, txt="RUN", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'input_path': lambda:self.input_dir.folder_path, 'output_path': lambda:self.save_dir.folder_path})

        self.rotate_video_frm = LabelFrame( self.main_frm, text="ROTATE SINGLE VIDEO", font=Formats.FONT_HEADER.value)
        self.input_file = FileSelect( self.rotate_video_frm, "Video path:", lblwidth=20, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])

        self.run_file = SimbaButton(parent=self.rotate_video_frm, txt="RUN", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'input_path': lambda:self.input_file.file_path, 'output_path': lambda:self.save_dir.folder_path})

        self.save_dir_frm.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=0, column=0, sticky=NW)

        self.setting_frm.grid(row=1, column=0, sticky=NW)

        self.rotate_dir_frm.grid(row=2, column=0, sticky=NW)
        self.input_dir.grid(row=0, column=0, sticky=NW)
        self.run_dir.grid(row=1, column=0, sticky=NW)

        self.rotate_video_frm.grid(row=3, column=0, sticky=NW)
        self.input_file.grid(row=0, column=0, sticky=NW)
        self.run_file.grid(row=1, column=0, sticky=NW)

    def run(self, input_path: str, output_path: str):
        check_if_dir_exists(in_dir=output_path)
        rotator = VideoRotator(
            input_path=input_path,
            output_dir=output_path,
            ffmpeg=self.use_ffmpeg_var.get(),
            gpu=self.use_gpu_var.get(),
        )
        rotator.run()


class VideoTemporalJoinPopUp(PopUpMixin):

    def __init__(self):
        super().__init__(title="TEMPORAL JOIN VIDEOS")
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm,
                                                     header="SETTINGS",
                                                     icon_name=Keys.DOCUMENTATION.value,
                                                     icon_link=Links.VIDEO_TOOLS.value)

        FPS_OPTIONS = list(range(2, 102, 2))
        FPS_OPTIONS.insert(0, 'SAME AS THE FIRST INPUT VIDEO IN INPUT DIRECTORY')
        FPS_OPTIONS.insert(0, 'KEEP INPUT VIDEO FPS')

        self.input_dir = FolderSelect( self.settings_frm, "INPUT DIRECTORY:", lblwidth=30)
        self.file_format = DropDownMenu(self.settings_frm, "INPUT VIDEO FORMAT:", Options.VIDEO_FORMAT_OPTIONS.value, "30")
        self.out_fps = DropDownMenu(self.settings_frm, "OUTPUT FPS:", FPS_OPTIONS, "30")
        self.out_fps.setChoices('SAME AS INPUT VIDEOS')
        use_gpu_cb, self.use_gpu_var = SimbaCheckbox(parent=self.settings_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')
        self.file_format.setChoices(Options.VIDEO_FORMAT_OPTIONS.value[0])
        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.input_dir.grid(row=0, column=0, sticky=NW)
        self.file_format.grid(row=1, column=0, sticky=NW)
        self.out_fps.grid(row=2, column=0, sticky=NW)
        use_gpu_cb.grid(row=3, column=0, sticky="NW")
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_if_dir_exists(in_dir=self.input_dir.folder_path)
        _ = find_files_of_filetypes_in_directory(directory=self.input_dir.folder_path, extensions=[f'.{self.file_format.getChoices()}'], raise_error=True)
        fps_setting = self.out_fps.getChoices()
        if fps_setting == 'KEEP INPUT VIDEO FPS':
            fps = None
        elif fps_setting == 'SAME AS THE FIRST INPUT VIDEO IN INPUT DIRECTORY':
            fps = '0'
        else:
            fps = int(fps_setting)

        print(f"Concatenating videos in {self.input_dir.folder_path} directory...")
        save_path = os.path.join(self.input_dir.folder_path, f"concatenated.mp4")
        concatenate_videos_in_folder(in_folder=self.input_dir.folder_path,
                                     save_path=save_path,
                                     remove_splits=False,
                                     fps=fps,
                                     video_format=self.file_format.getChoices(),
                                     gpu=self.use_gpu_var.get())

class ImportFrameDirectoryPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="IMPORT FRAME DIRECTORY", icon='import')
        ConfigReader.__init__(self, config_path=config_path)
        self.frame_folder = FolderSelect(self.main_frm, "FRAME DIRECTORY:", title="Select the main directory with frame folders")
        import_btn  = SimbaButton(parent=self.main_frm, txt="IMPORT FRAMES", img='import', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run)
        self.frame_folder.grid(row=0, column=0, sticky=NW)
        import_btn.grid(row=1, column=0, sticky=NW)
        self.main_frm.mainloop()

    def run(self):
        if not os.path.isdir(self.frame_folder.folder_path):
            raise NotDirectoryError(
                msg=f"SIMBA ERROR: {self.frame_folder.folder_path} is not a valid directory.",
                source=self.__class__.__name__,
            )
        copy_img_folder(
            config_path=self.config_path, source=self.frame_folder.folder_path
        )


#ImportFrameDirectoryPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")

class ExtractAnnotationFramesPopUp(PopUpMixin, ConfigReader):
    """
    :example:
    >>> ExtractAnnotationFramesPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path)
        if len(self.target_file_paths) == 0:
            raise NoFilesFoundError(msg=f'Cannot extract annotation images: No data files found in {self.targets_folder}.')
        self.video_dict = {}
        for file_path in self.target_file_paths:
            self.video_dict[get_fn_ext(filepath=file_path)[1]] = file_path

        PopUpMixin.__init__(self, config_path=config_path, title="EXTRACT ANNOTATED FRAMES", icon='frames')
        self.clf_frame = LabelFrame(self.main_frm, text="CHOOSE CLASSIFIERS", font=Formats.FONT_HEADER.value, pady=5, padx=5)
        self.create_clf_checkboxes(main_frm=self.clf_frame, clfs=self.clf_names)
        self.choose_video_frm = LabelFrame(self.main_frm, text="CHOOSE VIDEOS", font=Formats.FONT_HEADER.value, pady=5, padx=5)
        video_options = ["ALL"] + list(self.video_dict.keys())
        self.video_dropdown = DropDownMenu(self.choose_video_frm, "Video:", video_options, "25")
        self.video_dropdown.setChoices('ALL')
        self.settings_frm = LabelFrame(self.main_frm, text="STYLE SETTINGS", font=Formats.FONT_HEADER.value, pady=5, padx=5)
        down_sample_resolution_options = ["None", "2x", "3x", "4x", "5x"]
        img_format_options = ['png', 'jpg', 'webp']
        self.resolution_downsample_dropdown = DropDownMenu(self.settings_frm, "Down-sample images:", down_sample_resolution_options, "25")
        self.resolution_downsample_dropdown.setChoices(down_sample_resolution_options[0])
        self.img_format_dropdown = DropDownMenu(self.settings_frm, "Image format:", img_format_options, "25")
        self.img_format_dropdown.setChoices(img_format_options[0])
        self.greyscale_dropdown = DropDownMenu(self.settings_frm, "Image grayscale:", ['TRUE', 'FALSE'], "25")
        self.greyscale_dropdown.setChoices('FALSE')

        self.choose_video_frm.grid(row=self.children_cnt_main()+2, column=0, sticky=NW)
        self.video_dropdown.grid(row=0, column=0, sticky=NW)
        self.settings_frm.grid(row=self.children_cnt_main()+1, column=0, sticky=NW)
        self.resolution_downsample_dropdown.grid(row=0, column=0, sticky=NW)
        self.img_format_dropdown.grid(row=1, column=0, sticky=NW)
        self.greyscale_dropdown.grid(row=2, column=0, sticky=NW)



        self.run_btn = SimbaButton(parent=self.main_frm, txt="RUN", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=self.run)
        self.run_btn.grid(row=self.children_cnt_main()+3, column=0, sticky=NW)
        self.main_frm.mainloop()

    def run(self):
        downsample = self.resolution_downsample_dropdown.getChoices()
        if downsample != Dtypes.NONE.value: downsample = int("".join(filter(str.isdigit, downsample)))
        else: downsample = None
        greyscale = str_2_bool(input_str=self.greyscale_dropdown.getChoices())
        img_format = self.img_format_dropdown.getChoices()
        clfs = []
        for clf_name, selection in self.clf_selections.items():
            if selection.get():
                clfs.append(clf_name)
        if len(clfs) == 0:
            raise NoChoosenClassifierError(source=self.__class__.__name__)
        video_selection = self.video_dropdown.getChoices()
        if video_selection == 'ALL':
            data_paths = list(self.video_dict.values())
        else:
            data_paths = [self.video_dict[video_selection]]

        for data_path in data_paths:
            _ = find_video_of_file(video_dir=self.video_dir, filename=get_fn_ext(filepath=data_path)[1], raise_error=True)

        frame_extractor = AnnotationFrameExtractor(config_path=self.config_path,
                                                   clfs=clfs,
                                                   downsample=downsample,
                                                   img_format=img_format,
                                                   greyscale=greyscale,
                                                   data_paths=data_paths)
        frame_extractor.run()

class DownsampleVideoPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="DOWN-SAMPLE VIDEO RESOLUTION")
        instructions = Label(
            self.main_frm,
            text="Choose only one of the following method (Custom or Default)", font=Formats.FONT_REGULAR.value
        )
        choose_video_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SELECT VIDEO",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.DOWNSAMPLE.value,
        )
        self.video_path_selected = FileSelect(
            choose_video_frm,
            "Video path",
            title="Select a video file",
            file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)],
        )
        gpu_frm = LabelFrame(
            self.main_frm,
            text="GPU",
            font=Formats.FONT_HEADER.value,
            fg="black",
            padx=5,
            pady=5,
        )
        use_gpu_cb, self.use_gpu_var = SimbaCheckbox(parent=gpu_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')
        use_gpu_cb.grid(row=0, column=0, sticky="NW")
        custom_frm = LabelFrame(
            self.main_frm,
            text="Custom resolution",
            font=Formats.FONT_HEADER.value,
            fg="black",
            padx=5,
            pady=5,
        )
        self.entry_width = Entry_Box(custom_frm, "Width", "10", validation="numeric")
        self.entry_height = Entry_Box(custom_frm, "Height", "10", validation="numeric")




        self.custom_downsample_btn = SimbaButton(parent=custom_frm, txt="Downsample to custom resolution", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=self.custom_downsample)

        default_frm = LabelFrame(
            self.main_frm, text="Default resolution", font=Formats.FONT_HEADER.value, padx=5, pady=5
        )
        self.radio_btns = {}
        self.var = StringVar()
        for custom_cnt, resolution_radiobtn in enumerate(self.resolutions):
            self.radio_btns[resolution_radiobtn] = Radiobutton(
                default_frm,
                text=resolution_radiobtn,
                variable=self.var,
                value=resolution_radiobtn,
                font=Formats.FONT_REGULAR.value,
            )
            self.radio_btns[resolution_radiobtn].grid(row=custom_cnt, sticky=NW)

        self.default_downsample_btn = SimbaButton(parent=default_frm, txt="Downsample to default resolution", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=self.default_downsample)



        instructions.grid(row=0, sticky=NW, pady=10)
        choose_video_frm.grid(row=1, column=0, sticky=NW)
        gpu_frm.grid(row=2, column=0, sticky=NW)
        self.video_path_selected.grid(row=0, column=0, sticky=NW)
        custom_frm.grid(row=3, column=0, sticky=NW)
        self.entry_width.grid(row=0, column=0, sticky=NW)
        self.entry_height.grid(row=1, column=0, sticky=NW)
        self.custom_downsample_btn.grid(row=3, column=0, sticky=NW)
        default_frm.grid(row=5, column=0, sticky=NW)
        self.default_downsample_btn.grid(
            row=len(self.resolutions) + 1, column=0, sticky=NW
        )

    def custom_downsample(self):
        width = self.entry_width.entry_get
        height = self.entry_height.entry_get
        check_int(name="Video width", value=width)
        check_int(name="Video height", value=height)
        check_file_exist_and_readable(self.video_path_selected.file_path)
        downsample_video(
            file_path=self.video_path_selected.file_path,
            video_width=int(width),
            video_height=int(height),
            gpu=self.use_gpu_var.get(),
        )

    def default_downsample(self):
        resolution = self.var.get()
        width, height = (
            resolution.split("×", 2)[0].strip(),
            resolution.split("×", 2)[1].strip(),
        )
        check_file_exist_and_readable(self.video_path_selected.file_path)
        downsample_video(
            file_path=self.video_path_selected.file_path,
            video_width=int(width),
            video_height=int(height),
            gpu=self.use_gpu_var.get(),
        )


class ConvertROIDefinitionsPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CONVERT ROI DEFINITIONS")
        settings_frm = LabelFrame(self.main_frm, text="SETTINGS", font=Formats.FONT_HEADER.value)
        self.roi_definitions_file_select = FileSelect(settings_frm, "ROI DEFINITIONS PATH (H5)", title="SELECT H5 FILE", lblwidth=20, file_types=[("H5 FILE", (".h5", ".H5"))])
        self.save_dir = FolderSelect(settings_frm, "SAVE DIRECTORY", title="SELECT H5 FILE", lblwidth=20)
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.roi_definitions_file_select.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        check_file_exist_and_readable(file_path=self.roi_definitions_file_select.file_path)
        check_if_dir_exists(in_dir=self.save_dir.folder_path)
        convert_roi_definitions(roi_definitions_path=self.roi_definitions_file_select.file_path, save_dir=self.save_dir.folder_path)


class CropVideoCirclesPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CROP SINGLE VIDEO (CIRCLES)", icon='circle_small')
        crop_video_lbl_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="Crop Video (CIRCLES)", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.CIRCLE_CROP.value)
        selected_video = FileSelect(crop_video_lbl_frm, "Video path", title="Select a video file", lblwidth=20, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        button_crop_video_single = SimbaButton(parent=crop_video_lbl_frm, txt="CROP VIDEO", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=crop_single_video_circle, cmd_kwargs={'file_path': lambda:selected_video.file_path})
        crop_video_lbl_frm_multiple = LabelFrame(self.main_frm, text="Fixed CIRCLE coordinates crop for multiple videos", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        input_folder = FolderSelect(crop_video_lbl_frm_multiple,"Video directory:",title="Select Folder with videos",lblwidth=20)
        output_folder = FolderSelect(crop_video_lbl_frm_multiple, "Output directory:", title="Select a folder for your output videos", lblwidth=20)
        button_crop_video_multiple = SimbaButton(parent=crop_video_lbl_frm_multiple, txt="CROP VIDEOS", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=crop_single_video_circle, cmd_kwargs={'in_dir': lambda:input_folder.folder_path, 'out_dir': lambda:output_folder.folder_path})
        crop_video_lbl_frm.grid(row=0, sticky=NW)
        selected_video.grid(row=0, sticky=NW)
        button_crop_video_single.grid(row=3, sticky=NW)
        crop_video_lbl_frm_multiple.grid(row=1, sticky=NW)
        input_folder.grid(row=0, sticky=NW)
        output_folder.grid(row=1, sticky=NW)
        button_crop_video_multiple.grid(row=3, sticky=NW)


# _ = CropVideoCirclesPopUp()


class CropVideoPolygonsPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CROP SINGLE VIDEO (POLYGONS)", icon='polygon')
        crop_video_lbl_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="Crop Video (POLYGONS)", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.CIRCLE_CROP.value)
        selected_video = FileSelect( crop_video_lbl_frm, "Video path", title="Select a video file", lblwidth=20, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])

        button_crop_video_single = SimbaButton(parent=crop_video_lbl_frm, txt="CROP VIDEO", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=crop_single_video_polygon, cmd_kwargs={'file_path': lambda:selected_video.file_path})
        crop_video_lbl_frm_multiple = LabelFrame( self.main_frm, text="Fixed POLYGON coordinates crop for multiple videos", font=Formats.FONT_HEADER.value, padx=5, pady=5)
        input_folder = FolderSelect( crop_video_lbl_frm_multiple, "Video directory:", title="Select Folder with videos", lblwidth=20)
        output_folder = FolderSelect( crop_video_lbl_frm_multiple, "Output directory:", title="Select a folder for your output videos", lblwidth=20)
        button_crop_video_multiple = Button(crop_video_lbl_frm_multiple, text="Crop Videos", font=Formats.FONT_REGULAR.value, command=lambda: crop_multiple_videos_polygons(     in_dir=input_folder.folder_path, out_dir=output_folder.folder_path))

        crop_video_lbl_frm.grid(row=0, sticky=NW)
        selected_video.grid(row=0, sticky=NW)
        button_crop_video_single.grid(row=3, sticky=NW)
        crop_video_lbl_frm_multiple.grid(row=1, sticky=NW)
        input_folder.grid(row=0, sticky=NW)
        output_folder.grid(row=1, sticky=NW)
        button_crop_video_multiple.grid(row=3, sticky=NW)


class ClipSingleVideoByFrameNumbers(PopUpMixin):
    def __init__(self):
        super().__init__(title="CLIP SINGLE VIDEO BY FRAME NUMBERS")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(settings_frm, "VIDEO PATH: ", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], lblwidth=15)
        self.start_frm_eb = Entry_Box(settings_frm, "START FRAME: ", "15", validation="numeric")
        self.end_frm_eb = Entry_Box(settings_frm, "END FRAME: ", "15", validation="numeric")
        gpu_cb, self.gpu_var = SimbaCheckbox(parent=settings_frm, txt='USE GPU', txt_img='gpu_2')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.selected_video.grid(row=0, column=0, sticky=NW)
        self.start_frm_eb.grid(row=1, column=0, sticky=NW)
        self.end_frm_eb.grid(row=2, column=0, sticky=NW)
        gpu_cb.grid(row=3, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        check_file_exist_and_readable(file_path=self.selected_video.file_path)
        video_meta_data = get_video_meta_data(video_path=self.selected_video.file_path)
        check_int(name="Start frame", value=self.start_frm_eb.entry_get, min_value=0)
        check_int(name="End frame", value=self.end_frm_eb.entry_get, min_value=1)
        start_frame = int(self.start_frm_eb.entry_get)
        end_frame = int(self.end_frm_eb.entry_get)
        if start_frame >= end_frame:
            raise FrameRangeError(
                msg=f"Start frame ({start_frame}) is after or the same as the end frame ({end_frame})",
                source=__class__.__name__,
            )
        if (start_frame < 0) or (end_frame < 1):
            raise FrameRangeError(
                msg=f"Start frame has to be at least 0 and end frame has to be at least 1",
                source=__class__.__name__,
            )
        if start_frame > video_meta_data["frame_count"]:
            raise FrameRangeError(
                msg=f'The video  has {video_meta_data["frame_count"]} frames, which is less than the start frame: {start_frame}',
                source=__class__.__name__,
            )
        if end_frame > video_meta_data["frame_count"]:
            raise FrameRangeError(
                msg=f'The video  has {video_meta_data["frame_count"]} frames, which is less than the end frame: {end_frame}',
                source=__class__.__name__,
            )
        if self.gpu_var.get() and not check_nvidea_gpu_available():
            raise FFMPEGCodecGPUError('No GPU found on machine. Try unchecking the GPU checkbox', source=self.__class__.__name__)

        frm_ids = [[start_frame, end_frame]]
        _ = clip_videos_by_frame_ids(file_paths=[self.selected_video.file_path], frm_ids=frm_ids, save_dir=None, gpu=self.gpu_var.get())


#ClipSingleVideoByFrameNumbers()


class ClipMultipleVideosByFrameNumbersPopUp(PopUpMixin):

    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 save_dir: Union[str, os.PathLike]):
        """
        :param data_dir: Directory containing video files to be clipped.
        :param save_dir: Directory where to save the clipped videos.

        :example:
        >>> ClipMultipleVideosByFrameNumbersPopUp(data_dir='/Users/simon/Downloads/test__/test__', save_dir='/Users/simon/Downloads/test__/res')
        """

        check_if_dir_exists(in_dir=data_dir, source=self.__class__.__name__, create_if_not_exist=False )
        check_if_dir_exists(in_dir=save_dir, source=self.__class__.__name__, create_if_not_exist=True)
        self.video_paths = find_all_videos_in_directory(directory=data_dir, as_dict=True, raise_error=True)
        self.video_meta_data = [get_video_meta_data(video_path=x)["frame_count"]for x in list(self.video_paths.values())]
        max_video_name_len = len(max(list(self.video_paths.keys())))
        super().__init__(title="CLIP MULTIPLE VIDEOS BY FRAME NUMBERS")
        self.save_dir = save_dir
        data_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        data_frm.grid(row=0, column=0, sticky=NW)
        Label(data_frm, text="VIDEO NAME", width=max_video_name_len, font=Formats.FONT_REGULAR.value).grid(row=0, column=0, sticky=NW)
        Label(data_frm, text="START FRAME", font=Formats.FONT_REGULAR.value, width=10).grid(row=0, column=2)
        Label(data_frm, text="END FRAME", font=Formats.FONT_REGULAR.value, width=10).grid(row=0, column=3)
        self.entry_boxes = {}
        for cnt, video_name in enumerate(self.video_paths.keys()):
            self.entry_boxes[video_name] = {}
            Label(data_frm, text=video_name, width=max_video_name_len, font=Formats.FONT_REGULAR.value).grid(row=cnt + 1, column=0, sticky=NW)
            Label(data_frm, text=self.video_meta_data[cnt], width=max_video_name_len).grid(row=cnt + 1, column=1, sticky=NW)
            self.entry_boxes[video_name]["start"] = Entry_Box(data_frm, "", 5, validation="numeric")
            self.entry_boxes[video_name]["end"] = Entry_Box(data_frm, "", 5, validation="numeric")
            self.entry_boxes[video_name]["start"].grid(row=cnt + 1, column=2, sticky=NW)
            self.entry_boxes[video_name]["end"].grid(row=cnt + 1, column=3, sticky=NW)

        settings_frm = LabelFrame(self.main_frm, text="SETTINGS", font=Formats.FONT_HEADER.value, fg="black", padx=5, pady=5)
        gpu_cb, self.gpu_var = SimbaCheckbox(parent=settings_frm, txt='USE GPU (REDUCED RUNTIME)', txt_img='gpu_2')
        settings_frm.grid(row=1, column=0, sticky=NW)
        gpu_cb.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run, btn_txt_clr="blue")

    def run(self):
        video_paths, frame_ids = [], []
        for cnt, (video_name, v) in enumerate(self.entry_boxes.items()):
            video_paths.append(self.video_paths[video_name])
            video_frm_cnt = self.video_meta_data[cnt]
            check_int(
                name=f"START {video_name}", value=v["start"].entry_get, min_value=0
            )
            check_int(name=f"START {video_name}", value=v["end"].entry_get, min_value=1)
            start, end = int(v["start"].entry_get), int(v["end"].entry_get)
            if start >= end:
                raise FrameRangeError(
                    msg=f"For video {video_name}, the start frame ({start}) is after or the same as the end frame ({end})",
                    source=__class__.__name__,
                )
            if (start < 0) or (end < 1):
                raise FrameRangeError(
                    msg=f"For video {video_name}, start frame has to be at least 0 and end frame has to be at least 1",
                    source=__class__.__name__,
                )
            if start > video_frm_cnt:
                raise FrameRangeError(
                    msg=f"The video {video_name} has {video_frm_cnt} frames, which is less than the start frame: {start}",
                    source=__class__.__name__,
                )
            if end > video_frm_cnt:
                raise FrameRangeError(
                    msg=f"The video {video_name} has {video_frm_cnt} frames, which is less than the end frame: {end}",
                    source=__class__.__name__,
                )
            frame_ids.append([start, end])

        if self.gpu_var.get() and not check_nvidea_gpu_available():
            raise FFMPEGCodecGPUError('No GPU detected on machine. Try unchecking the GPU checkbox', source=self.__class__.__name__)
        _ = clip_videos_by_frame_ids(file_paths=video_paths, frm_ids=frame_ids, save_dir=self.save_dir, gpu=self.gpu_var.get())


class InitiateClipMultipleVideosByFrameNumbersPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CLIP MULTIPLE VIDEOS BY FRAME NUMBERS")
        data_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SELECT DATA DIRECTORIES",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        self.input_folder = FolderSelect(
            data_frm, "Video directory:", title="Select Folder with videos", lblwidth=20
        )
        self.output_folder = FolderSelect(
            data_frm,
            "Output directory:",
            title="Select a folder for your output videos",
            lblwidth=20,
        )
        data_frm.grid(row=0, column=0, sticky=NW)
        self.input_folder.grid(row=0, column=0, sticky=NW)
        self.output_folder.grid(row=1, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        check_if_dir_exists(
            in_dir=self.input_folder.folder_path,
            source=self.__class__.__name__,
            create_if_not_exist=False,
        )
        check_if_dir_exists(
            in_dir=self.output_folder.folder_path,
            source=self.__class__.__name__,
            create_if_not_exist=True,
        )
        self.video_paths = find_all_videos_in_directory(
            directory=self.input_folder.folder_path, as_dict=True, raise_error=True
        )
        self.root.destroy()
        _ = ClipMultipleVideosByFrameNumbersPopUp(
            data_dir=self.input_folder.folder_path,
            save_dir=self.output_folder.folder_path,
        )


class ClipMultipleVideosByTimestamps(PopUpMixin):
    """
    :example:
    >>> ClipMultipleVideosByTimestamps(data_dir=r"C:\troubleshooting\RAT_NOR\project_folder\videos\test", save_dir=r"C:\troubleshooting\RAT_NOR\project_folder\videos\test\out")
    >>> ClipMultipleVideosByTimestamps(data_dir=r"C:\troubleshooting\mitra\project_folder\videos", save_dir=r"C:\troubleshooting\mitra\project_folder\videos\temp_3")
    """
    def __init__(self, data_dir: Union[str, os.PathLike], save_dir: Union[str, os.PathLike]):

        check_if_dir_exists(in_dir=data_dir, source=self.__class__.__name__, create_if_not_exist=False)
        check_if_dir_exists(in_dir=save_dir, source=self.__class__.__name__, create_if_not_exist=True)
        self.video_paths = find_all_videos_in_directory(directory=data_dir, as_dict=True, raise_error=True)
        self.video_meta_data = [get_video_meta_data(video_path=x) for x in list(self.video_paths.values())]
        max_video_name_len = len(max(list(self.video_paths.keys()))) + 5
        super().__init__(title="CLIP MULTIPLE VIDEOS BY TIME-STAMPS")
        self.save_dir = save_dir
        batch_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="BATCH SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        batch_start_entry = Entry_Box(parent=batch_settings_frm, fileDescription='START TIME:', labelwidth=10, entry_box_width=12)
        batch_start_btn = SimbaButton(parent=batch_settings_frm, txt='SET', img='tick', cmd=self._batch_set_val, cmd_kwargs={'text': lambda: batch_start_entry.entry_get.strip(), 'box_type': lambda: 'start'})
        batch_end_entry = Entry_Box(parent=batch_settings_frm, fileDescription='END TIME:', labelwidth=10, entry_box_width=12)
        batch_end_btn = SimbaButton(parent=batch_settings_frm, txt='SET', img='tick', cmd=self._batch_set_val, cmd_kwargs={'text': lambda: batch_end_entry.entry_get.strip(), 'box_type': lambda: 'end'})
        batch_settings_frm.grid(row=0, column=0, sticky=NW)
        batch_start_entry.grid(row=0, column=0, sticky=NW)
        batch_start_btn.grid(row=0, column=1, sticky=NW)
        batch_end_entry.grid(row=1, column=0, sticky=NW)
        batch_end_btn.grid(row=1, column=1, sticky=NW)
        self.save_dir = save_dir
        data_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        data_frm.grid(row=1, column=0, sticky=NW)
        Label(data_frm, text="VIDEO NAME", width=max_video_name_len).grid(row=0, column=0, sticky=NW)
        Label(data_frm, text="VIDEO LENGTH", width=12).grid(row=0, column=1)
        Label(data_frm, text="START TIME (HH:MM:SS)", width=18).grid(row=0, column=2)
        Label(data_frm, text="END TIME (HH:MM:SS)", width=18).grid(row=0, column=3)

        self.entry_boxes = {}
        for cnt, video_name in enumerate(self.video_paths.keys()):
            self.entry_boxes[video_name] = {}
            Label(data_frm, text=video_name, width=max_video_name_len).grid(row=cnt + 1, column=0, sticky=NW)
            video_length = self.video_meta_data[cnt]["video_length_s"]
            video_length_hhmmss = seconds_to_timestamp(seconds=video_length)
            Label(data_frm, text=video_length_hhmmss, width=max_video_name_len).grid(row=cnt + 1, column=1, sticky=NW)
            self.entry_boxes[video_name]["start"] = Entry_Box(data_frm, "", 5)
            self.entry_boxes[video_name]["end"] = Entry_Box(data_frm, "", 5)
            self.entry_boxes[video_name]["start"].grid(row=cnt + 1, column=2, sticky=NW)
            self.entry_boxes[video_name]["end"].grid(row=cnt + 1, column=3, sticky=NW)

        gpu_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="GPU SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        gpu_cb, self.gpu_var = SimbaCheckbox(parent=gpu_frm, txt='USE GPU', txt_img='gpu')
        gpu_frm.grid(row=2, column=0, sticky=NW)
        gpu_cb.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run, btn_txt_clr="blue")
        self.main_frm.mainloop()

    def _batch_set_val(self, text: str, box_type: str):
        for cnt, video_name in enumerate(self.video_paths.keys()):
            if box_type == 'start':
                self.entry_boxes[video_name]["start"].entry_set(text)
            else:
                self.entry_boxes[video_name]["end"].entry_set(text)


    def run(self):
        timer = SimbaTimer(start=True)
        for cnt, (video_name, v) in enumerate(self.entry_boxes.items()):
            start, end = v["start"].entry_get, v["end"].entry_get
            check_if_string_value_is_valid_video_timestamp(value=start, name=f"{video_name} START TIME" )
            check_if_string_value_is_valid_video_timestamp(value=end, name=f"{video_name} END TIME")
            check_that_hhmmss_start_is_before_end(start_time=start, end_time=end, name=video_name)
            check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp=start, video_path=self.video_paths[video_name])
            check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp=end, video_path=self.video_paths[video_name])
            clip_video_in_range(file_path=self.video_paths[video_name], start_time=start, end_time=end, out_dir=self.save_dir, overwrite=True, include_clip_time_in_filename=False, gpu=self.gpu_var.get())
        timer.stop_timer()
        stdout_success(msg=f"{len(self.entry_boxes)} videos clipped by time-stamps and saved in {self.save_dir}", elapsed_time=timer.elapsed_time_str,)





class InitiateClipMultipleVideosByTimestampsPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CLIP MULTIPLE VIDEOS BY TIME-STAMPS")
        data_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SELECT DATA DIRECTORIES",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.VIDEO_TOOLS.value,
        )
        self.input_folder = FolderSelect(
            data_frm, "Video directory:", title="Select Folder with videos", lblwidth=20
        )
        self.output_folder = FolderSelect(
            data_frm,
            "Output directory:",
            title="Select a folder for your output videos",
            lblwidth=20,
        )
        data_frm.grid(row=0, column=0, sticky=NW)
        self.input_folder.grid(row=0, column=0, sticky=NW)
        self.output_folder.grid(row=1, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        check_if_dir_exists(
            in_dir=self.input_folder.folder_path,
            source=self.__class__.__name__,
            create_if_not_exist=False,
        )
        check_if_dir_exists(
            in_dir=self.output_folder.folder_path,
            source=self.__class__.__name__,
            create_if_not_exist=True,
        )
        self.video_paths = find_all_videos_in_directory(
            directory=self.input_folder.folder_path, as_dict=True, raise_error=True
        )
        self.root.destroy()
        _ = ClipMultipleVideosByTimestamps(
            data_dir=self.input_folder.folder_path,
            save_dir=self.output_folder.folder_path,
        )

class BrightnessContrastPopUp(PopUpMixin):
    """
    .. image:: _static/img/brightness_contrast_ui.gif
       :width: 700
       :align: center
    """
    def __init__(self):
        super().__init__(title="CHANGE BRIGHTNESS / CONTRAST")
        self.datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        setting_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.gpu_dropdown = DropDownMenu(setting_frm, "USE GPU:", ['TRUE', 'FALSE'], labelwidth=20)
        self.gpu_dropdown.setChoices('FALSE')
        setting_frm.grid(row=0, column=0, sticky="NW")
        self.gpu_dropdown.grid(row=0, column=0, sticky="NW")

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHANGE BRIGHTNESS / CONTRAST SINGLE VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], lblwidth=25)
        run_video_btn = SimbaButton(parent=single_video_frm, txt="RUN SINGLE VIDEO", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run_video)

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        run_video_btn.grid(row=1, column=0, sticky="NW")

        video_dir_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHANGE BRIGHTNESS / CONTRAST MULTIPLE VIDEOS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_dir = FolderSelect(video_dir_frm, "VIDEO DIRECTORY PATH:", lblwidth=25)

        run_dir_btn = SimbaButton(parent=video_dir_frm, txt="RUN VIDEO DIRECTORY", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run_directory)

        video_dir_frm.grid(row=2, column=0, sticky="NW")
        self.selected_dir.grid(row=0, column=0, sticky="NW")
        run_dir_btn.grid(row=1, column=0, sticky="NW")
        #self.main_frm.mainloop()

    def run_video(self):
        video_path = self.selected_video.file_path
        check_file_exist_and_readable(file_path=video_path)
        ui = BrightnessContrastUI(data=video_path)
        self.brightness, self.contrast = ui.run()
        self.video_paths = [video_path]
        self.apply()

    def run_directory(self):
        video_dir = self.selected_dir.folder_path
        check_if_dir_exists(in_dir=video_dir, source=self.__class__.__name__)
        self.video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        ui = BrightnessContrastUI(data=self.video_paths[0])
        self.brightness, self.contrast = ui.run()
        self.apply()

    def apply(self):
        timer = SimbaTimer(start=True)
        gpu = str_2_bool(self.gpu_dropdown.getChoices())
        for file_cnt, file_path in enumerate(self.video_paths):
            video_timer = SimbaTimer(start=True)
            dir, video_name, ext = get_fn_ext(filepath=file_path)
            print(f'Creating copy of {video_name}...')
            out_path = os.path.join(dir, f'{video_name}_eq_{self.datetime}{ext}')
            if not gpu:
                cmd = f'ffmpeg -i "{file_path}" -vf "eq=brightness={self.brightness}:contrast={self.contrast}" -loglevel error -stats "{out_path}" -y'
            else:
                cmd = f'ffmpeg -hwaccel auto -i "{file_path}" -vf "eq=brightness={self.brightness}:contrast={self.contrast}" -c:v h264_nvenc -loglevel error -stats "{out_path}" -y'
                #cmd = f'ffmpeg -i -hwaccel auto "{file_path}" -vf "eq=brightness={self.brightness}:contrast={self.contrast}" -c:v h264_nvenc -loglevel error -stats "{out_path}" -y'
            subprocess.call(cmd, shell=True, stdout=subprocess.PIPE)
            video_timer.stop_timer()
            stdout_success(msg=f'Video {out_path} complete!', elapsed_time=video_timer.elapsed_time_str)
        timer.stop_timer()
        stdout_success(f'{len(self.video_paths)} video(s) converted.', elapsed_time=timer.elapsed_time_str)


class InteractiveClahePopUp(PopUpMixin):
    """
    .. image:: _static/img/interactive_clahe_ui.gif
       :width: 500
       :align: center
    """

    def __init__(self):
        super().__init__(title="INTERACTIVE CLAHE")
        self.datetime = datetime.now().strftime("%Y%m%d%H%M%S")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings', icon_link=Links.VIDEO_TOOLS.value)
        self.core_cnt_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=list(range(1, find_core_cnt()[0]+1)), label='CORE COUNT:', label_width=25, dropdown_width=20, value=1)
        self.gpu_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label='USE GPU:', label_width=25, dropdown_width=20, value='FALSE')
        if not check_nvidea_gpu_available():
            self.gpu_dropdown.disable()

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=1, column=0, sticky=NW)



        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="INTERACTIVE CLAHE - SINGLE VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], lblwidth=25)
        run_video_btn = SimbaButton(parent=single_video_frm, txt="RUN SINGLE VIDEO", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run_video, width=160)


        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        run_video_btn.grid(row=1, column=0, sticky="NW")

        video_dir_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="INTERACTIVE CLAHE - MULTIPLE VIDEOS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_dir = FolderSelect(video_dir_frm, "VIDEO DIRECTORY PATH:", lblwidth=25)
        run_dir_btn = SimbaButton(parent=video_dir_frm, txt="RUN VIDEO DIRECTORY", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run_directory, width=160)
        video_dir_frm.grid(row=2, column=0, sticky="NW")
        self.selected_dir.grid(row=0, column=0, sticky="NW")
        run_dir_btn.grid(row=1, column=0, sticky="NW")

    def run_video(self):
        video_path = self.selected_video.file_path
        check_file_exist_and_readable(file_path=video_path)
        self.clip_limit, self.tile_size = interactive_clahe_ui(data=video_path)
        self.video_paths = [video_path]
        self.apply()

    def run_directory(self):
        video_dir = self.selected_dir.folder_path
        check_if_dir_exists(in_dir=video_dir, source=self.__class__.__name__)
        self.video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        self.clip_limit, self.tile_size = interactive_clahe_ui(data=self.video_paths[0])
        self.apply()

    def _get_settings(self):
        self.core_cnt = int(self.core_cnt_dropdown.get_value())
        self.gpu = str_2_bool(self.gpu_dropdown.get_value())

    def apply(self):
        timer = SimbaTimer(start=True)
        self._get_settings()
        for file_cnt, file_path in enumerate(self.video_paths):
            dir, video_name, ext = get_fn_ext(filepath=file_path)
            print(f'Creating CLAHE copy of {video_name}...')
            out_path = os.path.join(dir, f'{video_name}_CLAHE_CLIPLIMIT_{int(self.clip_limit)}_TILESIZE_{int(self.tile_size)}_{self.datetime}{ext}')
            if self.core_cnt == 1:
                clahe_enhance_video(file_path=file_path, clip_limit=int(self.clip_limit), tile_grid_size=(int(self.tile_size), int(self.tile_size)), out_path=out_path)
            else:
                clahe_enhance_video_mp(file_path=file_path, clip_limit=int(self.clip_limit), tile_grid_size=(int(self.tile_size), int(self.tile_size)), out_path=out_path, gpu=self.gpu)
        timer.stop_timer()
        stdout_success(f'{len(self.video_paths)} video(s) converted.', elapsed_time=timer.elapsed_time_str)


#_ = InteractiveClahePopUp()


class DownsampleSingleVideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self,title="DOWN-SAMPLE SINGLE VIDEO RESOLUTION", icon='minus')
        choose_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DOWNSAMPLE.value)
        self.video_path_selected = FileSelect(choose_video_frm, "VIDEO PATH: ", title="Select a video file", file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        choose_video_frm.grid(row=0, column=0, sticky=NW)
        self.video_path_selected.grid(row=0, column=0, sticky=NW)

        gpu_frm = LabelFrame(self.main_frm, text="GPU (REDUCED RUNTIMES)", font=Formats.FONT_HEADER.value, fg="black", padx=5, pady=5)
        use_gpu_cb, self.use_gpu_var = SimbaCheckbox(parent=gpu_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')

        gpu_frm.grid(row=1, column=0, sticky=NW)
        use_gpu_cb.grid(row=0, column=0, sticky=NW)

        custom_size_frm = LabelFrame(self.main_frm, text="CUSTOM RESOLUTION",font=Formats.FONT_HEADER.value,fg="black",padx=5,pady=5)
        self.entry_width = Entry_Box(custom_size_frm, "Width", "10", validation="numeric")
        self.entry_height = Entry_Box(custom_size_frm, "Height", "10", validation="numeric")
        self.custom_downsample_btn = SimbaButton(parent=custom_size_frm, txt="DOWN-SAMPLE USING CUSTOM RESOLUTION", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.downsample_custom)


        custom_size_frm.grid(row=2, column=0, sticky=NW)
        self.entry_width.grid(row=0, column=0, sticky=NW)
        self.entry_height.grid(row=1, column=0, sticky=NW)
        self.custom_downsample_btn.grid(row=2, column=0, sticky=NW)

        default_size_frm = LabelFrame(self.main_frm, text="DEFAULT RESOLUTION",font=Formats.FONT_HEADER.value,fg="black",padx=5,pady=5)
        self.width_dropdown = DropDownMenu(default_size_frm, "WIDTH:", Options.RESOLUTION_OPTIONS_2.value, labelwidth=20)
        self.height_dropdown = DropDownMenu(default_size_frm, "HEIGHT:", Options.RESOLUTION_OPTIONS_2.value, labelwidth=20)
        self.width_dropdown.setChoices(640)
        self.height_dropdown.setChoices("AUTO")

        self.default_downsample_btn = SimbaButton(parent=default_size_frm, txt="DOWN-SAMPLE USING DEFAULT RESOLUTION", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.downsample_default)

        default_size_frm.grid(row=3, column=0, sticky=NW)
        self.width_dropdown.grid(row=0, column=0, sticky=NW)
        self.height_dropdown.grid(row=1, column=0, sticky=NW)
        self.default_downsample_btn.grid(row=2, column=0, sticky=NW)

    def _checks(self):
        check_ffmpeg_available(raise_error=True)
        self.gpu = self.use_gpu_var.get()
        if self.gpu:
            check_nvidea_gpu_available()
        self.file_path = self.video_path_selected.file_path
        check_file_exist_and_readable(file_path=self.file_path)
        _ = get_video_meta_data(video_path=self.file_path)

    def downsample_custom(self):
        self._checks()
        width, height = self.entry_width.entry_get, self.entry_height.entry_get
        check_int(name=f'{self.__class__.__name__} width', value=width, min_value=1)
        check_int(name=f'{self.__class__.__name__} height', value=height, min_value=1)
        threading.Thread(target=downsample_video(file_path=self.file_path, video_height=height, video_width=width, gpu=self.gpu)).start()

    def downsample_default(self):
        self._checks()
        width, height = self.width_dropdown.getChoices(), self.height_dropdown.getChoices()
        if width == 'AUTO' and height == 'AUTO':
            raise InvalidInputError(msg='Both width and height cannot be AUTO', source=self.__class__.__name__)
        elif width == 'AUTO':
            resize_videos_by_height(video_paths=[self.file_path], height=int(height), overwrite=False, save_dir=None, gpu=self.gpu, suffix='downsampled', verbose=True)
        elif height == 'AUTO':
            resize_videos_by_width(video_paths=[self.file_path], width=int(width), overwrite=False, save_dir=None, gpu=self.gpu, suffix='downsampled', verbose=True)
        else:
            threading.Thread(target=downsample_video(file_path=self.file_path, video_height=height, video_width=width, gpu=self.gpu)).start()

class DownsampleMultipleVideosPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self,title="DOWN-SAMPLE MULTIPLE VIDEO RESOLUTION", icon='minus')
        choose_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DOWNSAMPLE.value)

        self.video_dir_selected = FolderSelect(choose_video_frm, "VIDEO DIRECTORY:",title="Select Folder with videos", lblwidth=20)
        choose_video_frm.grid(row=0, column=0, sticky=NW)
        self.video_dir_selected.grid(row=0, column=0, sticky=NW)

        gpu_frm = LabelFrame(self.main_frm, text="GPU (REDUCED RUNTIMES)", font=Formats.FONT_HEADER.value, fg="black", padx=5, pady=5)
        use_gpu_cb, self.use_gpu_var = SimbaCheckbox(parent=gpu_frm, txt="Use GPU (reduced runtime)", txt_img='gpu_2')

        gpu_frm.grid(row=1, column=0, sticky=NW)
        use_gpu_cb.grid(row=0, column=0, sticky=NW)

        custom_size_frm = LabelFrame(self.main_frm, text="CUSTOM RESOLUTION",font=Formats.FONT_HEADER.value,fg="black",padx=5,pady=5)
        self.entry_width = Entry_Box(custom_size_frm, "Width", "10", validation="numeric")
        self.entry_height = Entry_Box(custom_size_frm, "Height", "10", validation="numeric")

        self.custom_downsample_btn = SimbaButton(parent=custom_size_frm, txt="DOWN-SAMPLE USING CUSTOM RESOLUTION", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=self.downsample_custom)

        custom_size_frm.grid(row=2, column=0, sticky=NW)
        self.entry_width.grid(row=0, column=0, sticky=NW)
        self.entry_height.grid(row=1, column=0, sticky=NW)
        self.custom_downsample_btn.grid(row=2, column=0, sticky=NW)

        default_size_frm = LabelFrame(self.main_frm, text="DEFAULT RESOLUTION",font=Formats.FONT_HEADER.value,fg="black",padx=5,pady=5)
        self.width_dropdown = DropDownMenu(default_size_frm, "WIDTH:", Options.RESOLUTION_OPTIONS_2.value, labelwidth=20)
        self.height_dropdown = DropDownMenu(default_size_frm, "HEIGHT:", Options.RESOLUTION_OPTIONS_2.value, labelwidth=20)
        self.width_dropdown.setChoices(640)
        self.height_dropdown.setChoices("AUTO")

        self.default_downsample_btn = SimbaButton(parent=default_size_frm, txt="DOWN-SAMPLE USING DEFAULT RESOLUTION", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=self.downsample_default)

        default_size_frm.grid(row=3, column=0, sticky=NW)
        self.width_dropdown.grid(row=0, column=0, sticky=NW)
        self.height_dropdown.grid(row=1, column=0, sticky=NW)
        self.default_downsample_btn.grid(row=2, column=0, sticky=NW)

    def _checks(self):
        check_ffmpeg_available(raise_error=True)
        self.gpu = self.use_gpu_var.get()
        if self.gpu:
            check_nvidea_gpu_available()
        self.video_directory = self.video_dir_selected.folder_path
        check_if_dir_exists(in_dir=self.video_directory)
        self.video_paths = find_all_videos_in_directory(directory=self.video_directory, raise_error=True, as_dict=True)

    def downsample_custom(self):
        self._checks()
        width, height = self.entry_width.entry_get, self.entry_height.entry_get
        check_int(name=f'{self.__class__.__name__} width', value=width, min_value=1)
        check_int(name=f'{self.__class__.__name__} height', value=height, min_value=1)
        for file_path in self.video_paths.values():
            threading.Thread(target=downsample_video(file_path=file_path, video_height=height, video_width=width, gpu=self.gpu)).start()

    def downsample_default(self):
        self._checks()
        width, height = self.width_dropdown.getChoices(), self.height_dropdown.getChoices()
        if width == 'AUTO' and height == 'AUTO':
            raise InvalidInputError(msg='Both width and height cannot be AUTO', source=self.__class__.__name__)
        elif width == 'AUTO':
            resize_videos_by_height(video_paths=list(self.video_paths.values()), height=int(height), overwrite=False, save_dir=None, gpu=self.gpu, suffix='downsampled', verbose=True)
        elif height == 'AUTO':
            resize_videos_by_width(video_paths=list(self.video_paths.values()), width=int(width), overwrite=False, save_dir=None, gpu=self.gpu, suffix='downsampled', verbose=True)
        else:
            for file_path in self.video_paths.values():
                threading.Thread(target=downsample_video(file_path=file_path, video_height=height, video_width=width, gpu=self.gpu)).start()

class Convert2jpegPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CONVERT IMAGE DIRECTORY TO JPEG")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.quality_lbl = Label(settings_frm, text="JPEG OUTPUT QUALITY: ")
        self.quality_scale = Scale(settings_frm, from_=1, to=100, orient=HORIZONTAL, length=200, label='JPEG OUTPUT QUALITY', fg='blue', font=Formats.FONT_REGULAR.value,)
        self.quality_scale.set(95)
        settings_frm.grid(row=0, column=0, sticky="NW")
        self.quality_scale.grid(row=0, column=0, sticky="NW")

        img_dir_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CONVERT IMAGE DIRECTORY TO JPEG", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_frame_dir = FolderSelect(img_dir_frm, "IMAGE DIRECTORY PATH:", title="Select a image directory", lblwidth=25)

        run_btn_dir = SimbaButton(parent=img_dir_frm, txt="RUN DIRECTORY JPEG CONVERSION", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=self.run_dir)
        img_dir_frm.grid(row=1, column=0, sticky="NW")
        self.selected_frame_dir.grid(row=0, column=0, sticky="NW")
        run_btn_dir.grid(row=1, column=0, sticky="NW")

        img_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CONVERT IMAGE TO JPEG", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_file = FileSelect(img_frm, "IMAGE PATH:", title="Select an image file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_IMAGE_FORMAT_OPTIONS.value)])

        run_btn_img = SimbaButton(parent=img_frm, txt="RUN IMAGE JPEG CONVERSION", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=self.run_img)
        img_frm.grid(row=2, column=0, sticky="NW")
        self.selected_file.grid(row=0, column=0, sticky="NW")
        run_btn_img.grid(row=1, column=0, sticky="NW")

    def run_dir(self):
        folder_path = self.selected_frame_dir.folder_path
        check_if_dir_exists(in_dir=folder_path)
        _ = convert_to_jpeg(path=folder_path, quality=int(self.quality_scale.get()), verbose=True)

    def run_img(self):
        file_path = self.selected_file.file_path
        check_file_exist_and_readable(file_path)
        _ = convert_to_jpeg(path=file_path, quality=int(self.quality_scale.get()), verbose=True)


class Convert2bmpPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CONVERT IMAGES TO BMP")
        img_dir_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CONVERT IMAGE DIRECTORY TO BMP", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_frame_dir = FolderSelect(img_dir_frm, "IMAGE DIRECTORY PATH:", title="Select a image directory", lblwidth=25)

        run_btn_dir = SimbaButton(parent=img_dir_frm, txt="RUN DIRECTORY BMP CONVERSION", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=self.run_dir)
        img_dir_frm.grid(row=0, column=0, sticky="NW")
        self.selected_frame_dir.grid(row=0, column=0, sticky="NW")
        run_btn_dir.grid(row=1, column=0, sticky="NW")

        img_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CONVERT IMAGE TO BMP", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_file = FileSelect(img_frm, "IMAGE PATH:", title="Select an image file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_IMAGE_FORMAT_OPTIONS.value)])
        run_btn_img = Button(img_frm, text="RUN IMAGE BMP CONVERSION", font=Formats.FONT_REGULAR.value, command=lambda: self.run_img())

        img_frm.grid(row=1, column=0, sticky="NW")
        self.selected_file.grid(row=0, column=0, sticky="NW")
        run_btn_img.grid(row=1, column=0, sticky="NW")

    def run_dir(self):
        folder_path = self.selected_frame_dir.folder_path
        check_if_dir_exists(in_dir=folder_path)
        _ = convert_to_bmp(path=folder_path, verbose=True)

    def run_img(self):
        file_path = self.selected_file.file_path
        check_file_exist_and_readable(file_path)
        _ = convert_to_bmp(path=file_path, verbose=True)


class Convert2WEBPPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CONVERT IMAGES TO WEBP")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.quality_lbl = Label(settings_frm, text="WEBP OUTPUT QUALITY: ")
        self.quality_scale = Scale(settings_frm, from_=1, to=100, orient=HORIZONTAL, length=200, label='WEBP QUALITY', fg='blue', font=Formats.FONT_REGULAR.value,)
        self.quality_scale.set(95)
        settings_frm.grid(row=0, column=0, sticky="NW")
        self.quality_scale.grid(row=0, column=0, sticky="NW")

        convert_dir_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CONVERT IMAGE DIRECTORY TO WEBP", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_frame_dir = FolderSelect(convert_dir_frm, "IMAGE DIRECTORY PATH:", title="Select a image directory", lblwidth=25)

        run_btn_dir = SimbaButton(parent=convert_dir_frm, txt="RUN IMAGE DIRECTORY WEBP CONVERSION", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=self.run_dir)

        convert_dir_frm.grid(row=1, column=0, sticky="NW")
        self.selected_frame_dir.grid(row=0, column=0, sticky="NW")
        run_btn_dir.grid(row=1, column=0, sticky="NW")

        convert_img_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CONVERT IMAGE FILE TO WEBP", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_file = FileSelect(convert_img_frm, "IMAGE PATH:", title="Select an image file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_IMAGE_FORMAT_OPTIONS.value)])


        run_btn_frm = SimbaButton(parent=convert_img_frm, txt="RUN IMAGE WEBP CONVERSION", img='rocket', txt_clr='black', font=Formats.FONT_REGULAR.value, cmd=self.run_img)
        convert_img_frm.grid(row=2, column=0, sticky="NW")
        self.selected_file.grid(row=0, column=0, sticky="NW")
        run_btn_frm.grid(row=1, column=0, sticky="NW")

        #self.main_frm.mainloop()

    def run_dir(self):
        folder_path = self.selected_frame_dir.folder_path
        check_if_dir_exists(in_dir=folder_path)
        _ = convert_to_webp(path=folder_path, quality=int(self.quality_scale.get()), verbose=True)

    def run_img(self):
        file_path = self.selected_file.file_path
        check_file_exist_and_readable(file_path)
        _ = convert_to_webp(path=file_path, quality=int(self.quality_scale.get()), verbose=True)

class Convert2TIFFPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CONVERT IMAGE DIRECTORY TO TIFF")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_frame_dir = FolderSelect(settings_frm, "IMAGE DIRECTORY PATH:", title="Select a image directory", lblwidth=25)
        self.compression_dropdown = DropDownMenu(settings_frm, "COMPRESSION:", ['raw', 'tiff_deflate', 'tiff_lzw'], labelwidth=25)
        self.compression_dropdown.setChoices('raw')
        self.stack_dropdown = DropDownMenu(settings_frm, "STACK:", ['FALSE', 'TRUE'], labelwidth=25)
        self.stack_dropdown.setChoices('FALSE')
        self.create_run_frm(run_function=self.run, title='RUN TIFF CONVERSION')

        settings_frm.grid(row=0, column=0, sticky="NW")
        self.selected_frame_dir.grid(row=0, column=0, sticky="NW")
        self.compression_dropdown.grid(row=1, column=0, sticky="NW")
        self.stack_dropdown.grid(row=2, column=0, sticky="NW")
        #self.main_frm.mainloop()

    def run(self):
        folder_path = self.selected_frame_dir.folder_path
        check_if_dir_exists(in_dir=folder_path)
        stack = str_2_bool(self.stack_dropdown.getChoices())
        convert_to_tiff(directory=folder_path, compression=self.compression_dropdown.getChoices(), verbose=True, stack=stack)

class Convert2PNGPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CONVERT IMAGE TO PNG")
        img_dir_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CONVERT IMAGE DIRECTORY TO PNG", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_frame_dir = FolderSelect(img_dir_frm, "IMAGE DIRECTORY PATH:", title="Select a image directory", lblwidth=25)
        run_btn_dir = Button(img_dir_frm, text="RUN DIRECTORY PNG CONVERSION", font=Formats.FONT_REGULAR.value, command=lambda: self.run_dir())

        img_dir_frm.grid(row=0, column=0, sticky="NW")
        self.selected_frame_dir.grid(row=0, column=0, sticky="NW")
        run_btn_dir.grid(row=1, column=0, sticky="NW")

        img_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CONVERT IMAGE TO PNG", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_file_dir = FileSelect(img_frm, "IMAGE PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_IMAGE_FORMAT_OPTIONS.value)])
        run_btn_img = Button(img_frm, text="RUN PNG CONVERSION", font=Formats.FONT_REGULAR.value, command=lambda: self.run_img())

        img_frm.grid(row=1, column=0, sticky="NW")
        self.selected_file_dir.grid(row=0, column=0, sticky="NW")
        run_btn_img.grid(row=1, column=0, sticky="NW")
        self.main_frm.mainloop()

    def run_dir(self):
        folder_path = self.selected_frame_dir.folder_path
        check_if_dir_exists(in_dir=folder_path)
        _ = convert_to_png(path=folder_path, verbose=True)

    def run_img(self):
        file_path = self.selected_file_dir.file_path
        check_file_exist_and_readable(file_path)
        _ = convert_to_png(path=file_path, verbose=True)
class Convert2MP4PopUp(PopUpMixin):
    """
    :example:
    >>> Convert2MP4PopUp()
    """
    def __init__(self):

        self.MP4_CODEC_LK = {'HEVC (H.265)': 'libx265', 'H.264 (AVC)': 'libx264', 'VP9': 'vp9', 'GPU (h264_cuvid)': 'h264_cuvid', 'Guaranteed powerpoint compatible': 'powerpoint'}
        self.cpu_codec_qualities = list(range(10, 110, 10))
        self.gpu_codec_qualities = ['Low', 'Medium', 'High']
        super().__init__(title="CONVERT VIDEOS TO MP4", icon='mp4')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings', icon_link=Links.VIDEO_TOOLS.value, padx=5, pady=5, relief='solid')

        self.quality_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=self.cpu_codec_qualities, label="OUTPUT VIDEO QUALITY:", label_width=30, dropdown_width=40, value=60)
        self.codec_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=list(self.MP4_CODEC_LK.keys()), label="COMPRESSION CODEC:", label_width=30, dropdown_width=40, value='H.264 (AVC)', command=self.update_quality_dropdown)
        settings_frm.grid(row=0, column=0, sticky=NW, padx=10, pady=10)
        self.quality_dropdown.grid(row=0, column=0, sticky=NW)
        self.codec_dropdown.grid(row=1, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name='video', icon_link=Links.VIDEO_TOOLS.value, padx=5, pady=5, relief='solid')
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=30, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])

        single_video_run = SimbaButton(parent=single_video_frm, txt="RUN - SINGLE VIDEO", img='rocket', cmd=self.run, cmd_kwargs={'multiple': False})

        single_video_frm.grid(row=1, column=0, sticky=NW, padx=10, pady=10)
        self.selected_video.grid(row=0, column=0, sticky=NW)
        single_video_run.grid(row=1, column=0, sticky=NW)

        multiple_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO DIRECTORY", icon_name='stack', icon_link=Links.VIDEO_TOOLS.value, padx=5, pady=5, relief='solid')
        self.selected_video_dir = FolderSelect(multiple_video_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=30)

        multiple_video_run = SimbaButton(parent=multiple_video_frm, txt="RUN - VIDEO DIRECTORY", img='rocket', cmd=self.run, cmd_kwargs={'multiple': True})
        multiple_video_frm.grid(row=2, column=0, sticky=NW, padx=10, pady=10)
        multiple_video_run.grid(row=1, column=0, sticky=NW)
        self.selected_video_dir.grid(row=0, column=0, sticky=NW)
        self.main_frm.mainloop()

    def update_quality_dropdown(self, k):
        if k == 'GPU (h264_cuvid)': option_lst = self.gpu_codec_qualities
        else: option_lst = self.cpu_codec_qualities
        self.quality_dropdown.dropdown['values'] = option_lst
        self.quality_dropdown.dropdown.set(option_lst[0])

    def run(self, multiple: bool):
        if not multiple:
            video_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=video_path)
        else:
            video_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=video_path, source=self.__class__.__name__)
        codec = self.MP4_CODEC_LK[self.codec_dropdown.getChoices()]
        quality = self.quality_dropdown.getChoices()
        threading.Thread(target=convert_to_mp4(path=video_path, codec=codec, quality=quality))






class Convert2AVIPopUp(PopUpMixin):
    """
    :example:
    >>> Convert2AVIPopUp()
    """

    def __init__(self):
        super().__init__(title="CONVERT VIDEOS TO AVI")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.AVI_CODEC_LK = {'XviD': 'xvid', 'DivX': 'divx', 'MJPEG': 'mjpeg'}
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY:", list(range(10, 110, 10)), labelwidth=25)
        self.quality_dropdown.setChoices(60)
        self.codec_dropdown = DropDownMenu(settings_frm, "COMPRESSION CODEC:", list(self.AVI_CODEC_LK.keys()), labelwidth=25)
        self.codec_dropdown.setChoices('DivX')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.quality_dropdown.grid(row=0, column=0, sticky=NW)
        self.codec_dropdown.grid(row=1, column=0, sticky=NW)
        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))
        single_video_frm.grid(row=1, column=0, sticky=NW)
        self.selected_video.grid(row=0, column=0, sticky=NW)
        single_video_run.grid(row=1, column=0, sticky=NW)

        multiple_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO DIRECTORY", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_video_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_video_run = Button(multiple_video_frm, text="RUN - VIDEO DIRECTORY", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=True))
        multiple_video_frm.grid(row=2, column=0, sticky=NW)
        self.selected_video_dir.grid(row=0, column=0, sticky=NW)
        multiple_video_run.grid(row=1, column=0, sticky=NW)
        #self.main_frm.mainloop()

    def run(self, multiple: bool):
        if not multiple:
            video_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=video_path)
        else:
            video_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=video_path, source=self.__class__.__name__)
        codec = self.AVI_CODEC_LK[self.codec_dropdown.getChoices()]
        quality = int(self.quality_dropdown.getChoices())
        threading.Thread(target=convert_to_avi(path=video_path, codec=codec, quality=quality))

class Convert2WEBMPopUp(PopUpMixin):
    """
    :example:
    >>> Convert2WEBMPopUp()
    """

    def __init__(self):
        super().__init__(title="CONVERT VIDEOS TO WEBM")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.WEBM_CODEC_LK = {'VP8': 'vp8', 'VP9': 'vp9'}
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY:", list(range(10, 110, 10)), labelwidth=25)
        self.quality_dropdown.setChoices(60)
        self.codec_dropdown = DropDownMenu(settings_frm, "COMPRESSION CODEC:", list(self.WEBM_CODEC_LK.keys()), labelwidth=25)
        self.codec_dropdown.setChoices('VP9')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.quality_dropdown.grid(row=0, column=0, sticky=NW)
        self.codec_dropdown.grid(row=1, column=0, sticky=NW)
        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))
        single_video_frm.grid(row=1, column=0, sticky=NW)
        self.selected_video.grid(row=0, column=0, sticky=NW)
        single_video_run.grid(row=1, column=0, sticky=NW)

        multiple_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO DIRECTORY", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_video_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_video_run = Button(multiple_video_frm, text="RUN - VIDEO DIRECTORY", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=True))
        multiple_video_frm.grid(row=2, column=0, sticky=NW)
        self.selected_video_dir.grid(row=0, column=0, sticky=NW)
        multiple_video_run.grid(row=1, column=0, sticky=NW)
        #self.main_frm.mainloop()

    def run(self, multiple: bool):
        if not multiple:
            video_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=video_path)
        else:
            video_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=video_path, source=self.__class__.__name__)
        codec = self.WEBM_CODEC_LK[self.codec_dropdown.getChoices()]
        quality = int(self.quality_dropdown.getChoices())
        threading.Thread(target=convert_to_webm(path=video_path, codec=codec, quality=quality))

class Convert2MOVPopUp(PopUpMixin):
    """
    :example:
    >>> Convert2MOVPopUp()
    """
    def __init__(self):
        super().__init__(title="CONVERT VIDEOS TO MOV")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.MOV_CODEC_LK = {'ProRes Kostya Samanta': 'prores',
                             'Animation': 'animation',
                             'CineForm': 'cineform',
                             'DNxHD/DNxHR': 'dnxhd'}
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY:", list(range(10, 110, 10)), labelwidth=25)
        self.quality_dropdown.setChoices(60)
        self.codec_dropdown = DropDownMenu(settings_frm, "COMPRESSION CODEC:", list(self.MOV_CODEC_LK.keys()), labelwidth=25)
        self.codec_dropdown.setChoices('ProRes Kostya Samanta')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.quality_dropdown.grid(row=0, column=0, sticky=NW)
        self.codec_dropdown.grid(row=1, column=0, sticky=NW)
        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))
        single_video_frm.grid(row=1, column=0, sticky=NW)
        self.selected_video.grid(row=0, column=0, sticky=NW)
        single_video_run.grid(row=1, column=0, sticky=NW)

        multiple_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO DIRECTORY", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_video_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_video_run = Button(multiple_video_frm, text="RUN - VIDEO DIRECTORY", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=True))
        multiple_video_frm.grid(row=2, column=0, sticky=NW)
        self.selected_video_dir.grid(row=0, column=0, sticky=NW)
        multiple_video_run.grid(row=1, column=0, sticky=NW)
        #self.main_frm.mainloop()

    def run(self, multiple: bool):
        if not multiple:
            video_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=video_path)
        else:
            video_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=video_path, source=self.__class__.__name__)
        codec = self.MOV_CODEC_LK[self.codec_dropdown.getChoices()]
        quality = int(self.quality_dropdown.getChoices())
        threading.Thread(target=convert_to_mov(path=video_path, codec=codec, quality=quality))

class SuperimposeWatermarkPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="WATERMARK VIDEOS", icon='watermark_green')
        self.LOCATIONS = {'TOP LEFT': 'top_left', 'TOP RIGHT': 'top_right', 'BOTTOM LEFT': 'bottom_left', 'BOTTOM RIGHT': 'bottom_right', 'CENTER': 'center'}
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        opacities = [round(x, 1) for x in list(np.arange(0.1, 1.1, 0.1))]
        self.selected_img = FileSelect(settings_frm, "WATERMARK IMAGE PATH:", title="Select an image file", file_types=[("VIDEO", Options.ALL_IMAGE_FORMAT_OPTIONS.value)], lblwidth=25)
        self.location_dropdown = DropDownMenu(settings_frm, "WATERMARK LOCATION:", list(self.LOCATIONS.keys()), labelwidth=25)
        self.opacity_dropdown = DropDownMenu(settings_frm, "WATERMARK OPACITY:", opacities, labelwidth=25)
        self.size_dropdown = DropDownMenu(settings_frm, "WATERMARK SCALE %:", list(range(5, 100, 5)), labelwidth=25)
        self.gpu_dropdown = DropDownMenu(settings_frm, "USE GPU", ['TRUE', 'FALSE'], labelwidth=25)

        self.location_dropdown.setChoices('TOP LEFT')
        self.opacity_dropdown.setChoices(0.5)
        self.size_dropdown.setChoices(5)
        self.gpu_dropdown.setChoices('FALSE')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.selected_img.grid(row=0, column=0, sticky=NW)
        self.location_dropdown.grid(row=1, column=0, sticky=NW)
        self.opacity_dropdown.grid(row=2, column=0, sticky=NW)
        self.size_dropdown.grid(row=3, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=4, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE WATERMARK", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE WATERMARK", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        #self.main_frm.mainloop()

    def run(self, multiple: bool):
        gpu = str_2_bool(self.gpu_dropdown.getChoices())
        img_path = self.selected_img.file_path
        loc = self.location_dropdown.getChoices()
        loc = self.LOCATIONS[loc]
        opacity = float(self.opacity_dropdown.getChoices())
        size = float(int(self.size_dropdown.getChoices()) / 100)
        if size == 1.0: size = size - 0.001
        check_file_exist_and_readable(file_path=img_path)
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=watermark_video(video_path=data_path,
                                                watermark_path=img_path, position=loc,
                                                opacity=opacity,
                                                scale=size,
                                                gpu=gpu)).start()
#SuperimposeWatermarkPopUp()


class SuperimposeTimerPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="SUPER-IMPOSE TIME ON VIDEOS", icon='superimpose')
        self.LOCATIONS = {'TOP LEFT': 'top_left', 'TOP RIGHT': 'top_right', 'TOP MIDDLE': 'top_middle', 'BOTTOM LEFT': 'bottom_left', 'BOTTOM RIGHT': 'bottom_right', 'BOTTOM MIDDLE': 'bottom_middle'}
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.color_dict = get_color_dict()
        self.font_dict = get_fonts()
        gpu_available = NORMAL if check_nvidea_gpu_available() else DISABLED

        self.location_dropdown = SimBADropDown(parent=settings_frm, label="TIMER LOCATION:", dropdown_options=list(self.LOCATIONS.keys()), label_width=30, dropdown_width=35, value='TOP LEFT')
        self.font_dropdown = SimBADropDown(parent=settings_frm, label="TIMER FONT:", dropdown_options=list(self.font_dict.keys()), label_width=30, dropdown_width=35, value='Arial')
        self.font_size_dropdown = SimBADropDown(parent=settings_frm, label="FONT SIZE:", dropdown_options=list(range(20, 100, 5)), label_width=30, dropdown_width=35, value=20)
        self.font_color_dropdown = SimBADropDown(parent=settings_frm, label="FONT COLOR:", dropdown_options=list(self.color_dict.keys()), label_width=30, dropdown_width=35, value='White')
        self.font_border_dropdown = SimBADropDown(parent=settings_frm, label="FONT BORDER COLOR:", dropdown_options=list(self.color_dict.keys()), label_width=30, dropdown_width=35, value='Black')
        self.font_border_width_dropdown = SimBADropDown(parent=settings_frm, label="FONT BORDER WIDTH:", dropdown_options=list(range(2, 52, 2)), label_width=30, dropdown_width=35, value=2)
        self.timer_format_dropdown = SimBADropDown(parent=settings_frm, label="TIME FORMAT:", dropdown_options=['MM:SS', 'HH:MM:SS', 'SS.MMMMMM', 'HH:MM:SS.MMMM'], label_width=30, dropdown_width=35, value='HH:MM:SS.MMMM')
        self.gpu_dropdown = SimBADropDown(parent=settings_frm, label="USE GPU:", dropdown_options=['TRUE', 'FALSE'], label_width=30, dropdown_width=35, value='FALSE', state=gpu_available)

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.location_dropdown.grid(row=0, column=0, sticky=NW)
        self.font_dropdown.grid(row=1, column=0, sticky=NW)
        self.font_size_dropdown.grid(row=2, column=0, sticky=NW)
        self.font_color_dropdown.grid(row=3, column=0, sticky=NW)
        self.font_border_dropdown.grid(row=4, column=0, sticky=NW)
        self.font_border_width_dropdown.grid(row=5, column=0, sticky=NW)
        self.timer_format_dropdown.grid(row=6, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=7, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE TIMER", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE TIMER", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        #self.main_frm.mainloop()

    def run(self, multiple: bool):
        gpu = str_2_bool(self.gpu_dropdown.getChoices())
        loc = self.location_dropdown.getChoices()
        loc = self.LOCATIONS[loc]
        font_size = int(self.font_size_dropdown.getChoices())
        font = self.font_dropdown.getChoices()
        font_clr = self.font_color_dropdown.getChoices()
        font_border_clr = self.font_border_dropdown.getChoices()
        font_border_width = int(self.font_border_width_dropdown.getChoices())
        timer_format = self.timer_format_dropdown.getChoices()
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        superimpose_elapsed_time(video_path=data_path,
                                            font=font,
                                            font_size=font_size,
                                            font_color=font_clr,
                                            font_border_color=font_border_clr,
                                            font_border_width=font_border_width,
                                            time_format=timer_format,
                                            position=loc,
                                            gpu=gpu)


#SuperimposeTimerPopUp()


class SuperimposeProgressBarPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="SUPER-IMPOSE PROGRESS BAR ON VIDEOS", icon='superimpose')
        self.LOCATIONS = {'TOP': 'top', 'BOTTOM': 'bottom'}
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.color_dict = get_color_dict()
        size_lst = list(range(0, 110, 5))
        size_lst[0] = 1
        self.bar_loc_dropdown = DropDownMenu(settings_frm, "PROGRESS BAR LOCATION:", list(self.LOCATIONS.keys()), labelwidth=25)
        self.bar_color_dropdown = DropDownMenu(settings_frm, "PROGRESS BAR COLOR:", list(self.color_dict.keys()), labelwidth=25)
        self.bar_size_dropdown = DropDownMenu(settings_frm, "PROGRESS BAR HEIGHT (%):", size_lst, labelwidth=25)
        self.gpu_dropdown = DropDownMenu(settings_frm, "USE GPU:", ['TRUE', 'FALSE'], labelwidth=25)

        self.bar_color_dropdown.setChoices('Red')
        self.bar_size_dropdown.setChoices(10)
        self.bar_loc_dropdown.setChoices('BOTTOM')
        self.gpu_dropdown.setChoices('FALSE')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.bar_loc_dropdown.grid(row=0, column=0, sticky=NW)
        self.bar_color_dropdown.grid(row=1, column=0, sticky=NW)
        self.bar_size_dropdown.grid(row=2, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=3, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE PROGRESS BAR", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE PROGRESS BAR", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        #self.main_frm.mainloop()

    def run(self, multiple: bool):
        gpu = str_2_bool(self.gpu_dropdown.getChoices())
        loc = self.bar_loc_dropdown.getChoices()
        loc = self.LOCATIONS[loc]
        bar_clr = self.bar_color_dropdown.getChoices()
        bar_size = int(self.bar_size_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=superimpose_video_progressbar(video_path=data_path,
                                                              bar_height=bar_size,
                                                              color=bar_clr,
                                                              position=loc,
                                                              gpu=gpu)).start()

class SuperimposeVideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="SUPER-IMPOSE VIDEO ON VIDEO", icon='superimpose')
        self.LOCATIONS = {'TOP LEFT': 'top_left', 'TOP RIGHT': 'top_right', 'BOTTOM LEFT': 'bottom_left', 'BOTTOM RIGHT': 'bottom_right', 'CENTER': 'center'}
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        opacities = [round(x, 1) for x in list(np.arange(0.1, 1.1, 0.1))]
        scales = [round(x, 2) for x in list(np.arange(0.05, 1.0, 0.05))]
        self.main_video_path = FileSelect(settings_frm, "MAIN VIDEO PATH:", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_OPTIONS.value)], lblwidth=25)
        self.overlay_video_path = FileSelect(settings_frm, "OVERLAY VIDEO PATH:", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_OPTIONS.value)], lblwidth=25)
        self.location_dropdown = DropDownMenu(settings_frm, "OVERLAY VIDEO LOCATION:", list(self.LOCATIONS.keys()), labelwidth=25)
        self.opacity_dropdown = DropDownMenu(settings_frm, "OVERLAY VIDEO OPACITY:", opacities, labelwidth=25)
        self.size_dropdown = DropDownMenu(settings_frm, "OVERLAY VIDEO SCALE (%):", scales, labelwidth=25)
        self.gpu_dropdown = DropDownMenu(settings_frm, "USE GPU:", ['TRUE', 'FALSE'], labelwidth=25)

        self.location_dropdown.setChoices('TOP LEFT')
        self.opacity_dropdown.setChoices(0.5)
        self.size_dropdown.setChoices(0.05)
        self.gpu_dropdown.setChoices('FALSE')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.main_video_path.grid(row=0, column=0, sticky=NW)
        self.overlay_video_path.grid(row=1, column=0, sticky=NW)
        self.location_dropdown.grid(row=2, column=0, sticky=NW)
        self.opacity_dropdown.grid(row=3, column=0, sticky=NW)
        self.size_dropdown.grid(row=4, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=5, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        #self.main_frm.mainloop()

    def run(self):
        loc = self.location_dropdown.getChoices()
        loc = self.LOCATIONS[loc]
        gpu = str_2_bool(self.gpu_dropdown.getChoices())
        opacity = float(self.opacity_dropdown.getChoices())
        size = float(self.size_dropdown.getChoices())
        video_path = self.main_video_path.file_path
        overlay_path = self.overlay_video_path.file_path
        check_file_exist_and_readable(file_path=video_path)
        check_file_exist_and_readable(file_path=overlay_path)
        threading.Thread(target=superimpose_overlay_video(video_path=video_path,
                                                          overlay_video_path=overlay_path,
                                                          position=loc,
                                                          opacity=opacity,
                                                          scale=size,
                                                          gpu=gpu)).start()

class SuperimposeVideoNamesPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="SUPER-IMPOSE VIDEO NAMES ON VIDEOS", icon='superimpose')
        self.LOCATIONS = {'TOP LEFT': 'top_left', 'TOP RIGHT': 'top_right', 'TOP MIDDLE': 'top_middle', 'BOTTOM LEFT': 'bottom_left', 'BOTTOM RIGHT': 'bottom_right', 'BOTTOM MIDDLE': 'bottom_middle'}
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.color_dict = get_color_dict()
        self.font_dict = get_fonts()

        self.location_dropdown = DropDownMenu(settings_frm, "VIDEO NAME TEXT LOCATION:", list(self.LOCATIONS.keys()), labelwidth=25)
        self.font_dropdown = DropDownMenu(settings_frm, "FONT:", list(self.font_dict.keys()), labelwidth=25)
        self.font_size_dropdown = DropDownMenu(settings_frm, "FONT SIZE:", list(range(5, 105, 5)), labelwidth=25)
        self.font_color_dropdown = DropDownMenu(settings_frm, "FONT COLOR:", list(self.color_dict.keys()), labelwidth=25)
        self.font_border_dropdown = DropDownMenu(settings_frm, "FONT BORDER COLOR:", list(self.color_dict.keys()), labelwidth=25)
        self.font_border_width_dropdown = DropDownMenu(settings_frm, "FONT BORDER WIDTH:", list(range(2, 52, 2)), labelwidth=25)
        self.gpu_dropdown = DropDownMenu(settings_frm, "USE GPU:", ['TRUE', 'FALSE'], labelwidth=25)



        self.location_dropdown.setChoices('TOP LEFT')
        self.font_size_dropdown.setChoices(20)
        self.font_dropdown.setChoices('Arial')
        self.font_color_dropdown.setChoices('White')
        self.font_border_dropdown.setChoices('Black')
        self.font_border_width_dropdown.setChoices(2)
        self.gpu_dropdown.setChoices('FALSE')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.location_dropdown.grid(row=0, column=0, sticky=NW)
        self.font_dropdown.grid(row=1, column=0, sticky=NW)
        self.font_size_dropdown.grid(row=2, column=0, sticky=NW)
        self.font_color_dropdown.grid(row=3, column=0, sticky=NW)
        self.font_border_dropdown.grid(row=4, column=0, sticky=NW)
        self.font_border_width_dropdown.grid(row=5, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=6, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE VIDEO NAME", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE VIDEO NAME", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        #self.main_frm.mainloop()

    def run(self, multiple: bool):
        loc = self.location_dropdown.getChoices()
        loc = self.LOCATIONS[loc]
        gpu = str_2_bool(self.gpu_dropdown.getChoices())
        font_size = int(self.font_size_dropdown.getChoices())
        font = self.font_dropdown.getChoices()
        font_clr = self.font_color_dropdown.getChoices()
        font_border_clr = self.font_border_dropdown.getChoices()
        font_border_width = int(self.font_border_width_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=superimpose_video_names(video_path=data_path,
                                                        font=font,
                                                        font_size=font_size,
                                                        font_color=font_clr,
                                                        font_border_color=font_border_clr,
                                                        font_border_width=font_border_width,
                                                        position=loc,
                                                        gpu=gpu)).start()

class SuperimposeTextPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="SUPER-IMPOSE TEXT ON VIDEOS", icon='superimpose')
        self.LOCATIONS = {'TOP LEFT': 'top_left', 'TOP RIGHT': 'top_right', 'TOP MIDDLE': 'top_middle', 'BOTTOM LEFT': 'bottom_left', 'BOTTOM RIGHT': 'bottom_right', 'BOTTOM MIDDLE': 'bottom_middle'}
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.color_dict = get_color_dict()
        self.font_dict = get_fonts()

        self.location_dropdown = DropDownMenu(settings_frm, "TEXT LOCATION:", list(self.LOCATIONS.keys()), labelwidth=25)
        self.text_eb = Entry_Box(parent=settings_frm, labelwidth=25, entry_box_width=50, fileDescription='TEXT:')
        self.font_dropdown = DropDownMenu(settings_frm, "FONT:", list(self.font_dict.keys()), labelwidth=25)
        self.font_size_dropdown = DropDownMenu(settings_frm, "FONT SIZE:", list(range(5, 105, 5)), labelwidth=25)
        self.font_color_dropdown = DropDownMenu(settings_frm, "FONT COLOR:", list(self.color_dict.keys()), labelwidth=25)
        self.font_border_dropdown = DropDownMenu(settings_frm, "FONT BORDER COLOR:", list(self.color_dict.keys()), labelwidth=25)
        self.font_border_width_dropdown = DropDownMenu(settings_frm, "FONT BORDER WIDTH:", list(range(2, 52, 2)), labelwidth=25)
        self.gpu_dropdown = DropDownMenu(settings_frm, "USE GPU:", ['TRUE', 'FALSE'], labelwidth=25)



        self.location_dropdown.setChoices('TOP LEFT')
        self.font_size_dropdown.setChoices(20)
        self.font_dropdown.setChoices('Arial')
        self.font_color_dropdown.setChoices('White')
        self.font_border_dropdown.setChoices('Black')
        self.font_border_width_dropdown.setChoices(2)
        self.gpu_dropdown.setChoices('FALSE')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.location_dropdown.grid(row=0, column=0, sticky=NW)
        self.text_eb.grid(row=1, column=0, sticky=NW)
        self.font_dropdown.grid(row=2, column=0, sticky=NW)
        self.font_size_dropdown.grid(row=3, column=0, sticky=NW)
        self.font_color_dropdown.grid(row=4, column=0, sticky=NW)
        self.font_border_dropdown.grid(row=5, column=0, sticky=NW)
        self.font_border_width_dropdown.grid(row=6, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=7, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE TEXT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE TEXT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", font=Formats.FONT_REGULAR.value,  command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        #self.main_frm.mainloop()

    def run(self, multiple: bool):
        loc = self.location_dropdown.getChoices()
        loc = self.LOCATIONS[loc]
        text = self.text_eb.entry_get
        gpu = str_2_bool(self.gpu_dropdown.getChoices())
        check_str(name='text', value=text)
        font = self.font_dropdown.getChoices()
        font_size = int(self.font_size_dropdown.getChoices())
        font_clr = self.font_color_dropdown.getChoices()
        font_border_clr = self.font_border_dropdown.getChoices()
        font_border_width = int(self.font_border_width_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=superimpose_freetext(video_path=data_path,
                                                     text=text,
                                                     font=font,
                                                     font_size=font_size,
                                                     font_color=font_clr,
                                                     font_border_color=font_border_clr,
                                                     font_border_width=font_border_width,
                                                     position=loc,
                                                     gpu=gpu)).start()


class BoxBlurPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="BOX BLUR VIDEOS", icon='smooth')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        blur_lvl = [round(x, 2) for x in list(np.arange(0.05, 1.0, 0.05))]
        self.blur_lvl_dropdown = DropDownMenu(settings_frm, "BLUR LEVEL:", blur_lvl, labelwidth=25)
        self.invert_dropdown = DropDownMenu(settings_frm, "INVERT BLUR REGION:", ['TRUE', 'FALSE'], labelwidth=25)

        self.blur_lvl_dropdown.setChoices(0.02)
        self.invert_dropdown.setChoices('FALSE')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.blur_lvl_dropdown.grid(row=0, column=0, sticky=NW)
        self.invert_dropdown.grid(row=1, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="APPLY BOX-BLUR", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN", font=Formats.FONT_REGULAR.value, command=lambda: self.run())

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        #self.main_frm.mainloop()

    def run(self):
        video_path = self.selected_video.file_path
        check_file_exist_and_readable(file_path=video_path)
        blur_lvl = float(self.blur_lvl_dropdown.getChoices())
        invert = str_2_bool(self.invert_dropdown.getChoices())
        threading.Thread(target=roi_blurbox(video_path=video_path, blur_level=blur_lvl, invert=invert)).start()


class BackgroundRemoverSingleVideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="REMOVE BACKGROUND IN A VIDEO", icon='black_and_white')
        self.clr_dict = get_color_dict()
        self.foreground_clr_options = list(self.clr_dict.keys())
        self.foreground_clr_options.append('Original')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.video_path = FileSelect(settings_frm, "VIDEO PATH:", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_OPTIONS.value)], lblwidth=45)
        self.bg_video_path = FileSelect(settings_frm, "BACKGROUND REFERENCE VIDEO PATH (OPTIONAL):", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_OPTIONS.value)], lblwidth=45)
        self.bg_clr_dropdown = DropDownMenu(settings_frm, "BACKGROUND COLOR:", list(self.clr_dict.keys()), labelwidth=45)
        self.fg_clr_dropdown = DropDownMenu(settings_frm, "FOREGROUND COLOR:", self.foreground_clr_options, labelwidth=45)
        self.bg_threshold_dropdown = DropDownMenu(settings_frm, "BACKGROUND THRESHOLD:", list(range(1, 100)), labelwidth=45)
        self.bg_start_eb = Entry_Box(parent=settings_frm, labelwidth=45, entry_box_width=15, fileDescription='BACKGROUND VIDEO START (FRAME # OR TIME):')
        self.bg_end_eb = Entry_Box(parent=settings_frm, labelwidth=45, entry_box_width=15, fileDescription='BACKGROUND VIDEO END (FRAME # OR TIME):')
        self.bg_start_eb.set_state(DISABLED)
        self.bg_end_eb.set_state(DISABLED)
        self.entire_video_as_bg_var = BooleanVar(value=True)
        self.entire_video_as_bg_cb = Checkbutton(settings_frm, text="COMPUTE BACKGROUND FROM ENTIRE VIDEO", font=Formats.FONT_REGULAR.value, variable=self.entire_video_as_bg_var, command=lambda: self.enable_entrybox_from_checkbox(check_box_var=self.entire_video_as_bg_var, entry_boxes=[self.bg_start_eb, self.bg_end_eb], reverse=True))
        self.multiprocessing_var = BooleanVar(value=True)
        self.multiprocess_cb = Checkbutton(settings_frm, text="MULTIPROCESS VIDEO (FASTER)", font=Formats.FONT_REGULAR.value, variable=self.multiprocessing_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.multiprocessing_var, dropdown_menus=[self.multiprocess_dropdown]))
        self.multiprocess_dropdown = DropDownMenu(settings_frm, "CPU cores:", list(range(2, self.cpu_cnt)), "12")
        self.multiprocess_dropdown.setChoices(self.cpu_cnt)
        self.bg_clr_dropdown.setChoices('White')
        self.fg_clr_dropdown.setChoices('Original')
        self.bg_start_eb.entry_set('00:00:00')
        self.bg_end_eb.entry_set('00:00:20')
        self.bg_threshold_dropdown.setChoices('30')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.video_path.grid(row=0, column=0, sticky=NW)
        self.bg_video_path.grid(row=1, column=0, sticky=NW)
        self.bg_clr_dropdown.grid(row=2, column=0, sticky=NW)
        self.fg_clr_dropdown.grid(row=3, column=0, sticky=NW)
        self.bg_threshold_dropdown.grid(row=4, column=0, sticky=NW)
        self.entire_video_as_bg_cb.grid(row=5, column=0, sticky=NW)
        self.bg_start_eb.grid(row=6, column=0, sticky=NW)
        self.bg_end_eb.grid(row=7, column=0, sticky=NW)
        self.multiprocess_cb.grid(row=8, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=9, column=1, sticky=NW)

        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        video_path = self.video_path.file_path
        video_meta_data = get_video_meta_data(video_path=video_path)
        bg_video = self.bg_video_path.file_path
        bg_threshold = int(self.bg_threshold_dropdown.getChoices())
        bg_threshold = int((bg_threshold/100) * 255)
        bg_clr = self.colors_dict[self.bg_clr_dropdown.getChoices()]
        fg_clr = self.fg_clr_dropdown.getChoices()
        if fg_clr != 'Original':
            fg_clr = self.colors_dict[self.fg_clr_dropdown.getChoices()]
        else:
            fg_clr = None
        if bg_clr == fg_clr:
            raise DuplicationError(msg=f'The background and foreground color cannot be the same color ({fg_clr})', source=self.__class__.__name__)
        if not os.path.isfile(bg_video):
            bg_video = deepcopy(video_path)
        else:
            _ = get_video_meta_data(video_path=bg_video)
        if not self.entire_video_as_bg_var.get():
            start, end = self.bg_start_eb.entry_get.strip(), self.bg_end_eb.entry_get.strip()
            int_start, _ = check_int(name='', value=start, min_value=0, raise_error=False)
            int_end, _ = check_int(name='', value=end, min_value=0, raise_error=False)
            if int_start and int_end:
                bg_start_time, bg_end_time = None, None
                bg_start_frm, bg_end_frm = int(int_start), int(int_end)
                if bg_start_frm >= bg_end_frm:
                    raise InvalidInputError(msg=f'Start frame has to be before end frame (start: {bg_start_frm}, end: {bg_end_frm})', source=self.__class__.__name__)
            else:
                check_if_string_value_is_valid_video_timestamp(value=start, name='START FRAME')
                check_if_string_value_is_valid_video_timestamp(value=end, name='END FRAME')
                check_that_hhmmss_start_is_before_end(start_time=start, end_time=end, name='START AND END TIME')
                bg_start_frm, bg_end_frm = None, None
                bg_start_time, bg_end_time = start, end
        else:
            bg_video_meta_data = get_video_meta_data(video_path=bg_video)
            bg_start_frm, bg_end_frm = 0, bg_video_meta_data['frame_count']
            bg_start_time, bg_end_time = None, None


        print(f'Running background subtraction for video {video_meta_data["video_name"]}...')
        if not self.multiprocessing_var.get():
            video_bg_subtraction(video_path=video_path,
                                 bg_video_path=bg_video,
                                 bg_start_frm=bg_start_frm,
                                 bg_end_frm=bg_end_frm,
                                 bg_start_time=bg_start_time,
                                 bg_end_time=bg_end_time,
                                 bg_color=bg_clr,
                                 fg_color=fg_clr,
                                 threshold=bg_threshold)
        else:
            core_cnt = int(self.multiprocess_dropdown.getChoices())
            video_bg_subtraction_mp(video_path=video_path,
                                    bg_video_path=bg_video,
                                    bg_start_frm=bg_start_frm,
                                    bg_end_frm=bg_end_frm,
                                    bg_start_time=bg_start_time,
                                    bg_end_time=bg_end_time,
                                    bg_color=bg_clr,
                                    fg_color=fg_clr,
                                    core_cnt=core_cnt,
                                    threshold=bg_threshold)


class BackgroundRemoverDirectoryPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="REMOVE BACKGROUNDS IN MULTIPLE VIDEOS", icon='black_and_white')
        self.clr_dict = get_color_dict()
        self.foreground_clr_options = list(self.clr_dict.keys())
        self.foreground_clr_options.append('Original')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.dir_path = FolderSelect(settings_frm, "VIDEO DIRECTORY:", lblwidth=45)
        self.bg_dir_path = FolderSelect(settings_frm, "BACKGROUND VIDEO DIRECTORY (OPTIONAL):", lblwidth=45)
        self.bg_clr_dropdown = DropDownMenu(settings_frm, "BACKGROUND COLOR:", list(self.clr_dict.keys()), labelwidth=45)
        self.fg_clr_dropdown = DropDownMenu(settings_frm, "FOREGROUND COLOR:", self.foreground_clr_options, labelwidth=45)
        self.bg_threshold_dropdown = DropDownMenu(settings_frm, "BACKGROUND THRESHOLD:", list(range(1, 100)), labelwidth=45)
        self.bg_start_eb = Entry_Box(parent=settings_frm, labelwidth=45, entry_box_width=15, fileDescription='BACKGROUND VIDEO START (FRAME # OR TIME):')
        self.bg_end_eb = Entry_Box(parent=settings_frm, labelwidth=45, entry_box_width=15, fileDescription='BACKGROUND VIDEO END (FRAME # OR TIME):')
        self.bg_start_eb.set_state(DISABLED)
        self.bg_end_eb.set_state(DISABLED)
        self.entire_video_as_bg_var = BooleanVar(value=True)
        self.entire_video_as_bg_cb = Checkbutton(settings_frm, text="COMPUTE BACKGROUND FROM ENTIRE VIDEO", font=Formats.FONT_REGULAR.value, variable=self.entire_video_as_bg_var, command=lambda: self.enable_entrybox_from_checkbox(check_box_var=self.entire_video_as_bg_var, entry_boxes=[self.bg_start_eb, self.bg_end_eb], reverse=True))
        self.multiprocessing_var = BooleanVar(value=True)
        self.multiprocess_cb = Checkbutton(settings_frm, text="MULTIPROCESS VIDEO (FASTER)", font=Formats.FONT_REGULAR.value, variable=self.multiprocessing_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.multiprocessing_var, dropdown_menus=[self.multiprocess_dropdown]))
        self.multiprocess_dropdown = DropDownMenu(settings_frm, "CPU cores:", list(range(2, self.cpu_cnt)), "12")
        self.multiprocess_dropdown.setChoices(self.cpu_cnt)
        self.bg_clr_dropdown.setChoices('White')
        self.fg_clr_dropdown.setChoices('Original')
        self.bg_start_eb.entry_set('00:00:00')
        self.bg_end_eb.entry_set('00:00:20')
        self.bg_threshold_dropdown.setChoices('30')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.dir_path.grid(row=0, column=0, sticky=NW)
        self.bg_dir_path.grid(row=1, column=0, sticky=NW)
        self.bg_clr_dropdown.grid(row=2, column=0, sticky=NW)
        self.fg_clr_dropdown.grid(row=3, column=0, sticky=NW)
        self.bg_threshold_dropdown.grid(row=4, column=0, sticky=NW)
        self.entire_video_as_bg_cb.grid(row=5, column=0, sticky=NW)
        self.bg_start_eb.grid(row=6, column=0, sticky=NW)
        self.bg_end_eb.grid(row=7, column=0, sticky=NW)
        self.multiprocess_cb.grid(row=8, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=9, column=1, sticky=NW)
        self.create_run_frm(run_function=self.run)
        #self.main_frm.mainloop()

    def run(self):
        videos_directory_path = self.dir_path.folder_path
        bg_videos_directory_path = self.bg_dir_path.folder_path
        bg_threshold = int(self.bg_threshold_dropdown.getChoices())
        bg_threshold = int((bg_threshold/100) * 255)
        check_if_dir_exists(in_dir=videos_directory_path)
        bg_clr = self.colors_dict[self.bg_clr_dropdown.getChoices()]
        fg_clr = self.fg_clr_dropdown.getChoices()
        if fg_clr != 'Original':
            fg_clr = self.colors_dict[self.fg_clr_dropdown.getChoices()]
        else:
            fg_clr = None
        if bg_clr == fg_clr:
            raise DuplicationError(msg=f'The background and foreground color cannot be the same color ({fg_clr})', source=self.__class__.__name__)
        video_paths = find_all_videos_in_directory(directory=videos_directory_path, as_dict=True, raise_error=True)
        if os.path.isdir(bg_videos_directory_path):
            bg_video_paths = find_all_videos_in_directory(directory=bg_videos_directory_path, as_dict=True, raise_error=True)
            video_paths_names, bg_video_paths_names = list(video_paths.keys()), list(bg_video_paths.keys())
            missing_bg_videos = [x for x in video_paths_names if x not in bg_video_paths_names]
            if len(missing_bg_videos) > 0:
                raise NoDataError(msg=f'Not all videos in {videos_directory_path} directory are represented in the {bg_videos_directory_path} directory', source=self.__class__.__name__)
        else:
            bg_video_paths = deepcopy(video_paths)
        if not self.entire_video_as_bg_var.get():
            start, end = self.bg_start_eb.entry_get.strip(), self.bg_end_eb.entry_get.strip()
            int_start, _ = check_int(name='', value=start, min_value=0, raise_error=False)
            int_end, _ = check_int(name='', value=end, min_value=0, raise_error=False)
            if int_start and int_end:
                bg_start_time, bg_end_time = None, None
                bg_start_frm, bg_end_frm = int(int_start), int(int_end)
                if bg_start_frm >= bg_end_frm:
                    raise InvalidInputError(msg=f'Start frame has to be before end frame (start: {bg_start_frm}, end: {bg_end_frm})', source=self.__class__.__name__)
            else:
                check_if_string_value_is_valid_video_timestamp(value=start, name='START FRAME')
                check_if_string_value_is_valid_video_timestamp(value=end, name='END FRAME')
                check_that_hhmmss_start_is_before_end(start_time=start, end_time=end, name='START AND END TIME')
                bg_start_frm, bg_end_frm = None, None
                bg_start_time, bg_end_time = start, end

        for cnt, (video_name, video_path) in enumerate(video_paths.items()):
            print(f'Running background subtraction for video {video_name}... (Video {cnt+1}/{len(list(video_paths.keys()))})')
            bg_video_path = bg_video_paths[video_name]
            if self.entire_video_as_bg_var.get():
                bg_video_meta_data = get_video_meta_data(video_path=bg_video_path)
                bg_start_frm, bg_end_frm = 0, bg_video_meta_data['frame_count']
                bg_start_time, bg_end_time = None, None

            if not self.multiprocessing_var.get():
                video_bg_subtraction(video_path=video_path,
                                     bg_video_path=bg_video_path,
                                     bg_start_frm=bg_start_frm,
                                     bg_end_frm=bg_end_frm,
                                     bg_start_time=bg_start_time,
                                     bg_end_time=bg_end_time,
                                     bg_color=bg_clr,
                                     fg_color=fg_clr,
                                     threshold=bg_threshold)
            else:
                core_cnt = int(self.multiprocess_dropdown.getChoices())
                video_bg_subtraction_mp(video_path=video_path,
                                        bg_video_path=bg_video_path,
                                        bg_start_frm=bg_start_frm,
                                        bg_end_frm=bg_end_frm,
                                        bg_start_time=bg_start_time,
                                        bg_end_time=bg_end_time,
                                        bg_color=bg_clr,
                                        fg_color=fg_clr,
                                        core_cnt=core_cnt,
                                        threshold=bg_threshold)



class RotateVideoSetDegreesPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="ROTATE VIDEOS", icon='rotate')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.degrees_dropdown = DropDownMenu(settings_frm, "CLOCKWISE DEGREES:", list(range(1, 360, 1)), labelwidth=25)
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY (%):", list(range(10, 110, 10)), labelwidth=25)
        self.gpu_dropdown = DropDownMenu(settings_frm, "USE GPU:", ['TRUE', 'FALSE'], labelwidth=25)

        self.quality_dropdown.setChoices(60)
        self.degrees_dropdown.setChoices('90')
        self.gpu_dropdown.setChoices('FALSE')
        self.degrees_dropdown.grid(row=0, column=0, sticky=NW)

        settings_frm.grid(row=0, column=0, sticky="NW")
        self.degrees_dropdown.grid(row=0, column=0, sticky="NW")
        self.quality_dropdown.grid(row=1, column=0, sticky="NW")
        self.gpu_dropdown.grid(row=2, column=0, sticky="NW")

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - ROTATE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - ROTATE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")


    def run(self, multiple: bool):
        degrees = int(self.degrees_dropdown.getChoices())
        quality = int(self.quality_dropdown.getChoices())
        gpu = str_2_bool(self.gpu_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=rotate_video(video_path=data_path,
                                             degrees=degrees,
                                             quality=quality,
                                             gpu=True)).start()


class FlipVideosPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="FLIP VIDEOS", icon='flip_green')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.horizontal_dropdown = DropDownMenu(settings_frm, "HORIZONTAL FLIP:", ['TRUE', 'FALSE'], labelwidth=25)
        self.vertical_dropdown = DropDownMenu(settings_frm, "VERTICAL FLIP:", ['TRUE', 'FALSE'], labelwidth=25)
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY (%):", list(range(10, 110, 10)), labelwidth=25)
        self.gpu_dropdown = DropDownMenu(settings_frm, "USE GPU:", ['TRUE', 'FALSE'], labelwidth=25)

        self.horizontal_dropdown.setChoices('FALSE')
        self.vertical_dropdown.setChoices('FALSE')
        self.gpu_dropdown.setChoices('FALSE')
        self.quality_dropdown.setChoices(60)

        settings_frm.grid(row=0, column=0, sticky="NW")
        self.vertical_dropdown.grid(row=0, column=0, sticky="NW")
        self.horizontal_dropdown.grid(row=1, column=0, sticky="NW")
        self.quality_dropdown.grid(row=2, column=0, sticky="NW")
        self.gpu_dropdown.grid(row=3, column=0, sticky="NW")

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - FLIP", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - FLIP", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")

    def run(self, multiple: bool):
        vertical_flip = str_2_bool(self.vertical_dropdown.getChoices())
        horizontal_flip = str_2_bool(self.horizontal_dropdown.getChoices())
        gpu = str_2_bool(self.gpu_dropdown.getChoices())
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
                                            quality=quality,
                                            gpu=gpu)).start()


class UpsampleVideosPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="UPSAMPLE VIDEOS USING INTERPOLATION (WARNING: LONG RUN-TIMES)", icon='sample')
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
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS -  UP-SAMPLE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=True))

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



class ReverseVideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="REVERSE VIDEOS", icon='reverse_blue')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.MP4_CODEC_LK = {'HEVC (H.265)': 'libx265', 'H.264 (AVC)': 'libx264', 'VP9': 'vp9'}
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY:", list(range(10, 110, 10)), labelwidth=25)
        self.quality_dropdown.setChoices(60)
        self.codec_dropdown = DropDownMenu(settings_frm, "COMPRESSION CODEC:", list(self.MP4_CODEC_LK.keys()), labelwidth=25)
        self.codec_dropdown.setChoices('HEVC (H.265)')
        self.gpu_dropdown = DropDownMenu(settings_frm, "USE GPU:", ['TRUE', 'FALSE'], labelwidth=25)
        self.gpu_dropdown.setChoices('FALSE')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.quality_dropdown.grid(row=0, column=0, sticky=NW)
        self.codec_dropdown.grid(row=1, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=2, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - REVERSE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - REVERSE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        #self.main_frm.mainloop()

    def run(self, multiple: bool):
        target_quality = int(self.quality_dropdown.getChoices())
        codec = self.MP4_CODEC_LK[self.codec_dropdown.getChoices()]
        gpu = str_2_bool(self.gpu_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        reverse_videos(path=data_path,
                       quality=target_quality,
                       codec=codec,
                       gpu=gpu)

class Convert2BlackWhitePopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CONVERT VIDEOS TO BLACK AND WHITE (NOTE: NOT GRAYSCALE)", icon='black_and_white')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        threshold = [round(x, 2) for x in list(np.arange(0.01, 1.01, 0.01))]
        self.threshold_dropdown = DropDownMenu(settings_frm, "BLACK THRESHOLD:", threshold, labelwidth=25)
        self.gpu_dropdown = DropDownMenu(settings_frm, "USE GPU:", ['TRUE', 'FALSE'], labelwidth=25)

        self.threshold_dropdown.setChoices(0.5)
        self.gpu_dropdown.setChoices('FALSE')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.threshold_dropdown.grid(row=0, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=1, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - CONVERT TO BLACK & WHITE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - CONVERT TO BLACK & WHITE", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        #self.main_frm.mainloop()

    def run(self, multiple: bool):
        threshold = float(self.threshold_dropdown.getChoices())
        gpu = str_2_bool(self.gpu_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=video_to_bw(video_path=data_path,
                                            threshold=threshold,
                                            gpu=gpu)).start()


class CreateAverageFramePopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CREATE AVERAGE VIDEO FRAME", icon='average')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.save_dir = FolderSelect(settings_frm, "AVERAGE FRAME SAVE DIRECTORY:", title="Select a video directory", lblwidth=25)
        self.section_start_time_eb = Entry_Box(settings_frm, "SAMPLE START TIME:", "25")
        self.section_end_time_eb = Entry_Box(settings_frm, "SAMPLE END TIME:", "25")
        self.section_start_time_eb.entry_set('00:00:00')
        self.section_end_time_eb.entry_set('00:00:00')

        settings_frm.grid(row=0, column=0, sticky=NW, pady=10)
        self.save_dir.grid(row=0, column=0, sticky=NW)
        self.section_start_time_eb.grid(row=1, column=0, sticky=NW)
        self.section_end_time_eb.grid(row=2, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - CREATE AVERAGE FRAME", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - CREATE AVERAGE FRAMES", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", font=Formats.FONT_REGULAR.value, command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        self.main_frm.mainloop()

    def run(self, multiple: bool):
        start_time = self.section_start_time_eb.entry_get.strip()
        end_time = self.section_end_time_eb.entry_get.strip()
        save_dir = self.save_dir.folder_path
        check_if_dir_exists(in_dir=save_dir)
        if (start_time != '' and end_time == '') or (start_time == '' and end_time != ''):
            raise InvalidInputError(msg=f'Both start time and end time have to be either time-stamps or blank.', source=self.__class__.__name__)
        if start_time != '' and end_time != '':
            check_if_string_value_is_valid_video_timestamp(value=start_time, name='start_time')
            check_if_string_value_is_valid_video_timestamp(value=end_time, name='end_time')
            check_that_hhmmss_start_is_before_end(start_time=start_time, end_time=end_time, name=self.__class__.__name__)
        else:
            start_time, end_time = None, None
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
            data_path = [data_path]
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)
            data_path = list(find_all_videos_in_directory(directory=data_path, as_dict=True, raise_error=True).values())

        for video_cnt, video_path in enumerate(data_path):
            _, video_name, _ = get_fn_ext(filepath=video_path)
            save_path = os.path.join(save_dir, save_dir, f'{video_name}_avg_frm.png')
            _ = get_video_meta_data(video_path=video_path)
            if start_time != None and end_time != None:
                check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp=start_time, video_path=video_path)
                check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp=end_time, video_path=video_path)

            threading.Thread(target=create_average_frm(video_path=video_path,
                                                       start_time=start_time,
                                                       end_time=end_time,
                                                       save_path=save_path,
                                                       verbose=True)).start()

class ManualTemporalJoinPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="MANUAL TEMPORAL JOIN VIDEOS", icon='join_purple')
        video_cnt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="NUMBER OF VIDEOS TO JOIN", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.video_cnt_dropdown = DropDownMenu(video_cnt_frm, "NUMBER OF VIDEOS:", list(range(2, 101, 1)), labelwidth=25)
        self.select_video_cnt_btn = SimbaButton(parent=video_cnt_frm, txt='SELECT', img='tick', cmd=self.select)
        self.quality_dropdown = DropDownMenu(video_cnt_frm, "OUTPUT VIDEO QUALITY %:", list(range(10, 110, 10)), labelwidth=25)
        self.quality_dropdown.setChoices(60)
        self.out_format_dropdown = DropDownMenu(video_cnt_frm, "OUTPUT VIDEO FORMAT:", Options.ALL_VIDEO_FORMAT_OPTIONS.value, labelwidth=25)
        self.gpu_dropdown = DropDownMenu(video_cnt_frm, "USE GPU:", ['TRUE', 'FALSE'], labelwidth=25)
        self.gpu_dropdown.setChoices('FALSE')
        self.out_format_dropdown.setChoices('.mp4')
        self.video_cnt_dropdown.setChoices(2)
        video_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.video_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.select_video_cnt_btn.grid(row=0, column=1, sticky=NW)
        self.quality_dropdown.grid(row=1, column=0, sticky=NW)
        self.out_format_dropdown.grid(row=2, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=3, column=0, sticky=NW)
        self.main_frm.mainloop()

    def select(self):
        video_cnt = int(self.video_cnt_dropdown.getChoices())
        if hasattr(self, 'video_paths_frm'):
            self.video_paths_frm.destroy()
            self.run_frm.destroy()
        self.video_paths_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO PATHS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.video_paths_frm.grid(row=1, column=0, sticky=NW)
        self.video_paths = {}
        for video_cnt in range(video_cnt):
            self.video_paths[video_cnt] = FileSelect(self.video_paths_frm, f"VIDEO PATH {video_cnt+1}:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
            self.video_paths[video_cnt].grid(row=video_cnt, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        video_file_paths, meta = [], []
        for cnt, video_file_select in self.video_paths.items():
            video_path = video_file_select.file_path
            check_file_exist_and_readable(file_path=video_path)
            video_meta = get_video_meta_data(video_path=video_path)
            video_file_paths.append(video_path)
            meta.append(video_meta)
        fps, resolutions = [v['fps'] for v in meta], [v['resolution_str'] for v in meta]
        unique_fps, unique_res = list(set(fps)), list(set(resolutions))
        format = self.out_format_dropdown.getChoices()
        quality = self.quality_dropdown.getChoices()
        gpu = str_2_bool(self.gpu_dropdown.getChoices())
        if len(unique_fps) > 1: raise FrameRangeError(msg=f'The selected videos contain more than one unique FPS: {unique_fps}', source=self.__class__.__name__)
        if len(unique_res) > 1: raise ResolutionError(msg=f'The selected videos contain more than one unique resolutions: {unique_res}', source=self.__class__.__name__)
        threading.Thread(temporal_concatenation(video_paths=video_file_paths,
                                                save_format=format[1:],
                                                quality=quality,
                                                gpu=gpu)).start()


#ManualTemporalJoinPopUp()


class CrossfadeVideosPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="CROSS-FADE VIDEOS", icon='crossfade')
        crossfade_methods = get_ffmpeg_crossfade_methods()
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="NUMBER OF VIDEOS TO JOIN", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.video_path_1 = FileSelect(settings_frm, f"VIDEO PATH 1:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        self.video_path_2 = FileSelect(settings_frm, f"VIDEO PATH 2:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY:", list(range(10, 110, 10)), labelwidth=25)
        self.out_format_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO FORMAT:", ['mp4', 'avi', 'webm'], labelwidth=25)
        self.fade_method_dropdown = DropDownMenu(settings_frm, "CROSS-FADE METHOD:", crossfade_methods, labelwidth=25)
        self.duration_dropdown = DropDownMenu(settings_frm, "CROSS-FADE DURATION:", list(range(2, 22, 2)), labelwidth=25)
        self.offset_eb = Entry_Box(settings_frm, "CROSS-FADE OFFSET:", "25")

        self.offset_eb.entry_set('00:00:00')
        self.quality_dropdown.setChoices(60)
        self.out_format_dropdown.setChoices('mp4')
        self.fade_method_dropdown.setChoices('fade')
        self.duration_dropdown.setChoices(6)
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.video_path_1.grid(row=0, column=0, sticky=NW)
        self.video_path_2.grid(row=1, column=0, sticky=NW)
        self.quality_dropdown.grid(row=2, column=0, sticky=NW)
        self.out_format_dropdown.grid(row=3, column=0, sticky=NW)
        self.fade_method_dropdown.grid(row=4, column=0, sticky=NW)
        self.duration_dropdown.grid(row=5, column=0, sticky=NW)
        self.offset_eb.grid(row=6, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        video_1_path, video_2_path = self.video_path_1.file_path, self.video_path_2.file_path
        quality = int(self.quality_dropdown.getChoices())
        format = self.out_format_dropdown.getChoices()
        fade_method = self.fade_method_dropdown.getChoices()
        offset = self.offset_eb.entry_get
        check_if_string_value_is_valid_video_timestamp(value=offset, name='offset')
        duration = int(self.duration_dropdown.getChoices())
        check_if_hhmmss_timestamp_is_valid_part_of_video(timestamp=offset, video_path=video_1_path)
        offset = timestamp_to_seconds(timestamp=offset)
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
#_ = BrightnessContrastPopUp()
# FlipVideosPopUp()

# ClipMultipleVideosByFrameNumbers
# ClipMultipleVideosByFrameNumbers(data_dir='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/videos/test', save_dir='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/videos/clipped')
