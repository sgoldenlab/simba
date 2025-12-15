import datetime
import glob
import json
import os
from tkinter import *
from typing import Union

import cv2
import PIL
from PIL import ImageTk

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton, SimbaCheckbox,
                                        SimBADropDown, SimBALabel,
                                        SimBASeperator)
from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable, check_float,
                                check_if_string_value_is_valid_video_timestamp,
                                check_int, check_nvidea_gpu_available,
                                check_that_hhmmss_start_is_before_end)
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.errors import (FFMPEGCodecGPUError, FFMPEGNotFoundError,
                                NoFilesFoundError)
from simba.utils.lookups import (get_icons_paths, percent_to_crf_lookup,
                                 video_quality_to_preset_lookup, get_ffmpeg_encoders, get_fonts, get_color_dict)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (
    check_if_hhmmss_timestamp_is_valid_part_of_video, get_fn_ext,
    get_video_meta_data, str_2_bool)
from simba.video_processors.batch_process_create_ffmpeg_commands import \
    FFMPEGCommandCreator
from simba.video_processors.roi_selector import ROISelector

MENU_ICONS = get_icons_paths()

class BatchProcessFrame(PopUpMixin):
    """
    Interactive GUI that collect user-inputs for batch processing videos (e.g., cropping,
    clipping etc.). User-selected output is stored in json file format within the user-defined `output_dir`

    .. note::
       `Batch pre-process tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md>`__.

    :param str input_dir: Input folder path containing videos for bath processing.
    :param str output_dir: Output folder path for where to store the processed videos.


    :example:
    >>> batch_preprocessor = BatchProcessFrame(input_dir=r'MyInputVideosDir', output_dir=r'MyOutputVideosDir')
    >>> batch_preprocessor.create_main_window()
    >>> batch_preprocessor.create_video_table_headings()
    >>> batch_preprocessor.create_video_rows()
    >>> batch_preprocessor.create_execute_btn()
    >>> batch_preprocessor.main_frm.mainloop()

    """

    def __init__(self, input_dir: Union[str, os.PathLike],
                 output_dir: Union[str, os.PathLike]):

        if not check_ffmpeg_available(raise_error=False):
            raise FFMPEGNotFoundError(msg='Cannot perform batch video processing: FFMPEG not found', source=self.__class__.__name__)


        PopUpMixin.__init__(self, title="BATCH PRE-PROCESS VIDEOS IN SIMBA", size=(2000, 600), icon='factory')
        self.input_dir, self.output_dir = input_dir, output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.videos_in_dir_dict, self.crop_dict = {}, {}
        self.get_input_files()
        self.red_drop_img = ImageTk.PhotoImage(image=PIL.Image.open(MENU_ICONS["crop_red"]["icon_path"]))
        self.black_crop_img = ImageTk.PhotoImage(image=PIL.Image.open(MENU_ICONS["crop"]["icon_path"]))
        self.percent_to_crf_lookup = percent_to_crf_lookup()
        self.cpu_video_quality = list(range(10, 110, 10))
        self.cpu_video_quality = [str(x) for x in self.cpu_video_quality]
        self.video_quality_to_preset_lookup = video_quality_to_preset_lookup()
        self.clrs = list(get_color_dict().keys())
        self.gpu_available_state = NORMAL if check_nvidea_gpu_available() else DISABLED
        if len(list(self.videos_in_dir_dict.keys())) == 0:
            raise NoFilesFoundError(msg=f"The input directory {self.input_dir} contains ZERO video files in either .avi, .mp4, .mov, .flv, or m4v format", source=self.__class__.__name__)
        self.max_char_vid_name = len(max(list(self.videos_in_dir_dict.keys()), key=len))
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(50, lambda: self.root.attributes("-topmost", False))

    def get_input_files(self):
        for file_path in glob.glob(self.input_dir + "/*"):
            lower_str_name = file_path.lower()
            if lower_str_name.endswith(Options.ALL_VIDEO_FORMAT_OPTIONS.value):
                _, video_name, ext = get_fn_ext(file_path)
                self.videos_in_dir_dict[video_name] = get_video_meta_data(
                    video_path=file_path, fps_as_int=False
                )
                self.videos_in_dir_dict[video_name]["extension"] = ext
                self.videos_in_dir_dict[video_name]["video_length"] = str(
                    datetime.timedelta(
                        seconds=int(
                            self.videos_in_dir_dict[video_name]["frame_count"]
                            / self.videos_in_dir_dict[video_name]["fps"]
                        )
                    )
                )
                self.videos_in_dir_dict[video_name]["video_length"] = (
                    "0" + self.videos_in_dir_dict[video_name]["video_length"]
                )
                self.videos_in_dir_dict[video_name]["file_path"] = file_path

    def create_main_window(self):
        self.quick_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm,header="QUICK SETTINGS",icon_name=Keys.DOCUMENTATION.value,icon_link=Links.BATCH_PREPROCESS.value)

        self.clip_video_settings_frm = CreateLabelFrameWithIcon(parent=self.quick_settings_frm, header='CLIP VIDEOS SETTING', icon_name='clip', padx=5, pady=5)
        self.quick_clip_start_entry_box = Entry_Box(parent=self.clip_video_settings_frm, fileDescription='START TIME: ', labelwidth=15, value="00:00:00", justify='center', img='play', entry_box_width=12)
        self.quick_clip_end_entry_box = Entry_Box(parent=self.clip_video_settings_frm, fileDescription='END TIME: ', labelwidth=15, value="00:00:00", justify='center', img='finish', entry_box_width=12)
        self.quick_clip_apply = SimbaButton(parent=self.clip_video_settings_frm, txt='APPLY', img='arrow_down_green_2', cmd=self.apply_trim_to_all)

        self.quick_downsample_frm = CreateLabelFrameWithIcon(parent=self.quick_settings_frm, header='DOWNSAMPLE VIDEOS', icon_name='resize', padx=5, pady=5)
        self.quick_downsample_width = Entry_Box(parent=self.quick_downsample_frm, fileDescription='WIDTH: ', labelwidth=15, value=400, justify='center', img='width', entry_box_width=12)
        self.quick_downsample_height = Entry_Box(parent=self.quick_downsample_frm, fileDescription='HEIGHT: ', labelwidth=15, value=600, justify='center', img='height', entry_box_width=12)
        self.quick_downsample_apply = SimbaButton(parent=self.quick_downsample_frm, txt='APPLY', img='arrow_down_green_2', cmd=self.apply_resolution_to_all)


        self.quick_set_fps = CreateLabelFrameWithIcon(parent=self.quick_settings_frm, header="CHANGE FPS", icon_name='camera', padx=5, pady=12)
        self.quick_fps_entry_box = Entry_Box(parent=self.quick_set_fps, fileDescription='FPS: ', labelwidth=15, value=15.0, justify='center', img='camera', entry_box_width=12)
        self.quick_fps_apply = SimbaButton(parent=self.quick_set_fps, txt='APPLY', img='arrow_down_green_2', cmd=self.apply_fps_to_all)


        self.quick_set_quality = CreateLabelFrameWithIcon(parent=self.quick_settings_frm, header="OUTPUT VIDEO QUALITY", icon_name='star', padx=5, pady=12)
        self.use_gpu_dropdown = SimBADropDown(parent=self.quick_set_quality, label="USE GPU", label_width=20, dropdown_options=['TRUE', 'FALSE'], value='FALSE', img='gpu_3', state=self.gpu_available_state, dropdown_width=15)
        self.quick_set_quality_dropdown = SimBADropDown(parent=self.quick_set_quality, label='VIDEO QUALITY %', label_width=20, dropdown_options=self.cpu_video_quality, value=60, img='star', dropdown_width=15)
        self.quick_set_quality_apply = SimbaButton(parent=self.quick_set_quality, txt='APPLY', img='arrow_down_green_2', cmd=self.apply_quality_to_all)

        self.quick_settings_frm.grid(row=0, column=0, sticky=W, padx=10)
        self.quick_clip_start_entry_box.grid(row=0, column=0, sticky=NW)
        self.quick_clip_end_entry_box.grid(row=1, column=0, sticky=NW)
        self.quick_clip_apply.grid(row=2, column=0, sticky=NW)
        self.quick_downsample_width.grid(row=0, column=0, sticky=NW)
        self.quick_downsample_height.grid(row=1, column=0, sticky=NW)
        self.quick_downsample_apply.grid(row=2, column=0, sticky=NW)
        self.quick_fps_entry_box.grid(row=0, column=0, sticky=NW)
        self.quick_fps_apply.grid(row=2, column=0, sticky=NW)
        self.use_gpu_dropdown.grid(row=0, column=0, sticky=NW)
        self.quick_set_quality_dropdown.grid(row=1, column=0, sticky=NW)
        self.quick_set_quality_apply.grid(row=2, column=0, sticky=W)

        for col in range(4):
            self.quick_settings_frm.grid_columnconfigure(col, weight=1, uniform="quick_settings")

        for col, frm in enumerate([self.clip_video_settings_frm, self.quick_downsample_frm, self.quick_set_fps, self.quick_set_quality]):
            frm.grid(row=0, column=col, sticky="nsew", padx=2, pady=2)

        self.quick_settings_frm.update_idletasks()
        max_width = max(frm.winfo_reqwidth() for frm in [self.clip_video_settings_frm, self.quick_downsample_frm, self.quick_set_fps, self.quick_set_quality])
        max_height = max(frm.winfo_reqheight() for frm in [self.clip_video_settings_frm, self.quick_downsample_frm, self.quick_set_fps, self.quick_set_quality])

        for frm in [self.clip_video_settings_frm, self.quick_downsample_frm, self.quick_set_fps, self.quick_set_quality]:
            frm.config(width=max_width, height=max_height)


    def inverse_all_cb_ticks(self, variable_name=None):
        if self.headings[variable_name].get():
            for video_name in self.videos.keys():
                self.videos[video_name][variable_name].set(True)
        if not self.headings[variable_name].get():
            for video_name in self.videos.keys():
                self.videos[video_name][variable_name].set(False)

    def change_quality_options_cpu_gpu(self, x):
        gpu_state = str_2_bool(self.use_gpu_dropdown.get_value())
        if not gpu_state:
            self.quick_set_quality_dropdown.change_options(values=self.cpu_video_quality, set_str='100', auto_change_width=False)
        else:
            self.quick_set_quality_dropdown.change_options(values=["Low", "Medium", "High"], set_str='Medium', auto_change_width=False)
        #self.quick_set_quality_dropdown.grid(row=1, column=0, sticky=NW)
        for video_cnt, video_name in enumerate(self.videos.keys()):
            if not gpu_state:
                self.videos[video_name]["video_quality_dropdown"].change_options(values=self.cpu_video_quality, set_str='100', auto_change_width=False)
            else:
                self.videos[video_name]["video_quality_dropdown"].change_options(values=["Low", "Medium", "High"], set_str='Medium', auto_change_width=False)

    def apply_resolution_to_all(self):
        check_int(value=self.quick_downsample_width.entry_get, min_value=0, name=f"Quick set downsample WIDTH {self.quick_downsample_width.entry_get}",)
        check_int(value=self.quick_downsample_height.entry_get, min_value=0, name=f"Quick set downsample HEIGHT {self.quick_downsample_height.entry_get}")
        for video_name in self.videos.keys():
            self.videos[video_name]["width_entry"].entry_set(self.quick_downsample_width.entry_get)
            self.videos[video_name]["height_entry"].entry_set(self.quick_downsample_height.entry_get)

    def apply_trim_to_all(self):
        check_if_string_value_is_valid_video_timestamp(value=self.quick_clip_start_entry_box.entry_get, name=f"Quick set clip START time {self.quick_clip_start_entry_box.entry_get}",)
        check_if_string_value_is_valid_video_timestamp(value=self.quick_clip_end_entry_box.entry_get, name=f"Quick set clip END time {self.quick_clip_start_entry_box.entry_get}")
        check_that_hhmmss_start_is_before_end(start_time=self.quick_clip_start_entry_box.entry_get, end_time=self.quick_clip_end_entry_box.entry_get, name="Quick set START and END time")
        for video_name in self.videos.keys():
            self.videos[video_name]["start_entry"].entry_set(self.quick_clip_start_entry_box.entry_get)
            self.videos[video_name]["end_entry"].entry_set(self.quick_clip_end_entry_box.entry_get)

    def apply_fps_to_all(self):
        check_float(value=self.quick_fps_entry_box.entry_get, min_value=0, name=f"Quick set FPS setting {self.quick_fps_entry_box.entry_get}")
        for video_name in self.videos.keys():
            self.videos[video_name]["fps_entry"].entry_set(self.quick_fps_entry_box.entry_get)

    def apply_quality_to_all(self):
        for video_name in self.videos.keys():
            self.videos[video_name]["video_quality_dropdown"].setChoices(self.quick_set_quality_dropdown.getChoices())

    def create_video_table_headings(self):
        self.headings = {}
        self.videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='VIDEOS', font=Formats.FONT_HEADER.value, pady=5, padx=15, icon_name='stack')
        self.headings["video_name_col_head"] = SimBALabel(parent=self.videos_frm, txt='VIDEO NAME', font=Formats.FONT_REGULAR_BOLD.value, width=self.max_char_vid_name, justify='center')
        self.headings["crop_video_col_head"] = SimBALabel(parent=self.videos_frm, txt='CROP \n VIDEO', font=Formats.FONT_REGULAR_BOLD.value, justify='center', img='crop', padx=12)
        self.headings["start_time_col_head"] = SimBALabel(parent=self.videos_frm, txt='START \n TIME', font=Formats.FONT_REGULAR_BOLD.value, justify='center', padx=12, img='play')
        self.headings["end_time_col_head"] = SimBALabel(parent=self.videos_frm, txt='END \n TIME', font=Formats.FONT_REGULAR_BOLD.value, justify='center', img='stop', padx=12)
        self.headings["video_quality_head"] = SimBALabel(parent=self.videos_frm, txt='QUALITY', font=Formats.FONT_REGULAR_BOLD.value, justify='center', padx=12, img='star')
        self.headings["shorten_all_videos_cbox"], self.headings["clip_cb_var"] = SimbaCheckbox(parent=self.videos_frm, txt='APPLY \n CLIP', txt_img='clip', cmd=lambda: self.inverse_all_cb_ticks(variable_name="clip_cb_var"))
        self.headings["video_width_col_head"] = SimBALabel(parent=self.videos_frm, txt='WIDTH', img='width', font=Formats.FONT_REGULAR_BOLD.value, justify='center', padx=12)
        self.headings["video_height_col_head"] = SimBALabel(parent=self.videos_frm, txt='HEIGHT', img='height', font=Formats.FONT_REGULAR_BOLD.value, justify='center', padx=12)
        self.headings["downsample_all_videos_cbox"], self.headings["downsample_cb_var"] = SimbaCheckbox(parent=self.videos_frm, txt='APPLY \n DOWNSAMPLE', cmd=lambda: self.inverse_all_cb_ticks(variable_name="downsample_cb_var"))
        self.headings["fps_col_head"] = SimBALabel(parent=self.videos_frm, txt='FPS', img='camera', font=Formats.FONT_REGULAR_BOLD.value, justify='center', padx=12)
        self.headings["change_fps_all_videos_cbox"], self.headings["fps_cb_var"] = SimbaCheckbox(parent=self.videos_frm, txt='APPLY \n VIDEO FPS', cmd=lambda: self.inverse_all_cb_ticks(variable_name="fps_cb_var"))
        self.headings["grayscale_cbox"], self.headings["grayscale_cb_var"] = SimbaCheckbox(parent=self.videos_frm, txt='APPLY \n GREYSCALE', cmd=lambda: self.inverse_all_cb_ticks(variable_name="grayscale_cb_var"))
        self.headings["frame_cnt_cbox"], self.headings["frame_cnt_cb_var"] = SimbaCheckbox(parent=self.videos_frm, txt='APPLY \n FRAME COUNT', cmd=lambda: self.inverse_all_cb_ticks(variable_name="frame_cnt_cb_var"))
        self.headings["apply_clahe_cbox"], self.headings["apply_clahe_cb_var"] = SimbaCheckbox(parent=self.videos_frm, txt='APPLY \n CLAHE', cmd=lambda: self.inverse_all_cb_ticks(variable_name="apply_clahe_cb_var"))

        self.videos_frm.grid(row=1, column=0, sticky=W, padx=5, pady=15)
        self.headings["video_name_col_head"].grid(row=0, column=0, sticky=NW)
        self.headings["crop_video_col_head"].grid(row=0, column=2, sticky=NW)
        self.headings["start_time_col_head"].grid(row=0, column=3, sticky=NW)
        self.headings["end_time_col_head"].grid(row=0, column=4, sticky=W, padx=5)
        self.headings["shorten_all_videos_cbox"].grid(row=0, column=5, sticky=W, padx=5)
        self.headings["video_width_col_head"].grid(row=0, column=6, sticky=W, padx=5)
        self.headings["video_height_col_head"].grid(row=0, column=7, sticky=W, padx=5)
        self.headings["downsample_all_videos_cbox"].grid(row=0, column=8, sticky=W, padx=5)
        self.headings["fps_col_head"].grid(row=0, column=9, sticky=W, padx=5)
        self.headings["change_fps_all_videos_cbox"].grid(row=0, column=10, sticky=W, padx=5)
        self.headings["grayscale_cbox"].grid(row=0, column=11, sticky=W, padx=5)
        self.headings["frame_cnt_cbox"].grid(row=0, column=12, sticky=W, padx=5)
        self.headings["apply_clahe_cbox"].grid(row=0, column=13, sticky=W, padx=5)
        self.headings["video_quality_head"].grid(row=0, column=14, sticky=NW)

        seperator = SimBASeperator(parent=self.videos_frm, color=None, orient='horizontal', borderwidth=1)
        seperator.grid(row=1, column=0, columnspan=15, rowspan=1, sticky="ew")

        seperator = SimBASeperator(parent=self.videos_frm, orient='vertical', borderwidth=1)
        seperator.grid(row=0, column=1, rowspan=len(self.videos_in_dir_dict.keys()) + 400, sticky="ns")


    def get_file_menu(self):
        menu = Menu(self.root)
        file_menu = Menu(menu)
        menu.add_cascade(label="File", menu=file_menu, compound="left")
        file_menu.add_command(label="Preferences...", compound="left", image=self.menu_icons["settings"]["img"], command=lambda: self.preferences_pop_up())
        self.root.config(menu=menu)


    def create_video_rows(self):
        self.videos = {}
        for w in self.videos_frm.grid_slaves():
            if int(w.grid_info()["row"]) > 1:
                w.destroy()

        for video_cnt, (name, data) in enumerate(self.videos_in_dir_dict.items()):
            self.videos[name] = {}
            row = video_cnt * 2 + 2
            row_color = '#f8f8f8' if video_cnt % 2 == 0 else '#e5e5e5'
            self.videos[name]["video_name_lbl"] = SimBALabel(parent=self.videos_frm, txt=name, font=Formats.FONT_REGULAR_BOLD.value, width=self.max_char_vid_name, justify='center', bg_clr=row_color)
            self.videos[name]["crop_btn"] = SimbaButton(parent=self.videos_frm, txt='CROP', txt_clr='black', cmd=lambda k=self.videos[name]["video_name_lbl"]["text"]: self.batch_process_crop_function(k), img='crop_2')
            self.videos[name]["start_entry"] = Entry_Box(parent=self.videos_frm, fileDescription='', value="00:00:00", justify='center', entry_font=Formats.FONT_REGULAR_BOLD.value, entry_box_width=12)
            self.videos[name]["end_entry"] = Entry_Box(parent=self.videos_frm, fileDescription='', value=data["video_length"], justify='center', entry_font=Formats.FONT_REGULAR_BOLD.value, entry_box_width=12)
            self.videos[name]["clip_cb"], self.videos[name]["clip_cb_var"] = SimbaCheckbox(parent=self.videos_frm, txt='', cmd=None)
            self.videos[name]["width_entry"] = Entry_Box(parent=self.videos_frm, fileDescription='', value=data["width"], justify='center', entry_font=Formats.FONT_REGULAR_BOLD.value, entry_box_width=10)
            self.videos[name]["height_entry"] = Entry_Box(parent=self.videos_frm, fileDescription='', value=data["height"], justify='center', entry_font=Formats.FONT_REGULAR_BOLD.value, entry_box_width=10)
            self.videos[name]["downsample_cb"], self.videos[name]["downsample_cb_var"] = SimbaCheckbox(parent=self.videos_frm, txt='', cmd=None)
            self.videos[name]["fps_entry"] = Entry_Box(parent=self.videos_frm, fileDescription='', value=round(data["fps"], 4), justify='center', entry_font=Formats.FONT_REGULAR_BOLD.value, entry_box_width=8)
            self.videos[name]["fps_cb"], self.videos[name]["fps_cb_var"] = SimbaCheckbox(parent=self.videos_frm, txt='', cmd=None)
            self.videos[name]["grayscale_cbox"], self.videos[name]["grayscale_cb_var"] = SimbaCheckbox(parent=self.videos_frm, txt='', cmd=None)
            self.videos[name]["frame_cnt_cbox"], self.videos[name]["frame_cnt_cb_var"] = SimbaCheckbox(parent=self.videos_frm, txt='', cmd=None)
            self.videos[name]["apply_clahe_cbox"], self.videos[name]["apply_clahe_cb_var"] = SimbaCheckbox(parent=self.videos_frm, txt='', cmd=None)
            self.videos[name]["video_quality_dropdown"] = SimBADropDown(parent=self.videos_frm, label="", dropdown_options=self.cpu_video_quality, value=60, dropdown_width=10)

            self.videos[name]["video_name_lbl"].grid(row=row, column=0, sticky=W, pady=(3, 3))
            self.videos[name]["crop_btn"].grid(row=row, column=2, pady=(3, 3))
            self.videos[name]["start_entry"].grid(row=row, column=3, sticky=W, pady=(3, 3))
            self.videos[name]["end_entry"].grid(row=row, column=4, padx=5, pady=(3, 3))
            self.videos[name]["clip_cb"].grid(row=row, column=5, sticky=W, padx=5, pady=(3, 3))
            self.videos[name]["width_entry"].grid(row=row, column=6, padx=5, pady=(3, 3))
            self.videos[name]["height_entry"].grid(row=row, column=7, padx=5, pady=(3, 3))
            self.videos[name]["downsample_cb"].grid(row=row, column=8, sticky=W, padx=5, pady=(3, 3))
            self.videos[name]["fps_entry"].grid(row=row, column=9, padx=5, pady=(3, 3))
            self.videos[name]["fps_cb"].grid(row=row, column=10, sticky=W, padx=5, pady=(3, 3))
            self.videos[name]["grayscale_cbox"].grid(row=row, column=11, sticky=W, padx=5, pady=(3, 3))
            self.videos[name]["frame_cnt_cbox"].grid(row=row, column=12, sticky=W, padx=5, pady=(3, 3))
            self.videos[name]["apply_clahe_cbox"].grid(row=row, column=13, sticky=W, padx=5, pady=(3, 3))
            try:
                self.videos[name]["video_quality_dropdown"].grid_remove(row=row, column=14, sticky=W)
            except:
                pass
            self.videos[name]["video_quality_dropdown"].grid(row=row, column=14, sticky=W)
            if video_cnt != len(self.videos_in_dir_dict.keys()) -1:
                sep = SimBASeperator(parent=self.videos_frm, orient='horizontal', height=1, color="#ccc")
                sep.grid(row=row + 1, column=0, columnspan=15, sticky="ew")



    def create_execute_btn(self):
        self.execute_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="EXECUTE", icon_name='rocket', pady=5, padx=5, font=Formats.FONT_HEADER.value)
        self.reset_all_btn = SimbaButton(parent=self.execute_frm, txt='RESET ALL', txt_clr='red', img='trash', cmd=self.create_video_rows)
        self.reset_crop_btn = SimbaButton(parent=self.execute_frm, txt='RESET CROP', txt_clr='darkorange', img='trash', cmd=self.reset_crop)
        self.execute_btn = SimbaButton(parent=self.execute_frm, txt='EXECUTE', txt_clr='blue', img='rocket', cmd=self.execute)
        self.execute_frm.grid(row=2, column=0, sticky=W, padx=5, pady=30)
        self.reset_all_btn.grid(row=0, column=0, sticky=W, padx=5)
        self.reset_crop_btn.grid(row=0, column=1, sticky=W, padx=5)
        self.execute_btn.grid(row=0, column=2, sticky=W, padx=5)
        self.get_file_menu()

    def reset_crop(self):
        self.crop_dict = {}
        for video_name, video_data in self.videos_in_dir_dict.items():
            self.videos[video_name]["crop_btn"].configure(fg="black")
            self.videos[video_name]["crop_btn"].configure(image=self.black_crop_img, compound='left')


    def batch_process_crop_function(self, video_name):
        check_file_exist_and_readable(self.videos_in_dir_dict[video_name]["file_path"])
        roi_selector = ROISelector(path=self.videos_in_dir_dict[video_name]["file_path"], title=f"CROP {video_name} - Press ESC when ROI drawn")
        roi_selector.run()
        self.crop_dict[video_name] = {}
        self.crop_dict[video_name]["top_left_x"] = roi_selector.top_left[0]
        self.crop_dict[video_name]["top_left_y"] = roi_selector.top_left[1]
        self.crop_dict[video_name]["width"] = roi_selector.width
        self.crop_dict[video_name]["height"] = roi_selector.height
        self.crop_dict[video_name]["bottom_right_x"] = roi_selector.bottom_right[0]
        self.crop_dict[video_name]["bottom_right_y"] = roi_selector.bottom_right[1]
        k = cv2.waitKey(20) & 0xFF
        cv2.destroyAllWindows()
        self.videos[video_name]["crop_btn"].configure(fg="red", font=('Arial', 14, 'bold'))
        self.videos[video_name]["crop_btn"].configure(image=self.red_drop_img, compound='left')

    def preferences_pop_up(self):
        if hasattr(self, 'preferences_frm'):
            self.preferences_frm.destroy()

        self.preferences_frm = Toplevel()
        self.preferences_frm.minsize(400, 300)
        self.preferences_frm.wm_title("PREFERENCES")
        self.preferences_frm.iconphoto(False, self.menu_icons['settings']["img"])

        codecs = get_ffmpeg_encoders(alphabetically_sorted=True)
        fonts = list(get_fonts().keys())
        pref_frm = CreateLabelFrameWithIcon(parent=self.preferences_frm, header='SETTINGS', icon_name='settings')
        self.codec_dropdown = SimBADropDown(parent=pref_frm, dropdown_options=codecs, label='CODEC:', label_width=25, value=Formats.BATCH_CODEC.value, dropdown_width=30)
        self.font_dropdown = SimBADropDown(parent=pref_frm, dropdown_options=fonts, label='FONT:', label_width=25, value='Arial', dropdown_width=30)
        self.crop_thickness_dropdown = SimBADropDown(parent=pref_frm, dropdown_options=list(range(1, 31)), label='CROP THICKNESS:', label_width=25, value=10, dropdown_width=30)
        self.crop_color_dropdown = SimBADropDown(parent=pref_frm, dropdown_options=self.clrs, label='CROP COLOR:', label_width=25, value=10, dropdown_width=30)

        pref_frm.grid(row=0, column=0, sticky=NW)
        self.codec_dropdown.grid(row=0, column=0, sticky=NW)
        self.font_dropdown.grid(row=1, column=0, sticky=NW)
        self.crop_thickness_dropdown.grid(row=2, column=0, sticky=NW)
        self.crop_color_dropdown.grid(row=3, column=0, sticky=NW)

    def execute(self):
        out_video_dict = {}
        out_video_dict["meta_data"] = {}
        out_video_dict["video_data"] = {}
        out_video_dict["meta_data"]["in_dir"] = self.input_dir
        out_video_dict["meta_data"]["out_dir"] = self.output_dir
        out_video_dict["meta_data"]["gpu"] = str_2_bool(self.use_gpu_dropdown.get_value())
        if str_2_bool(self.use_gpu_dropdown.get_value()) and not check_nvidea_gpu_available():
            raise FFMPEGCodecGPUError(msg="No GPU found (as evaluated by nvidea-smi returning None)", source=self.__class__.__name__)
        for video_cnt, (name, data) in enumerate(self.videos_in_dir_dict.items()):
            out_video_dict["video_data"][name] = {}
            out_video_dict["video_data"][name]["video_info"] = self.videos_in_dir_dict[name]
            out_video_dict["video_data"][name]["output_quality"] = self.percent_to_crf_lookup[self.videos[name]["video_quality_dropdown"].getChoices()]
            if name in self.crop_dict.keys():
                out_video_dict["video_data"][name]["crop"] = True
                out_video_dict["video_data"][name]["crop_settings"] = self.crop_dict[name]
            else:
                out_video_dict["video_data"][name]["crop"] = False
                out_video_dict["video_data"][name]["crop_settings"] = None
            if self.videos[name]["clip_cb_var"].get():
                out_video_dict["video_data"][name]["clip"] = True
                out_video_dict["video_data"][name]["clip_settings"] = {"start": self.videos[name]["start_entry"].entry_get, "stop": self.videos[name]["end_entry"].entry_get}
            else:
                out_video_dict["video_data"][name]["clip"] = False
                out_video_dict["video_data"][name]["clip_settings"] = None
            if self.videos[name]["downsample_cb_var"].get():
                out_video_dict["video_data"][name]["downsample"] = True
                width, height = self.videos[name]["width_entry"].entry_get, self.videos[name]["height_entry"].entry_get
                check_int(name=f'{name} width', value=width, min_value=0, raise_error=True)
                check_int(name=f'{name} height', value=height, min_value=0, raise_error=True)
                width, height = int(width) + int(width) % 2, int(height) + int(height) % 2
                out_video_dict["video_data"][name]["downsample_settings"] = {"width": str(width), "height": str(height)}
            else:
                out_video_dict["video_data"][name]["downsample"] = False
                out_video_dict["video_data"][name]["downsample_settings"] = None
            if self.videos[name]["fps_cb_var"].get():
                out_video_dict["video_data"][name]["fps"] = True
                out_video_dict["video_data"][name]["fps_settings"] = {"fps": self.videos[name]["fps_entry"].entry_get}
            else:
                out_video_dict["video_data"][name]["fps"] = False
                out_video_dict["video_data"][name]["fps_settings"] = None
            if self.videos[name]["grayscale_cb_var"].get():
                out_video_dict["video_data"][name]["grayscale"] = True
                out_video_dict["video_data"][name]["grayscale_settings"] = None
            else:
                out_video_dict["video_data"][name]["grayscale"] = False
                out_video_dict["video_data"][name]["grayscale_settings"] = None
            if self.videos[name]["frame_cnt_cb_var"].get():
                out_video_dict["video_data"][name]["frame_cnt"] = True
                out_video_dict["video_data"][name]["frame_cnt_settings"] = None
            else:
                out_video_dict["video_data"][name]["frame_cnt"] = False
                out_video_dict["video_data"][name]["frame_cnt_settings"] = None
            if self.videos[name]["apply_clahe_cb_var"].get():
                out_video_dict["video_data"][name]["clahe"] = True
                out_video_dict["video_data"][name]["clahe_settings"] = None
            else:
                out_video_dict["video_data"][name]["clahe"] = False
                out_video_dict["video_data"][name]["clahe_settings"] = None
            out_video_dict["video_data"][name]["last_operation"] = None
            for operation in ["clahe", "frame_cnt", "grayscale", "fps", "downsample", "clip", "crop"]:
                if out_video_dict["video_data"][name][operation]:
                    out_video_dict["video_data"][name]["last_operation"] = operation

        self.save_path = os.path.join(self.output_dir, "batch_process_log.json")
        with open(self.save_path, "w") as fp:
            json.dump(out_video_dict, fp)
        self.perform_unit_tests(out_video_dict["video_data"])

    def perform_unit_tests(self, out_video_dict):
        timer = SimbaTimer(start=True)
        for video_name, video_data in out_video_dict.items():
            if video_data["crop"]:
                check_int(
                    value=video_data["crop_settings"]["width"],
                    min_value=1,
                    name=f"Crop width for video {video_name}",
                )
                check_int(
                    value=video_data["crop_settings"]["height"],
                    min_value=1,
                    name=f"Crop height for video {video_name}",
                )
            if video_data["clip"]:
                for variable in ["start", "stop"]:
                    check_if_string_value_is_valid_video_timestamp(
                        value=video_data["clip_settings"][variable],
                        name=f"Clip {variable} time for video {video_name}",
                    )
                check_that_hhmmss_start_is_before_end(
                    start_time=video_data["clip_settings"]["start"],
                    end_time=video_data["clip_settings"]["stop"],
                    name=f"Clip time for video {video_name}",
                )
                for variable in ["start", "stop"]:
                    check_if_hhmmss_timestamp_is_valid_part_of_video(
                        timestamp=video_data["clip_settings"][variable],
                        video_path=video_data["video_info"]["file_path"],
                    )
            if video_data["downsample"]:
                check_int(
                    value=video_data["downsample_settings"]["width"],
                    min_value=1,
                    name=f"Downsample width for video {video_name}",
                )
                check_int(
                    value=video_data["downsample_settings"]["height"],
                    min_value=1,
                    name=f"Downsample height for video {video_name}",
                )
            if video_data["fps"]:
                check_float(
                    value=video_data["fps_settings"]["fps"],
                    min_value=0,
                    name=f"FPS settings for video {video_name}",
                )

        ffmpeg_runner = FFMPEGCommandCreator(json_path=self.save_path)
        ffmpeg_runner.crop_videos()
        ffmpeg_runner.clip_videos()
        ffmpeg_runner.downsample_videos()
        ffmpeg_runner.apply_fps()
        ffmpeg_runner.apply_grayscale()
        ffmpeg_runner.apply_frame_count()
        ffmpeg_runner.apply_clahe()
        ffmpeg_runner.move_all_processed_files_to_output_folder()
        timer.stop_timer()
        stdout_success(msg=f"SimBA batch pre-process JSON saved at {self.save_path}", source=self.__class__.__name__)
        stdout_success(msg=f"Video batch pre-processing complete, new videos stored in {self.output_dir}", elapsed_time=timer.elapsed_time_str, source=self.__class__.__name__)


# test = BatchProcessFrame(input_dir=r'D:\troubleshooting\maplight_ri\project_folder\blob\videos', output_dir=r"D:\troubleshooting\maplight_ri\project_folder\blob\batch_out_6")
# test.create_main_window()
# test.create_video_table_headings()
# test.create_video_rows()
# test.create_execute_btn()
# test.main_frm.mainloop()
