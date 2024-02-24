import datetime
import glob
import json
import os
import re
from tkinter import *
from typing import Tuple, Union

import cv2

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, DropDownMenu
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_if_string_value_is_valid_video_timestamp,
                                check_int, check_nvidea_gpu_available,
                                check_that_hhmmss_start_is_before_end)
from simba.utils.enums import Keys, Links, Options
from simba.utils.errors import FFMPEGCodecGPUError, NoFilesFoundError
from simba.utils.lookups import (percent_to_crf_lookup,
                                 video_quality_to_preset_lookup)
from simba.utils.printing import SimbaTimer, stdout_success
from simba.utils.read_write import (
    check_if_hhmmss_timestamp_is_valid_part_of_video, get_fn_ext,
    get_video_meta_data)
from simba.video_processors.batch_process_create_ffmpeg_commands import \
    FFMPEGCommandCreator
from simba.video_processors.roi_selector import ROISelector


class BatchProcessFrame(PopUpMixin):
    """
    Interactive GUI that collect user-inputs for batch processing videos (e.g., cropping,
    clipping etc.). User-selected output is stored in json file format within the user-defined `output_dir`

    .. note::
       `Batch pre-process tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/tutorial_process_videos.md>`__.

    :param str input_dir: Input folder path containing videos for bath processing.
    :param str output_dir: Output folder path for where to store the processed videos.


    Examples
    ----------
    >>> batch_preprocessor = BatchProcessFrame(input_dir=r'MyInputVideosDir', output_dir=r'MyOutputVideosDir')
    >>> batch_preprocessor.create_main_window()
    >>> batch_preprocessor.create_video_table_headings()
    >>> batch_preprocessor.create_video_rows()
    >>> batch_preprocessor.create_execute_btn()
    >>> batch_preprocessor.main_frm.mainloop()

    """

    def __init__(
        self, input_dir: Union[str, os.PathLike], output_dir: Union[str, os.PathLike]
    ):
        PopUpMixin.__init__(
            self, title="BATCH PRE-PROCESS VIDEOS IN SIMBA", size=(2400, 600)
        )
        self.input_dir, self.output_dir = input_dir, output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.videos_in_dir_dict, self.crop_dict = {}, {}
        self.get_input_files()
        self.percent_to_crf_lookup = percent_to_crf_lookup()
        self.cpu_video_quality = list(range(10, 110, 10))
        self.cpu_video_quality = [str(x) for x in self.cpu_video_quality]
        self.video_quality_to_preset_lookup = video_quality_to_preset_lookup()
        if len(list(self.videos_in_dir_dict.keys())) == 0:
            raise NoFilesFoundError(
                msg=f"The input directory {self.input_dir} contains ZERO video files in either .avi, .mp4, .mov, .flv, or m4v format",
                source=self.__class__.__name__,
            )
        self.max_char_vid_name = len(max(list(self.videos_in_dir_dict.keys()), key=len))

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
        self.quick_settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="QUICK SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.BATCH_PREPROCESS.value,
        )
        self.clip_video_settings_frm = LabelFrame(
            self.quick_settings_frm, text="Clip Videos Settings", padx=5
        )
        self.quick_clip_start_entry_lbl = Label(
            self.clip_video_settings_frm, text="Start Time: "
        )
        self.quick_clip_start_entry_box_val = StringVar()
        self.quick_clip_start_entry_box_val.set("00:00:00")
        self.quick_clip_start_entry_box = Entry(
            self.clip_video_settings_frm,
            width=15,
            textvariable=self.quick_clip_start_entry_box_val,
        )
        self.quick_clip_end_entry_lbl = Label(
            self.clip_video_settings_frm, text="End Time: "
        )
        self.quick_clip_end_entry_box_val = StringVar()
        self.quick_clip_end_entry_box_val.set("00:00:00")
        self.quick_clip_end_entry_box = Entry(
            self.clip_video_settings_frm,
            width=15,
            textvariable=self.quick_clip_end_entry_box_val,
        )
        self.quick_clip_apply = Button(
            self.clip_video_settings_frm,
            text="Apply",
            command=lambda: self.apply_trim_to_all(),
        )
        self.quick_downsample_frm = LabelFrame(
            self.quick_settings_frm, text="Downsample Videos", padx=5
        )
        self.quick_downsample_width_lbl = Label(
            self.quick_downsample_frm, text="Width: "
        )
        self.quick_downsample_width_val = IntVar()
        self.quick_downsample_width_val.set(400)
        self.quick_downsample_width = Entry(
            self.quick_downsample_frm,
            width=15,
            textvariable=self.quick_downsample_width_val,
        )
        self.quick_downsample_height_lbl = Label(
            self.quick_downsample_frm, text="Height: "
        )
        self.quick_downsample_height_val = IntVar()
        self.quick_downsample_height_val.set(600)
        self.quick_downsample_height = Entry(
            self.quick_downsample_frm,
            width=15,
            textvariable=self.quick_downsample_height_val,
        )
        self.quick_downsample_apply = Button(
            self.quick_downsample_frm,
            text="Apply",
            command=lambda: self.apply_resolution_to_all(),
        )
        self.quick_set_fps = LabelFrame(
            self.quick_settings_frm, text="Change FPS", padx=5, pady=5
        )
        self.quick_fps_lbl = Label(self.quick_set_fps, text="FPS: ")
        self.quick_set_fps_val = DoubleVar()
        self.quick_set_fps_val.set(15.0)
        self.quick_fps_entry_box = Entry(
            self.quick_set_fps, width=15, textvariable=self.quick_set_fps_val
        )
        self.quick_set_fps_empty_row = Label(self.quick_set_fps, text=" ")
        self.quick_fps_apply = Button(
            self.quick_set_fps, text="Apply", command=lambda: self.apply_fps_to_all()
        )

        self.quick_set_quality = LabelFrame(
            self.quick_settings_frm, text="Output Video Quality", padx=5, pady=5
        )
        self.quick_set_quality_dropdown = DropDownMenu(
            self.quick_set_quality, "Video Quality % ", self.cpu_video_quality, "14"
        )
        self.quick_set_quality_dropdown.setChoices(100)
        self.quick_set_qualitys_empty_row = Label(self.quick_set_quality, text=" ")
        self.quick_set_quality_apply = Button(
            self.quick_set_quality,
            text="Apply",
            command=lambda: self.apply_quality_to_all(),
        )

        self.use_gpu_frm = LabelFrame(
            self.quick_settings_frm, text="GPU", padx=5, pady=10
        )
        self.use_gpu_var = BooleanVar(value=False)
        use_gpu_cb = Checkbutton(
            self.use_gpu_frm,
            text="Use GPU (reduced runtime)",
            variable=self.use_gpu_var,
            command=lambda: self.change_quality_options_cpu_gpu(),
        )
        use_gpu_cb.grid(row=0, column=0, sticky=NW)
        Label(self.use_gpu_frm, text=" ").grid(row=1, column=0)
        Label(self.use_gpu_frm, text=" ").grid(row=2, column=0)

        self.quick_settings_frm.grid(row=0, column=0, sticky=W, padx=10)
        self.clip_video_settings_frm.grid(row=0, column=0, sticky=W)
        self.quick_clip_start_entry_lbl.grid(row=0, column=0, sticky=W)
        self.quick_clip_start_entry_box.grid(row=0, column=1, sticky=W)
        self.quick_clip_end_entry_lbl.grid(row=1, column=0, sticky=W)
        self.quick_clip_end_entry_box.grid(row=1, column=1, sticky=W)
        self.quick_clip_apply.grid(row=2, column=0)
        self.quick_downsample_frm.grid(row=0, column=1, sticky=W)
        self.quick_downsample_width_lbl.grid(row=0, column=0, sticky=W)
        self.quick_downsample_width.grid(row=0, column=1, sticky=W)
        self.quick_downsample_height_lbl.grid(row=1, column=0, sticky=W)
        self.quick_downsample_height.grid(row=1, column=1, sticky=W)
        self.quick_downsample_apply.grid(row=2, column=0, sticky=W)
        self.quick_set_fps.grid(row=0, column=2, sticky=W)
        self.quick_fps_lbl.grid(row=0, column=0, sticky=W)
        self.quick_fps_entry_box.grid(row=0, column=1, sticky=W)
        self.quick_set_fps_empty_row.grid(row=1, column=1, sticky=W)
        self.quick_fps_apply.grid(row=2, column=0, sticky=W)
        self.quick_set_quality.grid(row=0, column=3, sticky=W)
        self.quick_set_quality_dropdown.grid(row=0, column=0, sticky=W)
        self.quick_set_qualitys_empty_row.grid(row=1, column=0, sticky=W)
        self.quick_set_quality_apply.grid(row=2, column=0, sticky=W)
        self.use_gpu_frm.grid(row=0, column=4, sticky=W)

    def inverse_all_cb_ticks(self, variable_name=None):
        if self.headings[variable_name].get():
            for video_name in self.videos.keys():
                self.videos[video_name][variable_name].set(True)
        if not self.headings[variable_name].get():
            for video_name in self.videos.keys():
                self.videos[video_name][variable_name].set(False)

    def change_quality_options_cpu_gpu(self):
        self.quick_set_quality_dropdown.destroy()
        if not self.use_gpu_var.get():
            self.quick_set_quality_dropdown = DropDownMenu(
                self.quick_set_quality, "Video Quality % ", self.cpu_video_quality, "14"
            )
            self.quick_set_quality_dropdown.setChoices(100)
        else:
            self.quick_set_quality_dropdown = DropDownMenu(
                self.quick_set_quality,
                "Video Quality % ",
                ["Low", "Medium", "High"],
                "14",
            )
            self.quick_set_quality_dropdown.setChoices("Medium")
        self.quick_set_quality_dropdown.grid(row=0, column=0, sticky=W)

        for video_cnt, video_name in enumerate(self.videos.keys()):
            self.videos[video_name]["video_quality_dropdown"].destroy()
            if not self.use_gpu_var.get():
                self.videos[video_name]["video_quality_dropdown"] = DropDownMenu(
                    self.videos_frm, "", self.cpu_video_quality, "5"
                )
                self.videos[video_name]["video_quality_dropdown"].setChoices(100)
            else:
                self.videos[video_name]["video_quality_dropdown"] = DropDownMenu(
                    self.videos_frm, "", ["Low", "Medium", "High"], "5"
                )
                self.videos[video_name]["video_quality_dropdown"].setChoices("Medium")
            self.videos[video_name]["video_quality_dropdown"].grid(
                row=video_cnt + 1, column=14, sticky=W
            )

    def apply_resolution_to_all(self):
        check_int(
            value=self.quick_downsample_width_val.get(),
            min_value=0,
            name=f"Quick set downsample WIDTH {self.quick_downsample_width_val.get()}",
        )
        check_int(
            value=self.quick_downsample_height_val.get(),
            min_value=0,
            name=f"Quick set downsample HEIGHT {self.quick_downsample_height_val.get()}",
        )
        for video_name in self.videos.keys():
            self.videos[video_name]["width_var"].set(
                self.quick_downsample_width_val.get()
            )
            self.videos[video_name]["height_var"].set(
                self.quick_downsample_height_val.get()
            )

    def apply_trim_to_all(self):
        check_if_string_value_is_valid_video_timestamp(
            value=self.quick_clip_start_entry_box.get(),
            name=f"Quick set clip START time {self.quick_clip_start_entry_box.get()}",
        )
        check_if_string_value_is_valid_video_timestamp(
            value=self.quick_clip_end_entry_box.get(),
            name=f"Quick set clip END time {self.quick_clip_start_entry_box.get()}",
        )
        check_that_hhmmss_start_is_before_end(
            start_time=self.quick_clip_start_entry_box.get(),
            end_time=self.quick_clip_end_entry_box.get(),
            name="Quick set START and END time",
        )
        for video_name in self.videos.keys():
            self.videos[video_name]["start_time_var"].set(
                self.quick_clip_start_entry_box.get()
            )
            self.videos[video_name]["end_time_var"].set(
                self.quick_clip_end_entry_box.get()
            )

    def apply_fps_to_all(self):
        check_float(
            value=self.quick_set_fps_val.get(),
            min_value=0,
            name=f"Quick set FPS setting {self.quick_set_fps_val.get()}",
        )
        for video_name in self.videos.keys():
            self.videos[video_name]["fps_var"].set(self.quick_set_fps_val.get())

    def apply_quality_to_all(self):
        for video_name in self.videos.keys():
            self.videos[video_name]["video_quality_dropdown"].setChoices(
                self.quick_set_quality_dropdown.getChoices()
            )

    def create_video_table_headings(self):
        self.headings = {}
        self.videos_frm = LabelFrame(
            self.main_frm,
            text="VIDEOS",
            font=("Helvetica", 15, "bold"),
            pady=5,
            padx=15,
        )
        self.headings["video_name_col_head"] = Label(
            self.videos_frm, text="Video Name", width=self.max_char_vid_name + 5
        )
        self.headings["crop_video_col_head"] = Label(
            self.videos_frm, text="Crop Video", width=8
        )
        self.headings["start_time_col_head"] = Label(
            self.videos_frm, text="Start Time", width=8
        )
        self.headings["end_time_col_head"] = Label(
            self.videos_frm, text="End Time", width=8
        )
        self.headings["video_quality_head"] = Label(
            self.videos_frm, text="Quality %", width=15
        )
        self.headings["clip_cb_var"] = BooleanVar()
        self.headings["shorten_all_videos_cbox"] = Checkbutton(
            self.videos_frm,
            text="Clip All Videos",
            variable=self.headings["clip_cb_var"],
            command=lambda: self.inverse_all_cb_ticks(variable_name="clip_cb_var"),
        )
        self.headings["video_width_col_head"] = Label(
            self.videos_frm, text="Video Width", width=8
        )
        self.headings["video_height_col_head"] = Label(
            self.videos_frm, text="Video Height", width=8
        )
        self.headings["downsample_cb_var"] = BooleanVar()
        self.headings["downsample_all_videos_cbox"] = Checkbutton(
            self.videos_frm,
            text="Downsample All Videos",
            variable=self.headings["downsample_cb_var"],
            command=lambda: self.inverse_all_cb_ticks(
                variable_name="downsample_cb_var"
            ),
        )
        self.headings["fps_col_head"] = Label(
            self.videos_frm, text="Video FPS", width=8
        )
        self.headings["fps_cb_var"] = BooleanVar()
        self.headings["change_fps_all_videos_cbox"] = Checkbutton(
            self.videos_frm,
            text="Change FPS",
            variable=self.headings["fps_cb_var"],
            command=lambda: self.inverse_all_cb_ticks(variable_name="fps_cb_var"),
        )
        self.headings["grayscale_cb_var"] = BooleanVar()
        self.headings["grayscale_cbox"] = Checkbutton(
            self.videos_frm,
            text="Apply Greyscale",
            variable=self.headings["grayscale_cb_var"],
            command=lambda: self.inverse_all_cb_ticks(variable_name="grayscale_cb_var"),
        )
        self.headings["frame_cnt_cb_var"] = BooleanVar()
        self.headings["frame_cnt_cbox"] = Checkbutton(
            self.videos_frm,
            text="Print Frame Count",
            variable=self.headings["frame_cnt_cb_var"],
            command=lambda: self.inverse_all_cb_ticks(variable_name="frame_cnt_cb_var"),
        )
        self.headings["apply_clahe_cb_var"] = BooleanVar()
        self.headings["apply_clahe_cbox"] = Checkbutton(
            self.videos_frm,
            text="Apply CLAHE",
            variable=self.headings["apply_clahe_cb_var"],
            command=lambda: self.inverse_all_cb_ticks(
                variable_name="apply_clahe_cb_var"
            ),
        )

        self.videos_frm.grid(row=1, column=0, sticky=W, padx=5, pady=15)
        self.headings["video_name_col_head"].grid(row=0, column=1, sticky=W, padx=5)
        self.headings["crop_video_col_head"].grid(row=0, column=2, sticky=W, padx=5)
        self.headings["start_time_col_head"].grid(row=0, column=3, sticky=W, padx=5)
        self.headings["end_time_col_head"].grid(row=0, column=4, sticky=W, padx=5)
        self.headings["shorten_all_videos_cbox"].grid(row=0, column=5, sticky=W, padx=5)
        self.headings["video_width_col_head"].grid(row=0, column=6, sticky=W, padx=5)
        self.headings["video_height_col_head"].grid(row=0, column=7, sticky=W, padx=5)
        self.headings["downsample_all_videos_cbox"].grid(
            row=0, column=8, sticky=W, padx=5
        )
        self.headings["fps_col_head"].grid(row=0, column=9, sticky=W, padx=5)
        self.headings["change_fps_all_videos_cbox"].grid(
            row=0, column=10, sticky=W, padx=5
        )
        self.headings["grayscale_cbox"].grid(row=0, column=11, sticky=W, padx=5)
        self.headings["frame_cnt_cbox"].grid(row=0, column=12, sticky=W, padx=5)
        self.headings["apply_clahe_cbox"].grid(row=0, column=13, sticky=W, padx=5)
        self.headings["video_quality_head"].grid(row=0, column=14, sticky=NW)

    def create_video_rows(self):
        self.videos = {}
        for w in self.videos_frm.grid_slaves():
            if w.grid_info()["column"] == 14 and w.grid_info()["row"] > 0:
                w.grid_remove()
        for video_cnt, (name, data) in enumerate(self.videos_in_dir_dict.items()):
            self.videos[name] = {}
            row = video_cnt + 1
            self.videos[name]["video_name_lbl"] = Label(
                self.videos_frm, text=name, width=self.max_char_vid_name + 5
            )
            self.videos[name]["crop_btn"] = Button(
                self.videos_frm,
                text="Crop",
                fg="black",
                command=lambda k=self.videos[name]["video_name_lbl"][
                    "text"
                ]: self.batch_process_crop_function(k),
            )
            self.videos[name]["start_time_var"] = StringVar()
            self.videos[name]["start_time_var"].set("00:00:00")
            self.videos[name]["start_entry"] = Entry(
                self.videos_frm,
                width=6,
                textvariable=self.videos[name]["start_time_var"],
            )
            self.videos[name]["end_time_var"] = StringVar()
            self.videos[name]["end_time_var"].set(data["video_length"])
            self.videos[name]["end_entry"] = Entry(
                self.videos_frm, width=6, textvariable=self.videos[name]["end_time_var"]
            )
            self.videos[name]["clip_cb_var"] = BooleanVar()
            self.videos[name]["clip_cb"] = Checkbutton(
                self.videos_frm, variable=self.videos[name]["clip_cb_var"], command=None
            )
            self.videos[name]["width_var"] = IntVar()
            self.videos[name]["width_var"].set(data["width"])
            self.videos[name]["width_entry"] = Entry(
                self.videos_frm, width=6, textvariable=self.videos[name]["width_var"]
            )
            self.videos[name]["height_var"] = IntVar()
            self.videos[name]["height_var"].set(data["height"])
            self.videos[name]["height_entry"] = Entry(
                self.videos_frm, width=6, textvariable=self.videos[name]["height_var"]
            )
            self.videos[name]["downsample_cb_var"] = BooleanVar()
            self.videos[name]["downsample_cb"] = Checkbutton(
                self.videos_frm,
                variable=self.videos[name]["downsample_cb_var"],
                command=None,
            )
            self.videos[name]["fps_var"] = DoubleVar()
            self.videos[name]["fps_var"].set(round(data["fps"], 4))
            self.videos[name]["fps_entry"] = Entry(
                self.videos_frm, width=6, textvariable=self.videos[name]["fps_var"]
            )
            self.videos[name]["fps_cb_var"] = BooleanVar()
            self.videos[name]["fps_cb"] = Checkbutton(
                self.videos_frm, variable=self.videos[name]["fps_cb_var"], command=None
            )
            self.videos[name]["grayscale_cb_var"] = BooleanVar()
            self.videos[name]["grayscale_cbox"] = Checkbutton(
                self.videos_frm,
                variable=self.videos[name]["grayscale_cb_var"],
                command=None,
            )
            self.videos[name]["frame_cnt_cb_var"] = BooleanVar()
            self.videos[name]["frame_cnt_cbox"] = Checkbutton(
                self.videos_frm,
                variable=self.videos[name]["frame_cnt_cb_var"],
                command=None,
            )
            self.videos[name]["apply_clahe_cb_var"] = BooleanVar()
            self.videos[name]["apply_clahe_cbox"] = Checkbutton(
                self.videos_frm,
                variable=self.videos[name]["apply_clahe_cb_var"],
                command=None,
            )
            self.videos[name]["video_quality_dropdown"] = DropDownMenu(
                self.videos_frm, "", self.cpu_video_quality, "5"
            )
            self.videos[name]["video_quality_dropdown"].setChoices(100)

            self.videos[name]["video_name_lbl"].grid(
                row=row, column=1, sticky=W, padx=5
            )
            self.videos[name]["crop_btn"].grid(row=row, column=2, padx=5)
            self.videos[name]["start_entry"].grid(row=row, column=3, padx=5)
            self.videos[name]["end_entry"].grid(row=row, column=4, padx=5)
            self.videos[name]["clip_cb"].grid(row=row, column=5, sticky=W, padx=5)
            self.videos[name]["width_entry"].grid(row=row, column=6, padx=5)
            self.videos[name]["height_entry"].grid(row=row, column=7, padx=5)
            self.videos[name]["downsample_cb"].grid(row=row, column=8, sticky=W, padx=5)
            self.videos[name]["fps_entry"].grid(row=row, column=9, padx=5)
            self.videos[name]["fps_cb"].grid(row=row, column=10, sticky=W, padx=5)
            self.videos[name]["grayscale_cbox"].grid(
                row=row, column=11, sticky=W, padx=5
            )
            self.videos[name]["frame_cnt_cbox"].grid(
                row=row, column=12, sticky=W, padx=5
            )
            self.videos[name]["apply_clahe_cbox"].grid(
                row=row, column=13, sticky=W, padx=5
            )
            try:
                self.videos[name]["video_quality_dropdown"].grid_remove(
                    row=row, column=14, sticky=NW
                )
            except:
                pass
            self.videos[name]["video_quality_dropdown"].grid(
                row=row, column=14, sticky=NW
            )

    def create_execute_btn(self):
        self.execute_frm = LabelFrame(
            self.main_frm,
            text="EXECUTE",
            font=("Helvetica", 15, "bold"),
            pady=5,
            padx=15,
        )
        self.reset_all_btn = Button(
            self.execute_frm,
            text="RESET ALL",
            fg="red",
            command=lambda: self.create_video_rows(),
        )
        self.reset_crop_btn = Button(
            self.execute_frm,
            text="RESET CROP",
            fg="orange",
            command=lambda: self.reset_crop(),
        )
        self.execute_btn = Button(
            self.execute_frm, text="EXECUTE", fg="blue", command=lambda: self.execute()
        )

        self.execute_frm.grid(row=2, column=0, sticky=W, padx=5, pady=30)
        self.reset_all_btn.grid(row=0, column=0, sticky=W, padx=5)
        self.reset_crop_btn.grid(row=0, column=1, sticky=W, padx=5)
        self.execute_btn.grid(row=0, column=2, sticky=W, padx=5)

    def reset_crop(self):
        self.crop_dict = {}
        for video_name, video_data in self.videos_in_dir_dict.items():
            self.videos[video_name]["crop_btn"].configure(fg="black")

    def batch_process_crop_function(self, video_name):
        check_file_exist_and_readable(self.videos_in_dir_dict[video_name]["file_path"])
        roi_selector = ROISelector(
            path=self.videos_in_dir_dict[video_name]["file_path"],
            title=f"CROP {video_name} - Press ESC when ROI drawn",
        )
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
        self.videos[video_name]["crop_btn"].configure(fg="red")

    def execute(self):
        out_video_dict = {}
        out_video_dict["meta_data"] = {}
        out_video_dict["video_data"] = {}
        out_video_dict["meta_data"]["in_dir"] = self.input_dir
        out_video_dict["meta_data"]["out_dir"] = self.output_dir
        out_video_dict["meta_data"]["gpu"] = self.use_gpu_var.get()
        if self.use_gpu_var.get() and not check_nvidea_gpu_available():
            raise FFMPEGCodecGPUError(
                msg="No GPU found (as evaluated by nvidea-smi returning None)",
                source=self.__class__.__name__,
            )
        for video_cnt, (name, data) in enumerate(self.videos_in_dir_dict.items()):
            out_video_dict["video_data"][name] = {}
            out_video_dict["video_data"][name]["video_info"] = self.videos_in_dir_dict[
                name
            ]
            if not self.use_gpu_var.get():
                out_video_dict["video_data"][name]["output_quality"] = (
                    self.percent_to_crf_lookup[
                        self.videos[name]["video_quality_dropdown"].getChoices()
                    ]
                )
            else:
                out_video_dict["video_data"][name]["output_quality"] = (
                    self.video_quality_to_preset_lookup[
                        self.videos[name]["video_quality_dropdown"].getChoices()
                    ]
                )
            if name in self.crop_dict.keys():
                out_video_dict["video_data"][name]["crop"] = True
                out_video_dict["video_data"][name]["crop_settings"] = self.crop_dict[
                    name
                ]
            else:
                out_video_dict["video_data"][name]["crop"] = False
                out_video_dict["video_data"][name]["crop_settings"] = None
            if self.videos[name]["clip_cb_var"].get():
                out_video_dict["video_data"][name]["clip"] = True
                out_video_dict["video_data"][name]["clip_settings"] = {
                    "start": self.videos[name]["start_time_var"].get(),
                    "stop": self.videos[name]["end_time_var"].get(),
                }
            else:
                out_video_dict["video_data"][name]["clip"] = False
                out_video_dict["video_data"][name]["clip_settings"] = None
            if self.videos[name]["downsample_cb_var"].get():
                out_video_dict["video_data"][name]["downsample"] = True
                out_video_dict["video_data"][name]["downsample_settings"] = {
                    "width": self.videos[name]["width_var"].get(),
                    "height": self.videos[name]["height_var"].get(),
                }
            else:
                out_video_dict["video_data"][name]["downsample"] = False
                out_video_dict["video_data"][name]["downsample_settings"] = None
            if self.videos[name]["fps_cb_var"].get():
                out_video_dict["video_data"][name]["fps"] = True
                out_video_dict["video_data"][name]["fps_settings"] = {
                    "fps": self.videos[name]["fps_var"].get()
                }
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
            for operation in [
                "clahe",
                "frame_cnt",
                "grayscale",
                "fps",
                "downsample",
                "clip",
                "crop",
            ]:
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
        stdout_success(
            msg=f"SimBA batch pre-process JSON saved at {self.save_path}",
            source=self.__class__.__name__,
        )
        stdout_success(
            msg=f"Video batch pre-processing complete, new videos stored in {self.output_dir}",
            elapsed_time=timer.elapsed_time_str,
            source=self.__class__.__name__,
        )


# test = BatchProcessFrame(input_dir=r'/Users/simon/Desktop',
#                          output_dir=r'/Users/simon/Desktop/edited')
# test.create_main_window()
# test.create_video_table_headings()
# test.create_video_rows()
# test.create_execute_btn()
# test.main_frm.mainloop()
