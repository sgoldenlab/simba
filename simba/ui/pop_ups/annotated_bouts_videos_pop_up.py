__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from tkinter import *
from typing import Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.annotation_videos import PlotAnnotatedBouts
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton, SimbaCheckbox,
                                        SimBADropDown)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_int, check_nvidea_gpu_available)
from simba.utils.enums import Formats, Options
from simba.utils.errors import NoChoosenClassifierError, NoFilesFoundError
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import find_video_of_file, get_fn_ext, str_2_bool

ALL_VIDEOS = "ALL VIDEOS"
AUTO = "AUTO"
TEXT_SIZE_OPTIONS = list(range(1, 101))
TEXT_SIZE_OPTIONS.insert(0, AUTO)
OPACITY_OPTIONS = [round(x, 1) for x in np.arange(0.1, 1.1, 0.1)]
FALSE = "FALSE"


class AnnotatedBoutsVideoPopUp(PopUpMixin, ConfigReader):
    """
    :example:
    >>> _ = AnnotatedBoutsVideoPopUp(config_path=r"C:\\troubleshooting\\project_folder\\project_config.ini")
    """

    def __init__(self, config_path: Union[str, os.PathLike]):
        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        if len(self.target_file_paths) == 0:
            raise NoFilesFoundError(msg=f"Cannot create annotated bout videos: No data files found in {self.targets_folder}.", source=self.__class__.__name__)
        if len(self.clf_names) == 0:
            raise NoFilesFoundError(msg=f"The SimBA project {config_path} does not have any defined classifier names.", source=self.__class__.__name__)
        self.video_dict = {}
        for file_path in self.target_file_paths:
            self.video_dict[get_fn_ext(filepath=file_path)[1]] = file_path
        self.video_options = [ALL_VIDEOS] + list(self.video_dict.keys())
        self.clr_dict = get_color_dict()
        self.pose_palettes = Options.PALETTE_OPTIONS_CATEGORICAL.value + Options.PALETTE_OPTIONS.value
        gpu_available = check_nvidea_gpu_available()
        PopUpMixin.__init__(self, title="VISUALIZE ANNOTATION BOUTS", config_path=config_path, icon="video", size=(800, 900))

        self.clf_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHOOSE CLASSIFIERS", icon_name="forest")
        self.classifiers_cbs = {}
        for clf_cnt, clf_name in enumerate(self.clf_names):
            self.classifiers_cbs[clf_name] = SimbaCheckbox(parent=self.clf_frm, txt=clf_name, val=True)
            self.classifiers_cbs[clf_name][0].grid(row=clf_cnt, column=0, sticky=NW)
        self.clf_frm.grid(row=0, column=0, sticky=NW)

        self.video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHOOSE VIDEOS", icon_name="video")
        self.video_dropdown = SimBADropDown(parent=self.video_frm, dropdown_options=self.video_options, label="VIDEO:", label_width=35, dropdown_width=30, value=ALL_VIDEOS, img="video_2", tooltip_key="ANNOTATION_BOUTS_VIDEO")
        self.video_frm.grid(row=1, column=0, sticky=NW)
        self.video_dropdown.grid(row=0, column=0, sticky=NW)

        self.window_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="BOUT WINDOW SETTINGS", icon_name="timer")
        self.pre_window_entry = Entry_Box(parent=self.window_frm, fileDescription="PRE-BOUT WINDOW (SECONDS):", labelwidth=35, entry_box_width=15, value=5, justify="center", img="timer", validation="numeric", tooltip_key="ANNOTATION_BOUTS_PRE_WINDOW")
        self.post_window_entry = Entry_Box(parent=self.window_frm, fileDescription="POST-BOUT WINDOW (SECONDS):", labelwidth=35, entry_box_width=15, value=5, justify="center", img="timer", validation="numeric", tooltip_key="ANNOTATION_BOUTS_POST_WINDOW")
        self.window_frm.grid(row=2, column=0, sticky=NW)
        self.pre_window_entry.grid(row=0, column=0, sticky=NW)
        self.post_window_entry.grid(row=1, column=0, sticky=NW)

        self.style_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name="style")
        self.text_size_dropdown = SimBADropDown(parent=self.style_frm, dropdown_options=TEXT_SIZE_OPTIONS, label="TEXT SIZE:", label_width=35, dropdown_width=15, value=AUTO, img="text", tooltip_key="ANNOTATION_BOUTS_TEXT_SIZE")
        self.text_spacing_dropdown = SimBADropDown(parent=self.style_frm, dropdown_options=TEXT_SIZE_OPTIONS, label="TEXT SPACING:", label_width=35, dropdown_width=15, value=AUTO, img="text_spacing", tooltip_key="ANNOTATION_BOUTS_TEXT_SPACING")
        self.text_thickness_dropdown = SimBADropDown(parent=self.style_frm, dropdown_options=TEXT_SIZE_OPTIONS, label="TEXT THICKNESS:", label_width=35, dropdown_width=15, value=AUTO, img="bold", tooltip_key="ANNOTATION_BOUTS_TEXT_THICKNESS")
        self.circle_size_dropdown = SimBADropDown(parent=self.style_frm, dropdown_options=TEXT_SIZE_OPTIONS, label="CIRCLE SIZE:", label_width=35, dropdown_width=15, value=AUTO, img="circle_small", tooltip_key="ANNOTATION_BOUTS_CIRCLE_SIZE")
        self.text_opacity_dropdown = SimBADropDown(parent=self.style_frm, dropdown_options=OPACITY_OPTIONS, label="TEXT OPACITY:", label_width=35, dropdown_width=15, value=0.8, img="opacity", tooltip_key="ANNOTATION_BOUTS_TEXT_OPACITY")
        self.text_clr_dropdown = SimBADropDown(parent=self.style_frm, dropdown_options=list(self.clr_dict.keys()), label="TEXT COLOR:", label_width=35, dropdown_width=15, value="White", img="text_color", tooltip_key="ANNOTATION_BOUTS_TEXT_COLOR")
        self.text_bg_clr_dropdown = SimBADropDown(parent=self.style_frm, dropdown_options=list(self.clr_dict.keys()), label="TEXT BG COLOR:", label_width=35, dropdown_width=15, value="Black", img="fill", tooltip_key="ANNOTATION_BOUTS_TEXT_BG_COLOR")
        self.tracking_clr_palette_dropdown = SimBADropDown(parent=self.style_frm, dropdown_options=self.pose_palettes, label="TRACKING COLOR PALETTE:", label_width=35, dropdown_width=15, value="Set1", img="color_wheel", tooltip_key="ANNOTATION_BOUTS_TRACKING_PALETTE")
        self.style_frm.grid(row=3, column=0, sticky=NW)
        self.text_size_dropdown.grid(row=0, column=0, sticky=NW)
        self.text_spacing_dropdown.grid(row=1, column=0, sticky=NW)
        self.text_thickness_dropdown.grid(row=2, column=0, sticky=NW)
        self.circle_size_dropdown.grid(row=3, column=0, sticky=NW)
        self.text_opacity_dropdown.grid(row=4, column=0, sticky=NW)
        self.text_clr_dropdown.grid(row=5, column=0, sticky=NW)
        self.text_bg_clr_dropdown.grid(row=6, column=0, sticky=NW)
        self.tracking_clr_palette_dropdown.grid(row=7, column=0, sticky=NW)

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZATION SETTINGS", icon_name="eye")
        self.multiprocess_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(2, self.cpu_cnt + 1)), label="CPU CORES:", label_width=35, dropdown_width=15, value=int(max(2, self.cpu_cnt / 3)), img="cpu_small", tooltip_key="ANNOTATION_BOUTS_CPU_CORES")
        self.gpu_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=["TRUE", FALSE], label="USE GPU:", label_width=35, dropdown_width=15, value=FALSE, state=DISABLED if not gpu_available else NORMAL, img="gpu_3", tooltip_key="ANNOTATION_BOUTS_USE_GPU")
        self.bbox_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=[FALSE, Options.AXIS_ALIGNED.value, Options.ANIMAL_ALIGNED.value], label="SHOW ANIMAL BBOX:", label_width=35, dropdown_width=15, value=FALSE, img="rectangle", tooltip_key="ANNOTATION_BOUTS_SHOW_BBOX")
        self.timer_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=[FALSE, Options.SECONDS.value, Options.HHMMSSSSSS.value], label="SHOW VIDEO TIMER:", label_width=35, dropdown_width=15, value=Options.HHMMSSSSSS.value, img="timer", tooltip_key="ANNOTATION_BOUTS_TIMER")
        self.show_pose_cb, self.show_pose_var = SimbaCheckbox(parent=self.settings_frm, txt="SHOW TRACKING (POSE)", font=Formats.FONT_REGULAR.value, txt_img="pose", val=True)
        self.show_animal_names_cb, self.show_animal_names_var = SimbaCheckbox(parent=self.settings_frm, txt="SHOW ANIMAL NAME(S)", font=Formats.FONT_REGULAR.value, txt_img="label", val=False)
        self.verbose_cb, self.verbose_var = SimbaCheckbox(parent=self.settings_frm, txt="VERBOSE OUTPUT", font=Formats.FONT_REGULAR.value, txt_img="details", val=True)
        self.settings_frm.grid(row=4, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=0, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=1, column=0, sticky=NW)
        self.bbox_dropdown.grid(row=2, column=0, sticky=NW)
        self.timer_dropdown.grid(row=3, column=0, sticky=NW)
        self.show_pose_cb.grid(row=4, column=0, sticky=NW)
        self.show_animal_names_cb.grid(row=5, column=0, sticky=NW)
        self.verbose_cb.grid(row=6, column=0, sticky=NW)

        self.run_btn = SimbaButton(parent=self.main_frm, txt="RUN", img="rocket", txt_clr="blue", font=Formats.FONT_REGULAR.value, cmd=self.run)
        self.run_btn.grid(row=5, column=0, sticky=NW)
        self.main_frm.mainloop()

    def __get_int_value(self, value: str, name: str, min_value: int = 1) -> int:
        value = str(value).strip()
        check_int(name=name, value=value, min_value=min_value)
        return int(value)

    def run(self):
        clfs = []
        for clf_name, selections in self.classifiers_cbs.items():
            if selections[1].get(): clfs.append(clf_name)
        if len(clfs) == 0: raise NoChoosenClassifierError(source=self.__class__.__name__)

        video_selection = self.video_dropdown.getChoices()
        if video_selection == ALL_VIDEOS:
            data_paths = list(self.video_dict.values())
        else:
            data_paths = [self.video_dict[video_selection]]
        for data_path in data_paths:
            _ = find_video_of_file(video_dir=self.video_dir, filename=get_fn_ext(filepath=data_path)[1], raise_error=True)

        font_size = None if self.text_size_dropdown.get_value() == AUTO else float(self.text_size_dropdown.get_value())
        space_size = None if self.text_spacing_dropdown.get_value() == AUTO else float(self.text_spacing_dropdown.get_value())
        text_thickness = None if self.text_thickness_dropdown.get_value() == AUTO else float(self.text_thickness_dropdown.get_value())
        circle_size = None if self.circle_size_dropdown.get_value() == AUTO else float(self.circle_size_dropdown.get_value())
        text_opacity = float(self.text_opacity_dropdown.get_value())
        text_clr = self.clr_dict[self.text_clr_dropdown.get_value()]
        text_bg_clr = self.clr_dict[self.text_bg_clr_dropdown.get_value()]
        pose_palette = self.tracking_clr_palette_dropdown.get_value()
        bbox = None if self.bbox_dropdown.get_value() == FALSE else self.bbox_dropdown.get_value()
        timer = None if self.timer_dropdown.get_value() == FALSE else self.timer_dropdown.get_value()
        pre_window = self.__get_int_value(value=self.pre_window_entry.entry_get, name="PRE BOUT WINDOW (SECONDS)")
        post_window = self.__get_int_value(value=self.post_window_entry.entry_get, name="POST BOUT WINDOW (SECONDS)")

        visualizer = PlotAnnotatedBouts(
            config_path=self.config_path,
            data_paths=data_paths,
            animal_names=self.show_animal_names_var.get(),
            show_pose=self.show_pose_var.get(),
            pre_window=pre_window,
            post_window=post_window,
            font_size=font_size,
            space_size=space_size,
            text_thickness=text_thickness,
            text_opacity=text_opacity,
            circle_size=circle_size,
            pose_palette=pose_palette,
            clf_names=clfs,
            video_timer=timer.lower(),
            overwrite=True,
            bbox=bbox,
            text_clr=text_clr,
            text_bg_clr=text_bg_clr,
            gpu=str_2_bool(self.gpu_dropdown.get_value()),
            verbose=self.verbose_var.get(),
            core_cnt=int(self.multiprocess_dropdown.get_value()),
        )
        visualizer.run()


# _ = AnnotatedBoutsVideoPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
