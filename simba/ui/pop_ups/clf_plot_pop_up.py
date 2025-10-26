__author__ = "Simon Nilsson"
__email__ = "sronilsson@gmail.com"

import os.path
from tkinter import *
from typing import Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.plot_clf_results import PlotSklearnResultsSingleCore
from simba.plotting.plot_clf_results_mp import PlotSklearnResultsMultiProcess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        FileSelect, SimbaButton, SimbaCheckbox,
                                        SimBADropDown, SimBALabel)
from simba.utils.checks import check_float, check_nvidea_gpu_available
from simba.utils.enums import Formats, Links, Options
from simba.utils.errors import NoFilesFoundError, NoSpecifiedOutputError
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import (find_all_videos_in_directory,
                                    find_video_of_file, str_2_bool)

AUTO = 'AUTO'
TEXT_SIZE_OPTIONS = list(range(1, 101))
TEXT_SIZE_OPTIONS.insert(0, 'AUTO')

OPACITY_OPTIONS = list(np.arange(0.1, 1.1, 0.1))
OPACITY_OPTIONS = [round(x, 1) for x in OPACITY_OPTIONS]
GANTT_OPTIONS = {'NO GANTT': None, 'Static Gantt (final frame, faster)': 1, 'Dynamic Gantt (updated per frame)': 2}

class SklearnVisualizationPopUp(PopUpMixin, ConfigReader):

    """
    :example:
    >>> _ = SklearnVisualizationPopUp(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")
    """

    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path)
        self.video_dict = find_all_videos_in_directory(directory=self.video_dir, as_dict=True, raise_error=True)
        self.video_lst = list(self.video_dict.keys())
        if len(self.video_lst) == 0:
            raise NoFilesFoundError(msg=f'Cannot create classification videos: No video files found in {self.video_dir} directory', source=self.__class__.__name__)
        if len(self.machine_results_paths) == 0:
            raise NoFilesFoundError(msg=f'Cannot create classification videos: No data files found in {self.machine_results_dir} directory', source=self.__class__.__name__)
        self.max_len = max(len(s) for s in self.video_lst)
        gpu_available = check_nvidea_gpu_available()
        self.clr_dict = get_color_dict()
        pose_palettes = Options.PALETTE_OPTIONS_CATEGORICAL.value + Options.PALETTE_OPTIONS.value

        PopUpMixin.__init__(self, title="VISUALIZE CLASSIFICATION (SKLEARN) RESULTS", icon='photos')
        bp_threshold_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="BODY-PART VISUALIZATION THRESHOLD", icon_name='threshold', icon_link=Links.SKLEARN_PLOTS.value, padx=5, pady=5, relief='solid')
        self.bp_threshold_lbl = SimBALabel(parent=bp_threshold_frm, txt="Body-parts detected below the set threshold won't be shown in the output videos.", font=Formats.FONT_REGULAR_ITALICS.value)
        self.bp_threshold_entry = Entry_Box(parent=bp_threshold_frm, fileDescription='BODY-PART PROBABILITY THRESHOLD: ', labelwidth=40, entry_box_width=15, value=0.00)
        self.get_bp_probability_threshold()

        bp_threshold_frm.grid(row=0, column=0, sticky=NW)
        self.bp_threshold_lbl.grid(row=0, column=0, sticky=NW)
        self.bp_threshold_entry.grid(row=1, column=0, sticky=NW)

        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='style', icon_link=Links.SKLEARN_PLOTS.value, padx=5, pady=5, relief='solid')
        self.text_size_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=TEXT_SIZE_OPTIONS, label='TEXT SIZE: ', label_width=40, dropdown_width=15, value='AUTO')
        self.text_spacing_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=TEXT_SIZE_OPTIONS, label='TEXT SPACING: ', label_width=40, dropdown_width=15, value='AUTO')
        self.text_thickness_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=TEXT_SIZE_OPTIONS, label='TEXT THICKNESS: ', label_width=40, dropdown_width=15, value='AUTO')
        self.circle_size_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=TEXT_SIZE_OPTIONS, label='CIRCLE SIZE: ', label_width=40, dropdown_width=15, value='AUTO')
        self.text_opacity_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=OPACITY_OPTIONS, label='TEXT OPACITY: ', label_width=40, dropdown_width=15, value=0.8)
        self.text_clr_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=list(self.clr_dict.keys()), label='TEXT COLOR: ', label_width=40, dropdown_width=15, value='White')
        self.bg_clr_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=list(self.clr_dict.keys()), label='TEXT BACKGROUND COLOR: ', label_width=40, dropdown_width=15, value='Black')
        self.tracking_clr_palette_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=pose_palettes, label='TRACKING COLOR PALETTE: ', label_width=40, dropdown_width=15, value='Set1')

        self.style_settings_frm.grid(row=1, column=0, sticky=NW)
        self.text_size_dropdown.grid(row=0, column=0, sticky=NW)
        self.text_spacing_dropdown.grid(row=1, column=0, sticky=NW)
        self.text_thickness_dropdown.grid(row=2, column=0, sticky=NW)
        self.circle_size_dropdown.grid(row=3, column=0, sticky=NW)
        self.text_opacity_dropdown.grid(row=4, column=0, sticky=NW)
        self.text_clr_dropdown.grid(row=5, column=0, sticky=NW)
        self.bg_clr_dropdown.grid(row=6, column=0, sticky=NW)
        self.tracking_clr_palette_dropdown.grid(row=7, column=0, sticky=NW)

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZATION SETTINGS",  icon_name='eye', icon_link=Links.SKLEARN_PLOTS.value, padx=5,  pady=5, relief='solid')
        self.multiprocess_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(1, self.cpu_cnt+1)), label='CPU CORES: ', label_width=40, dropdown_width=30, value=int(self.cpu_cnt/2))
        self.gpu_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label='USE GPU: ', label_width=40, dropdown_width=30, value='FALSE', state=DISABLED if not gpu_available else NORMAL)
        self.gantt_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(GANTT_OPTIONS.keys()), label='SHOW GANTT PLOT:', label_width=40, dropdown_width=30, value=list(GANTT_OPTIONS.keys())[0])

        self.create_videos_cb, self.create_videos_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE VIDEO', font=Formats.FONT_REGULAR.value, txt_img='video', val=True)
        self.create_frames_cb, self.create_frames_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE FRAMES', font=Formats.FONT_REGULAR.value, txt_img='frames', val=False)
        self.timers_cb, self.include_timers_var = SimbaCheckbox(parent=self.settings_frm, txt='INCLUDE TIMERS OVERLAY', font=Formats.FONT_REGULAR.value, txt_img='timer', val=True)
        self.rotate_cb, self.rotate_img_var = SimbaCheckbox(parent=self.settings_frm, txt="ROTATE VIDEO 90°", font=Formats.FONT_REGULAR.value, txt_img='rotate', val=False)
        self.show_pose_cb, self.show_pose_var = SimbaCheckbox(parent=self.settings_frm, txt="SHOW TRACKING (POSE)", font=Formats.FONT_REGULAR.value, txt_img='pose', val=True)
        self.show_animal_names_cb, self.show_animal_names_var = SimbaCheckbox(parent=self.settings_frm, txt="SHOW ANIMAL NAME(S)", font=Formats.FONT_REGULAR.value, txt_img='label', val=False)
        self.show_bboxes_cb, self.show_bboxes_var = SimbaCheckbox(parent=self.settings_frm, txt="SHOW ANIMAL BOUNDING BOXES", font=Formats.FONT_REGULAR.value, txt_img='rectangle_2', val=False)

        self.settings_frm.grid(row=2, column=0,  sticky=NW)
        self.multiprocess_dropdown.grid(row=0, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=1, column=0, sticky=NW)
        self.gantt_dropdown.grid(row=2, column=0, sticky=NW)
        self.show_pose_cb.grid(row=3, column=0, sticky=NW)
        self.show_bboxes_cb.grid(row=4, column=0, sticky=NW)
        self.show_animal_names_cb.grid(row=5, column=0, sticky=NW)
        self.create_videos_cb.grid(row=6, column=0,  sticky=NW)
        self.create_frames_cb.grid(row=7, column=0,  sticky=NW)
        self.timers_cb.grid(row=8, column=0, sticky=NW)
        self.rotate_cb.grid(row=9, column=0, sticky=NW)


        self.run_single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO",  icon_name='video', icon_link=Links.SKLEARN_PLOTS.value, padx=5,  pady=5, relief='solid')
        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, txt="CREATE SINGLE VIDEO", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.__run, cmd_kwargs={'multiple': lambda: False})

        self.single_video_dropdown = SimBADropDown(parent=self.run_single_video_frm, dropdown_options=self.video_lst, label='VIDEO: ', label_width=12, value=self.video_lst[0], dropdown_width=self.max_len)
        self.select_video_file_select = FileSelect(self.run_single_video_frm, "", lblwidth="1", file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], dropdown=self.single_video_dropdown, initialdir=self.video_dir, initial_path=self.video_lst[0])

        self.run_multiple_videos = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS", icon_name='stack', icon_link=Links.SKLEARN_PLOTS.value, padx=5, pady=5, relief='solid')
        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, txt=f"CREATE MULTIPLE VIDEOS ({len(self.machine_results_paths)} found)", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.__run, cmd_kwargs={'multiple': lambda: True})

        self.run_single_video_frm.grid(row=3, sticky=NW)
        self.run_single_video_btn.grid(row=1, sticky=NW)
        self.single_video_dropdown.grid(row=1, column=1, sticky=NW)
        self.select_video_file_select.grid(row=1, column=2, sticky=NW)
        self.run_multiple_videos.grid(row=4, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)
        self.main_frm.mainloop()

    def __update_single_video_file_path(self, filename: str):
        self.select_video_file_select.filePath.set(filename)

    def get_bp_probability_threshold(self):
        try:
            self.bp_threshold_entry.entry_set(self.config.getfloat("threshold_settings", "bp_threshold_sklearn"))
        except:
            self.bp_threshold_entry.entry_set(0.0)

    def __run(self, multiple: bool = False):
        check_float(name="BODY-PART PROBABILITY THRESHOLD", value=self.bp_threshold_entry.entry_get, min_value=0.0, max_value=1. )
        self.config.set("threshold_settings", "bp_threshold_sklearn", self.bp_threshold_entry.entry_get)
        with open(self.config_path, "w") as f: self.config.write(f)

        font_size = float(self.text_size_dropdown.get_value()) if self.text_size_dropdown.get_value() != AUTO else None
        circle_size = float(self.circle_size_dropdown.get_value())  if self.circle_size_dropdown.get_value() != AUTO else None
        space_size = float(self.text_spacing_dropdown.get_value()) if self.text_spacing_dropdown.get_value() != AUTO else None
        text_thickness = float(self.text_thickness_dropdown.get_value()) if self.text_thickness_dropdown.get_value() != AUTO else None
        text_opacity = float(self.text_opacity_dropdown.get_value())
        text_clr = self.clr_dict[self.text_clr_dropdown.get_value()]
        text_bg_clr = self.clr_dict[self.bg_clr_dropdown.get_value()]
        pose_palette = self.tracking_clr_palette_dropdown.get_value()
        bbox = self.show_bboxes_var.get()
        gantt = GANTT_OPTIONS[self.gantt_dropdown.get_value()]
        show_pose, show_animal_names = self.show_pose_var.get(), self.show_animal_names_var.get()
        gpu = str_2_bool(self.gpu_dropdown.get_value())
        core_cnt = int(self.multiprocess_dropdown.get_value())
        frm_setting, video_setting = self.create_frames_var.get(), self.create_videos_var.get()
        if not frm_setting and not video_setting:
            raise NoSpecifiedOutputError(msg="Please choose to create a video and/or frames. SimBA found that you ticked neither video and/or frames", source=self.__class__.__name__)

        if not multiple:
            video_name = self.single_video_dropdown.getChoices()
            video_path = find_video_of_file(video_dir=self.video_dir, filename=video_name, raise_error=True)
        else:
            video_path = None

        if core_cnt == 1:
            plotter = PlotSklearnResultsSingleCore(config_path=self.config_path,
                                                   video_setting=self.create_videos_var.get(),
                                                   rotate=self.rotate_img_var.get(),
                                                   video_paths=video_path,
                                                   frame_setting=self.create_frames_var.get(),
                                                   print_timers=self.include_timers_var.get(),
                                                   font_size=font_size,
                                                   space_size=space_size,
                                                   text_thickness=text_thickness,
                                                   circle_size=circle_size,
                                                   show_pose=show_pose,
                                                   animal_names=show_animal_names,
                                                   text_opacity=text_opacity,
                                                   text_clr=text_clr,
                                                   text_bg_clr=text_bg_clr,
                                                   pose_palette=pose_palette,
                                                   show_bbox=bbox,
                                                   show_gantt=gantt)

        else:
            plotter = PlotSklearnResultsMultiProcess(config_path=self.config_path,
                                                     video_setting=self.create_videos_var.get(),
                                                     rotate=self.rotate_img_var.get(),
                                                     video_paths=video_path,
                                                     frame_setting=self.create_frames_var.get(),
                                                     print_timers=self.include_timers_var.get(),
                                                     core_cnt=core_cnt,
                                                     font_size=font_size,
                                                     space_size=space_size,
                                                     text_thickness=text_thickness,
                                                     circle_size=circle_size,
                                                     show_pose=show_pose,
                                                     animal_names=show_animal_names,
                                                     text_opacity=text_opacity,
                                                     text_clr=text_clr,
                                                     text_bg_clr=text_bg_clr,
                                                     gpu=gpu,
                                                     pose_palette=pose_palette,
                                                     show_bbox=bbox,
                                                     show_gantt=gantt)

        plotter.run()


#_ = SklearnVisualizationPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
