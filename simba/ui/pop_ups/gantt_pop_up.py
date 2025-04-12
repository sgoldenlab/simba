__author__ = "Simon Nilsson"

import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.gantt_creator import GanttCreatorSingleProcess
from simba.plotting.gantt_creator_mp import GanttCreatorMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        SimbaButton, SimbaCheckbox,
                                        SimBADropDown)
from simba.utils.checks import check_if_filepath_list_is_empty, check_int
from simba.utils.enums import Formats, Keys, Links, Options, Paths
from simba.utils.errors import NoSpecifiedOutputError
from simba.utils.read_write import find_files_of_filetypes_in_directory


class GanttPlotPopUp(PopUpMixin, ConfigReader):

    """
    :example:
    >>>  _ = GanttPlotPopUp(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")
    """
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        check_if_filepath_list_is_empty(filepaths=self.machine_results_paths,error_msg=f"SIMBA ERROR: Zero files found in the {self.machine_results_dir} directory. Create classification results before visualizing gantt charts",)
        palettes = Options.PALETTE_OPTIONS_CATEGORICAL.value + Options.PALETTE_OPTIONS.value
        self.data_paths = find_files_of_filetypes_in_directory(directory=self.machine_results_dir, extensions=[f'.{self.file_type}'], as_dict=True)
        max_file_name_len = max(len(k) for k in self.data_paths) + 5

        PopUpMixin.__init__(self, config_path=config_path, title="VISUALIZE GANTT PLOTS", icon='gantt_small')
        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name='settings', icon_link=Links.GANTT_PLOTS.value, relief='solid', padx=5, pady=5)
        self.resolution_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.resolutions, label='GANTT PLOT RESOLUTION: ', label_width=30, dropdown_width=30, value='640×480')
        self.font_size_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=list(range(1, 26)), label='TEXT SIZE: ', label_width=30, dropdown_width=30, value=8)
        self.font_rotation_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=list(range(0, 182, 2)), label='TEXT ROTATION (°): ', label_width=30, dropdown_width=30, value=0)
        self.palette_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=palettes, label='COLOR PALETTE: ', label_width=30, dropdown_width=30, value='Set1')
        self.core_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=list(range(1, self.cpu_cnt+1)), label='CPU CORES: ', label_width=30, dropdown_width=30, value=int(self.cpu_cnt/2))

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZATION SETTINGS", icon_name='eye', icon_link=Links.GANTT_PLOTS.value, relief='solid', padx=5, pady=5)
        gantt_frames_cb, self.gantt_frames_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE FRAMES', txt_img='frames', val=False)
        gantt_videos_cb, self.gantt_videos_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE VIDEOS', txt_img='video', val=False)
        gantt_last_frame_cb, self.gantt_last_frame_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE LAST FRAME', txt_img='finish', val=True)

        self.run_single_video_frm= CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name='eye', icon_link=Links.GANTT_PLOTS.value, relief='solid', padx=5, pady=5)
        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, txt="VIDEO", txt_clr="blue", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.__create_gantt_plots, cmd_kwargs={'multiple': False})
        self.single_video_dropdown = SimBADropDown(parent=self.run_single_video_frm, dropdown_options=list(self.data_paths.keys()), label='VIDEO', label_width=20, dropdown_width=max_file_name_len, value=list(self.data_paths.keys())[0])

        self.run_multiple_videos = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEO(S)", icon_name='stack', icon_link=Links.GANTT_PLOTS.value, relief='solid', padx=5, pady=5)
        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, txt=f"Create multiple videos ({len(list(self.data_paths.keys()))} video(s) found)", txt_clr="blue", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.__create_gantt_plots, cmd_kwargs={'multiple': True})
        self.style_settings_frm.grid(row=0, sticky=NW, padx=10, pady=10)
        self.resolution_dropdown.grid(row=0, sticky=NW)
        self.font_size_dropdown.grid(row=1, sticky=NW)
        self.font_rotation_dropdown.grid(row=2, sticky=NW)
        self.palette_dropdown.grid(row=3, sticky=NW)
        self.core_dropdown.grid(row=4, sticky=NW)

        self.settings_frm.grid(row=1, sticky=NW, padx=10, pady=10)
        gantt_videos_cb.grid(row=0, sticky=NW)
        gantt_frames_cb.grid(row=1, sticky=W)
        gantt_last_frame_cb.grid(row=2, sticky=NW)

        self.run_single_video_frm.grid(row=2, sticky=NW)
        self.run_single_video_btn.grid(row=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=3, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        self.main_frm.mainloop()

    def __create_gantt_plots(self,
                             multiple: bool):

        width = int(self.resolution_dropdown.getChoices().split("×")[0])
        height = int(self.resolution_dropdown.getChoices().split("×")[1])
        font_size = int(self.font_size_dropdown.get_value())
        font_rotation = int(self.font_rotation_dropdown.get_value())
        core_cnt = int(self.core_dropdown.get_value())
        frame_setting = self.gantt_frames_var.get()
        video_setting = self.gantt_videos_var.get()
        last_frm_setting = self.gantt_last_frame_var.get()
        palette = self.palette_dropdown.get_value()

        if not frame_setting and not video_setting and not last_frm_setting:
            raise NoSpecifiedOutputError(msg="SIMBA ERROR: Please select gantt videos, frames, and/or last frame.")

        if multiple:
            data_paths = list(self.data_paths.values())
        else:
            data_paths = [self.data_paths[self.single_video_dropdown.getChoices()]]

        if core_cnt > 1:
            gantt_creator = GanttCreatorMultiprocess(config_path=self.config_path,
                                                     frame_setting=frame_setting,
                                                     video_setting=video_setting,
                                                     last_frm_setting=last_frm_setting,
                                                     data_paths=data_paths,
                                                     width=width,
                                                     height=height,
                                                     font_size=font_size,
                                                     font_rotation=font_rotation,
                                                     core_cnt=core_cnt,
                                                     palette=palette)

        else:
            gantt_creator = GanttCreatorSingleProcess(config_path=self.config_path,
                                                      frame_setting=frame_setting,
                                                      video_setting=video_setting,
                                                      last_frm_setting=last_frm_setting,
                                                      data_paths=data_paths,
                                                      width=width,
                                                      height=height,
                                                      font_size=font_size,
                                                      font_rotation=font_rotation,
                                                      palette=palette)
        gantt_creator.run()



#_ = GanttPlotPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")

 # _ = GanttPlotPopUp(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")
