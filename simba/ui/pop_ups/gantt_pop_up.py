__author__ = "Simon Nilsson"

import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.gantt_creator import GanttCreatorSingleProcess
from simba.plotting.gantt_creator_mp import GanttCreatorMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, SimbaButton, SimbaCheckbox,
                                        SimBADropDown)
from simba.utils.checks import check_if_filepath_list_is_empty, check_int
from simba.utils.enums import Formats, Keys, Links, Paths
from simba.utils.read_write import get_file_name_info_in_directory


class GanttPlotPopUp(PopUpMixin, ConfigReader):

    """
    :example:
    >>>  _ = GanttPlotPopUp(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")
    """
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.data_path = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)
        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)
        check_if_filepath_list_is_empty(filepaths=list(self.files_found_dict.keys()),error_msg="SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. Create classification results before visualizing gantt charts",)
        PopUpMixin.__init__(self, config_path=config_path, title="VISUALIZE GANTT PLOTS", icon='gantt_small')
        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name='settings', icon_link=Links.GANTT_PLOTS.value)
        self.resolution_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.resolutions, label='GANTT PLOT RESOLUTION: ', label_width=30, dropdown_width=30, value='640×480')
        self.font_size_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=list(range(1, 26)), label='FONT SIZE: ', label_width=30, dropdown_width=30, value=8)
        self.font_rotation_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=list(range(1, 181, 2)), label='FONT ROTATION (°): ', label_width=30, dropdown_width=30, value=45)
        self.core_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=list(range(1, self.cpu_cnt+1)), label='CPU CORES: ', label_width=30, dropdown_width=30, value=int(self.cpu_cnt/2))

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZATION SETTINGS", icon_name='eye')
        gantt_frames_cb, self.gantt_frames_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE FRAMES', txt_img='frames')
        gantt_videos_cb, self.gantt_videos_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE VIDEOS', txt_img='video')
        gantt_last_frame_cb, self.gantt_last_frame_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE LAST FRAME', txt_img='finish')


        self.run_frm = LabelFrame( self.main_frm, text="RUN", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black",)
        self.run_single_video_frm = LabelFrame(self.run_frm,text="SINGLE VIDEO",font=Formats.FONT_HEADER.value,pady=5,padx=5,fg="black",)
        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, txt="Create single video", txt_clr="blue", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.__create_gantt_plots, cmd_kwargs={'multiple_videos': False})
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm,"Video:",list(self.files_found_dict.keys()),"12",)
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])

        self.run_multiple_videos = LabelFrame(self.run_frm, text="MULTIPLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black",)
        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, txt="Create multiple videos ({} video(s) found)".format(str(len(list(self.files_found_dict.keys())))), txt_clr="blue", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.__create_gantt_plots, cmd_kwargs={'multiple_videos': True})
        self.style_settings_frm.grid(row=0, sticky=NW)
        self.resolution_dropdown.grid(row=0, sticky=NW)
        self.font_size_dropdown.grid(row=1, sticky=NW)
        self.font_rotation_dropdown.grid(row=2, sticky=NW)
        self.core_dropdown.grid(row=3, sticky=NW)

        self.settings_frm.grid(row=1, sticky=NW)
        gantt_videos_cb.grid(row=0, sticky=NW)
        gantt_frames_cb.grid(row=1, sticky=W)
        gantt_last_frame_cb.grid(row=2, sticky=NW)

        self.run_frm.grid(row=2, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        self.main_frm.mainloop()

    def __create_gantt_plots(self, multiple_videos: bool):
        width = int(self.resolution_dropdown.getChoices().split("×")[0])
        height = int(self.resolution_dropdown.getChoices().split("×")[1])
        font_size = int(self.font_size_dropdown.get_value())
        font_rotation = int(self.font_rotation_dropdown.get_value())
        core_cnt = int(self.core_dropdown.get_value())

        style_attr = {"width": width,
                      "height": height,
                      "font size": font_size,
                      "font rotation": font_rotation}

        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        if core_cnt > 1:
            gantt_creator = GanttCreatorMultiprocess(config_path=self.config_path,
                                                     frame_setting=self.gantt_frames_var.get(),
                                                     video_setting=self.gantt_videos_var.get(),
                                                     last_frm_setting=self.gantt_last_frame_var.get(),
                                                     data_paths=data_paths,
                                                     cores=core_cnt,
                                                     style_attr=style_attr)
        else:
            gantt_creator = GanttCreatorSingleProcess(config_path=self.config_path,
                                                      frame_setting=self.gantt_frames_var.get(),
                                                      video_setting=self.gantt_videos_var.get(),
                                                      last_frm_setting=self.gantt_last_frame_var.get(),
                                                      data_paths=data_paths,
                                                      style_attr=style_attr)
        gantt_creator.run()



#_ = GanttPlotPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")

 # _ = GanttPlotPopUp(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")
