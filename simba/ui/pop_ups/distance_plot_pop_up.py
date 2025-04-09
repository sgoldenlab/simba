__author__ = "Simon Nilsson"

import os
import threading
from collections import defaultdict
from tkinter import *
from typing import Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.distance_plotter import DistancePlotterSingleCore
from simba.plotting.distance_plotter_mp import DistancePlotterMultiCore
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, SimbaButton, SimbaCheckbox,
                                        SimBADropDown)
from simba.utils.checks import check_if_filepath_list_is_empty, check_int
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import DuplicationError
from simba.utils.read_write import get_file_name_info_in_directory

AUTO = 'AUTO'

class DistancePlotterPopUp(PopUpMixin, ConfigReader):

    """
    :example:
    >>> _ = DistancePlotterPopUp(config_path=r'C:\troubleshooting\RAT_NOR\project_folder\project_config.ini')

    """
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.max_y_lst = list(range(10, 1010, 10))
        self.max_y_lst.insert(0, AUTO)
        font_size_options = list(range(1, 33))
        opacity_options = list(np.round(np.arange(0.0, 1.1, 0.1), 1))
        self.number_of_distances_options = list(range(1, len(self.body_parts_lst) * 2))

        self.files_found_dict = get_file_name_info_in_directory(directory=self.outlier_corrected_dir, file_type=self.file_type)
        check_if_filepath_list_is_empty(filepaths=list(self.files_found_dict.keys()), error_msg="SIMBA ERROR: Zero files found in the project_folder/csv/outlier_corrected_movement_location directory. ",)

        PopUpMixin.__init__(self, title="CREATE DISTANCE PLOTS", icon='distance')

        self.style_settings_frm = CreateLabelFrameWithIcon( parent=self.main_frm, header="STYLE SETTINGS", icon_name='style', icon_link=Links.DISTANCE_PLOTS.value,)
        self.resolution_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.resolutions, label='RESOLUTION: ', label_width=25, dropdown_width=20, value=self.resolutions[1])
        self.font_size_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=font_size_options, label='FONT SIZE: ', label_width=25, dropdown_width=20, value=8)
        self.line_width_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=font_size_options, label='LINE WIDTH: ', label_width=25, dropdown_width=20, value=6)
        self.opacity_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=opacity_options, label='LINE OPACITY: ', label_width=25, dropdown_width=20, value=0.5)
        self.max_y_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.max_y_lst, label='MAX Y-AXIS: ', label_width=25, dropdown_width=20, value=AUTO)

        self.distances_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHOOSE DISTANCES", icon_name='distance', icon_link=Links.DISTANCE_PLOTS.value)
        self.number_of_distances_dropdown = SimBADropDown(parent=self.distances_frm, dropdown_options=self.number_of_distances_options, label='# DISTANCES: ', label_width=25, dropdown_width=20, value=1, command=self.__populate_distances_menu)
        self.__populate_distances_menu(1)

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZATION SETTING", icon_name='eye', icon_link=Links.DISTANCE_PLOTS.value)

        distance_frames_cb, self.distance_frames_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE FRAMES', txt_img='frames', val=False)
        distance_videos_cb, self.distance_videos_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE VIDEOS', txt_img='video', val=False)
        distance_final_img_cb, self.distance_final_img_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE LAST FRAME', txt_img='finish', val=True)
        self.multiprocess_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(1, self.cpu_cnt)), label='CPU CORE COUNT: ', label_width=25, dropdown_width=20, value=int(self.cpu_cnt/2))

        self.run_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="RUN", icon_name='run', icon_link=Links.DISTANCE_PLOTS.value)
        self.run_single_video_frm = LabelFrame(self.run_frm, text="SINGLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black")

        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, txt="Create single video", txt_clr="blue", font=Formats.FONT_REGULAR.value, cmd=self.__create_distance_plots, cmd_kwargs={'multiple_videos': False})
        self.single_video_dropdown = DropDownMenu( self.run_single_video_frm, "Video:", list(self.files_found_dict.keys()), "12")
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])

        self.run_multiple_videos = LabelFrame(self.run_frm, text="MULTIPLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black",)
        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, txt="Create multiple videos ({} video(s) found)".format(str(len(list(self.files_found_dict.keys())))), txt_clr="blue", font=Formats.FONT_REGULAR.value, cmd=self.__create_distance_plots, cmd_kwargs={'multiple_videos': True})

        self.style_settings_frm.grid(row=0, sticky=NW)
        self.resolution_dropdown.grid(row=0, sticky=NW)
        self.font_size_dropdown.grid(row=1, sticky=NW)
        self.line_width_dropdown.grid(row=2, sticky=NW)
        self.opacity_dropdown.grid(row=3, sticky=NW)
        self.max_y_dropdown.grid(row=4, sticky=NW)

        self.distances_frm.grid(row=1, sticky=NW)
        self.number_of_distances_dropdown.grid(row=0, sticky=NW)

        self.settings_frm.grid(row=2, sticky=NW)
        self.multiprocess_dropdown.grid(row=0, column=0, sticky=NW)
        distance_frames_cb.grid(row=1, sticky=NW)
        distance_videos_cb.grid(row=2, sticky=NW)
        distance_final_img_cb.grid(row=3, sticky=NW)

        self.run_frm.grid(row=3, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        self.main_frm.mainloop()

    def __populate_distances_menu(self, choice):
        if hasattr(self, "bp_1"):
            for k, v in self.bp_1.items():
                self.bp_1[k].destroy()
                self.bp_2[k].destroy()
                self.distance_clrs[k].destroy()

        self.bp_1, self.bp_2, self.distance_clrs = {}, {}, {}
        for distance_cnt in range(int(self.number_of_distances_dropdown.getChoices())):
            self.bp_1[distance_cnt] = SimBADropDown(parent=self.distances_frm, dropdown_options=self.body_parts_lst, label=f'DISTANCE {distance_cnt + 1}: ', label_width=25, dropdown_width=20, value=self.body_parts_lst[distance_cnt])
            self.bp_1[distance_cnt].grid(row=distance_cnt + 1, column=0, sticky=NW)
            self.bp_2[distance_cnt] = SimBADropDown(parent=self.distances_frm, dropdown_options=self.body_parts_lst, label='', label_width=0, dropdown_width=20, value=self.body_parts_lst[distance_cnt])
            self.bp_2[distance_cnt].grid(row=distance_cnt + 1, column=1, sticky=NW)
            self.distance_clrs[distance_cnt] = SimBADropDown(parent=self.distances_frm, dropdown_options=list(self.colors_dict.keys()), label='', label_width=0, dropdown_width=20, value=list(self.colors_dict.keys())[distance_cnt])
            self.distance_clrs[distance_cnt].grid(row=distance_cnt + 1, column=3, sticky=NW)

    def __create_distance_plots(self, multiple_videos: bool):
        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        line_attr = defaultdict(list)
        for attr in (self.bp_1, self.bp_2, self.distance_clrs):
            for key, value in attr.items():
                line_attr[key].append(value.getChoices())
        line_attr = list(line_attr.values())

        for cnt, v in enumerate(line_attr):
            if v[0] == v[1]:
                raise DuplicationError(msg=f"DISTANCE LINE {cnt+1} ERROR: The two body-parts cannot be the same body-part.",source=self.__class__.__name__,)

        width = int(self.resolution_dropdown.getChoices().split("×")[0])
        height = int(self.resolution_dropdown.getChoices().split("×")[1])
        font_size = int(self.font_size_dropdown.get_value())
        line_width = int(self.line_width_dropdown.get_value())
        opacity = float(self.opacity_dropdown.getChoices())
        core_cnt = int(self.multiprocess_dropdown.get_value())
        max_y = -1 if self.max_y_dropdown.getChoices() == AUTO else int(self.max_y_dropdown.getChoices())

        style_attr = {"width": width,
                      "height": height,
                      "line width": line_width,
                      "font size": font_size,
                      "opacity": opacity,
                      "y_max": max_y}

        if core_cnt == 1:
            distance_plotter = DistancePlotterSingleCore(config_path=self.config_path,
                                                         frame_setting=self.distance_frames_var.get(),
                                                         video_setting=self.distance_videos_var.get(),
                                                         final_img=self.distance_final_img_var.get(),
                                                         style_attr=style_attr,
                                                         data_paths=data_paths,
                                                         line_attr=line_attr)
        else:
            distance_plotter = DistancePlotterMultiCore(config_path=self.config_path,
                                                        frame_setting=self.distance_frames_var.get(),
                                                        video_setting=self.distance_videos_var.get(),
                                                        final_img=self.distance_final_img_var.get(),
                                                        style_attr=style_attr,
                                                        data_paths=data_paths,
                                                        line_attr=line_attr,
                                                        core_cnt=core_cnt)

        threading.Thread(distance_plotter.run()).start()



#_ = DistancePlotterPopUp(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")


# _ = DistancePlotterPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# _ = DistancePlotterPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini')
