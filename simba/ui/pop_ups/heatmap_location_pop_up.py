__author__ = "Simon Nilsson"

import multiprocessing
import os
from tkinter import *
from typing import Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.heat_mapper_location import HeatmapperLocationSingleCore
from simba.plotting.heat_mapper_location_mp import \
    HeatMapperLocationMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        SimbaButton, SimbaCheckbox,
                                        SimBADropDown)
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.enums import Formats, Keys, Links, Paths
from simba.utils.read_write import get_file_name_info_in_directory


class HeatmapLocationPopup(PopUpMixin, ConfigReader):

    """
    :example:
    >>> _ = HeatmapLocationPopup(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")
    >>> _ = HeatmapLocationPopup(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")

    """
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.data_path = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)
        check_if_filepath_list_is_empty(filepaths=list(self.files_found_dict.keys()),error_msg="SIMBA ERROR: Zero files found in the project_folder/csv/outlier_corrected_movement_location directory. ",)
        PopUpMixin.__init__(self, title="HEATMAPS: LOCATION", icon='heatmap')
        max_scales = list(np.arange(5, 105, 5))
        max_scales.insert(0, "Auto-compute")
        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.HEATMAP_LOCATION.value)

        self.palette_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.palette_options, label='PALETTE: ', label_width=25, dropdown_width=30, value=self.palette_options[0])
        self.shading_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.shading_options, label='SHADING: ', label_width=25, dropdown_width=30, value=self.shading_options[0])
        self.bp_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.body_parts_lst, label='BODY-PART: ', label_width=25, dropdown_width=30, value=self.body_parts_lst[0])
        self.max_time_scale_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=max_scales, label='MAX TIME SCALE (S): ', label_width=25, dropdown_width=30, value=max_scales[0])
        self.bin_size_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.heatmap_bin_size_options, label='BIN SIZE (MM): ', label_width=25, dropdown_width=30, value="80×80")

        self.settings_frm = LabelFrame(self.main_frm, text="VISUALIZATION SETTINGS", font=Formats.FONT_HEADER.value, pady=5, padx=5)
        self.multiprocess_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(1, self.cpu_cnt)), label='CPU CORES: ', label_width=25, dropdown_width=30, value=int(self.cpu_cnt/2))
        heatmap_frames_cb, self.heatmap_frames_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE FRAMES', txt_img='frames')
        heatmap_videos_cb, self.heatmap_videos_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE VIDEOS', txt_img='video')
        heatmap_last_frm_cb, self.heatmap_last_frm_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE LAST FRAME', txt_img='finish')

        self.run_frm = LabelFrame(self.main_frm, text="RUN", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black")
        self.run_single_video_frm = LabelFrame(self.run_frm, text="SINGLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black")


        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, txt="Create single video", txt_clr="blue", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.__create_heatmap_plots, cmd_kwargs={'multiple_videos': False})
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm, "Video:", list(self.files_found_dict.keys()), "12")
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])
        self.run_multiple_videos = LabelFrame(self.run_frm, text="MULTIPLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black")

        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, txt="Create multiple videos ({} video(s) found)".format(str(len(list(self.files_found_dict.keys())))), img='rocket', txt_clr="blue", font=Formats.FONT_REGULAR.value, cmd=self.__create_heatmap_plots, cmd_kwargs={'multiple_videos': True})
        self.style_settings_frm.grid(row=0, sticky=NW)
        self.palette_dropdown.grid(row=0, sticky=NW)
        self.shading_dropdown.grid(row=1, sticky=NW)
        self.bp_dropdown.grid(row=2, sticky=NW)
        self.max_time_scale_dropdown.grid(row=3, sticky=NW)
        self.bin_size_dropdown.grid(row=4, sticky=NW)

        self.settings_frm.grid(row=1, sticky=NW)
        heatmap_frames_cb.grid(row=0, sticky=NW)
        heatmap_videos_cb.grid(row=1, sticky=NW)
        heatmap_last_frm_cb.grid(row=2, sticky=NW)
        self.multiprocess_dropdown.grid(row=3, column=0, sticky=NW)

        self.run_frm.grid(row=2, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)
        #self.main_frm.mainloop()

    def __create_heatmap_plots(self, multiple_videos: bool):
        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        if self.max_time_scale_dropdown.getChoices() != "Auto-compute":
            max_scale = int(float(self.max_time_scale_dropdown.getChoices()))
        else:
            max_scale = "auto"

        bin_size = int(self.bin_size_dropdown.getChoices().split("×")[0])

        style_attr = {"palette": self.palette_dropdown.getChoices(),
                      "shading": self.shading_dropdown.getChoices(),
                      "max_scale": max_scale,
                      "bin_size": bin_size}

        if int(self.multiprocess_dropdown.get_value()) == 1:
            heatmapper_clf = HeatmapperLocationSingleCore(config_path=self.config_path,
                                                          style_attr=style_attr,
                                                          final_img_setting=self.heatmap_last_frm_var.get(),
                                                          video_setting=self.heatmap_videos_var.get(),
                                                          frame_setting=self.heatmap_frames_var.get(),
                                                          bodypart=self.bp_dropdown.getChoices(),
                                                          data_paths=data_paths)

            heatmapper_clf.run()

        else:
            heatmapper_clf = HeatMapperLocationMultiprocess(config_path=self.config_path,
                                                            style_attr=style_attr,
                                                            final_img_setting=self.heatmap_last_frm_var.get(),
                                                            video_setting=self.heatmap_videos_var.get(),
                                                            frame_setting=self.heatmap_frames_var.get(),
                                                            bodypart=self.bp_dropdown.getChoices(),
                                                            data_paths=data_paths,
                                                            core_cnt=int(self.multiprocess_dropdown.getChoices()))

            heatmapper_clf.run()
