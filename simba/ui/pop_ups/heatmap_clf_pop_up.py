__author__ = "Simon Nilsson"

import multiprocessing
import os
from tkinter import *
from typing import Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.heat_mapper_clf import HeatMapperClfSingleCore
from simba.plotting.heat_mapper_clf_mp import HeatMapperClfMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        SimbaButton, SimbaCheckbox,
                                        SimBADropDown)
from simba.utils.checks import check_if_filepath_list_is_empty
from simba.utils.enums import Formats, Keys, Links, Paths
from simba.utils.read_write import get_file_name_info_in_directory

AUTO = "auto"

class HeatmapClfPopUp(PopUpMixin, ConfigReader):
    """
    :example:
    >>> _ = HeatmapClfPopUp(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")
    """

    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.data_path = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)

        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)
        check_if_filepath_list_is_empty( filepaths=list(self.files_found_dict.keys()), error_msg="SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. ",)
        PopUpMixin.__init__(self, title="CREATE CLASSIFICATION HEATMAP PLOTS", icon='heatmap')
        max_scales_option = list(np.linspace(5, 600, 5).astype(np.int32))
        max_scales_option.insert(0, AUTO)


        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name='style', icon_link=Links.HEATMAP_CLF.value)

        self.palette_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.palette_options, label='PALETTE: ', label_width=30, dropdown_width=35, value=self.palette_options[0])
        self.shading_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.shading_options, label='SHADING: ', label_width=30, dropdown_width=35, value=self.shading_options[0])
        self.clf_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.clf_names, label='CLASSIFIER: ', label_width=30, dropdown_width=35, value=self.clf_names[0])
        self.bp_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.body_parts_lst, label='BODY-PART: ', label_width=30, dropdown_width=35, value=self.body_parts_lst[0])
        self.max_time_scale_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=max_scales_option, label='MAX TIME SCALE (S): ', label_width=30, dropdown_width=35, value=AUTO)
        self.bin_size_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.heatmap_bin_size_options, label='BIN SIZE (MM): ', label_width=30, dropdown_width=35, value="80×80")

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZATION SETTINGS", icon_name='eye', icon_link=Links.HEATMAP_CLF.value)
        self.core_cnt_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(1, self.cpu_cnt)), label='CPU CORE COUNT: ', label_width=30, dropdown_width=35, value=int(self.cpu_cnt/2))

        heatmap_frames_cb, self.heatmap_frames_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE FRAMES', txt_img='frames')
        heatmap_videos_cb, self.heatmap_videos_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE VIDEOS', txt_img='video')
        heatmap_last_frm_cb, self.heatmap_last_frm_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE LAST FRAME', txt_img='finish', val=True)

        self.run_frm = LabelFrame(self.main_frm, text="RUN", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black",)
        self.run_single_video_frm = LabelFrame(self.run_frm, text="SINGLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black",)

        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, txt="Create single video", txt_clr="blue", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.__create_heatmap_plots, cmd_kwargs={'multiple_videos': False})
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm, "Video:", list(self.files_found_dict.keys()), "12",)
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])
        self.run_multiple_videos = LabelFrame(self.run_frm, text="MULTIPLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black",)

        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, txt="Create multiple videos ({} video(s) found)".format(str(len(list(self.files_found_dict.keys())))), img='rocket', txt_clr="blue", font=Formats.FONT_REGULAR.value, cmd=self.__create_heatmap_plots, cmd_kwargs={'multiple_videos': True})
        self.style_settings_frm.grid(row=0, sticky=NW)
        self.palette_dropdown.grid(row=0, sticky=NW)
        self.shading_dropdown.grid(row=1, sticky=NW)
        self.clf_dropdown.grid(row=2, sticky=NW)
        self.bp_dropdown.grid(row=3, sticky=NW)
        self.max_time_scale_dropdown.grid(row=4, sticky=NW)
        self.bin_size_dropdown.grid(row=5, sticky=NW)

        self.settings_frm.grid(row=1, sticky=NW)
        self.core_cnt_dropdown.grid(row=0, sticky=NW)
        heatmap_frames_cb.grid(row=1, sticky=NW)
        heatmap_videos_cb.grid(row=2, sticky=NW)
        heatmap_last_frm_cb.grid(row=3, sticky=NW)

        self.run_frm.grid(row=2, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        self.main_frm.mainloop()

    def __create_heatmap_plots(self, multiple_videos: bool):
        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        max_scale = int(self.max_time_scale_dropdown.getChoices().split("×")[0]) if self.max_time_scale_dropdown.getChoices() != AUTO else AUTO
        bin_size = int(self.bin_size_dropdown.getChoices().split("×")[0])
        core_cnt = int(self.core_cnt_dropdown.get_value())

        style_attr = {"palette": self.palette_dropdown.getChoices(),
                      "shading": self.shading_dropdown.getChoices(),
                      "max_scale": max_scale,
                      "bin_size": bin_size}

        if core_cnt == 1:
            heatmapper_clf = HeatMapperClfSingleCore(
                config_path=self.config_path,
                style_attr=style_attr,
                final_img_setting=self.heatmap_last_frm_var.get(),
                video_setting=self.heatmap_videos_var.get(),
                frame_setting=self.heatmap_frames_var.get(),
                bodypart=self.bp_dropdown.getChoices(),
                data_paths=data_paths,
                clf_name=self.clf_dropdown.getChoices(),
            )

            heatmapper_clf_processor = multiprocessing.Process(heatmapper_clf.run())
            heatmapper_clf_processor.start()

        else:
            heatmapper_clf = HeatMapperClfMultiprocess(
                config_path=self.config_path,
                style_attr=style_attr,
                final_img_setting=self.heatmap_last_frm_var.get(),
                video_setting=self.heatmap_videos_var.get(),
                frame_setting=self.heatmap_frames_var.get(),
                bodypart=self.bp_dropdown.getChoices(),
                data_paths=data_paths,
                clf_name=self.clf_dropdown.getChoices(),
                core_cnt=core_cnt,
            )

            heatmapper_clf.run()



#_ = HeatmapClfPopUp(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")