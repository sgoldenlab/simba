__author__ = "Simon Nilsson"

import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.probability_plot_creator import \
    TresholdPlotCreatorSingleProcess
from simba.plotting.probability_plot_creator_mp import \
    TresholdPlotCreatorMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, SimbaButton,
                                        SimbaCheckbox, SimBADropDown)
from simba.utils.enums import Formats, Links
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import (check_if_filepath_list_is_empty,
                                    get_file_name_info_in_directory)

STYLE_WIDTH = 'width'
STYLE_HEIGHT = 'height'
STYLE_FONT_SIZE = 'font size'
STYLE_LINE_WIDTH = 'line width'
STYLE_YMAX = 'y_max'
STYLE_COLOR = 'color'
STYLE_OPACITY = 'opacity'
AUTO = 'AUTO'

MAX_Y_OPTIONS = list(range(100, 0, -10))
MAX_Y_OPTIONS.insert(0, AUTO)
OPACITY_OPTIONS = [round(x * 0.1, 1) for x in range(1, 11)]

class VisualizeClassificationProbabilityPopUp(PopUpMixin, ConfigReader):

    """
    :example:
    >>> _ = VisualizeClassificationProbabilityPopUp(config_path=r'C:\troubleshooting\RAT_NOR\project_folder\project_config.ini')
    """


    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        color_names = list(get_color_dict().keys())

        self.files_found_dict = get_file_name_info_in_directory(directory=self.machine_results_dir, file_type=self.file_type)
        check_if_filepath_list_is_empty(filepaths=list(self.files_found_dict.keys()), error_msg=f"Cannot visualize probabilities, no data in {self.machine_results_dir} directory")
        max_file_name_len = max(len(k) for k in self.files_found_dict) + 5
        PopUpMixin.__init__(self, title="CREATE CLASSIFICATION PROBABILITY PLOTS", icon='probability')

        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name='style', icon_link=Links.VISUALIZE_CLF_PROBABILITIES.value, pady=5, padx=5, relief='solid')
        self.resolution_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.resolutions, label='RESOLUTION: ', label_width=25, dropdown_width=35, value=self.resolutions[1])
        self.max_y_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=MAX_Y_OPTIONS, label='MAX Y-AXIS: ', label_width=25, dropdown_width=35, value=AUTO)
        self.line_clr_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=color_names, label='LINE COLOR: ', label_width=25, dropdown_width=35, value='Red')
        self.font_size_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=list(range(1, 26)), label='TEXT SIZE: ', label_width=25, dropdown_width=35, value=10)
        self.line_width_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=list(range(1, 26)), label='LINE WIDTH: ', label_width=25, dropdown_width=35, value=6)
        self.line_opacity_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=OPACITY_OPTIONS, label='LINE OPACITY: ', label_width=25, dropdown_width=35, value=1.0)


        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZATION SETTINGS", icon_name='eye', icon_link=Links.VISUALIZE_CLF_PROBABILITIES.value, pady=5, padx=5, relief='solid')
        self.clf_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=self.clf_names, label='CLASSIFIER: ', label_width=25, dropdown_width=35, value=self.clf_names[0])
        self.core_cnt_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(1, self.cpu_cnt+1)), label='CPU CORE COUNT: ', label_width=25, dropdown_width=35, value=int(self.cpu_cnt/2))

        probability_frames_cb, self.probability_frames_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE FRAMES', font=Formats.FONT_REGULAR.value, txt_img='frames', val=False)
        probability_videos_cb, self.probability_videos_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE VIDEOS', font=Formats.FONT_REGULAR.value, txt_img='video', val=False)
        probability_last_frm_cb, self.probability_last_frm_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE FINAL FRAME', font=Formats.FONT_REGULAR.value, txt_img='finish', val=True)


        self.run_single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name='video', icon_link=Links.VISUALIZE_CLF_PROBABILITIES.value, pady=5, padx=5, relief='solid')
        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, txt="CREATE SINGLE VIDEO", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.__run, cmd_kwargs={'multiple': False})
        self.single_video_dropdown = SimBADropDown(parent=self.run_single_video_frm, dropdown_options=list(self.files_found_dict.keys()), label='VIDEO: ', label_width=25, dropdown_width=max_file_name_len, value=list(self.files_found_dict.keys())[0])

        self.run_multiple_videos = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEO", icon_name='video', icon_link=Links.VISUALIZE_CLF_PROBABILITIES.value, pady=5, padx=5, relief='solid')
        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, txt=f"Create multiple videos ({len(list(self.files_found_dict.keys()))} video(s) found)", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.__run, cmd_kwargs={'multiple': True})

        self.style_settings_frm.grid(row=0, sticky=NW, padx=10, pady=10)
        self.resolution_dropdown.grid(row=0, sticky=NW)
        self.line_clr_dropdown.grid(row=1, sticky=NW)
        self.font_size_dropdown.grid(row=2, sticky=NW)
        self.line_width_dropdown.grid(row=3, sticky=NW)
        self.max_y_dropdown.grid(row=4, sticky=NW)
        self.line_opacity_dropdown.grid(row=5, sticky=NW)

        self.settings_frm.grid(row=1, sticky=NW, padx=10, pady=10)
        self.clf_dropdown.grid(row=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=1, sticky=NW)
        probability_frames_cb.grid(row=2, sticky=NW)
        probability_videos_cb.grid(row=3, sticky=NW)
        probability_last_frm_cb.grid(row=4, sticky=NW)


        self.run_single_video_frm.grid(row=2, sticky=NW, padx=10, pady=10)
        self.run_single_video_btn.grid(row=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=3, sticky=NW, padx=10, pady=10)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        self.main_frm.mainloop()

    def __run(self, multiple: bool):
        width = int(self.resolution_dropdown.getChoices().split("×")[0])
        height = int(self.resolution_dropdown.getChoices().split("×")[1])
        font_size = int(self.font_size_dropdown.get_value())
        line_width = int(self.line_width_dropdown.get_value())
        core_cnt = int(self.core_cnt_dropdown.get_value())
        opacity = float(self.line_opacity_dropdown.get_value())


        style_attr = {
            STYLE_WIDTH: width,
            STYLE_HEIGHT: height,
            STYLE_FONT_SIZE: font_size,
            STYLE_LINE_WIDTH: line_width,
            STYLE_COLOR: self.line_clr_dropdown.getChoices(),
            STYLE_YMAX: self.max_y_dropdown.getChoices(),
            STYLE_OPACITY: opacity}

        if multiple:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        if core_cnt == 1:
            probability_plot_creator = TresholdPlotCreatorSingleProcess(config_path=self.config_path,
                                                                        frame_setting=self.probability_frames_var.get(),
                                                                        video_setting=self.probability_videos_var.get(),
                                                                        last_image=self.probability_last_frm_var.get(),
                                                                        files_found=data_paths,
                                                                        clf_name=self.clf_dropdown.getChoices(),
                                                                        style_attr=style_attr)
        else:
            probability_plot_creator = TresholdPlotCreatorMultiprocess(config_path=self.config_path,
                                                                       frame_setting=self.probability_frames_var.get(),
                                                                       video_setting=self.probability_videos_var.get(),
                                                                       last_frame=self.probability_last_frm_var.get(),
                                                                       files_found=data_paths,
                                                                       clf_name=self.clf_dropdown.getChoices(),
                                                                       cores=core_cnt,
                                                                       style_attr=style_attr)

        probability_plot_creator.run()

        #threading.Thread(target=probability_plot_creator.run).start()




#_ = VisualizeClassificationProbabilityPopUp(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")
#_ = VisualizeClassificationProbabilityPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
