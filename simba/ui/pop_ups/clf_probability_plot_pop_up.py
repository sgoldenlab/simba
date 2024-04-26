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
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box)
from simba.utils.checks import check_int
from simba.utils.enums import Formats, Keys, Links, Paths
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import (check_if_filepath_list_is_empty,
                                    get_file_name_info_in_directory)

STYLE_WIDTH = 'width'
STYLE_HEIGHT = 'height'
STYLE_FONT_SIZE = 'font size'
STYLE_LINE_WIDTH = 'line width'
STYLE_YMAX = 'y_max'
STYLE_COLOR = 'color'

class VisualizeClassificationProbabilityPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="CREATE CLASSIFICATION PROBABILITY PLOTS")
        ConfigReader.__init__(self, config_path=config_path)
        color_names = get_color_dict().keys()
        self.max_y_lst = [x for x in range(10, 110, 10)]
        self.max_y_lst.insert(0, "auto")
        self.files_found_dict = get_file_name_info_in_directory(directory=self.machine_results_dir, file_type=self.file_type)
        check_if_filepath_list_is_empty(filepaths=list(self.files_found_dict.keys()), error_msg=f"Cannot visualize probabilities, no data in {self.machine_results_dir} directory")

        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VISUALIZE_CLF_PROBABILITIES.value)
        self.resolution_dropdown = DropDownMenu(self.style_settings_frm, "Resolution:", self.resolutions, "16")
        self.max_y_dropdown = DropDownMenu(self.style_settings_frm, "Max Y-axis:", self.max_y_lst, "16")

        self.line_clr_dropdown = DropDownMenu(self.style_settings_frm, "Line color:", color_names, "16")
        self.font_size_entry = Entry_Box(self.style_settings_frm, "Font size: ", "16", validation="numeric")
        self.line_width = Entry_Box(self.style_settings_frm, "Line width: ", "16", validation="numeric")
        self.line_clr_dropdown.setChoices("Red")
        self.resolution_dropdown.setChoices(self.resolutions[1])
        self.font_size_entry.entry_set(val=10)
        self.line_width.entry_set(val=6)
        self.max_y_dropdown.setChoices(choice="auto")

        self.settings_frm = LabelFrame(self.main_frm, text="VISUALIZATION SETTINGS", font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5)
        self.probability_frames_var = BooleanVar()
        self.probability_videos_var = BooleanVar()
        self.probability_last_frm_var = BooleanVar()
        self.probability_multiprocess_var = BooleanVar()

        self.clf_dropdown = DropDownMenu(self.settings_frm, "Classifier:", self.clf_names, "16")
        self.clf_dropdown.setChoices(self.clf_names[0])
        probability_frames_cb = Checkbutton(self.settings_frm,text="Create frames",variable=self.probability_frames_var)
        probability_videos_cb = Checkbutton(self.settings_frm,text="Create videos",variable=self.probability_videos_var,)
        probability_last_frm_cb = Checkbutton(self.settings_frm,text="Create last frame",variable=self.probability_last_frm_var)
        probability_multiprocess_cb = Checkbutton(
            self.settings_frm,
            text="Multi-process (faster)",
            variable=self.probability_multiprocess_var,
            command=lambda: self.enable_dropdown_from_checkbox(
                check_box_var=self.probability_multiprocess_var,
                dropdown_menus=[self.multiprocess_dropdown],
            ),
        )

        self.multiprocess_dropdown = DropDownMenu(
            self.settings_frm, "CPU cores:", list(range(2, self.cpu_cnt)), "12"
        )
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()

        self.run_frm = LabelFrame(self.main_frm,text="RUN",font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5,padx=5,fg="black")
        self.run_single_video_frm = LabelFrame(self.run_frm,text="SINGLE VIDEO",font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5,padx=5,fg="black")
        self.run_single_video_btn = Button(self.run_single_video_frm,text="Create single video",fg="blue",command=lambda: self.__create_probability_plots(multiple_videos=False))
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm,"Video:",list(self.files_found_dict.keys()),"12")
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])

        self.run_multiple_videos = LabelFrame(
            self.run_frm,
            text="MULTIPLE VIDEO",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
            fg="black",
        )
        self.run_multiple_video_btn = Button(
            self.run_multiple_videos,
            text="Create multiple videos ({} video(s) found)".format(
                str(len(list(self.files_found_dict.keys())))
            ),
            fg="blue",
            command=lambda: self.__create_probability_plots(multiple_videos=True),
        )

        self.style_settings_frm.grid(row=0, sticky=NW)
        self.resolution_dropdown.grid(row=0, sticky=NW)
        self.line_clr_dropdown.grid(row=1, sticky=NW)
        self.font_size_entry.grid(row=2, sticky=NW)
        self.line_width.grid(row=3, sticky=NW)
        self.max_y_dropdown.grid(row=4, sticky=NW)

        self.settings_frm.grid(row=1, sticky=NW)
        self.clf_dropdown.grid(row=0, sticky=NW)
        probability_frames_cb.grid(row=1, sticky=NW)
        probability_videos_cb.grid(row=2, sticky=NW)
        probability_last_frm_cb.grid(row=3, sticky=NW)
        probability_multiprocess_cb.grid(row=4, column=0, sticky=W)
        self.multiprocess_dropdown.grid(row=4, column=1, sticky=W)

        self.run_frm.grid(row=2, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        self.main_frm.mainloop()

    def __create_probability_plots(self, multiple_videos: bool):
        width = int(self.resolution_dropdown.getChoices().split("×")[0])
        height = int(self.resolution_dropdown.getChoices().split("×")[1])
        check_int(name="PLOT FONT SIZE", value=self.font_size_entry.entry_get, min_value=1)
        check_int(name="PLOT LINE WIDTH", value=self.line_width.entry_get, min_value=1)
        style_attr = {
            STYLE_WIDTH: width,
            STYLE_HEIGHT: height,
            STYLE_FONT_SIZE: int(self.font_size_entry.entry_get),
            STYLE_LINE_WIDTH: int(self.line_width.entry_get),
            STYLE_COLOR: self.line_clr_dropdown.getChoices(),
            STYLE_YMAX: self.max_y_dropdown.getChoices()}

        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        if not self.probability_multiprocess_var.get():
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
                                                                       cores=int(self.multiprocess_dropdown.getChoices()),
                                                                       style_attr=style_attr)

        threading.Thread(target=probability_plot_creator.run).start()

#_ = VisualizeClassificationProbabilityPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')



#_ = VisualizeClassificationProbabilityPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
