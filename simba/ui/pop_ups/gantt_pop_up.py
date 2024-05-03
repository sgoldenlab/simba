__author__ = "Simon Nilsson"

import os
from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.gantt_creator import GanttCreatorSingleProcess
from simba.plotting.gantt_creator_mp import GanttCreatorMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box)
from simba.utils.checks import check_if_filepath_list_is_empty, check_int
from simba.utils.enums import Formats, Keys, Links, Paths
from simba.utils.read_write import get_file_name_info_in_directory


class GanttPlotPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(
            self, config_path=config_path, title="VISUALIZE GANTT PLOTS"
        )
        ConfigReader.__init__(self, config_path=config_path)
        self.data_path = os.path.join(
            self.project_path, Paths.MACHINE_RESULTS_DIR.value
        )
        self.files_found_dict = get_file_name_info_in_directory(
            directory=self.data_path, file_type=self.file_type
        )
        check_if_filepath_list_is_empty(
            filepaths=list(self.files_found_dict.keys()),
            error_msg="SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. Create classification results before visualizing gantt charts",
        )

        self.style_settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="STYLE SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.GANTT_PLOTS.value,
        )
        self.use_default_style_bool = BooleanVar(value=True)
        self.auto_compute_style_cb = Checkbutton(
            self.style_settings_frm,
            text="Use default style",
            variable=self.use_default_style_bool,
            command=lambda: self.enable_text_settings(),
        )
        self.resolution_dropdown = DropDownMenu(
            self.style_settings_frm, "Resolution:", self.resolutions, "16"
        )
        self.font_size_entry = Entry_Box(
            self.style_settings_frm, "Font size: ", "16", validation="numeric"
        )
        self.font_rotation_entry = Entry_Box(
            self.style_settings_frm,
            "Font rotation degree: ",
            "16",
            validation="numeric",
        )
        self.font_size_entry.entry_set(val=8)
        self.font_rotation_entry.entry_set(val=45)
        self.resolution_dropdown.setChoices(self.resolutions[1])
        self.resolution_dropdown.disable()
        self.font_size_entry.set_state("disable")
        self.font_rotation_entry.set_state("disable")

        self.settings_frm = LabelFrame(
            self.main_frm,
            text="VISUALIZATION SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
        )
        self.gantt_frames_var = BooleanVar()
        self.gantt_last_frame_var = BooleanVar()
        self.gantt_videos_var = BooleanVar()
        self.gantt_multiprocess_var = BooleanVar()

        gantt_frames_cb = Checkbutton(
            self.settings_frm, text="Create frames", variable=self.gantt_frames_var
        )
        gantt_videos_cb = Checkbutton(
            self.settings_frm, text="Create videos", variable=self.gantt_videos_var
        )
        gantt_last_frame_cb = Checkbutton(
            self.settings_frm,
            text="Create last frame",
            variable=self.gantt_last_frame_var,
        )
        gantt_multiprocess_cb = Checkbutton(
            self.settings_frm,
            text="Multi-process (faster)",
            variable=self.gantt_multiprocess_var,
            command=lambda: self.enable_dropdown_from_checkbox(
                check_box_var=self.gantt_multiprocess_var,
                dropdown_menus=[self.multiprocess_dropdown],
            ),
        )

        self.multiprocess_dropdown = DropDownMenu(
            self.settings_frm, "CPU cores:", list(range(2, self.cpu_cnt)), "12"
        )
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()

        self.run_frm = LabelFrame(
            self.main_frm,
            text="RUN",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
            fg="black",
        )
        self.run_single_video_frm = LabelFrame(
            self.run_frm,
            text="SINGLE VIDEO",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
            fg="black",
        )
        self.run_single_video_btn = Button(
            self.run_single_video_frm,
            text="Create single video",
            fg="blue",
            command=lambda: self.__create_gantt_plots(multiple_videos=False),
        )
        self.single_video_dropdown = DropDownMenu(
            self.run_single_video_frm,
            "Video:",
            list(self.files_found_dict.keys()),
            "12",
        )
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
            command=lambda: self.__create_gantt_plots(multiple_videos=True),
        )

        self.style_settings_frm.grid(row=0, sticky=NW)
        self.auto_compute_style_cb.grid(row=0, sticky=NW)
        self.resolution_dropdown.grid(row=1, sticky=NW)
        self.font_size_entry.grid(row=2, sticky=NW)
        self.font_rotation_entry.grid(row=3, sticky=NW)

        self.settings_frm.grid(row=1, sticky=NW)
        gantt_videos_cb.grid(row=0, sticky=NW)
        gantt_frames_cb.grid(row=1, sticky=W)
        gantt_last_frame_cb.grid(row=2, sticky=NW)
        gantt_multiprocess_cb.grid(row=3, column=0, sticky=W)
        self.multiprocess_dropdown.grid(row=3, column=1, sticky=W)

        self.run_frm.grid(row=2, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

    def enable_text_settings(self):
        if not self.use_default_style_bool.get():
            self.resolution_dropdown.enable()
            self.font_rotation_entry.set_state("normal")
            self.font_size_entry.set_state("normal")
        else:
            self.resolution_dropdown.disable()
            self.font_rotation_entry.set_state("disable")
            self.font_size_entry.set_state("disable")

    def __create_gantt_plots(self, multiple_videos: bool):
        width = int(self.resolution_dropdown.getChoices().split("×")[0])
        height = int(self.resolution_dropdown.getChoices().split("×")[1])
        check_int(name="FONT SIZE", value=self.font_size_entry.entry_get, min_value=1)
        check_int(
            name="FONT ROTATION DEGREES",
            value=self.font_rotation_entry.entry_get,
            min_value=0,
            max_value=360,
        )
        style_attr = {
            "width": width,
            "height": height,
            "font size": int(self.font_size_entry.entry_get),
            "font rotation": int(self.font_rotation_entry.entry_get),
        }

        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        if self.gantt_multiprocess_var.get():
            gantt_creator = GanttCreatorMultiprocess(config_path=self.config_path,
                                                     frame_setting=self.gantt_frames_var.get(),
                                                     video_setting=self.gantt_videos_var.get(),
                                                     last_frm_setting=self.gantt_last_frame_var.get(),
                                                     data_paths=data_paths,
                                                     cores=int(self.multiprocess_dropdown.getChoices()),
                                                     style_attr=style_attr)
        else:
            gantt_creator = GanttCreatorSingleProcess(config_path=self.config_path,
                                                      frame_setting=self.gantt_frames_var.get(),
                                                      video_setting=self.gantt_videos_var.get(),
                                                      last_frm_setting=self.gantt_last_frame_var.get(),
                                                      data_paths=data_paths,
                                                      style_attr=style_attr)
        gantt_creator.run()


# _ = GanttPlotPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
