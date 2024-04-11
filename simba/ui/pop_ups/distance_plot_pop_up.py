__author__ = "Simon Nilsson"

import os
import threading
from collections import defaultdict
from tkinter import *

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.distance_plotter import DistancePlotterSingleCore
from simba.plotting.distance_plotter_mp import DistancePlotterMultiCore
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box)
from simba.utils.checks import check_if_filepath_list_is_empty, check_int
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import DuplicationError
from simba.utils.read_write import get_file_name_info_in_directory


class DistancePlotterPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="CREATE DISTANCE PLOTS")
        ConfigReader.__init__(self, config_path=config_path)

        self.data_path = os.path.join(
            self.project_path, "csv", "outlier_corrected_movement_location"
        )
        self.max_y_lst = list(range(10, 510, 10))
        self.max_y_lst.insert(0, "auto")
        self.files_found_dict = get_file_name_info_in_directory(
            directory=self.data_path, file_type=self.file_type
        )
        check_if_filepath_list_is_empty(
            filepaths=list(self.files_found_dict.keys()),
            error_msg="SIMBA ERROR: Zero files found in the project_folder/csv/outlier_corrected_movement_location directory. ",
        )

        self.number_of_distances = list(range(1, len(self.body_parts_lst) * 2))
        self.style_settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="STYLE SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.DISTANCE_PLOTS.value,
        )
        self.resolution_dropdown = DropDownMenu(
            self.style_settings_frm, "Resolution:", self.resolutions, "16"
        )
        self.font_size_entry = Entry_Box(
            self.style_settings_frm, "Font size: ", "16", validation="numeric"
        )
        self.line_width = Entry_Box(
            self.style_settings_frm, "Line width: ", "16", validation="numeric"
        )
        self.opacity_dropdown = DropDownMenu(
            self.style_settings_frm,
            "Line opacity:",
            list(np.round(np.arange(0.0, 1.1, 0.1), 1)),
            "16",
        )
        self.max_y_dropdown = DropDownMenu(
            self.style_settings_frm, "Max Y-axis:", self.max_y_lst, "16"
        )
        self.resolution_dropdown.setChoices(self.resolutions[1])
        self.font_size_entry.entry_set(val=8)
        self.line_width.entry_set(val=6)
        self.opacity_dropdown.setChoices(0.5)
        self.max_y_dropdown.setChoices(choice="auto")
        self.distances_frm = LabelFrame(
            self.main_frm,
            text="CHOOSE DISTANCES",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
        )
        self.number_of_distances_dropdown = DropDownMenu(
            self.distances_frm,
            "# Distances:",
            self.number_of_distances,
            "16",
            com=self.__populate_distances_menu,
        )
        self.number_of_distances_dropdown.setChoices(self.number_of_distances[0])
        self.__populate_distances_menu(1)

        self.settings_frm = LabelFrame(
            self.main_frm,
            text="VISUALIZATION SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
        )
        self.distance_frames_var = BooleanVar()
        self.distance_videos_var = BooleanVar()
        self.distance_final_img_var = BooleanVar()
        self.multiprocess_var = BooleanVar()
        distance_frames_cb = Checkbutton(
            self.settings_frm, text="Create frames", variable=self.distance_frames_var
        )
        distance_videos_cb = Checkbutton(
            self.settings_frm, text="Create videos", variable=self.distance_videos_var
        )
        distance_final_img_cb = Checkbutton(
            self.settings_frm,
            text="Create last frame",
            variable=self.distance_final_img_var,
        )
        self.multiprocess_cb = Checkbutton(
            self.settings_frm,
            text="Multiprocess (faster)",
            variable=self.multiprocess_var,
            command=lambda: self.enable_dropdown_from_checkbox(
                check_box_var=self.multiprocess_var,
                dropdown_menus=[self.multiprocess_dropdown],
            ),
        )
        self.multiprocess_dropdown = DropDownMenu(
            self.settings_frm, "Cores:", list(range(2, self.cpu_cnt)), "12"
        )
        self.multiprocess_dropdown.setChoices(choice=2)
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
            command=lambda: self.__create_distance_plots(multiple_videos=False),
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
            command=lambda: self.__create_distance_plots(multiple_videos=True),
        )

        self.style_settings_frm.grid(row=0, sticky=NW)
        self.resolution_dropdown.grid(row=0, sticky=NW)
        self.font_size_entry.grid(row=1, sticky=NW)
        self.line_width.grid(row=2, sticky=NW)
        self.opacity_dropdown.grid(row=3, sticky=NW)
        self.max_y_dropdown.grid(row=4, sticky=NW)

        self.distances_frm.grid(row=1, sticky=NW)
        self.number_of_distances_dropdown.grid(row=0, sticky=NW)

        self.settings_frm.grid(row=2, sticky=NW)
        distance_frames_cb.grid(row=0, sticky=NW)
        distance_videos_cb.grid(row=1, sticky=NW)
        distance_final_img_cb.grid(row=2, sticky=NW)
        self.multiprocess_cb.grid(row=3, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=3, column=1, sticky=NW)

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
            self.bp_1[distance_cnt] = DropDownMenu(
                self.distances_frm,
                "Distance {}:".format(str(distance_cnt + 1)),
                self.body_parts_lst,
                "16",
            )
            self.bp_1[distance_cnt].setChoices(self.body_parts_lst[distance_cnt])
            self.bp_1[distance_cnt].grid(row=distance_cnt + 1, column=0, sticky=NW)

            self.bp_2[distance_cnt] = DropDownMenu(
                self.distances_frm, "", self.body_parts_lst, "2"
            )
            self.bp_2[distance_cnt].setChoices(self.body_parts_lst[distance_cnt])
            self.bp_2[distance_cnt].grid(row=distance_cnt + 1, column=1, sticky=NW)

            self.distance_clrs[distance_cnt] = DropDownMenu(
                self.distances_frm, "", self.colors_dict, "2"
            )
            self.distance_clrs[distance_cnt].setChoices(
                list(self.colors_dict.keys())[distance_cnt]
            )
            self.distance_clrs[distance_cnt].grid(
                row=distance_cnt + 1, column=3, sticky=NW
            )

    def __create_distance_plots(self, multiple_videos: bool):
        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [
                self.files_found_dict[self.single_video_dropdown.getChoices()]
            ]

        line_attr = defaultdict(list)
        for attr in (self.bp_1, self.bp_2, self.distance_clrs):
            for key, value in attr.items():
                line_attr[key].append(value.getChoices())
        line_attr = list(line_attr.values())

        for cnt, v in enumerate(line_attr):
            if v[0] == v[1]:
                raise DuplicationError(
                    msg=f"DISTANCE LINE {cnt+1} ERROR: The two body-parts cannot be the same body-part.",
                    source=self.__class__.__name__,
                )

        width = int(self.resolution_dropdown.getChoices().split("×")[0])
        height = int(self.resolution_dropdown.getChoices().split("×")[1])
        check_int(
            name="DISTANCE FONT SIZE", value=self.font_size_entry.entry_get, min_value=1
        )
        check_int(
            name="DISTANCE LINE WIDTH", value=self.line_width.entry_get, min_value=1
        )
        max_y = self.max_y_dropdown.getChoices()
        if max_y == "auto":
            max_y = -1
        else:
            max_y = int(max_y)

        style_attr = {
            "width": width,
            "height": height,
            "line width": int(self.line_width.entry_get),
            "font size": int(self.font_size_entry.entry_get),
            "opacity": float(self.opacity_dropdown.getChoices()),
            "y_max": max_y,
        }

        if not self.multiprocess_var.get():
            distance_plotter = DistancePlotterSingleCore(
                config_path=self.config_path,
                frame_setting=self.distance_frames_var.get(),
                video_setting=self.distance_videos_var.get(),
                final_img=self.distance_final_img_var.get(),
                style_attr=style_attr,
                data_paths=data_paths,
                line_attr=line_attr,
            )
        else:
            distance_plotter = DistancePlotterMultiCore(
                config_path=self.config_path,
                frame_setting=self.distance_frames_var.get(),
                video_setting=self.distance_videos_var.get(),
                final_img=self.distance_final_img_var.get(),
                style_attr=style_attr,
                data_paths=data_paths,
                line_attr=line_attr,
                core_cnt=int(self.multiprocess_dropdown.getChoices()),
            )

        threading.Thread(distance_plotter.run()).start()


# _ = DistancePlotterPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')


# _ = DistancePlotterPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# _ = DistancePlotterPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini')
