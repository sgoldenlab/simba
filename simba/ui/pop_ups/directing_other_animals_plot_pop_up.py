__author__ = "Simon Nilsson"

import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.directing_animals_visualizer import \
    DirectingOtherAnimalsVisualizer
from simba.plotting.directing_animals_visualizer_mp import \
    DirectingOtherAnimalsVisualizerMultiprocess
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, DropDownMenu
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import AnimalNumberError
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import find_all_videos_in_directory

DIRECTION_THICKNESS = "direction_thickness"
DIRECTIONALITY_COLOR = "directionality_color"
CIRCLE_SIZE = "circle_size"
HIGHLIGHT_ENDPOINTS = "highlight_endpoints"
SHOW_POSE = "show_pose"
ANIMAL_NAMES = "animal_names"


class DirectingOtherAnimalsVisualizerPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path)
        if self.animal_cnt == 1:
            raise AnimalNumberError(
                msg="Cannot visualize directionality between animals in a 1 animal project.",
                source=self.__class__.__name__,
            )
        PopUpMixin.__init__(self, title="CREATE ANIMAL DIRECTION VIDEOS")
        self.color_dict = get_color_dict()
        self.color_lst = list(self.color_dict.keys())
        self.color_lst.insert(0, "random")
        self.size_lst = list(range(1, 11))
        self.files_found_dict = find_all_videos_in_directory(
            directory=self.video_dir, as_dict=True
        )
        self.show_pose_var = BooleanVar(value=True)
        self.show_animal_names_var = BooleanVar(value=True)
        self.highlight_direction_endpoints_var = BooleanVar(value=True)
        self.multiprocess_var = BooleanVar(value=False)

        self.style_settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="STYLE SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.DIRECTING_ANIMALS_PLOTS.value,
        )
        self.show_pose_cb = Checkbutton(
            self.style_settings_frm,
            text="Show pose-estimated body-parts",
            variable=self.show_pose_var,
        )
        self.highlight_direction_endpoints_cb = Checkbutton(
            self.style_settings_frm,
            text="Highlight direction end-points",
            variable=self.highlight_direction_endpoints_var,
        )
        self.show_animal_names_cb = Checkbutton(
            self.style_settings_frm,
            text="Show animal names",
            variable=self.show_animal_names_var,
        )

        self.direction_clr_dropdown = DropDownMenu(
            self.style_settings_frm, "Direction color:", self.color_lst, "16"
        )
        self.pose_size_dropdown = DropDownMenu(
            self.style_settings_frm, "Pose circle size:", self.size_lst, "16"
        )
        self.line_thickness = DropDownMenu(
            self.style_settings_frm, "Line thickness:", self.size_lst, "16"
        )
        self.line_thickness.setChoices(choice=4)
        self.pose_size_dropdown.setChoices(choice=3)
        self.direction_clr_dropdown.setChoices(choice="random")
        multiprocess_cb = Checkbutton(
            self.style_settings_frm,
            text="Multi-process (faster)",
            variable=self.multiprocess_var,
            command=lambda: self.enable_dropdown_from_checkbox(
                check_box_var=self.multiprocess_var,
                dropdown_menus=[self.multiprocess_dropdown],
            ),
        )
        self.multiprocess_dropdown = DropDownMenu(
            self.style_settings_frm, "CPU cores:", list(range(2, self.cpu_cnt)), "12"
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
            command=lambda: self.__create_directionality_plots(multiple_videos=False),
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
            text=f"Create multiple videos ({len(list(self.files_found_dict.keys()))} video(s) found)",
            fg="blue",
            command=lambda: self.__create_directionality_plots(multiple_videos=True),
        )

        self.style_settings_frm.grid(row=0, column=0, sticky=NW)
        self.show_pose_cb.grid(row=0, column=0, sticky=NW)
        self.highlight_direction_endpoints_cb.grid(row=1, column=0, sticky=NW)
        self.show_animal_names_cb.grid(row=2, column=0, sticky=NW)
        self.direction_clr_dropdown.grid(row=3, column=0, sticky=NW)
        self.pose_size_dropdown.grid(row=4, column=0, sticky=NW)
        self.line_thickness.grid(row=5, column=0, sticky=NW)
        multiprocess_cb.grid(row=6, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=6, column=1, sticky=NW)

        self.run_frm.grid(row=1, column=0, sticky=NW)
        self.run_single_video_frm.grid(row=0, column=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)
        self.run_multiple_videos.grid(row=1, column=0, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, column=0, sticky=NW)

        self.main_frm.mainloop()

    def __create_directionality_plots(self, multiple_videos: bool):
        if multiple_videos:
            video_paths = list(self.files_found_dict.values())
        else:
            video_paths = [
                self.files_found_dict[self.single_video_dropdown.getChoices()]
            ]

        direction_clr = self.direction_clr_dropdown.getChoices()
        if direction_clr != "random":
            direction_clr = self.color_dict[direction_clr]

        style_attr = {
            SHOW_POSE: self.show_pose_var.get(),
            CIRCLE_SIZE: int(self.pose_size_dropdown.getChoices()),
            DIRECTIONALITY_COLOR: direction_clr,
            DIRECTION_THICKNESS: int(self.line_thickness.getChoices()),
            HIGHLIGHT_ENDPOINTS: self.highlight_direction_endpoints_var.get(),
            ANIMAL_NAMES: self.show_animal_names_var.get(),
        }

        for video_path in video_paths:
            if not self.multiprocess_var.get():
                visualizer = DirectingOtherAnimalsVisualizer(
                    config_path=self.config_path,
                    video_path=video_path,
                    style_attr=style_attr,
                )
            else:
                visualizer = DirectingOtherAnimalsVisualizerMultiprocess(
                    config_path=self.config_path,
                    video_path=video_path,
                    style_attr=style_attr,
                    core_cnt=int(self.multiprocess_dropdown.getChoices()),
                )

            threading.Thread(target=visualizer.run()).start()


# _ = DirectingOtherAnimalsVisualizerPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# _ = DirectingOtherAnimalsVisualizerPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
