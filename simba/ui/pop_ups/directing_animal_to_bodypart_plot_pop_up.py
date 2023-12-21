__author__ = "Tzuk Polinsky"

import os
from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.directing_animals_to_bodypart_visualizer import \
    DirectingAnimalsToBodyPartVisualizer
from simba.plotting.directing_animals_visualizer import \
    DirectingOtherAnimalsVisualizer
from simba.plotting.directing_animals_visualizer_mp import \
    DirectingOtherAnimalsVisualizerMultiprocess
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, DropDownMenu
from simba.utils.enums import Formats, Keys, Links, Paths
from simba.utils.errors import AnimalNumberError
from simba.utils.read_write import get_file_name_info_in_directory


class DirectingAnimalToBodyPartVisualizerPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        ConfigReader.__init__(self, config_path=config_path)
        PopUpMixin.__init__(self, title="CREATE ANIMAL DIRECTION TO BODY PART VIDEOS")
        self.color_lst = list(self.colors_dict.keys())
        self.color_lst.insert(0, "Random")
        self.size_lst = list(range(1, 11))
        self.data_path = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.files_found_dict = get_file_name_info_in_directory(
            directory=self.data_path, file_type=self.file_type
        )

        self.show_pose_var = BooleanVar(value=True)
        self.merge_directionality_lines_var = BooleanVar(value=False)
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
        self.merge_directionality_lines_cb = Checkbutton(
            self.style_settings_frm,
            text="Polyfill direction lines",
            variable=self.merge_directionality_lines_var,
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
        self.direction_clr_dropdown.setChoices(choice="Random")
        # multiprocess_cb = Checkbutton(
        #     self.style_settings_frm,
        #     text="Multi-process (faster)",
        #     variable=self.multiprocess_var,
        #     command=lambda: self.enable_dropdown_from_checkbox(
        #         check_box_var=self.multiprocess_var,
        #         dropdown_menus=[self.multiprocess_dropdown],
        #     ),
        # )
        # self.multiprocess_dropdown = DropDownMenu(
        #     self.style_settings_frm, "CPU cores:", list(range(2, self.cpu_cnt)), "12"
        # )
        # self.multiprocess_dropdown.setChoices(2)
        # self.multiprocess_dropdown.disable()

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
            text="Create multiple videos ({} video(s) found)".format(
                str(len(list(self.files_found_dict.keys())))
            ),
            fg="blue",
            command=lambda: self.__create_directionality_plots(multiple_videos=True),
        )

        self.style_settings_frm.grid(row=0, column=0, sticky=NW)
        self.show_pose_cb.grid(row=0, column=0, sticky=NW)
        self.merge_directionality_lines_cb.grid(row=2, column=0, sticky=NW)
        self.direction_clr_dropdown.grid(row=3, column=0, sticky=NW)
        self.pose_size_dropdown.grid(row=4, column=0, sticky=NW)
        self.line_thickness.grid(row=5, column=0, sticky=NW)
        # multiprocess_cb.grid(row=6, column=0, sticky=NW)
        # self.multiprocess_dropdown.grid(row=6, column=1, sticky=NW)

        self.run_frm.grid(row=1, column=0, sticky=NW)
        self.run_single_video_frm.grid(row=0, column=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)
        self.run_multiple_videos.grid(row=1, column=0, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, column=0, sticky=NW)

    def __create_directionality_plots(self, multiple_videos: bool):
        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [
                self.files_found_dict[self.single_video_dropdown.getChoices()]
            ]

        style_attr = {
            "Show_pose": self.show_pose_var.get(),
            "Pose_circle_size": int(self.pose_size_dropdown.getChoices()),
            "Direction_color": self.direction_clr_dropdown.getChoices(),
            "Direction_thickness": int(self.line_thickness.getChoices()),
            "Polyfill": self.merge_directionality_lines_var.get(),
        }

        for data_path in data_paths:
            # if not self.multiprocess_var.get():
            directing_other_animal_visualizer = DirectingAnimalsToBodyPartVisualizer(
                config_path=self.config_path,
                data_path=data_path,
                style_attr=style_attr,
            )
            # else:
            #     directing_other_animal_visualizer = (
            #         DirectingOtherAnimalsVisualizerMultiprocess(
            #             config_path=self.config_path,
            #             data_path=data_path,
            #             style_attr=style_attr,
            #             core_cnt=int(self.multiprocess_dropdown.getChoices()),
            #         )
            #     )
            directing_other_animal_visualizer.run()
