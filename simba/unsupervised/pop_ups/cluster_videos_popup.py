__author__ = "Simon Nilsson"

import os

""" Tkinter pop-up classes for unsupervised ML"""

import glob
from tkinter import *
from typing import Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (DropDownMenu, Entry_Box, FileSelect,
                                        FolderSelect)
from simba.unsupervised.cluster_visualizer import ClusterVisualizer
from simba.unsupervised.enums import UMLOptions
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_int)
from simba.utils.enums import Formats


class ClusterVisualizerPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="CLUSTER VIDEO VISUALIZATIONS")
        ConfigReader.__init__(self, config_path=config_path)
        self.include_pose_var = BooleanVar(value=False)

        self.data_frm = LabelFrame(
            self.main_frm,
            text="DATA",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.videos_dir_select = FolderSelect(
            self.data_frm,
            "VIDEOS DIRECTORY:",
            lblwidth=25,
            initialdir=self.project_path,
        )
        self.dataset_file_selected = FileSelect(
            self.data_frm,
            "DATASET (PICKLE): ",
            lblwidth=25,
            initialdir=self.project_path,
            file_types=[("SimBA model", f"*.{Formats.PICKLE.value}")],
        )
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.videos_dir_select.grid(row=0, column=0, sticky=NW)

        self.settings_frm = LabelFrame(
            self.main_frm,
            text="SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.include_pose = Checkbutton(
            self.settings_frm,
            text="INCLUDE POSE-ESTIMATION",
            variable=self.include_pose_var,
            command=lambda: self.enable_entrybox_from_checkbox(
                check_box_var=self.include_pose_var,
                entry_boxes=[self.circle_size_entry],
            ),
        )
        self.circle_size_entry = Entry_Box(
            self.settings_frm, "CIRCLE SIZE: ", "25", validation="numeric"
        )
        self.circle_size_entry.entry_set(val=5)
        self.circle_size_entry.set_state(setstatus="disable")

        self.speed_dropdown = DropDownMenu(
            self.settings_frm, "VIDEO SPEED:", UMLOptions.SPEED_OPTIONS.value, "25"
        )
        self.speed_dropdown.setChoices(1.0)
        self.settings_frm.grid(row=1, column=0, sticky=NW)
        self.include_pose.grid(row=0, column=0, sticky=NW)
        self.circle_size_entry.grid(row=1, column=0, sticky=NW)
        self.speed_dropdown.grid(row=2, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_file_exist_and_readable(file_path=self.dataset_file_selected.file_path)
        check_if_dir_exists(in_dir=self.videos_dir_select.folder_path)
        speed = float(self.speed_dropdown.getChoices())
        if self.include_pose_var.get():
            check_int(name="CIRCLE SIZE", value=self.circle_size_entry.entry_get)
            circle_size = self.circle_size_entry.entry_get
        else:
            circle_size = np.inf
        settings = {
            "videos_speed": speed,
            "pose": {
                "include": self.include_pose_var.get(),
                "circle_size": circle_size,
            },
        }
        cluster_visualizer = ClusterVisualizer(
            config_path=self.config_path,
            settings=settings,
            video_dir=self.videos_dir_select.folder_path,
            data_path=self.dataset_file_selected.file_path,
        )
        cluster_visualizer.run()


# _ = ClusterVisualizerPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
