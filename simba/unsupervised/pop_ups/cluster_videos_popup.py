__author__ = "Simon Nilsson"

import multiprocessing
import os
import platform
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (DropDownMenu, Entry_Box, FileSelect,
                                        FolderSelect)
from simba.unsupervised.cluster_video_visualizer import ClusterVideoVisualizer
from simba.unsupervised.enums import UMLOptions
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import Formats
from simba.utils.lookups import get_color_dict


class ClusterVisualizerPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="CLUSTER VIDEO VISUALIZATIONS")
        ConfigReader.__init__(self, config_path=config_path)
        self.bg_clr_dict = get_color_dict()
        self.include_pose_var = BooleanVar(value=False)
        self.data_frm = LabelFrame(
            self.main_frm,
            text="DATA",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        plt_types = ["SKELETON", "HULL", "POINTS"]
        self.plot_type_dropdown = DropDownMenu(
            self.data_frm,
            "PLOT TYPE:",
            plt_types,
            "25",
            com=lambda x: self._activate_video(selection=x),
        )
        self.plot_type_dropdown.setChoices("SKELETON")
        self.dataset_file_selected = FileSelect(
            self.data_frm,
            "CLUSTER DATASET (PICKLE): ",
            lblwidth=25,
            initialdir=self.project_path,
            file_types=[("SimBA model", f"*.{Formats.PICKLE.value}")],
        )
        self.plot_type_dropdown.grid(row=0, column=0, sticky=NW)
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.dataset_file_selected.grid(row=1, column=0, sticky=NW)
        self.settings_frm = LabelFrame(
            self.main_frm,
            text="SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )

        max_vid_lst = [str(x) for x in range(0, 105, 5)]
        max_vid_lst.insert(0, "None")
        self.max_vid_per_cluster = DropDownMenu(
            self.settings_frm, "MAX VIDEOS PER CLUSTER:", max_vid_lst, "25"
        )
        self.max_vid_per_cluster.setChoices(max_vid_lst[0])
        self.speed_dropdown = DropDownMenu(
            self.settings_frm, "VIDEO SPEED:", UMLOptions.SPEED_OPTIONS.value, "25"
        )
        self.bg_clr_dropdown = DropDownMenu(
            self.settings_frm, "BACKGROUND COLOR:", list(self.bg_clr_dict.keys()), "25"
        )
        self.bg_clr_dropdown.setChoices("White")
        self.speed_dropdown.setChoices(1.0)
        self.settings_frm.grid(row=1, column=0, sticky=NW)
        self.max_vid_per_cluster.grid(row=0, column=0, sticky=NW)
        self.speed_dropdown.grid(row=1, column=0, sticky=NW)
        self.bg_clr_dropdown.grid(row=2, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def _activate_video(self, selection):
        if selection == "VIDEO":
            self.videos_dir_select = FolderSelect(
                self.data_frm,
                "VIDEOS DIRECTORY:",
                lblwidth=15,
                initialdir=self.video_dir,
            )
            self.videos_dir_select.grid(row=0, column=1, sticky=NW)
        else:
            self.videos_dir_select.destroy()

    def run(self):
        check_file_exist_and_readable(file_path=self.dataset_file_selected.file_path)
        speed = float(self.speed_dropdown.getChoices())
        data_path = self.dataset_file_selected.file_path
        max_videos = self.max_vid_per_cluster.getChoices()
        if max_videos == "None":
            max_videos = None
        else:
            max_videos = int(max_videos)
        plt_type = self.plot_type_dropdown.getChoices()
        bg_clr = self.color_dict[self.bg_clr_dropdown.getChoices()]

        cluster_visualizer = ClusterVideoVisualizer(
            config_path=self.config_path,
            data_path=data_path,
            plot_type=plt_type,
            bg_clr=bg_clr,
            max_videos=max_videos,
            speed=speed,
        )
        if platform.system() == "Darwin":
            multiprocessing.set_start_method("spawn", force=True)
        threading.Thread(target=cluster_visualizer.run()).start()


# _ = ClusterVisualizerPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
