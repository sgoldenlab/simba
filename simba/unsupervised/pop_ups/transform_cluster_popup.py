__author__ = "Simon Nilsson"

import os

""" Tkinter pop-up classes for unsupervised ML"""

import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import DropDownMenu, FileSelect, FolderSelect
from simba.unsupervised.enums import UMLOptions, Unsupervised
from simba.unsupervised.hdbscan_clusterer import HDBSCANClusterer
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.enums import Formats


class TransformClustererPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="CLUSTERING: TRANSFORM")
        ConfigReader.__init__(self, config_path=config_path)
        self.settings_frm = LabelFrame(
            self.main_frm,
            text="SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.model_select = FileSelect(
            self.settings_frm,
            "CLUSTER MODEL (PICKLE):",
            lblwidth=25,
            file_types=[("SimBA model pickle", f".{Formats.PICKLE.value}")],
            initialdir=self.project_path,
        )
        self.data_select = FileSelect(
            self.settings_frm,
            "DATASET (PICKLE):",
            lblwidth=25,
            file_types=[("SimBA data pickle", f".{Formats.PICKLE.value}")],
            initialdir=self.project_path,
        )
        self.save_dir = FolderSelect(
            self.settings_frm,
            "SAVE DIRECTORY:",
            lblwidth=25,
            initialdir=self.project_path,
        )
        self.additional_data_drpdwn = DropDownMenu(
            self.settings_frm,
            "INCLUDE FEATURES:",
            UMLOptions.UMAP_ADDITIONAL_DATA.value,
            "25",
        )
        self.additional_data_drpdwn.setChoices("SCALED FEATURES & CLASSIFICATIONS")
        self.save_format_dropdown = DropDownMenu(
            self.settings_frm, "SAVE FORMATS:", UMLOptions.SAVE_FORMATS.value, "25"
        )
        self.save_format_dropdown.setChoices(UMLOptions.SAVE_FORMATS.value[0])
        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.model_select.grid(row=0, column=0, sticky=NW)
        self.data_select.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=2, column=0, sticky=NW)
        self.additional_data_drpdwn.grid(row=3, column=0, sticky=NW)
        self.save_format_dropdown.grid(row=4, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_file_exist_and_readable(self.model_select.file_path)
        check_file_exist_and_readable(self.data_select.file_path)
        check_if_dir_exists(self.save_dir.folder_path)
        settings = {"DATA_FORMAT": None, "CLASSIFICATIONS": False}
        settings_choice = self.additional_data_drpdwn.getChoices()
        if settings_choice == "SCALED FEATURES":
            settings["DATA_FORMAT"] = "scaled"
        elif settings_choice == "RAW FEATURES":
            settings["DATA_FORMAT"] = "RAW"
        elif settings_choice == "SCALED FEATURES & CLASSIFICATIONS":
            settings["DATA_FORMAT"] = "scaled"
            settings["CLASSIFICATIONS"] = True
        elif settings_choice == "RAW FEATURES & CLASSIFICATIONS":
            settings["DATA_FORMAT"] = "RAW"
            settings["CLASSIFICATIONS"] = True

        clusterer = HDBSCANClusterer()
        threading.Thread(
            target=clusterer.transform(
                data_path=self.data_select.file_path,
                model=self.model_select.file_path,
                save_dir=self.save_dir.folder_path,
                settings=settings,
            )
        ).start()


# _ = TransformClustererPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
