__author__ = "Simon Nilsson"

""" Tkinter pop-up classes for unsupervised ML"""

import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import DropDownMenu, FileSelect, FolderSelect
from simba.unsupervised.enums import UMLOptions
from simba.unsupervised.umap_embedder import UmapEmbedder
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.enums import Formats


class TransformDimReductionPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="DIMENSIONALITY REDUCTION: TRANSFORM")
        ConfigReader.__init__(self, config_path=config_path)
        self.dim_reduction_frm = LabelFrame(
            self.main_frm,
            text="SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.model_select = FileSelect(
            self.dim_reduction_frm,
            "MODEL (PICKLE):",
            lblwidth=25,
            file_types=[("SimBA model pickle", f".{Formats.PICKLE.value}")],
            initialdir=self.project_path,
        )
        self.dataset_select = FileSelect(
            self.dim_reduction_frm,
            "DATASET (PICKLE):",
            lblwidth=25,
            file_types=[("SimBA model pickle", f".{Formats.PICKLE.value}")],
            initialdir=self.project_path,
        )
        self.save_dir = FolderSelect(
            self.dim_reduction_frm,
            "SAVE DIRECTORY: ",
            lblwidth=25,
            initialdir=self.project_path,
        )
        self.additional_data = DropDownMenu(
            self.dim_reduction_frm,
            "ADDITIONAL DATA:",
            UMLOptions.UMAP_ADDITIONAL_DATA.value,
            "25",
        )
        self.additional_data.setChoices("SCALED FEATURES & CLASSIFICATIONS")
        self.save_format_dropdown = DropDownMenu(
            self.dim_reduction_frm, "SAVE FORMATS:", UMLOptions.SAVE_FORMATS.value, "25"
        )
        self.save_format_dropdown.setChoices(UMLOptions.SAVE_FORMATS.value[0])

        self.dim_reduction_frm.grid(row=0, column=0, sticky=NW)
        self.model_select.grid(row=0, column=0, sticky=NW)
        self.dataset_select.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=2, column=0, sticky=NW)
        self.additional_data.grid(row=3, column=0, sticky=NW)
        self.save_format_dropdown.grid(row=4, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_if_dir_exists(in_dir=self.save_dir.folder_path)
        check_file_exist_and_readable(file_path=self.model_select.file_path)
        check_file_exist_and_readable(file_path=self.dataset_select.file_path)
        settings = {"DATA_FORMAT": None, "CLASSIFICATIONS": False}
        settings_choice = self.features_dropdown.getChoices()
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

        umap_searcher = UmapEmbedder()
        threading.Thread(
            target=umap_searcher.transform(
                data_path=self.dataset_select.file_path,
                model=self.model_select.file_path,
                save_dir=self.save_dir.folder_path,
                settings=settings,
            )
        ).start()


# _ = TransformDimReductionPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
