__author__ = "Simon Nilsson"

import os

""" Tkinter pop-up classes for unsupervised ML"""

import glob
from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.ui.tkinter_functions import FolderSelect
from simba.unsupervised.dbcv_calculator import DBCVCalculator
from simba.utils.checks import (check_if_dir_exists,
                                check_if_filepath_list_is_empty)
from simba.utils.enums import Formats


class DBCVPopUp(PopUpMixin, ConfigReader, UnsupervisedMixin):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="DENSITY BASED CLUSTER VALIDATION")
        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        self.data_frm = LabelFrame(
            self.main_frm,
            text="DATA",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.folder_selected = FolderSelect(
            self.data_frm,
            "DATASETS (DIRECTORY WITH PICKLES):",
            lblwidth=35,
            initialdir=self.project_path,
        )
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.folder_selected.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_if_dir_exists(in_dir=self.folder_selected.folder_path)
        data_paths = glob.glob(
            self.folder_selected.folder_path + f"/*.{Formats.PICKLE.value}"
        )
        check_if_filepath_list_is_empty(
            filepaths=data_paths,
            error_msg=f"No pickle files in {self.folder_selected.folder_path}",
        )
        dbcv_calculator = DBCVCalculator(
            data_path=self.folder_selected.folder_path, config_path=self.config_path
        )
        dbcv_calculator.run()


# _ = DBCVPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
