__author__ = "Simon Nilsson"

import os

""" Tkinter pop-up classes for unsupervised ML"""

import glob
from tkinter import *
from typing import Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.unsupervised_mixin import UnsupervisedMixin
from simba.ui.tkinter_functions import FileSelect
from simba.unsupervised.enums import Unsupervised
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import Formats


class PrintEmBeddingInfoPopUp(PopUpMixin, ConfigReader, UnsupervisedMixin):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="PRINT EMBEDDING MODEL INFO")
        ConfigReader.__init__(self, config_path=config_path)
        UnsupervisedMixin.__init__(self)
        self.data_frm = LabelFrame(
            self.main_frm,
            text="DATA",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.dataset_file_selected = FileSelect(
            self.data_frm, "DATASET (PICKLE): ", lblwidth=25
        )
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.dataset_file_selected.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_file_exist_and_readable(file_path=self.dataset_file_selected.file_path)
        data = self.read_pickle(data_path=self.dataset_file_selected.file_path)
        parameters = {
            **data[Unsupervised.DR_MODEL.value][Unsupervised.PARAMETERS.value]
        }
        print(parameters)


# _ = PrintEmBeddingInfoPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
