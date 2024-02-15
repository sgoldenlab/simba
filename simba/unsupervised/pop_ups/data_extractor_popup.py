__author__ = "Simon Nilsson"

import glob
import os
import threading
from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import DropDownMenu, FolderSelect
from simba.unsupervised.data_extractor import DataExtractor
from simba.unsupervised.enums import UMLOptions
from simba.utils.checks import check_if_dir_exists
from simba.utils.enums import Formats, Options


class DataExtractorPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="DATA EXTRACTOR")
        ConfigReader.__init__(self, config_path=config_path)
        data_frm = LabelFrame(
            self.main_frm,
            text="DATA",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.data_dir_select = FolderSelect(
            data_frm,
            "DATA DIRECTORY (PICKLES):",
            lblwidth=25,
            initialdir=self.project_path,
        )
        data_frm.grid(row=0, column=0, sticky=NW)
        self.data_dir_select.grid(row=0, column=0, sticky=NW)

        data_type_frm = LabelFrame(
            self.main_frm,
            text="DATA TYPE",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.data_type_dropdown = DropDownMenu(
            data_type_frm, "DATA TYPE:", UMLOptions.DATA_TYPES.value, "25"
        )
        self.data_type_dropdown.setChoices(UMLOptions.DATA_TYPES.value[0])
        data_type_frm.grid(row=1, column=0, sticky=NW)
        self.data_type_dropdown.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run, title="RUN")
        self.main_frm.mainloop()

    def run(self):
        check_if_dir_exists(self.data_dir_select.folder_path)
        data_extractor = DataExtractor(
            config_path=self.config_path,
            data_path=self.data_dir_select.folder_path,
            data_type=self.data_type_dropdown.getChoices(),
        )
        threading.Thread(target=data_extractor.run()).start()


# _ = DataExtractorPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
