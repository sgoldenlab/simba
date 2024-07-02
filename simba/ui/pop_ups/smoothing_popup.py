__author__ = "Simon Nilsson"

import os
from copy import deepcopy
from tkinter import *
from typing import Union

from simba.data_processors.smoothing import Smoothing
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (DropDownMenu, Entry_Box, FileSelect,
                                        FolderSelect)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists, check_int)
from simba.utils.enums import Formats, Options
from simba.utils.read_write import str_2_bool

SMOOTHING_OPTION = {'Savitzky Golay': "savitzky-golay", "Gaussian": "gaussian"}

class SmoothingPopUp(PopUpMixin, ConfigReader):

    """
    :example:
    >>> SmoothingPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
    """
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="SMOOTH POSE-ESTIMATION DATA")
        ConfigReader.__init__(self, config_path=config_path)
        self.config_path = config_path

        self.settings_frm = LabelFrame(self.main_frm, text="SETTINGS", font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.time_window = Entry_Box(self.settings_frm, "TIME WINDOW (MILLISECONDS):", "35", validation="numeric")
        self.method_dropdown = DropDownMenu(self.settings_frm, "SMOOTHING METHOD:", Options.SMOOTHING_OPTIONS.value, "35")
        self.save_originals_dropdown = DropDownMenu(self.settings_frm, "SAVE ORIGINALS:", Options.BOOL_STR_OPTIONS.value, "35")
        self.save_originals_dropdown.setChoices(Options.BOOL_STR_OPTIONS.value[0])

        self.method_dropdown.setChoices(Options.SMOOTHING_OPTIONS.value[0])
        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.time_window.grid(row=0, column=0, sticky=NW)
        self.method_dropdown.grid(row=1, column=0, sticky=NW)
        self.save_originals_dropdown.grid(row=2, column=0, sticky=NW)

        self.single_file_frm = LabelFrame(self.main_frm, text="SMOOTH SINGLE DATA FILE", font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.selected_file = FileSelect(self.single_file_frm, "DATA PATH:", lblwidth=35, file_types=[("VIDEO FILE", ".csv .parquet")], initialdir=self.project_path)
        self.run_btn_single = Button(self.single_file_frm, text="RUN SINGLE DATA FILE SMOOTHING", fg="blue", command=lambda: self.run(multiple=False))

        self.single_file_frm.grid(row=1, column=0, sticky=NW)
        self.selected_file.grid(row=0, column=0, sticky=NW)
        self.run_btn_single.grid(row=1, column=0, sticky=NW)

        self.multiple_file_frm = LabelFrame(self.main_frm, text="SMOOTH DIRECTORY OF DATA", font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.selected_dir = FolderSelect(self.multiple_file_frm, "SELECT DIRECTORY OF DATA FILES:", lblwidth=35, initialdir=self.project_path)
        self.run_btn_multiple = Button(self.multiple_file_frm, text="RUN SINGLE DATA FILE SMOOTHING", fg="blue", command=lambda: self.run(multiple=True))

        self.multiple_file_frm.grid(row=2, column=0, sticky=NW)
        self.selected_dir.grid(row=0, column=0, sticky=NW)
        self.run_btn_multiple.grid(row=1, column=0, sticky=NW)
        self.main_frm.mainloop()

    def run(self, multiple):
        smooth_time = self.time_window.entry_get
        smooth_method = SMOOTHING_OPTION[self.method_dropdown.getChoices()]
        copy_originals = str_2_bool(self.save_originals_dropdown.getChoices())
        check_int(name='TIME WINDOW (MILLISECONDS)', value=smooth_time, min_value=1)

        if not multiple:
            data_path = self.selected_file.file_path
            check_file_exist_and_readable(file_path=data_path)
            data_dir = os.path.dirname(data_path)
        else:
            data_path = self.selected_dir.folder_path
            check_if_dir_exists(in_dir=data_path)
            data_dir = deepcopy(data_path)
        multi_index_df_headers = False
        if data_dir == self.input_csv_dir: multi_index_df_headers = True

        smoothing = Smoothing(config_path=self.config_path,
                              data_path=data_path,
                              time_window=int(smooth_time),
                              method=smooth_method,
                              multi_index_df_headers=multi_index_df_headers,
                              copy_originals=copy_originals)
        smoothing.run()