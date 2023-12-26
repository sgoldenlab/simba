__author__ = "Simon Nilsson"

from tkinter import *
import os

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.enums import Options
from simba.ui.tkinter_functions import DropDownMenu, Entry_Box, FolderSelect
from simba.data_processors.interpolation_smoothing import Interpolate, Smooth
from simba.utils.checks import check_int
from simba.utils.errors import NotDirectoryError

class InterpolatePopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='INTERPOLATE POSE')
        ConfigReader.__init__(self, config_path=config_path)
        self.input_dir = FolderSelect(self.main_frm, 'DATA DIRECTORY:', lblwidth=25)
        self.method_dropdown = DropDownMenu(self.main_frm, 'INTERPOLATION METHOD:', Options.INTERPOLATION_OPTIONS.value, '25')
        self.method_dropdown.setChoices(Options.INTERPOLATION_OPTIONS.value[0])
        run_btn = Button(self.main_frm, text='RUN INTERPOLATION', fg='blue', command=lambda: self.run())
        self.input_dir.grid(row=0, column=0, sticky=NW)
        self.method_dropdown.grid(row=1, column=0, sticky=NW)
        run_btn.grid(row=2, column=0, sticky=NW)

    def run(self):
        if not os.path.isdir(self.input_dir.folder_path):
            raise NotDirectoryError(msg=f'{self.input_dir.folder_path} is not a valid directory.', source=self.__class__.__name__)
        Interpolate(config_path=self.config_path,
                    method=self.method_dropdown.getChoices(),
                    input_path=self.input_dir.folder_path)


class SmoothingPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='SMOOTH POSE')
        ConfigReader.__init__(self, config_path=config_path)
        self.input_dir = FolderSelect(self.main_frm, 'DATA DIRECTORY:', lblwidth=20)
        self.time_window = Entry_Box(self.main_frm, 'TIME WINDOW (MS):', '20', validation='numeric')
        self.method_dropdown = DropDownMenu(self.main_frm, 'SMOOTHING METHOD:', Options.SMOOTHING_OPTIONS.value, '20')
        self.method_dropdown.setChoices(Options.SMOOTHING_OPTIONS.value[0])
        run_btn = Button(self.main_frm, text='RUN SMOOTHING', fg='blue', command=lambda: self.run())

        self.input_dir.grid(row=0, column=0, sticky=NW)
        self.method_dropdown.grid(row=1, column=0, sticky=NW)
        self.time_window.grid(row=2, column=0, sticky=NW)
        run_btn.grid(row=3, column=0, sticky=NW)

    def run(self):
        if not os.path.isdir(self.input_dir.folder_path):
            raise NotDirectoryError(msg=f'{self.input_dir.folder_path} is not a valid directory.', source=self.__class__.__name__)
        check_int(name='TIME WINDOW', value=self.time_window.entry_get, min_value=1)
        _ = Smooth(config_path=self.config_path,
                   input_path=self.input_dir.folder_path,
                   time_window=self.time_window.entry_get,
                   smoothing_method=self.method_dropdown.getChoices())
