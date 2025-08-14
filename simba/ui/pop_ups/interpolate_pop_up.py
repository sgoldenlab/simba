__author__ = "Simon Nilsson"

import os
from pathlib import Path
from tkinter import *
from typing import Union

from simba.data_processors.interpolate import Interpolate
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (DropDownMenu, FileSelect, FolderSelect,
                                        SimbaButton)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.enums import Formats, Options
from simba.utils.read_write import str_2_bool

INTERPOLATOR_METHOD = {'MISSING BODY-PARTS': 'body-parts', 'MISSING ANIMALS': 'animals'}

class InterpolatePopUp(PopUpMixin, ConfigReader):
    """
    :example:
    >>> InterpolatePopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')

    """

    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="INTERPOLATE POSE", icon='line_chart_blue')
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.config_path = config_path
        self.settings_frm = LabelFrame(self.main_frm, text="SETTINGS", font=Formats.FONT_HEADER.value)
        self.type_dropdown = DropDownMenu(self.settings_frm, "INTERPOLATION TYPE:", ['MISSING BODY-PARTS', 'MISSING ANIMALS'], "35")
        self.method_dropdown = DropDownMenu(self.settings_frm, "INTERPOLATION METHOD:", ['NEAREST', 'LINEAR', 'QUADRATIC'], "35")
        self.save_originals_dropdown = DropDownMenu(self.settings_frm, "SAVE ORIGINALS:", Options.BOOL_STR_OPTIONS.value, "35")
        self.type_dropdown.setChoices('MISSING BODY-PARTS')
        self.method_dropdown.setChoices('NEAREST')
        self.save_originals_dropdown.setChoices(Options.BOOL_STR_OPTIONS.value[0])

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.type_dropdown.grid(row=0, column=0, sticky=NW)
        self.method_dropdown.grid(row=1, column=0, sticky=NW)
        self.save_originals_dropdown.grid(row=2, column=0, sticky=NW)

        self.single_file_frm = LabelFrame(self.main_frm, text="INTERPOLATE SINGLE DATA FILE", font=Formats.FONT_HEADER.value)
        self.selected_file = FileSelect(self.single_file_frm, "DATA PATH:", lblwidth=35, file_types=[("VIDEO FILE", ".csv .parquet")], initialdir=self.project_path)
        self.run_btn_single = SimbaButton(parent=self.single_file_frm, txt="RUN SINGLE DATA FILE INTERPOLATION", img='rocket', txt_clr="blue", font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'multiple': False})

        self.single_file_frm.grid(row=1, column=0, sticky=NW)
        self.selected_file.grid(row=0, column=0, sticky=NW)
        self.run_btn_single.grid(row=1, column=0, sticky=NW)

        self.multiple_file_frm = LabelFrame(self.main_frm, text="INTERPOLATE DIRECTORY OF DATA", font=Formats.FONT_HEADER.value)
        self.selected_dir = FolderSelect(self.multiple_file_frm, "SELECT DIRECTORY OF DATA FILES:", lblwidth=35, initialdir=self.project_path)


        self.run_btn_multiple = SimbaButton(parent=self.multiple_file_frm, txt="RUN DATA DIRECTORY INTERPOLATION", img='rocket', txt_clr="blue", font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'multiple': True})
        self.multiple_file_frm.grid(row=2, column=0, sticky=NW)
        self.selected_dir.grid(row=0, column=0, sticky=NW)
        self.run_btn_multiple.grid(row=1, column=0, sticky=NW)
        self.main_frm.mainloop()

    def run(self, multiple):
        interpolation_type = INTERPOLATOR_METHOD[self.type_dropdown.getChoices()]
        interpolation_method = self.method_dropdown.getChoices().lower()
        copy_originals = str_2_bool(self.save_originals_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_file.file_path
            check_file_exist_and_readable(file_path=data_path)
            data_dir = Path(os.path.dirname(data_path))
        else:
            data_path = self.selected_dir.folder_path
            check_if_dir_exists(in_dir=data_path)
            data_dir = Path(data_path)

        multi_index_df_headers = False
        if data_dir.resolve().absolute() == Path(self.input_csv_dir).resolve().absolute():
            multi_index_df_headers = True
        interpolator = Interpolate(config_path=self.config_path,
                                   data_path=data_path,
                                   type=interpolation_type,
                                   method=interpolation_method,
                                   copy_originals=copy_originals,
                                   multi_index_df_headers=multi_index_df_headers)
        interpolator.run()


#InterpolatePopUp(config_path=r"C:\troubleshooting\my_project_tutorial\project_folder\project_config.ini")