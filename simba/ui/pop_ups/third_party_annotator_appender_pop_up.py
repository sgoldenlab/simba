__author__ = "Simon Nilsson"

import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.third_party_appender import \
    ThirdPartyLabelAppender
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        FolderSelect)
from simba.utils.checks import check_if_dir_exists
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.lookups import get_third_party_appender_file_formats


class ThirdPartyAnnotatorAppenderPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="APPEND THIRD-PARTY ANNOTATIONS")
        ConfigReader.__init__(self, config_path=config_path)
        apps_lst = Options.THIRD_PARTY_ANNOTATION_APPS_OPTIONS.value
        warnings_lst = Options.THIRD_PARTY_ANNOTATION_ERROR_OPTIONS.value
        app_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="THIRD-PARTY APPLICATION", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.THIRD_PARTY_ANNOTATION_NEW.value)
        self.app_dropdown = DropDownMenu(app_frm, "THIRD-PARTY APPLICATION:", apps_lst, "35")
        self.app_dropdown.setChoices(apps_lst[0])
        app_frm.grid(row=0, column=0, sticky=NW)
        self.app_dropdown.grid(row=0, column=0, sticky=NW)
        select_data_frm = LabelFrame(self.main_frm, text="SELECT DATA", font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.data_folder = FolderSelect(select_data_frm, "DATA DIRECTORY:", lblwidth=35, initialdir=self.project_path)
        select_data_frm.grid(row=1, column=0, sticky=NW)
        self.data_folder.grid(row=0, column=0, sticky=NW)

        self.error_dropdown_dict = self.create_dropdown_frame( main_frm=self.main_frm, drop_down_titles=warnings_lst, drop_down_options=["WARNING", "ERROR"], frm_title="WARNINGS AND ERRORS")
        log_frm = LabelFrame(self.main_frm, text="LOGGING", font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.log_var = BooleanVar(value=True)
        self.log_cb = Checkbutton(log_frm, text="CREATE IMPORT LOG", variable=self.log_var)
        log_frm.grid(row=5, column=0, sticky=NW)
        self.log_cb.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        log = self.log_var.get()
        file_format = get_third_party_appender_file_formats()[self.app_dropdown.getChoices()]
        file_format = f'.{file_format}'
        app = self.app_dropdown.getChoices()
        error_settings = {}
        for error_name, error_dropdown in self.error_dropdown_dict.items():
            error_settings[error_name] = error_dropdown.getChoices()
        data_dir = self.data_folder.folder_path
        check_if_dir_exists(in_dir=self.data_folder.folder_path)
        third_party_importer = ThirdPartyLabelAppender(config_path=self.config_path,
                                                       data_dir=data_dir,
                                                       app=app,
                                                       file_format=file_format,
                                                       error_settings=error_settings,
                                                       log=log)

        threading.Thread(target=third_party_importer.run()).start()


#_ = ThirdPartyAnnotatorAppenderPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')