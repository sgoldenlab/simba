__author__ = "Simon Nilsson"

import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.third_party_appender import \
    ThirdPartyLabelAppender
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FolderSelect,
                                        SimbaCheckbox, SimBADropDown)
from simba.utils.checks import check_if_dir_exists
from simba.utils.enums import Defaults, Links, Methods, Options
from simba.utils.errors import NoDataError
from simba.utils.lookups import get_third_party_appender_file_formats


class ThirdPartyAnnotatorAppenderPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if len(self.feature_file_paths) == 0:
            raise NoDataError(msg=f'Cannot append third-party annotations: No data found in {self.features_dir} directory to append annotations too.', source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="APPEND THIRD-PARTY ANNOTATIONS", icon='draw')
        warnings_lst = Options.THIRD_PARTY_ANNOTATION_ERROR_OPTIONS.value
        app_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="THIRD-PARTY APPLICATION", icon_name='application', icon_link=Links.THIRD_PARTY_ANNOTATION_NEW.value, pady=5, padx=5, relief='solid')
        self.app_dropdown = SimBADropDown(parent=app_frm, dropdown_options=Options.THIRD_PARTY_ANNOTATION_APPS_OPTIONS.value, label="THIRD-PARTY APPLICATION:", label_width=50, dropdown_width=30, value='BORIS')
        app_frm.grid(row=0, column=0, sticky=NW, padx=10, pady=10)
        self.app_dropdown.grid(row=0, column=0, sticky=NW)

        select_data_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT DATA", icon_name='data_table', icon_link=Links.THIRD_PARTY_ANNOTATION_NEW.value, pady=5, padx=5, relief='solid')
        self.data_folder = FolderSelect(select_data_frm, "DATA DIRECTORY:", lblwidth=50, initialdir=self.project_path)
        select_data_frm.grid(row=1, column=0, sticky=NW, padx=10, pady=10)
        self.data_folder.grid(row=0, column=0, sticky=NW)

        warnings_and_errors_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="WARNINGS & ERRORS", icon_name='warning', icon_link=Links.THIRD_PARTY_ANNOTATION_NEW.value, pady=5, padx=5, relief='solid')

        self.invalid_data_file_format_dropdown = SimBADropDown(parent=warnings_and_errors_frm, dropdown_options=["WARNING", "ERROR"], label="INVALID annotations file data format", label_width=50, dropdown_width=30, value='WARNING')
        self.additional_third_party_behavior_dropdown = SimBADropDown(parent=warnings_and_errors_frm, dropdown_options=["WARNING", "ERROR"], label="ADDITIONAL third-party behavior detected", label_width=50, dropdown_width=30, value='WARNING')
        self.annot_overlap_conflict_dropdown = SimBADropDown(parent=warnings_and_errors_frm, dropdown_options=["WARNING", "ERROR"], label="Annotations OVERLAP conflict", label_width=50, dropdown_width=30, value='WARNING')
        self.zero_third_party_behavior_detected_dropdown = SimBADropDown(parent=warnings_and_errors_frm, dropdown_options=["WARNING", "ERROR"], label="ZERO third-party video behavior annotations found", label_width=50, dropdown_width=30, value='WARNING')
        self.annotation_pose_conflict_frm_cnt_dropdown = SimBADropDown(parent=warnings_and_errors_frm, dropdown_options=["WARNING", "ERROR"], label="Annotations and pose FRAME COUNT conflict", label_width=50, dropdown_width=30, value='WARNING')
        self.annotation_event_cnt_conflict = SimBADropDown(parent=warnings_and_errors_frm, dropdown_options=["WARNING", "ERROR"], label="Annotations EVENT COUNT conflict", label_width=50, dropdown_width=30, value='WARNING')
        self.data_file_not_found = SimBADropDown(parent=warnings_and_errors_frm, dropdown_options=["WARNING", "ERROR"], label="Annotations data file NOT FOUND", label_width=50, dropdown_width=30, value='WARNING')

        warnings_and_errors_frm.grid(row=2, column=0, sticky=NW, padx=10, pady=10)
        self.invalid_data_file_format_dropdown.grid(row=0, column=0, sticky=NW)
        self.additional_third_party_behavior_dropdown.grid(row=1, column=0, sticky=NW)
        self.annot_overlap_conflict_dropdown.grid(row=2, column=0, sticky=NW)
        self.zero_third_party_behavior_detected_dropdown.grid(row=3, column=0, sticky=NW)
        self.annotation_pose_conflict_frm_cnt_dropdown.grid(row=4, column=0, sticky=NW)
        self.annotation_event_cnt_conflict.grid(row=5, column=0, sticky=NW)
        self.data_file_not_found.grid(row=6, column=0, sticky=NW)

        log_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="LOGGING", icon_name='log', icon_link=Links.THIRD_PARTY_ANNOTATION_NEW.value, pady=5, padx=5, relief='solid')
        log_cb, self.log_var = SimbaCheckbox(parent=log_frm, txt='CREATE IMPORT LOG', val=False)
        log_frm.grid(row=3, column=0, sticky=NW, padx=10, pady=10)
        log_cb.grid(row=0, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        log = self.log_var.get()
        file_format = get_third_party_appender_file_formats()[self.app_dropdown.getChoices()]
        file_format = f'.{file_format}'
        app = self.app_dropdown.getChoices()
        data_dir = self.data_folder.folder_path
        check_if_dir_exists(in_dir=self.data_folder.folder_path)

        error_settings = {Methods.INVALID_THIRD_PARTY_APPENDER_FILE.value: self.invalid_data_file_format_dropdown.get_value(),
                          Methods.ADDITIONAL_THIRD_PARTY_CLFS.value: self.additional_third_party_behavior_dropdown.get_value(),
                          Methods.ZERO_THIRD_PARTY_VIDEO_ANNOTATIONS.value: self.zero_third_party_behavior_detected_dropdown.get_value(),
                          Methods.THIRD_PARTY_EVENT_COUNT_CONFLICT.value: self.annotation_event_cnt_conflict.get_value(),
                          Methods.THIRD_PARTY_EVENT_OVERLAP.value: self.annot_overlap_conflict_dropdown.get_value(),
                          Methods.ZERO_THIRD_PARTY_VIDEO_BEHAVIOR_ANNOTATIONS.value: self.zero_third_party_behavior_detected_dropdown.get_value(),
                          Methods.THIRD_PARTY_FRAME_COUNT_CONFLICT.value: self.annotation_pose_conflict_frm_cnt_dropdown.get_value(),
                          Methods.THIRD_PARTY_ANNOTATION_FILE_NOT_FOUND.value: self.data_file_not_found.get_value()}

        third_party_importer = ThirdPartyLabelAppender(config_path=self.config_path,
                                                       data_dir=data_dir,
                                                       app=app,
                                                       file_format=file_format,
                                                       error_settings=error_settings,
                                                       log=log)

        if self.cpu_cnt > Defaults.THREADSAFE_CORE_COUNT.value:
            third_party_importer.run()
        else:
            threading.Thread(target=third_party_importer.run()).start()


# _ = ThirdPartyAnnotatorAppenderPopUp(config_path=r"C:\troubleshooting\boris_test_3\project_folder\project_config.ini")

#_ = ThirdPartyAnnotatorAppenderPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')