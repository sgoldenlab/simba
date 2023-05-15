__author__ = "Simon Nilsson"

from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.enums import Options, Formats, Keys, Links
from simba.ui.tkinter_functions import DropDownMenu, FolderSelect, CreateLabelFrameWithIcon
from simba.utils.lookups import get_third_party_appender_file_formats
from simba.utils.checks import check_if_dir_exists
from simba.third_party_label_appenders.third_party_appender import ThirdPartyLabelAppender


class ThirdPartyAnnotatorAppenderPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self,  title='APPEND THIRD-PARTY ANNOTATIONS')
        ConfigReader.__init__(self, config_path=config_path)
        apps_lst = Options.THIRD_PARTY_ANNOTATION_APPS_OPTIONS.value
        warnings_lst = Options.THIRD_PARTY_ANNOTATION_ERROR_OPTIONS.value
        app_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='THIRD-PARTY APPLICATION', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.THIRD_PARTY_ANNOTATION_NEW.value)
        self.app_dropdown = DropDownMenu(app_frm, 'THIRD-PARTY APPLICATION:', apps_lst, '35')
        self.app_dropdown.setChoices(apps_lst[0])
        app_frm.grid(row=0, column=0, sticky=NW)
        self.app_dropdown.grid(row=0, column=0, sticky=NW)

        select_data_frm = LabelFrame(self.main_frm, text='SELECT DATA', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.data_folder = FolderSelect(select_data_frm, 'DATA DIRECTORY:', lblwidth=35)
        select_data_frm.grid(row=1, column=0, sticky=NW)
        self.data_folder.grid(row=0, column=0, sticky=NW)

        self.error_dropdown_dict = self.create_dropdown_frame(main_frm=self.main_frm, drop_down_titles=warnings_lst, drop_down_options=['WARNING', 'ERROR'], frm_title='WARNINGS AND ERRORS')
        log_frm = LabelFrame(self.main_frm, text='LOGGING', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.log_var = BooleanVar(value=True)
        self.log_cb = Checkbutton(log_frm, text='CREATE IMPORT LOG', variable=self.log_var)
        log_frm.grid(row=5, column=0, sticky=NW)
        self.log_cb.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        settings = {'log': self.log_var.get()}
        settings['file_format'] = get_third_party_appender_file_formats()[self.app_dropdown.getChoices()]
        settings['errors'], app_choice = {}, self.app_dropdown.getChoices()
        for error_name, error_dropdown in self.error_dropdown_dict.items():
            settings['errors'][error_name] = error_dropdown.getChoices()
        check_if_dir_exists(in_dir=self.data_folder.folder_path)

        third_party_importer = ThirdPartyLabelAppender(app=self.app_dropdown.getChoices(),
                                                       config_path=self.config_path,
                                                       data_dir=self.data_folder.folder_path,
                                                       settings=settings)
        third_party_importer.run()
