__author__ = "Simon Nilsson"

import threading
from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.converters import dlc_to_labelme
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FolderSelect,
                                        SimBADropDown)
from simba.utils.checks import check_if_dir_exists
from simba.utils.read_write import str_2_bool


class DLCAnnotations2LabelMePopUp(PopUpMixin):

    """
    :example:
    >>> DLCAnnotations2LabelMePopUp()
    """
    def __init__(self):
        super().__init__(title="DLC ANNOTATIONS TO LABELME")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        self.dlc_dir = FolderSelect(settings_frm, folderDescription="DLC ANNOTATIONS DIRECTORY:", lblwidth=35)
        self.labelme_dir = FolderSelect(settings_frm, folderDescription="LABELME SAVE DIR:", lblwidth=35)
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE:", label_width=35, dropdown_width=40, value='TRUE')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.dlc_dir.grid(row=0, column=0, sticky=NW)
        self.labelme_dir.grid(row=1, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=2, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()


    def run(self):
        dlc_dir = self.dlc_dir.folder_path
        labelme_dir = self.labelme_dir.folder_path
        verbose = str_2_bool(self.verbose_dropdown.get_value())

        check_if_dir_exists(in_dir=dlc_dir, source=f'{self.__class__.__name__} DLC ANNOTATIONS DIRECTORY')
        check_if_dir_exists(in_dir=labelme_dir, source=f'{self.__class__.__name__} LABELME SAVE DIR')

        thread = threading.Thread(target=dlc_to_labelme, kwargs={'dlc_dir': dlc_dir, 'save_dir': labelme_dir, 'verbose': verbose})
        thread.start()


