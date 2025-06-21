__author__ = "Simon Nilsson"

from tkinter import *

import numpy as np

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.transform.dlc_to_labelme import \
    DLC2Labelme
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FolderSelect,
                                        SimBADropDown)
from simba.utils.checks import check_if_dir_exists
from simba.utils.read_write import str_2_bool


class DLC2LabelmePopUp(PopUpMixin):

    """
    GUI for converting DeepLabCut (DLC) annotations to YOLO keypoint format.

    :example:
    >>> DLC2LabelmePopUp()
    """

    def __init__(self):
        PopUpMixin.__init__(self, title="DLC MULTI-ANIMAL H5 PREDICTIONS TO YOLO", icon='dlc_1')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        self.dlc_dir = FolderSelect(settings_frm, folderDescription="DLC ANNOTATION DIRECTORY:", lblwidth=35, entry_width=40, initialdir=r'D:\troubleshooting\dlc_h5_multianimal_to_yolo\data')
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35, entry_width=40, initialdir='D:\troubleshooting\dlc_h5_multianimal_to_yolo\yolo')
        self.grey_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GREYSCALE: ", label_width=35, dropdown_width=40, value='FALSE')
        self.clahe_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="CLAHE: ", label_width=35, dropdown_width=40, value='FALSE')
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE: ", label_width=35, dropdown_width=40, value='TRUE')


        settings_frm.grid(row=0, column=0, sticky=NW)
        self.dlc_dir .grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)
        self.grey_dropdown.grid(row=2, column=0, sticky=NW)
        self.clahe_dropdown.grid(row=3, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=4, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()


    def run(self):
        dlc_dir = self.dlc_dir.folder_path
        save_dir = self.save_dir.folder_path

        grey = str_2_bool(self.grey_dropdown.get_value())
        clahe = str_2_bool(self.clahe_dropdown.get_value())
        verbose = str_2_bool(self.verbose_dropdown.get_value())
        check_if_dir_exists(in_dir=dlc_dir, source=f'{self.__class__.__name__} DLC ANNOTATION DIRECTORY', raise_error=True)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} SAVE DIRECTORY', raise_error=True)

        runner = DLC2Labelme(dlc_dir=dlc_dir, save_dir=save_dir, verbose=verbose, greyscale=grey, clahe=clahe)
        runner.run()


#DLC2LabelmePopUp()