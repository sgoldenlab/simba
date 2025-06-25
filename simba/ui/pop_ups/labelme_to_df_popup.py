__author__ = "Simon Nilsson"

import os
from datetime import datetime
from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.transform.labelme_to_df import \
    LabelMe2DataFrame
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FolderSelect,
                                        SimBADropDown)
from simba.utils.checks import check_if_dir_exists
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    str_2_bool)


class Labelme2DataFramePopUp(PopUpMixin):

    """
    :example:
    >>> Labelme2DataFramePopUp()
    """

    def __init__(self):
        PopUpMixin.__init__(self, title="LABELME JSONS TO DATAFRAME", icon='labelme')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        self.labelme_dir = FolderSelect(settings_frm, folderDescription="LABELME DIRECTORY (JSONS):", lblwidth=35, entry_width=40, initialdir=r"C:\troubleshooting\coco_data\labels\test_2",)
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY :", lblwidth=35, entry_width=40, initialdir=r"C:\troubleshooting\RAT_NOR")
        self.grey_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GREYSCALE: ", label_width=35, dropdown_width=40, value='FALSE')
        self.clahe_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="CLAHE: ", label_width=35, dropdown_width=40, value='FALSE')
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE: ", label_width=35, dropdown_width=40, value='TRUE')
        self.pad_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="PAD: ", label_width=35, dropdown_width=40, value='FALSE')
        self.normalize_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="NORMALIZE: ", label_width=35, dropdown_width=40, value='FALSE')
        self.size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['MIN', 'MAX', 'NONE'], label="SIZE: ", label_width=35, dropdown_width=40, value='MAX')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.labelme_dir .grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)
        self.grey_dropdown.grid(row=2, column=0, sticky=NW)
        self.clahe_dropdown.grid(row=3, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=4, column=0, sticky=NW)
        self.pad_dropdown.grid(row=5, column=0, sticky=NW)
        self.normalize_dropdown.grid(row=6, column=0, sticky=NW)
        self.size_dropdown.grid(row=7, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()


    def run(self):
        labelme_dir = self.labelme_dir.folder_path
        save_dir = self.save_dir.folder_path
        grey = str_2_bool(self.grey_dropdown.get_value())
        clahe = str_2_bool(self.clahe_dropdown.get_value())
        verbose = str_2_bool(self.verbose_dropdown.get_value())
        pad = str_2_bool(self.pad_dropdown.get_value())
        normalize = str_2_bool(self.normalize_dropdown.get_value())
        size = self.size_dropdown.get_value()
        size = None if size == 'NONE' else size.lower()
        _ = find_files_of_filetypes_in_directory(directory=labelme_dir, extensions=['.json'], raise_error=True)
        check_if_dir_exists(in_dir=labelme_dir, source=f'{self.__class__.__name__} LABELME DIRECTORY (JSONS)', raise_error=True)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} SAVE DIRECTORY', raise_error=True)

        save_path = os.path.join(save_dir, f'labelme_dataframe_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
        runner = LabelMe2DataFrame(labelme_dir=labelme_dir, greyscale=grey, pad=pad, size=size, normalize=normalize, save_path=save_path, verbose=verbose, clahe=clahe)
        runner.run()


#Labelme2DataFramePopUp()