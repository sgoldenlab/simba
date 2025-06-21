__author__ = "Simon Nilsson"

import threading
from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.converters import labelme_to_df
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FolderSelect,
                                        SimBADropDown)
from simba.utils.checks import check_if_dir_exists
from simba.utils.read_write import str_2_bool


class LabelmeDirectory2CSVPopUp(PopUpMixin):

    """
    :example:
    >>> LabelmeDirectory2CSVPopUp()
    """
    def __init__(self):
        super().__init__(title="LABELME DIRECTORY TO CSV")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        self.labelme_dir = FolderSelect(settings_frm, folderDescription="LABELME DIRECTORY:", lblwidth=35)
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35)
        self.greyscale_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GREYSCALE: ", label_width=35, dropdown_width=40, value='FALSE')
        self.pad_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="PAD IMAGES: ", label_width=35, dropdown_width=40, value='FALSE')
        self.size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['FALSE', 'MINIMUM', 'MAXIMUM'], label="RESIZE IMAGES: ", label_width=35, dropdown_width=40, value='FALSE')
        self.normalize_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'],  label="NORMALIZE IMAGES: ", label_width=35, dropdown_width=40, value='FALSE')
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'],  label="VERBOSE: ", label_width=35, dropdown_width=40, value='TRUE')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.labelme_dir.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)
        self.greyscale_dropdown.grid(row=2, column=0, sticky=NW)
        self.pad_dropdown.grid(row=3, column=0, sticky=NW)
        self.size_dropdown.grid(row=4, column=0, sticky=NW)
        self.normalize_dropdown.grid(row=5, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=6, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        labelme_dir = self.labelme_dir.folder_path
        save_path = self.save_dir.folder_path
        greyscale = str_2_bool(self.greyscale_dropdown.get_value())
        pad = str_2_bool(self.pad_dropdown.get_value())
        normalize = str_2_bool(self.normalize_dropdown.get_value())
        verbose = str_2_bool(self.verbose_dropdown.get_value())

        size = self.size_dropdown.get_value()
        size = None if size == 'FALSE' else 'min' if size == 'MINIMUM' else 'max' if size == 'MAXIMUM' else size
        check_if_dir_exists(in_dir=labelme_dir, source=f'{self.__class__.__name__} LABELME DIRECTORY')
        check_if_dir_exists(in_dir=labelme_dir, source=f'{self.__class__.__name__} SAVE DIRECTORY')
        thread = threading.Thread(target=labelme_to_df, kwargs={'labelme_dir': labelme_dir, 'save_path': save_path, 'greyscale': greyscale, 'pad': pad, 'normalize': normalize, 'size': size, 'verbose': verbose})
        thread.start()