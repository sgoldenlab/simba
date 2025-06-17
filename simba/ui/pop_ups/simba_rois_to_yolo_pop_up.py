__author__ = "Simon Nilsson"

import threading
from tkinter import *

import numpy as np

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.converters import simba_rois_to_yolo
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        FolderSelect, SimBADropDown)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.read_write import str_2_bool

train_size_options = np.arange(10, 110, 0.1)

class SimBAROIs2YOLOPopUp(PopUpMixin):

    """
    :example:
    >>> SimBAROIs2YOLOPopUp()
    """
    def __init__(self):
        super().__init__(title="SIMBA ROIS TO YOLO")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        self.roi_path = FileSelect(parent=settings_frm, fileDescription='ROI PATH (h5): ', lblwidth=35, file_types=[("H5 FILE", (".h5", ".H5",))], entry_width=40)
        self.video_dir = FolderSelect(settings_frm, folderDescription="VIDEO DIRECTORY:", lblwidth=35, width=40)
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35, width=40)
        self.greyscale_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GREYSCALE: ", label_width=35, dropdown_width=40, value='FALSE')
        self.obb_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="OBB: ", label_width=35, dropdown_width=40, value='TRUE')
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE: ", label_width=35, dropdown_width=40, value='TRUE')
        self.frm_cnt_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=list(range(50, 650, 50)), label="FRAM COUNT (PER VIDEO): ", label_width=35, dropdown_width=40, value=100)
        self.train_size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=list(range(10, 110, 10)), label="TRAIN SIZE (%): ", label_width=35, dropdown_width=40, value=70)

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.roi_path.grid(row=0, column=0, sticky=NW)
        self.video_dir.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=2, column=0, sticky=NW)
        self.greyscale_dropdown.grid(row=3, column=0, sticky=NW)
        self.obb_dropdown.grid(row=4, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=5, column=0, sticky=NW)
        self.frm_cnt_dropdown.grid(row=6, column=0, sticky=NW)
        self.train_size_dropdown.grid(row=7, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        roi_path = self.roi_path.file_path
        video_dir = self.video_dir.folder_path
        save_dir = self.save_dir.folder_path
        greyscale = str_2_bool(self.greyscale_dropdown.get_value())
        obb = str_2_bool(self.obb_dropdown.get_value())
        verbose = str_2_bool(self.verbose_dropdown.get_value())
        frm_cnt = int(self.frm_cnt_dropdown.get_value())
        train_size = int(self.train_size_dropdown.get_value()) / 100

        check_if_dir_exists(in_dir=video_dir, source=f'{self.__class__.__name__} VIDEO DIRECTORY')
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} SAVE DIRECTORY')
        check_file_exist_and_readable(file_path=roi_path)

        thread = threading.Thread(target=simba_rois_to_yolo, kwargs={'roi_path': roi_path, 'video_dir': video_dir, 'save_dir': save_dir, 'greyscale': greyscale, 'obb': obb, 'roi_frm_cnt': frm_cnt, 'verbose': verbose, 'train_size': train_size})
        thread.start()



