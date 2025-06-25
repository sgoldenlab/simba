__author__ = "Simon Nilsson"

import threading
from tkinter import *

import numpy as np

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.transform.coco_keypoints_to_yolo import \
    COCOKeypoints2Yolo
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        FolderSelect, SimBADropDown)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.enums import Options
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    str_2_bool)

TRAIN_SIZE_OPTIONS = list(np.arange(10, 110, 10))

class COCOKeypoints2YOLOkeypointsPopUp(PopUpMixin):

    def __init__(self):
        PopUpMixin.__init__(self, title="COCO KEYPOINTS TO YOLO KEYPOINTS", icon='coco_small')

        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        self.coco_file_path = FileSelect(parent=settings_frm, fileDescription='COCO FILE (JSON):', lblwidth=30, entry_width=45)
        self.img_dir = FolderSelect(settings_frm, folderDescription="IMAGE DIRECTORY:", lblwidth=35)
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35)
        self.train_size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=TRAIN_SIZE_OPTIONS, label="TRAIN SIZE (%): ", label_width=35, dropdown_width=40, value=70)
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE:", label_width=35, dropdown_width=40, value='TRUE')
        self.greyscale_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GREYSCALE:", label_width=35, dropdown_width=40, value='FALSE')
        self.clahe_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="CLAHE:", label_width=35, dropdown_width=40, value='FALSE')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.coco_file_path.grid(row=0, column=0, sticky=NW)
        self.img_dir.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=2, column=0, sticky=NW)
        self.train_size_dropdown.grid(row=3, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=4, column=0, sticky=NW)
        self.clahe_dropdown.grid(row=5, column=0, sticky=NW)
        self.greyscale_dropdown.grid(row=6, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()


    def run(self):

        save_dir = self.save_dir.folder_path
        img_dir = self.img_dir.folder_path
        coco_file_path = self.coco_file_path.file_path
        train_size = float(self.train_size_dropdown.get_value()) / 100
        verbose = str_2_bool(self.verbose_dropdown.get_value())
        clahe = str_2_bool(self.clahe_dropdown.get_value())
        grey = str_2_bool(self.greyscale_dropdown.get_value())

        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} SAVE DIRECTORY')
        check_if_dir_exists(in_dir=img_dir, source=f'{self.__class__.__name__} IMAGE DIRECTORY')

        check_file_exist_and_readable(file_path=coco_file_path)
        _ = find_files_of_filetypes_in_directory(directory=img_dir, extensions=Options.ALL_IMAGE_FORMAT_OPTIONS.value, raise_error=True)

        runner = COCOKeypoints2Yolo(coco_path=coco_file_path, img_dir=img_dir, save_dir=save_dir, train_size=train_size, verbose=verbose, greyscale=grey, clahe=clahe)
        runner.run()