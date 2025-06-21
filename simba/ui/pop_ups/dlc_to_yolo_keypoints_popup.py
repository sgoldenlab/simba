__author__ = "Simon Nilsson"

from tkinter import *

import numpy as np
import pandas as pd

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.transform.dlc_to_yolo import DLC2Yolo
from simba.third_party_label_appenders.transform.utils import (
    get_yolo_keypoint_bp_id_idx, get_yolo_keypoint_flip_idx)
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FolderSelect,
                                        SimBADropDown)
from simba.utils.checks import check_if_dir_exists
from simba.utils.read_write import recursive_file_search, str_2_bool

TRAIN_SIZE_OPTIONS = np.arange(10, 110, 10)
SAMPLE_SIZE_OPTIONS = list(np.arange(50, 650, 50))

PADDING_OPTIONS = list(np.round(np.arange(0.01, 10.05, 0.02),2).astype(str))
PADDING_OPTIONS = list(np.insert(PADDING_OPTIONS, 0, 'None'))

class DLCYoloKeypointsPopUp(PopUpMixin):

    """
    GUI for converting DeepLabCut (DLC) annotations to YOLOv8 keypoint format.

    :example:
    >>> DLCYoloKeypointsPopUp()
    """
    def __init__(self):

        PopUpMixin.__init__(self, title="DLC ANNOTATIONS TO YOLO KEY-KEYPOINTS", icon='dlc_2')
        #super().__init__(title="DLC ANNOTATIONS TO YOLO key-KEYPOINTS")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        self.dlc_dir = FolderSelect(settings_frm, folderDescription="DLC DATA DIRECTORY:", lblwidth=35, entry_width=40, initialdir=r'D:\rat_resident_intruder\dlc_data') #r'D:\rat_resident_intruder\dlc_data'
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35, entry_width=40, initialdir=r'D:\troubleshooting\dlc_h5_multianimal_to_yolo\yolo') #r'D:\rat_resident_intruder\yolo_1'
        self.train_size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=TRAIN_SIZE_OPTIONS, label="TRAIN SIZE (%): ", label_width=35, dropdown_width=40, value=70)
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE: ", label_width=35, dropdown_width=40, value='TRUE')
        self.padding_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=PADDING_OPTIONS, label="PADDING: ", label_width=35, dropdown_width=40, value='None')
        self.grey_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GREYSCALE: ", label_width=35, dropdown_width=40, value='FALSE')
        self.clahe_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="CLAHE: ", label_width=35, dropdown_width=40, value='FALSE')
        self.animal_cnt_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=list(range(1, 10, 1)), label="ANIMAL COUNT: ", label_width=35, dropdown_width=40, value=1)

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.dlc_dir.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)
        self.train_size_dropdown.grid(row=2, column=0, sticky=NW)

        self.verbose_dropdown.grid(row=3, column=0, sticky=NW)
        self.padding_dropdown.grid(row=4, column=0, sticky=NW)
        self.grey_dropdown.grid(row=6, column=0, sticky=NW)
        self.clahe_dropdown.grid(row=7, column=0, sticky=NW)
        self.animal_cnt_dropdown.grid(row=8, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()


    def run(self):
        dlc_dir = self.dlc_dir.folder_path
        save_dir = self.save_dir.folder_path
        animal_cnt = int(self.animal_cnt_dropdown.get_value())
        check_if_dir_exists(in_dir=dlc_dir, source=f'{self.__class__.__name__} DLC DATA DIRECTORY', raise_error=True)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} SAVE DIRECTORY', raise_error=True)
        padding = float(self.padding_dropdown.get_value()) if self.padding_dropdown.get_value() != 'None' else 0.0
        grey = str_2_bool(self.grey_dropdown.get_value())
        clahe = str_2_bool(self.clahe_dropdown.get_value())
        train_size = int(self.train_size_dropdown.get_value()) / 100
        verbose = str_2_bool(self.verbose_dropdown.get_value())

        annotation_paths = recursive_file_search(directory=dlc_dir, substrings=['CollectedData'], extensions=['csv'], case_sensitive=False, raise_error=True)
        animal_bp_col_cnt = int(int((len(pd.read_csv(annotation_paths[0], header=[0, 1, 2]).columns) / 2) - 1) / animal_cnt)
        animal_bp_names = list(pd.read_csv(annotation_paths[0], header=[0, 1, 2]).columns)
        animal_bp_names = list(dict.fromkeys([x[1] for x in animal_bp_names[1:]]))
        flip_idx = get_yolo_keypoint_flip_idx(x=animal_bp_names[0:animal_bp_col_cnt+1])
        names = tuple([f'animal_{x+1}' for x in range(animal_cnt)])

        runner = DLC2Yolo(dlc_dir=dlc_dir, save_dir=save_dir, train_size=train_size, verbose=verbose, padding=padding, flip_idx=flip_idx, names=names, greyscale=grey, clahe=clahe)
        runner.run()


#DLCYoloKeypointsPopUp()