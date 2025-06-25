__author__ = "Simon Nilsson"

from tkinter import *

import numpy as np

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.transform.sleap_csv_to_yolo import \
    Sleap2Yolo
from simba.third_party_label_appenders.transform.sleap_to_yolo import \
    SleapAnnotations2Yolo
from simba.third_party_label_appenders.transform.utils import \
    get_yolo_keypoint_flip_idx
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FolderSelect,
                                        SimbaCheckbox, SimBADropDown)
from simba.utils.checks import check_if_dir_exists
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    str_2_bool)

TRAIN_SIZE_OPTIONS = list(np.arange(10, 110, 10))
SAMPLE_SIZE_OPTIONS = list(np.arange(50, 650, 50))

PADDING_OPTIONS = list(np.round(np.arange(0.01, 10.05, 0.05),2).astype(str))
PADDING_OPTIONS = list(np.insert(PADDING_OPTIONS, 0, 'None'))



class SLEAPAnnotations2YoloPopUp(PopUpMixin):

    """
    GUI for converting DeepLabCut (DLC) annotations to YOLOv8 keypoint format.

    :example:
    >>> SLEAPAnnotations2YoloPopUp()
    """

    def __init__(self):
        PopUpMixin.__init__(self, title="SLEAP ANNOTATIONS TO YOLO POSE ESTIMATION ANNOTATIONS", icon='sleap_small')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        self.sleap_dir = FolderSelect(settings_frm, folderDescription="SLEAP DATA DIRECTORY (.SLP):", lblwidth=35, entry_width=40, initialdir=r"D:\troubleshooting\two_animals_sleap\import_data")
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35, entry_width=40, initialdir=r"D:\troubleshooting\two_animals_sleap\yolo_kpts_2")

        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE: ", label_width=35, dropdown_width=40, value='TRUE')
        self.train_size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=TRAIN_SIZE_OPTIONS, label="TRAIN SIZE (%): ", label_width=35, dropdown_width=40, value=70)
        self.grey_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GREYSCALE: ", label_width=35, dropdown_width=40, value='FALSE')
        self.padding_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=PADDING_OPTIONS, label="PADDING: ", label_width=35, dropdown_width=40, value='None')
        self.clahe_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="CLAHE: ", label_width=35, dropdown_width=40, value='FALSE')
        single_id_cb, self.single_id_var = SimbaCheckbox(parent=settings_frm, txt="REMOVE ANIMAL ID'S", val=False)


        settings_frm.grid(row=0, column=0, sticky=NW)
        self.sleap_dir .grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)

        self.verbose_dropdown.grid(row=2, column=0, sticky=NW)
        self.train_size_dropdown.grid(row=3, column=0, sticky=NW)
        self.grey_dropdown.grid(row=4, column=0, sticky=NW)
        self.clahe_dropdown.grid(row=5, column=0, sticky=NW)
        self.padding_dropdown.grid(row=6, column=0, sticky=NW)
        single_id_cb.grid(row=7, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()


    def run(self):
        sleap_dir = self.sleap_dir.folder_path
        save_dir = self.save_dir.folder_path

        check_if_dir_exists(in_dir=sleap_dir, source=f'{self.__class__.__name__} SLEAP DATA DIRECTORY', raise_error=True)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} SAVE DIRECTORY', raise_error=True)
        _ = find_files_of_filetypes_in_directory(directory=sleap_dir, extensions=['.slp'], raise_error=True)

        grey = str_2_bool(self.grey_dropdown.get_value())
        train_size = int(self.train_size_dropdown.get_value()) / 100
        verbose = str_2_bool(self.verbose_dropdown.get_value())
        clahe = str_2_bool(self.clahe_dropdown.get_value())
        padding = float(self.padding_dropdown.get_value()) if self.padding_dropdown.get_value() != 'None' else 0.0
        single_id = 'animal_1' if self.single_id_var.get() else None

        runner = SleapAnnotations2Yolo(sleap_dir=sleap_dir, save_dir=save_dir, padding=padding, train_size=train_size, verbose=verbose, greyscale=grey, clahe=clahe, single_id=single_id)
        runner.run()


#SLEAPAnnotations2YoloPopUp()

