__author__ = "Simon Nilsson"

from tkinter import *

import numpy as np
import pandas as pd

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.transform.dlc_ma_h5_to_yolo import \
    MADLCH52Yolo
from simba.third_party_label_appenders.transform.utils import \
    get_yolo_keypoint_flip_idx
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FolderSelect,
                                        SimbaCheckbox, SimBADropDown)
from simba.utils.checks import check_if_dir_exists, check_valid_dataframe
from simba.utils.enums import Formats, Options
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    str_2_bool)

TRAIN_SIZE_OPTIONS = list(np.arange(10, 110, 10))
SAMPLE_SIZE_OPTIONS = list(np.arange(50, 650, 50))

PADDING_OPTIONS = list(np.round(np.arange(0.01, 10.05, 0.05),2).astype(str))
PADDING_OPTIONS = np.insert(PADDING_OPTIONS, 0, 'None')

THRESHOLD_OPTION = list(range(10, 110, 10))

class DLCH5Inference2YoloPopUp(PopUpMixin):

    """
    GUI for converting DeepLabCut (DLC) annotations to YOLO keypoint format.

    :example:
    >>> DLCH5Inference2YoloPopUp()
    """

    def __init__(self):
        PopUpMixin.__init__(self, title="DLC MULTI-ANIMAL H5 PREDICTIONS TO YOLO", icon='dlc_1')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        self.dlc_dir = FolderSelect(settings_frm, folderDescription="DLC DATA DIRECTORY (H5):", lblwidth=35, entry_width=40, initialdir=r'D:\troubleshooting\dlc_h5_multianimal_to_yolo\data')
        self.video_dir = FolderSelect(settings_frm, folderDescription="VIDEO DIRECTORY:", lblwidth=35, entry_width=40, initialdir=r'D:\troubleshooting\dlc_h5_multianimal_to_yolo\videos')
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35, entry_width=40, initialdir='D:\troubleshooting\dlc_h5_multianimal_to_yolo\yolo')

        self.frm_cnt_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=SAMPLE_SIZE_OPTIONS, label="FRAMES (PER VIDEO): ", label_width=35, dropdown_width=40, value=100)
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE: ", label_width=35, dropdown_width=40, value='TRUE')
        self.threshold_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=THRESHOLD_OPTION, label="THRESHOLD (%): ", label_width=35, dropdown_width=40, value=90)
        self.train_size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=TRAIN_SIZE_OPTIONS, label="TRAIN SIZE (%): ", label_width=35, dropdown_width=40, value=70)
        self.grey_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GREYSCALE: ", label_width=35, dropdown_width=40, value='FALSE')
        self.clahe_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="CLAHE: ", label_width=35, dropdown_width=40, value='FALSE')
        self.padding_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=PADDING_OPTIONS, label="PADDING: ", label_width=35, dropdown_width=40, value='None')
        single_id_cb, self.single_id_var = SimbaCheckbox(parent=settings_frm, txt="REMOVE ANIMAL ID'S", val=False)

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.dlc_dir .grid(row=0, column=0, sticky=NW)
        self.video_dir.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=2, column=0, sticky=NW)

        self.frm_cnt_dropdown.grid(row=3, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=4, column=0, sticky=NW)
        self.threshold_dropdown.grid(row=5, column=0, sticky=NW)
        self.train_size_dropdown.grid(row=6, column=0, sticky=NW)
        self.grey_dropdown.grid(row=7, column=0, sticky=NW)
        self.clahe_dropdown.grid(row=8, column=0, sticky=NW)
        self.padding_dropdown.grid(row=9, column=0, sticky=NW)
        single_id_cb.grid(row=10, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        dlc_dir = self.dlc_dir.folder_path
        video_dir = self.video_dir.folder_path
        save_dir = self.save_dir.folder_path

        check_if_dir_exists(in_dir=dlc_dir, source=f'{self.__class__.__name__} SLEAP DATA DIRECTORY', raise_error=True)
        check_if_dir_exists(in_dir=video_dir, source=f'{self.__class__.__name__} VIDEO DIRECTORY', raise_error=True)
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} SAVE DIRECTORY', raise_error=True)
        h5_files = find_files_of_filetypes_in_directory(directory=dlc_dir, extensions=['.h5', '.H5'], raise_error=True)
        video_files = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)

        grey = str_2_bool(self.grey_dropdown.get_value())
        clahe = str_2_bool(self.clahe_dropdown.get_value())
        train_size = int(self.train_size_dropdown.get_value()) / 100
        verbose = str_2_bool(self.verbose_dropdown.get_value())
        frm_cnt = int(self.frm_cnt_dropdown.get_value())
        padding = float(self.padding_dropdown.get_value()) if self.padding_dropdown.get_value() != 'None' else 0.0
        threshold = float(self.threshold_dropdown.get_value()) / 100
        single_id = 'animal_1' if self.single_id_var.get() else None

        data_df = pd.DataFrame(pd.read_hdf(h5_files[0], header=[0, 1, 2, 3]))
        check_valid_dataframe(df=data_df, source=h5_files[0], valid_dtypes=Formats.NUMERIC_DTYPES.value)
        animal_names = tuple(list(set([x[1] for x in list(data_df.columns)])))
        bp_names = [x[2] for x in data_df.columns if x[1] == animal_names[0]]
        bp_names = list(dict.fromkeys([x for x in bp_names]))
        flip_idx = get_yolo_keypoint_flip_idx(x=bp_names)

        runner = MADLCH52Yolo(data_dir=dlc_dir,
                              video_dir=video_dir,
                              save_dir=save_dir,
                              frms_cnt=frm_cnt,
                              verbose=verbose,
                              threshold=threshold,
                              train_size=train_size,
                              flip_idx=flip_idx,
                              greyscale=grey,
                              clahe=clahe,
                              padding=padding,
                              single_id=single_id)
        runner.run()

#DLCH5Inference2YoloPopUp()