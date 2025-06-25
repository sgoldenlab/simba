__author__ = "Simon Nilsson"

from tkinter import *

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.transform.simba_to_yolo import \
    SimBA2Yolo
from simba.third_party_label_appenders.transform.utils import (
    get_yolo_keypoint_bp_id_idx, get_yolo_keypoint_flip_idx)
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        FolderSelect, SimBADropDown)
from simba.utils.read_write import str_2_bool

TRAIN_SIZE_OPTIONS = np.arange(10, 110, 10)
SAMPLE_SIZE_OPTIONS = list(np.arange(50, 650, 50))

THRESHOLD_OPTIONS = list(np.arange(0.1, 1.1, 0.1))

PADDING_OPTIONS = list(np.round(np.arange(0.01, 10.05, 0.05),2).astype(str))
PADDING_OPTIONS = list(np.insert(PADDING_OPTIONS, 0, 'None'))

class SimBA2YoloKeypointsPopUp(PopUpMixin):

    """
    :example:
    >>> SimBA2YoloKeypointsPopUp()
    """
    def __init__(self):
        PopUpMixin.__init__(self, title="SIMBA TO YOLO KEYPOINTS", icon='SimBA_logo_3_small')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        self.config_select = FileSelect(parent=settings_frm, fileDescription='SIMBA PROJECT CONFIG (.INI): ', lblwidth=35, file_types=[("INI FILE", (".ini", ".INI",))], entry_width=40, initialdir=r"C:\troubleshooting\mitra\project_folder")
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35, entry_width=40, initialdir=r'C:\troubleshooting\mitra\yolo')
        self.train_size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=TRAIN_SIZE_OPTIONS, label="TRAIN SIZE (%): ", label_width=35, dropdown_width=40, value=70)
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE: ", label_width=35, dropdown_width=40, value='TRUE')
        self.padding_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=PADDING_OPTIONS, label="PADDING: ", label_width=35, dropdown_width=40, value='None')
        self.sample_size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=SAMPLE_SIZE_OPTIONS, label="FRAME SAMPLES PER VIDEO: ", label_width=35, dropdown_width=40, value=100)
        self.threshold_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=THRESHOLD_OPTIONS, label="CONFIDENCE THRESHOLD: ", label_width=35, dropdown_width=40, value='None')
        self.grey_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GREYSCALE: ", label_width=35, dropdown_width=40, value='FALSE')
        self.clahe_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="CLAHE: ", label_width=35, dropdown_width=40, value='FALSE')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.config_select.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)
        self.train_size_dropdown.grid(row=2, column=0, sticky=NW)

        self.verbose_dropdown.grid(row=3, column=0, sticky=NW)
        self.padding_dropdown.grid(row=4, column=0, sticky=NW)
        self.sample_size_dropdown.grid(row=5, column=0, sticky=NW)
        self.threshold_dropdown.grid(row=6, column=0, sticky=NW)
        self.grey_dropdown.grid(row=7, column=0, sticky=NW)
        self.clahe_dropdown.grid(row=8, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()


    def run(self):
        config_path = self.config_select.file_path
        config = ConfigReader(config_path=config_path)
        animal_names = list(config.animal_bp_dict.keys())
        bps = [x[:-2] for x in config.animal_bp_dict[animal_names[0]]['X_bps']]
        flip_idx = get_yolo_keypoint_flip_idx(x=bps)
        map_dict = {c: k for c, k in enumerate(animal_names)}
        bp_id_idx = None
        if len(animal_names) > 1:
            bp_id_idx = get_yolo_keypoint_bp_id_idx(animal_bp_dict=config.animal_bp_dict)
        train_size = int(self.train_size_dropdown.get_value()) / 100
        verbose = str_2_bool(self.verbose_dropdown.get_value())
        save_dir = self.save_dir.folder_path
        padding = float(self.padding_dropdown.get_value()) if self.padding_dropdown.get_value() != 'None' else 0.0
        sample_size = int(self.sample_size_dropdown.get_value())
        grey = str_2_bool(self.grey_dropdown.get_value())
        clahe = str_2_bool(self.clahe_dropdown.get_value())
        threshold = float(self.threshold_dropdown.get_value())

        runner = SimBA2Yolo(config_path=config_path,
                            save_dir=save_dir,
                            data_dir=None,
                            train_size=train_size,
                            threshold=threshold,
                            verbose=verbose,
                            greyscale=grey,
                            padding=padding,
                            flip_idx=flip_idx,
                            names=map_dict,
                            sample_size=sample_size,
                            bp_id_idx=bp_id_idx,
                            clahe=clahe)
        runner.run()


#SimBA2YoloKeypointsPopUp()
#simba_to_yolo_keypoints(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini", save_dir=r'C:\troubleshooting\mitra\yolo', sample_size=150, verbose=True)
