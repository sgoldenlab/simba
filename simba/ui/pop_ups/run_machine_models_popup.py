__author__ = "Simon Nilsson"

import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.model.inference_batch import InferenceBatch
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        FileSelect, SimbaButton, SimBADropDown,
                                        SimBALabel)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_int)
from simba.utils.enums import ConfigKey, Dtypes, Formats, Keys, Links
from simba.utils.errors import InvalidInputError, NoDataError
from simba.utils.printing import stdout_success
from simba.utils.read_write import find_core_cnt, read_config_entry
from simba.utils.warnings import NoFileFoundWarning

CORE_CNT_OPTIONS = list(range(1, find_core_cnt()[0]))
MIN_BOUT, THRESHOLD, PATH = "min_bout", "threshold", "path"

class RunMachineModelsPopUp(PopUpMixin, ConfigReader):

    """
    :example:
    >>> _ = RunMachineModelsPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')

    """
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        PopUpMixin.__init__(self, title="SET MODEL PARAMETERS", icon='equation_small')
        self.clf_table_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.SET_RUN_ML_PARAMETERS.value)
        clf_header = SimBALabel(parent=self.clf_table_frm, txt="CLASSIFIER", font=Formats.FONT_HEADER.value)
        mdl_path_header = SimBALabel(parent=self.clf_table_frm, txt="MODEL PATH (.SAV)", font=Formats.FONT_HEADER.value)
        threshold_header = SimBALabel(parent=self.clf_table_frm, txt="THRESHOLD (0.0 - 1.0)", font=Formats.FONT_HEADER.value)
        min_bout_header = SimBALabel(parent=self.clf_table_frm, txt="MINIMUM BOUT LENGTH (MS)", font=Formats.FONT_HEADER.value)
        clf_header.grid(row=0, column=0, sticky=W, padx=(0, 10))
        mdl_path_header.grid(row=0, column=1, sticky=NW)
        threshold_header.grid(row=0, column=2, sticky=NW)
        min_bout_header.grid(row=0, column=3, sticky=NW)

        self.clf_data = {}
        for clf_cnt, clf_name in enumerate(self.clf_names):
            self.clf_data[clf_name] = {}
            SimBALabel(parent=self.clf_table_frm, txt=clf_name, font=Formats.FONT_REGULAR_ITALICS.value).grid(row=clf_cnt + 1, column=0, sticky=W, padx=(0, 10))
            mdl_path = read_config_entry(config=self.config, section=ConfigKey.SML_SETTINGS.value, option=f"model_path_{clf_cnt + 1}", default_value='Select model (.sav) file', data_type=Dtypes.STR.value)
            self.clf_data[clf_name][PATH] = FileSelect(self.clf_table_frm, title="Select model (.sav) file", initialdir=self.project_path, file_types=[("SimBA Classifier", "*.sav")], initial_path=mdl_path)
            threshold = read_config_entry(config=self.config, section=ConfigKey.THRESHOLD_SETTINGS.value, option=f"threshold_{clf_cnt + 1}", default_value='', data_type=Dtypes.STR.value)
            self.clf_data[clf_name][THRESHOLD] = Entry_Box(parent=self.clf_table_frm, fileDescription='', labelwidth=0, entry_box_width=20, value=threshold)
            bout_length = read_config_entry(config=self.config, section=ConfigKey.MIN_BOUT_LENGTH.value, option=f"min_bout_{clf_cnt + 1}", default_value='', data_type=Dtypes.STR.value)
            self.clf_data[clf_name][MIN_BOUT] = Entry_Box(parent=self.clf_table_frm, fileDescription='', labelwidth=0, entry_box_width=20, value=bout_length)
            self.clf_data[clf_name][PATH].grid(row=clf_cnt + 1, column=1, sticky=NW, padx=(0, 5))
            self.clf_data[clf_name][THRESHOLD].grid(row=clf_cnt + 1, column=2, sticky=NW, padx=(0, 5))
            self.clf_data[clf_name][MIN_BOUT].grid(row=clf_cnt + 1, column=3, sticky=NW, padx=(0, 5))
        self.clf_table_frm.grid(row=0, sticky=W, pady=5, padx=5)
        # self.runtime_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="RUNTIME SETTINGS", icon_name='run', icon_link=Links.SET_RUN_ML_PARAMETERS.value)
        # self.core_cnt_dropdown = SimBADropDown(parent=self.runtime_settings_frm, dropdown_options=CORE_CNT_OPTIONS, label='CPU CORE COUNT:', label_width=30, dropdown_width=20, value=int(find_core_cnt()[0]/3))
        # self.runtime_settings_frm.grid(row=1, sticky=W, pady=5, padx=5)
        # self.core_cnt_dropdown.grid(row=0, sticky=W, pady=5, padx=5)
        run_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header=f"ANALYZE {len(self.feature_file_paths)} FILES(S)", icon_name='rocket')
        run_btn = SimbaButton(parent=run_frm, txt="RUN", img='rocket', txt_clr='red', font=Formats.FONT_REGULAR.value, hover_font=Formats.FONT_REGULAR.value, cmd=self.run)
        run_frm.grid(row=2, sticky=W, pady=5, padx=5)
        run_btn.grid(row=0, sticky=W, pady=5, padx=5)

        self.main_frm.mainloop()

    def run(self):
        filtered_clf_data = {}
        for model_name, model_settings in self.clf_data.items():
            print('Updating settings in SimBA project config...')
            valid_path = check_file_exist_and_readable(model_settings[PATH].file_path, raise_error=False)
            valid_threshold, _ = check_float(name=f"Classifier {model_name} threshold", value=model_settings[THRESHOLD].entry_get, max_value=1.0, min_value=0.0, raise_error=False)
            valid_min_bout, _ = check_int(name=f"Classifier {model_name} minimum bout", value=model_settings[MIN_BOUT].entry_get, min_value=0, raise_error=False)
            if not valid_path:
                NoFileFoundWarning(msg=f'SKIPPING CLASSIFIER {model_name}: The path "{model_settings[PATH].file_path}" is not a valid file path for classifier {model_name}', source=self.__class__.__name__)
                filtered_clf_data[model_name] = {PATH: None, THRESHOLD: None, MIN_BOUT: None}
            elif not valid_threshold:
                NoFileFoundWarning(msg=f'SKIPPING CLASSIFIER {model_name}: Classifier {model_name} threshold {model_settings["threshold"].entry_get} is not a valid threshold between 0.0 - 1.0.', source=self.__class__.__name__)
                filtered_clf_data[model_name] = {PATH: None, THRESHOLD: None, MIN_BOUT: None}
            elif not valid_min_bout:
                NoFileFoundWarning(msg=f'SKIPPING CLASSIFIER {model_name}: Classifier {model_name} minimum bout length {model_settings["min_bout"].entry_get} is not a valid integer number', source=self.__class__.__name__)
                filtered_clf_data[model_name] = {PATH: None, THRESHOLD: None, MIN_BOUT: None}
            else:
                filtered_clf_data[model_name] = {PATH: model_settings["path"].file_path, 'threshold': model_settings["threshold"].entry_get, 'min_bout': model_settings["min_bout"].entry_get}
        if len(list(filtered_clf_data.keys())) == 0:
            raise InvalidInputError(msg=f'None of the {len(self.clf_names)} classifier(s) have valid paths, minimum bout lengths, and/or thresholds', source=self.__class__.__name__)

        for cnt, (model_name, model_settings) in enumerate(filtered_clf_data.items()):
            self.config.set(ConfigKey.SML_SETTINGS.value, f"model_path_{cnt + 1}", str(model_settings["path"]))
            self.config.set(ConfigKey.THRESHOLD_SETTINGS.value, f"threshold_{cnt + 1}", str(model_settings["threshold"]))
            self.config.set(ConfigKey.MIN_BOUT_LENGTH.value, f"min_bout_{cnt + 1}", str(model_settings["min_bout"]))

        with open(self.config_path, "w") as f:
            self.config.write(f)

        stdout_success(msg="Model paths/settings saved in project_config.ini", source=self.__class__.__name__)

        if len(self.feature_file_paths) == 0:
            raise NoDataError(msg=f'Cannot run machine model predictions: No data files found in {self.features_dir} directory', source=self.__class__.__name__)

        inferencer = InferenceBatch(config_path=self.config_path, features_dir=None, save_dir=None, minimum_bout_length=None)
        inferencer.run()

#_ = RunMachineModelsPopUp(config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini")
