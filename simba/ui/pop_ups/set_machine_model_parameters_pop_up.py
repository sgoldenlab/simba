__author__ = "Simon Nilsson"

import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        FileSelect, SimbaButton)
from simba.utils.checks import (check_file_exist_and_readable, check_float,
                                check_int)
from simba.utils.enums import Formats, Keys, Links
from simba.utils.printing import stdout_success


class SetMachineModelParameters(PopUpMixin, ConfigReader):

    """
    :example:
    >>> _ = SetMachineModelParameters(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')

    """
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        PopUpMixin.__init__(self, title="SET MODEL PARAMETERS", icon='equation_small')
        self.clf_table_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.SET_RUN_ML_PARAMETERS.value)
        Label(self.clf_table_frm,text="CLASSIFIER",font=Formats.FONT_HEADER.value).grid(row=0, column=0)
        Label(self.clf_table_frm, text="MODEL PATH (.SAV)", font=Formats.FONT_HEADER.value).grid(row=0, column=1, sticky=NW)
        Label(self.clf_table_frm, text="DISCRIMINATION THRESHOLD", font=Formats.FONT_HEADER.value).grid(row=0, column=2, sticky=NW)
        Label(self.clf_table_frm, text="MINIMUM BOUT LENGTH (MS)", font=Formats.FONT_HEADER.value).grid(row=0, column=3, sticky=NW)
        self.clf_data = {}
        for clf_cnt, clf_name in enumerate(self.clf_names):
            self.clf_data[clf_name] = {}
            Label(self.clf_table_frm, text=clf_name, font=Formats.FONT_HEADER.value).grid(row=clf_cnt + 1, column=0, sticky=NW)
            self.clf_data[clf_name]["path"] = FileSelect(self.clf_table_frm, title="Select model (.sav) file", initialdir=self.project_path, file_types=[("SimBA Classifier", "*.sav")])
            self.clf_data[clf_name]["threshold"] = Entry_Box(self.clf_table_frm, "", "0")
            self.clf_data[clf_name]["min_bout"] = Entry_Box(self.clf_table_frm, "", "0")
            self.clf_data[clf_name]["path"].grid(row=clf_cnt + 1, column=1, sticky=NW)
            self.clf_data[clf_name]["threshold"].grid(row=clf_cnt + 1, column=2, sticky=NW)
            self.clf_data[clf_name]["min_bout"].grid(row=clf_cnt + 1, column=3, sticky=NW)


        set_btn = SimbaButton(parent=self.main_frm, txt="SET MODEL(S)", img='tick', txt_clr='red', font=Formats.FONT_REGULAR.value, cmd=self.set)
        self.clf_table_frm.grid(row=0, sticky=W, pady=5, padx=5)
        set_btn.grid(row=1, pady=10)
        self.main_frm.mainloop()

    def set(self):
        for model_name, model_settings in self.clf_data.items():
            check_file_exist_and_readable(model_settings["path"].file_path)
            check_float(
                name="Classifier {} threshhold".format(model_name),
                value=model_settings["threshold"].entry_get,
                max_value=1.0,
                min_value=0.0,
            )
            check_int(
                name="Classifier {} minimum bout".format(model_name),
                value=model_settings["min_bout"].entry_get,
                min_value=0.0,
            )

        for cnt, (model_name, model_settings) in enumerate(self.clf_data.items()):
            self.config.set(
                "SML settings",
                "model_path_{}".format(str(cnt + 1)),
                model_settings["path"].file_path,
            )
            self.config.set(
                "threshold_settings",
                "threshold_{}".format(str(cnt + 1)),
                model_settings["threshold"].entry_get,
            )
            self.config.set(
                "Minimum_bout_lengths",
                "min_bout_{}".format(str(cnt + 1)),
                model_settings["min_bout"].entry_get,
            )

        with open(self.config_path, "w") as f:
            self.config.write(f)

        stdout_success(
            msg="Model paths/settings saved in project_config.ini",
            source=self.__class__.__name__,
        )


#_ = SetMachineModelParameters(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
