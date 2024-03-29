__author__ = "Simon Nilsson"

import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import DropDownMenu, FileSelect
from simba.unsupervised.cluster_xai_calculator import ClusterXAICalculator
from simba.unsupervised.enums import UMLOptions
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import Formats


class ClusterXAIPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="CLUSTER XAI STATISTICS")
        ConfigReader.__init__(self, config_path=config_path)
        self.gini_importance_var = BooleanVar(value=True)
        self.permutation_importance_var = BooleanVar(value=True)
        self.shap_var = BooleanVar(value=False)

        self.data_frm = LabelFrame(
            self.main_frm,
            text="DATA",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.model_select = FileSelect(
            self.data_frm,
            "MODEL PATH:",
            lblwidth=25,
            initialdir=self.project_path,
            file_types=[("SimBA model", f"*.{Formats.PICKLE.value}")],
        )
        self.data_frm.grid(row=0, column=0, sticky=NW)
        self.model_select.grid(row=0, column=0, sticky=NW)

        self.settings_frm = LabelFrame(
            self.main_frm,
            text="SETTINGS",
            pady=5,
            padx=5,
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.gini_importance_cb = Checkbutton(
            self.settings_frm,
            text="CLUSTER RF GINI IMPORTANCE",
            variable=self.gini_importance_var,
        )
        self.permutation_cb = Checkbutton(
            self.settings_frm,
            text="CLUSTER RF PERMUTATION IMPORTANCE",
            variable=self.permutation_importance_var,
        )
        self.shap_method_dropdown = DropDownMenu(
            self.settings_frm,
            "SHAP METHOD:",
            UMLOptions.SHAP_CLUSTER_METHODS.value,
            "25",
        )
        self.shap_method_dropdown.setChoices(UMLOptions.SHAP_CLUSTER_METHODS.value[0])
        self.shap_sample_dropdown = DropDownMenu(
            self.settings_frm,
            "SHAP SAMPLES:",
            UMLOptions.SHAP_SAMPLE_OPTIONS.value,
            "25",
        )
        self.shap_sample_dropdown.setChoices(100)
        self.shap_method_dropdown.disable()
        self.shap_sample_dropdown.disable()
        self.shap_cb = Checkbutton(
            self.settings_frm,
            text="CLUSTER RF SHAP VALUES",
            variable=self.shap_var,
            command=lambda: self.enable_dropdown_from_checkbox(
                check_box_var=self.shap_var,
                dropdown_menus=[self.shap_method_dropdown, self.shap_sample_dropdown],
            ),
        )
        self.settings_frm.grid(row=1, column=0, sticky=NW)
        self.gini_importance_cb.grid(row=0, column=0, sticky=NW)
        self.permutation_cb.grid(row=1, column=0, sticky=NW)
        self.shap_cb.grid(row=2, column=0, sticky=NW)
        self.shap_method_dropdown.grid(row=3, column=0, sticky=NW)
        self.shap_sample_dropdown.grid(row=4, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        check_file_exist_and_readable(file_path=self.model_select.file_path)
        settings = {
            "gini_importance": self.gini_importance_var.get(),
            "permutation_importance": self.permutation_importance_var.get(),
            "shap": {
                "method": self.shap_method_dropdown.getChoices(),
                "run": self.shap_var.get(),
                "sample": int(self.shap_sample_dropdown.getChoices()),
            },
        }
        xai_calculator = ClusterXAICalculator(
            data_path=self.model_select.file_path,
            settings=settings,
            config_path=self.config_path,
        )
        threading.Thread(target=xai_calculator.run()).start()


# _ = ClusterXAIPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/NG_Unsupervised/project_folder/project_config.ini')
