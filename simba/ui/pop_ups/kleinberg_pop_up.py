__author__ = "Simon Nilsson"

import os
from tkinter import *
from typing import Union

from simba.data_processors.kleinberg_calculator import KleinbergCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton, SimbaCheckbox)
from simba.utils.checks import check_float, check_int
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import NoChoosenClassifierError


class KleinbergPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="APPLY KLEINBERG BEHAVIOR CLASSIFICATION SMOOTHING")
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        kleinberg_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="KLEINBERG SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.KLEINBERG.value)
        self.k_sigma = Entry_Box(kleinberg_settings_frm, "SIGMA", "10")
        self.k_sigma.entry_set("2")
        self.k_gamma = Entry_Box(kleinberg_settings_frm, "GAMMA", "10")
        self.k_gamma.entry_set("0.3")
        self.k_hierarchy = Entry_Box(kleinberg_settings_frm, "HIERARCHY", "10")
        self.k_hierarchy.entry_set("1")
        self.h_search_lbl_val_cb, self.h_search_lbl_val = SimbaCheckbox(parent=kleinberg_settings_frm, txt="HIERACHICAL SEARCH", txt_img='hierarchy')
        kleinberg_table_frame = LabelFrame(self.main_frm, text="CHOOSE CLASSIFIER(S) FOR KLEINBERG SMOOTHING", font=Formats.FONT_HEADER.value)
        clf_var_dict, clf_cb_dict = {}, {}
        for clf_cnt, clf in enumerate(self.clf_names):
            clf_var_dict[clf] = BooleanVar()
            clf_cb_dict[clf] = Checkbutton(kleinberg_table_frame, text=clf, font=Formats.FONT_REGULAR.value, variable=clf_var_dict[clf])
            clf_cb_dict[clf].grid(row=clf_cnt, sticky=NW)

        run_kleinberg_btn = SimbaButton(parent=self.main_frm, txt="APPLY KLEINBERG SMOOTHER", img='rocket', txt_clr="blue", font=Formats.FONT_REGULAR.value, cmd=self.run_kleinberg, cmd_kwargs={'behaviors_dict': lambda: clf_var_dict, 'hierarchical_search': lambda: self.h_search_lbl_val.get()})
        kleinberg_settings_frm.grid(row=0, sticky=W, padx=10)
        self.k_sigma.grid(row=0, sticky=W)
        self.k_gamma.grid(row=1, sticky=W)
        self.k_hierarchy.grid(row=2, sticky=W)
        self.h_search_lbl_val_cb.grid(row=3, column=0, sticky=W)
        kleinberg_table_frame.grid(row=1, pady=10, padx=10)
        run_kleinberg_btn.grid(row=2)
        self.main_frm.mainloop()

    def run_kleinberg(self, behaviors_dict: dict, hierarchical_search: bool):
        targets = []
        for behaviour, behavior_val in behaviors_dict.items():
            if behavior_val.get():
                targets.append(behaviour)

        if len(targets) == 0:
            raise NoChoosenClassifierError(source=self.__class__.__name__)

        check_int(name="Hierarchy", value=self.k_hierarchy.entry_get)
        check_float(name="Sigma", value=self.k_sigma.entry_get)
        check_float(name="Gamma", value=self.k_gamma.entry_get)

        try:
            print(
                "Applying kleinberg hyperparameter Setting: Sigma: {}, Gamma: {}, Hierarchy: {}".format(
                    str(self.k_sigma.entry_get),
                    str(self.k_gamma.entry_get),
                    str(self.k_hierarchy.entry_get),
                )
            )
        except:
            print("Please insert accurate values for all hyperparameters.")

        kleinberg_analyzer = KleinbergCalculator(
            config_path=self.config_path,
            classifier_names=targets,
            sigma=self.k_sigma.entry_get,
            gamma=self.k_gamma.entry_get,
            hierarchy=self.k_hierarchy.entry_get,
            hierarchical_search=hierarchical_search,
        )
        kleinberg_analyzer.run()


#_ = KleinbergPopUp(config_path=r"C:\troubleshooting\platea\project_folder\project_config.ini")
