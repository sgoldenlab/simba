__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from tkinter import *
from typing import Union

from simba.data_processors.kleinberg_calculator import KleinbergCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box, SimbaButton, SimBADropDown, SimBALabel)
from simba.utils.checks import check_float, check_int
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import NoChoosenClassifierError, NoDataError
from simba.utils.read_write import str_2_bool, get_current_time

INSTRUCTIONS_TXT = 'Results in the project_folder/csv/machine_results folder are overwritten.\n If saving the originals, the original un-smoothened data is saved in a subdirectory of \nthe project_folder/csv/machine_results folder'

class KleinbergPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if len(self.machine_results_paths) == 0:
            raise NoDataError(msg=f'Cannot perform Kleinberg smoothing: No data files found in {self.machine_results_dir} directory', source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="APPLY KLEINBERG BEHAVIOR CLASSIFICATION SMOOTHING", icon='smooth')
        kleinberg_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="KLEINBERG SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.KLEINBERG.value)
        self.k_sigma = Entry_Box(kleinberg_settings_frm, fileDescription="SIGMA", img='sigma', value='2', justify='center', labelwidth=35, entry_box_width=35, tooltip_key='KLEINBERG_SIGMA')
        self.k_gamma = Entry_Box(kleinberg_settings_frm, fileDescription="GAMMA", img='gamma', value='0.3', justify='center', labelwidth=35, entry_box_width=35, tooltip_key='KLEINBERG_GAMMA')
        self.k_hierarchy = Entry_Box(kleinberg_settings_frm, fileDescription="HIERARCHY", value=1, img='hierarchy_2', justify='center', labelwidth=35, entry_box_width=35, validation='numeric', tooltip_key='KLEINBERG_HIERARCHY')
        self.h_search_dropdown = SimBADropDown(parent=kleinberg_settings_frm, dropdown_options=['TRUE', 'FALSE'], label="HIERARCHICAL SEARCH", value='FALSE', img='hierarchy', label_width=35, dropdown_width=35, tooltip_key='KLEINBERG_HIERARCHY_SEARCH')
        self.save_originals_dropdown = SimBADropDown(parent=kleinberg_settings_frm, dropdown_options=['TRUE', 'FALSE'], label="SAVE ORIGINAL DATA:", value='TRUE', img='save', label_width=35, dropdown_width=35, tooltip_key='KLEINBERG_SAVE_ORIGINALS')
        self.instructions_lbl = SimBALabel(parent=kleinberg_settings_frm, txt=INSTRUCTIONS_TXT, justify='center', txt_clr='blue', font=Formats.FONT_REGULAR_ITALICS.value)


        kleinberg_table_frame = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHOOSE CLASSIFIER(S) FOR KLEINBERG SMOOTHING", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.KLEINBERG.value)
        clf_var_dict, clf_cb_dict = {}, {}
        for clf_cnt, clf in enumerate(self.clf_names):
            clf_var_dict[clf] = BooleanVar()
            clf_cb_dict[clf] = Checkbutton(kleinberg_table_frame, text=clf, font=Formats.FONT_REGULAR.value, variable=clf_var_dict[clf])
            clf_cb_dict[clf].grid(row=clf_cnt, column=0, sticky=NW)

        run_kleinberg_btn = SimbaButton(parent=self.main_frm, txt="APPLY KLEINBERG SMOOTHER", img='rocket', txt_clr="blue", font=Formats.FONT_REGULAR.value, cmd=self.run_kleinberg, cmd_kwargs={'behaviors_dict': lambda: clf_var_dict, 'hierarchical_search': lambda: str_2_bool(self.h_search_dropdown.get_value())})
        kleinberg_settings_frm.grid(row=0, sticky=W, pady=(15, 0))
        self.instructions_lbl.grid(row=0, sticky=W)
        self.k_sigma.grid(row=1, sticky=W)
        self.k_gamma.grid(row=2, sticky=W)
        self.k_hierarchy.grid(row=3, sticky=W)
        self.h_search_dropdown.grid(row=4, column=0, sticky=W)
        self.save_originals_dropdown.grid(row=5, column=0, sticky=W)
        kleinberg_table_frame.grid(row=1, column=0, sticky=NW, pady=(15, 0))
        run_kleinberg_btn.grid(row=2, column=0, sticky=NW, pady=(15, 0))
        self.main_frm.mainloop()

    def run_kleinberg(self, behaviors_dict: dict, hierarchical_search: bool):
        targets = []
        for behaviour, behavior_val in behaviors_dict.items():
            if behavior_val.get(): targets.append(behaviour)

        if len(targets) == 0:
            raise NoChoosenClassifierError(source=self.__class__.__name__)

        k_hierarchy = self.k_hierarchy.entry_get
        k_sigma = self.k_sigma.entry_get
        k_gamma = self.k_gamma.entry_get
        save_originals = str_2_bool(self.save_originals_dropdown.get_value())

        check_int(name="Hierarchy", value=k_hierarchy, min_value=1, allow_negative=False, allow_zero=False, raise_error=True)
        check_float(name="Sigma", value=k_sigma, allow_negative=False, allow_zero=False, raise_error=True)
        check_float(name="Gamma", value=k_gamma, allow_negative=False, allow_zero=False, raise_error=True)

        print(f"[{get_current_time()}] Applying kleinberg hyperparameter Setting: Sigma: {k_sigma}, Gamma: {k_gamma}, Hierarchy: {k_hierarchy}")

        kleinberg_analyzer = KleinbergCalculator(config_path=self.config_path,
                                                 classifier_names=targets,
                                                 sigma=float(k_sigma),
                                                 gamma=float(k_gamma),
                                                 hierarchy=int(k_hierarchy),
                                                 hierarchical_search=hierarchical_search,
                                                 save_originals=save_originals)
        kleinberg_analyzer.run()



#_ = KleinbergPopUp(config_path=r"D:\troubleshooting\maplight_ri\project_folder\project_config.ini")

#_ = KleinbergPopUp(config_path=r"C:\troubleshooting\platea\project_folder\project_config.ini")
