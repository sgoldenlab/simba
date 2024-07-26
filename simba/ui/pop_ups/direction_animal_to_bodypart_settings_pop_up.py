__author__ = "Tzuk Polinsky"

import configparser
import os
from tkinter import *

from simba.data_processors.directing_animal_to_bodypart import \
    DirectingAnimalsToBodyPartAnalyzer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box)
from simba.utils.checks import check_float
from simba.utils.enums import ConfigKey, Formats, Keys, Links
from simba.utils.printing import stdout_success


class DirectionAnimalToBodyPartSettingsPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="ANIMAL To BODY PART SETTINGS")
        ConfigReader.__init__(self, config_path=config_path)
        self.animal_bps = {}
        for animal_name, animal_data in self.animal_bp_dict.items():
            self.animal_bps[animal_name] = [x[:-2] for x in animal_data["X_bps"]]

        self.location_correction_frm = CreateLabelFrameWithIcon( parent=self.main_frm, header="BODY PART SELECTION", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.OULIERS.value,)
        bp_entry_cnt, self.criterion_dropdowns = 0, {}
        for animal_cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
            self.criterion_dropdowns[animal_name] = {}
            self.criterion_dropdowns[animal_name]["bp"] = DropDownMenu(self.location_correction_frm, "Choose {} body part:".format(animal_name), self.animal_bps[animal_name], "30",)
            self.criterion_dropdowns[animal_name]["bp"].setChoices(self.animal_bps[animal_name][0])
            self.criterion_dropdowns[animal_name]["bp"].grid(row=bp_entry_cnt, column=0, sticky=NW)
            bp_entry_cnt += 1
        self.location_correction_frm.grid(row=0, column=0, sticky=NW)
        run_btn = Button( self.main_frm, text="RUN ANALYSIS", font=Formats.FONT_HEADER.value, fg="red", command=lambda: self.run(),)
        run_btn.grid(row=3, column=0, sticky=NW)

    def run(self):
        for animal_cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
            try:
                self.config.set(
                    ConfigKey.DIRECTIONALITY_SETTINGS.value,
                    "bodypart_direction",
                    self.criterion_dropdowns[animal_name]["bp"].getChoices(),
                )
            except configparser.NoSectionError as e:
                self.config.add_section(ConfigKey.DIRECTIONALITY_SETTINGS.value)
                self.config.set(
                    ConfigKey.DIRECTIONALITY_SETTINGS.value,
                    "bodypart_direction",
                    self.criterion_dropdowns[animal_name]["bp"].getChoices(),
                )
        with open(self.config_path, "w") as f:
            self.config.write(f)

        stdout_success(
            msg="Body part directionality settings updated in the project_config.ini"
        )
        directing_animals_analyzer = DirectingAnimalsToBodyPartAnalyzer(
            config_path=self.config_path
        )
        directing_animals_analyzer.process_directionality()
        directing_animals_analyzer.create_directionality_dfs()
        directing_animals_analyzer.save_directionality_dfs()
        directing_animals_analyzer.summary_statistics()
        self.root.destroy()
