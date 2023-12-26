__author__ = "Simon Nilsson"

import os
from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box)
from simba.utils.checks import check_float
from simba.utils.enums import ConfigKey, Formats, Keys, Links
from simba.utils.printing import stdout_success


class OutlierSettingsPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="OUTLIER SETTINGS")
        ConfigReader.__init__(self, config_path=config_path)
        self.animal_bps = {}
        for animal_name, animal_data in self.animal_bp_dict.items():
            self.animal_bps[animal_name] = [x[:-2] for x in animal_data["X_bps"]]

        self.location_correction_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="LOCATION CORRECTION",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.OULIERS.value,
        )

        bp_entry_cnt, self.criterion_dropdowns = 0, {}
        for animal_cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
            self.criterion_dropdowns[animal_name] = {}
            self.criterion_dropdowns[animal_name]["location_bp_1"] = DropDownMenu(
                self.location_correction_frm,
                "Choose {} body part 1:".format(animal_name),
                self.animal_bps[animal_name],
                "30",
            )
            self.criterion_dropdowns[animal_name]["location_bp_2"] = DropDownMenu(
                self.location_correction_frm,
                "Choose {} body part 2:".format(animal_name),
                self.animal_bps[animal_name],
                "30",
            )
            self.criterion_dropdowns[animal_name]["location_bp_1"].setChoices(
                self.animal_bps[animal_name][0]
            )
            self.criterion_dropdowns[animal_name]["location_bp_2"].setChoices(
                self.animal_bps[animal_name][1]
            )
            self.criterion_dropdowns[animal_name]["location_bp_1"].grid(
                row=bp_entry_cnt, column=0, sticky=NW
            )
            bp_entry_cnt += 1
            self.criterion_dropdowns[animal_name]["location_bp_2"].grid(
                row=bp_entry_cnt, column=0, sticky=NW
            )
            bp_entry_cnt += 1
        self.location_criterion = Entry_Box(
            self.location_correction_frm, "Location criterion: ", "15"
        )
        self.location_criterion.grid(row=bp_entry_cnt, column=0, sticky=NW)
        self.location_correction_frm.grid(row=0, column=0, sticky=NW)

        self.movement_correction_frm = LabelFrame(
            self.main_frm,
            text="MOVEMENT CORRECTION",
            font=("Times", 12, "bold"),
            pady=5,
            padx=5,
        )
        bp_entry_cnt = 0
        for animal_cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
            self.criterion_dropdowns[animal_name]["movement_bp_1"] = DropDownMenu(
                self.movement_correction_frm,
                "Choose {} body part 1:".format(animal_name),
                self.animal_bps[animal_name],
                "30",
            )
            self.criterion_dropdowns[animal_name]["movement_bp_2"] = DropDownMenu(
                self.movement_correction_frm,
                "Choose {} body part 2:".format(animal_name),
                self.animal_bps[animal_name],
                "30",
            )
            self.criterion_dropdowns[animal_name]["movement_bp_1"].setChoices(
                self.animal_bps[animal_name][0]
            )
            self.criterion_dropdowns[animal_name]["movement_bp_2"].setChoices(
                self.animal_bps[animal_name][1]
            )
            self.criterion_dropdowns[animal_name]["movement_bp_1"].grid(
                row=bp_entry_cnt, column=0, sticky=NW
            )
            bp_entry_cnt += 1
            self.criterion_dropdowns[animal_name]["movement_bp_2"].grid(
                row=bp_entry_cnt, column=0, sticky=NW
            )
            bp_entry_cnt += 1
        self.movement_criterion = Entry_Box(
            self.movement_correction_frm, "Location criterion: ", "15"
        )
        self.movement_criterion.grid(row=bp_entry_cnt, column=0, sticky=NW)
        self.movement_correction_frm.grid(row=1, column=0, sticky=NW)

        agg_type_frm = LabelFrame(
            self.main_frm,
            text="AGGREGATION METHOD",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
        )
        self.agg_type_dropdown = DropDownMenu(
            agg_type_frm, "Aggregation method:", ["mean", "median"], "15"
        )
        self.agg_type_dropdown.setChoices("median")
        self.agg_type_dropdown.grid(row=0, column=0, sticky=NW)
        agg_type_frm.grid(row=2, column=0, sticky=NW)

        run_btn = Button(
            self.main_frm,
            text="CONFIRM",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="red",
            command=lambda: self.run(),
        )
        run_btn.grid(row=3, column=0, sticky=NW)

        # self.main_frm.mainloop()

    def run(self):
        if self.config.has_section(ConfigKey.OUTLIER_SETTINGS.value):
            self.config.remove_section(ConfigKey.OUTLIER_SETTINGS.value)
        self.config.add_section(ConfigKey.OUTLIER_SETTINGS.value)
        check_float(
            name="LOCATION CRITERION",
            value=self.location_criterion.entry_get,
            min_value=0.0,
        )
        check_float(
            name="MOVEMENT CRITERION",
            value=self.movement_criterion.entry_get,
            min_value=0.0,
        )
        if not self.config.has_section("Outlier settings"):
            self.config.add_section("Outlier settings")
        self.config.set(
            ConfigKey.OUTLIER_SETTINGS.value,
            ConfigKey.MOVEMENT_CRITERION.value,
            str(self.movement_criterion.entry_get),
        )
        self.config.set(
            ConfigKey.OUTLIER_SETTINGS.value,
            ConfigKey.LOCATION_CRITERION.value,
            str(self.location_criterion.entry_get),
        )
        self.config.set(
            ConfigKey.OUTLIER_SETTINGS.value,
            "mean_or_median",
            str(self.agg_type_dropdown.getChoices()),
        )
        for animal_cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
            self.config.set(
                ConfigKey.OUTLIER_SETTINGS.value,
                "movement_bodyPart1_{}".format(animal_name),
                self.criterion_dropdowns[animal_name]["movement_bp_1"].getChoices(),
            )
            self.config.set(
                ConfigKey.OUTLIER_SETTINGS.value,
                "movement_bodyPart2_{}".format(animal_name),
                self.criterion_dropdowns[animal_name]["movement_bp_2"].getChoices(),
            )
            self.config.set(
                ConfigKey.OUTLIER_SETTINGS.value,
                "location_bodyPart1_{}".format(animal_name),
                self.criterion_dropdowns[animal_name]["location_bp_1"].getChoices(),
            )
            self.config.set(
                ConfigKey.OUTLIER_SETTINGS.value,
                "location_bodyPart2_{}".format(animal_name),
                self.criterion_dropdowns[animal_name]["location_bp_2"].getChoices(),
            )
        with open(self.config_path, "w") as f:
            self.config.write(f)

        stdout_success(
            msg="Outlier correction settings updated in the project_config.ini",
            source=self.__class__.__name__,
        )
        self.root.destroy()


# _ = OutlierSettingsPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
