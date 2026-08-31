__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, SimbaButton, SimBADropDown)
from simba.utils.checks import check_float
from simba.utils.enums import ConfigKey, Formats, Keys, Links
from simba.utils.printing import stdout_success


VALID_CLR = 'yellowgreen'
INVALID_CLR = 'lightsalmon'

class OutlierSettingsPopUp(PopUpMixin, ConfigReader):
    """
    Pop-up for setting the movement and location outlier correction criteria of a SimBA project.

    Per animal, two body-parts are chosen for the movement criterion and two for the location criterion, together with
    the criteria multipliers and the aggregation method (mean or median). Pressing CONFIRM writes the settings to the
    ``[Outlier settings]`` section of the project config, where the outlier correctors read them.

    .. seealso::
       `Outlier correction documentation <https://github.com/sgoldenlab/simba/blob/master/misc/Outlier_settings.pdf>`__.

    :param Union[str, os.PathLike] config_path: Path to the SimBA project config file. Criteria already stored in the file are pre-filled in the entry boxes.

    :example:

    >>> OutlierSettingsPopUp(config_path='project_folder/project_config.ini')
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        PopUpMixin.__init__(self, title="OUTLIER SETTINGS", icon='outlier')
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.animal_bps = {}
        for animal_name, animal_data in self.animal_bp_dict.items(): self.animal_bps[animal_name] = [x[:-2] for x in animal_data["X_bps"]]
        movement_criterion = self.read_config_entry(config=self.config, section=ConfigKey.OUTLIER_SETTINGS.value, option=ConfigKey.MOVEMENT_CRITERION.value, default_value='', data_type='float')
        location_criterion = self.read_config_entry(config=self.config, section=ConfigKey.OUTLIER_SETTINGS.value, option=ConfigKey.LOCATION_CRITERION.value, default_value='', data_type='float')

        self.location_correction_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="LOCATION CORRECTION", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.OULIERS.value)
        bp_entry_cnt, self.criterion_dropdowns = 0, {}
        for animal_cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
            self.criterion_dropdowns[animal_name] = {}
            self.criterion_dropdowns[animal_name]["location_bp_1"] = SimBADropDown(parent=self.location_correction_frm, dropdown_options=self.animal_bps[animal_name], label=f"Choose {animal_name} body part 1:", label_width=30, dropdown_width=30, value=self.animal_bps[animal_name][0], img='point')
            self.criterion_dropdowns[animal_name]["location_bp_2"] = SimBADropDown(parent=self.location_correction_frm, dropdown_options=self.animal_bps[animal_name], label=f"Choose {animal_name} body part 2:", label_width=30, dropdown_width=30, value=self.animal_bps[animal_name][1], img='point')
            self.criterion_dropdowns[animal_name]["location_bp_1"].grid(row=bp_entry_cnt, column=0, sticky=NW)
            bp_entry_cnt += 1
            self.criterion_dropdowns[animal_name]["location_bp_2"].grid(row=bp_entry_cnt, column=0, sticky=NW)
            bp_entry_cnt += 1

        self.location_criterion = Entry_Box(parent=self.location_correction_frm, fileDescription="Location criterion: ", labelwidth=30, entry_box_width=30, img='threshold', trace=self._check_float_bg, value='' if location_criterion is None else location_criterion)
        self.location_criterion.grid(row=bp_entry_cnt, column=0, sticky=NW)
        self.location_correction_frm.grid(row=0, column=0, sticky=NW)

        self.movement_correction_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MOVEMENT CORRECTION", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.OULIERS.value)
        bp_entry_cnt = 0
        for animal_cnt, animal_name in enumerate(self.animal_bp_dict.keys()):
            self.criterion_dropdowns[animal_name]["movement_bp_1"] = SimBADropDown(parent=self.movement_correction_frm, dropdown_options=self.animal_bps[animal_name], label=f"Choose {animal_name} body part 1:", label_width=30, dropdown_width=30, value=self.animal_bps[animal_name][0], img='point')
            self.criterion_dropdowns[animal_name]["movement_bp_2"] = SimBADropDown(parent=self.movement_correction_frm, dropdown_options=self.animal_bps[animal_name], label=f"Choose {animal_name} body part 2:", label_width=30, dropdown_width=30, value=self.animal_bps[animal_name][0], img='point')
            self.criterion_dropdowns[animal_name]["movement_bp_1"].grid(row=bp_entry_cnt, column=0, sticky=NW)
            bp_entry_cnt += 1
            self.criterion_dropdowns[animal_name]["movement_bp_2"].grid(row=bp_entry_cnt, column=0, sticky=NW)
            bp_entry_cnt += 1
        self.movement_criterion = Entry_Box(parent=self.movement_correction_frm, fileDescription="Movement criterion: ", labelwidth=30, entry_box_width=30, img='threshold', trace=self._check_float_bg, value='' if movement_criterion is None else movement_criterion)
        self.movement_criterion.grid(row=bp_entry_cnt, column=0, sticky=NW)
        self.movement_correction_frm.grid(row=1, column=0, sticky=NW)


        agg_type_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="AGGREGATION METHOD", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.OULIERS.value)
        self.agg_type_dropdown = SimBADropDown(parent=agg_type_frm, dropdown_options=["mean", "median"], label="Aggregation method:", label_width=30, dropdown_width=30, value='median', img='equation_small')
        self.agg_type_dropdown.grid(row=0, column=0, sticky=NW)
        agg_type_frm.grid(row=2, column=0, sticky=NW)


        run_btn = SimbaButton(parent=self.main_frm, txt="CONFIRM", img='tick', txt_clr="red", font=Formats.FONT_REGULAR.value, cmd=self.run)
        run_btn.grid(row=3, column=0, sticky=NW)

        self.main_frm.mainloop()

    def _check_float_bg(self, entry_box: Entry_Box, allow_blank: bool = False):
        allow_blank = allow_blank or getattr(entry_box, 'allow_blank', False)
        value = entry_box.entry_get
        if value.strip() == '' and allow_blank:
            entry_box.set_bg_clr(clr=VALID_CLR)
        else:
            bg_clr = VALID_CLR if check_float(value=value, name='', allow_zero=False, allow_negative=False, raise_error=False)[0] else INVALID_CLR
            entry_box.set_bg_clr(clr=bg_clr)

    def run(self):
        check_float(name="LOCATION CRITERION", value=self.location_criterion.entry_get, allow_zero=False, allow_negative=False)   # NOTE: checked before the config section is touched, and with the same criteria as the entry-box background check - the criteria are multipliers of the mean/median body-part distance, and a criterion of zero flags every frame as an outlier.
        check_float(name="MOVEMENT CRITERION", value=self.movement_criterion.entry_get, allow_zero=False, allow_negative=False)
        if self.config.has_section(ConfigKey.OUTLIER_SETTINGS.value):
            self.config.remove_section(ConfigKey.OUTLIER_SETTINGS.value)
        self.config.add_section(ConfigKey.OUTLIER_SETTINGS.value)
        self.config.set(ConfigKey.OUTLIER_SETTINGS.value, ConfigKey.MOVEMENT_CRITERION.value, str(self.movement_criterion.entry_get))
        self.config.set(ConfigKey.OUTLIER_SETTINGS.value, ConfigKey.LOCATION_CRITERION.value, str(self.location_criterion.entry_get))
        self.config.set(ConfigKey.OUTLIER_SETTINGS.value, "mean_or_median", str(self.agg_type_dropdown.getChoices()))
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
#_ = OutlierSettingsPopUp(config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini")
