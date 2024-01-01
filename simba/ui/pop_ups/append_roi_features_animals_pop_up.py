import os
from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.ROI_feature_analyzer import ROIFeatureCreator
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box)
from simba.utils.enums import ConfigKey, Formats, Keys, Links
from simba.utils.errors import NoROIDataError


class AppendROIFeaturesByAnimalPopUp(ConfigReader, PopUpMixin):
    def __init__(self, config_path: str):
        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise NoROIDataError(
                msg=f"SIMBA ERROR: No ROIs have been defined. Please define ROIs before appending ROI-based features (no data file found at path {self.roi_coordinates_path})",
                source=self.__class__.__name__,
            )
        PopUpMixin.__init__(
            self, title="APPEND ROI FEATURES: BY ANIMALS", size=(400, 400)
        )
        self.animal_cnt_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SELECT NUMBER OF ANIMALS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.ROI_FEATURES.value,
        )
        self.animal_cnt_dropdown = DropDownMenu(
            self.animal_cnt_frm,
            "# of animals",
            list(range(1, self.animal_cnt + 1)),
            labelwidth=20,
        )
        self.animal_cnt_dropdown.setChoices(1)
        self.animal_cnt_confirm_btn = Button(
            self.animal_cnt_frm,
            text="Confirm",
            command=lambda: self.create_settings_frm(),
        )
        self.animal_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_confirm_btn.grid(row=0, column=1, sticky=NW)
        self.main_frm.mainloop()

    def create_settings_frm(self):
        if hasattr(self, "setting_frm"):
            self.setting_frm.destroy()
            self.body_part_frm.destroy()
        self.setting_frm = LabelFrame(
            self.main_frm, text="SETTINGS", font=Formats.LABELFRAME_HEADER_FORMAT.value
        )
        self.choose_bp_frm(parent=self.setting_frm, bp_options=self.body_parts_lst)
        self.setting_frm.grid(row=1, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        if self.config.has_section(ConfigKey.ROI_SETTINGS.value):
            self.config.remove_section(ConfigKey.ROI_SETTINGS.value)
        self.config.add_section(ConfigKey.ROI_SETTINGS.value)

        for bp_cnt, bp_dropdown in self.body_parts_dropdowns.items():
            self.config.set(
                ConfigKey.ROI_SETTINGS.value,
                f"animal_{bp_cnt+1}_bp",
                bp_dropdown.getChoices(),
            )
        with open(self.config_path, "w") as f:
            self.config.write(f)

        roi_feature_creator = ROIFeatureCreator(config_path=self.config_path)
        roi_feature_creator.run()
        roi_feature_creator.save()


# AppendROIFeaturesByAnimalPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/project_config.ini')


# AppendROIFeaturesByAnimalPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
