__author__ = "Simon Nilsson"

import os.path
from tkinter import *

from simba.feature_extractors.feature_subsets import FeatureSubsetsCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        FolderSelect)
from simba.utils.checks import check_if_dir_exists
from simba.utils.enums import Links
from simba.utils.errors import InvalidInputError


class FeatureSubsetExtractorPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="EXTRACT FEATURE SUBSETS", size=(500, 500))
        ConfigReader.__init__(self, config_path=config_path)
        self.feature_subset_options = [
            "Two-point body-part distances (mm)",
            "Within-animal three-point body-part angles (degrees)",
            "Within-animal three-point convex hull perimeters (mm)",
            "Within-animal four-point convex hull perimeters (mm)",
            "Entire animal convex hull perimeters (mm)",
            "Entire animal convex hull area (mm2)",
            "Frame-by-frame body-part movements (mm)",
            "Frame-by-frame distance to ROI centers (mm)",
            "Frame-by-frame body-parts inside ROIs (Boolean)",
        ]

        self.settings_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SETTINGS",
            icon_name="documentation",
            icon_link=Links.FEATURE_SUBSETS.value,
        )
        self.save_dir = FolderSelect(self.settings_frm, "SAVE DIRECTORY:", lblwidth=20)
        self.append_to_features_extracted_var = BooleanVar(value=False)
        self.append_to_targets_inserted_var = BooleanVar(value=False)
        self.include_file_checks = BooleanVar(value=False)
        self.append_to_features_cb = Checkbutton(
            self.settings_frm,
            text="APPEND RESULTS TO FEATURES EXTRACTED FILES",
            variable=self.append_to_features_extracted_var,
        )
        self.append_to_targets_inserted_cb = Checkbutton(
            self.settings_frm,
            text="APPEND RESULTS TO TARGET INSERTED FILES",
            variable=self.append_to_targets_inserted_var,
        )
        self.include_file_checks_cb = Checkbutton(
            self.settings_frm,
            text="INCLUDE INTEGRITY CHECKS BEFORE APPENDING NEW DATA (RECOMMENDED)",
            variable=self.include_file_checks,
        )
        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)
        self.append_to_features_cb.grid(row=2, column=0, sticky=NW)
        self.append_to_targets_inserted_cb.grid(row=3, column=0, sticky=NW)
        self.include_file_checks_cb.grid(row=4, column=0, sticky=NW)
        self.feature_subset_selections = self.create_cb_frame(
            main_frm=self.main_frm,
            cb_titles=self.feature_subset_options,
            frm_title="FEATURE SUB-SETS",
        )
        self.create_run_frm(run_function=self.run)
        # self.main_frm.mainloop()

    def run(self):
        selected_features = []
        for feature_name, feature_var in self.feature_subset_selections.items():
            if feature_var.get():
                selected_features.append(feature_name)
        if len(selected_features) == 0:
            raise InvalidInputError(
                msg="Please select at least ONE feature subset family.",
                source=self.__class__.__name__,
            )
        if (
            not self.append_to_targets_inserted_var.get()
            and not self.append_to_features_extracted_var.get()
        ):
            if not os.path.isdir(self.save_dir.folder_path):
                raise InvalidInputError(
                    msg="You must select a valid path where to save the results OR select to append the features to the targets_inserted and/or features_extracted files.",
                    source=self.__class__.__name__,
                )

        feature_extractor = FeatureSubsetsCalculator(
            config_path=self.config_path,
            feature_families=selected_features,
            save_dir=self.save_dir.folder_path,
            include_file_checks=self.include_file_checks.get(),
            append_to_features_extracted=self.append_to_features_extracted_var.get(),
            append_to_targets_inserted=self.append_to_targets_inserted_var.get(),
        )
        feature_extractor.run()


# FeatureSubsetExtractorPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
