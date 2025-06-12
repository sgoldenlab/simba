__author__ = "Simon Nilsson"

import os.path
from tkinter import *
from typing import Union

from simba.feature_extractors.feature_subsets import FeatureSubsetsCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        FolderSelect, SimbaCheckbox)
from simba.utils.checks import check_if_dir_exists
from simba.utils.enums import Formats, Links
from simba.utils.errors import InvalidInputError, NoDataError

TWO_POINT_BP_DISTANCES = 'TWO-POINT BODY-PART DISTANCES (MM)'
WITHIN_ANIMAL_THREE_POINT_ANGLES = 'WITHIN-ANIMAL THREE-POINT BODY-PART ANGLES (DEGREES)'
WITHIN_ANIMAL_THREE_POINT_HULL = "WITHIN-ANIMAL THREE-POINT CONVEX HULL PERIMETERS (MM)"
WITHIN_ANIMAL_FOUR_POINT_HULL = "WITHIN-ANIMAL FOUR-POINT CONVEX HULL PERIMETERS (MM)"
ANIMAL_CONVEX_HULL_PERIMETER = 'ENTIRE ANIMAL CONVEX HULL PERIMETERS (MM)'
ANIMAL_CONVEX_HULL_AREA = "ENTIRE ANIMAL CONVEX HULL AREA (MM2)"
FRAME_BP_MOVEMENT = "FRAME-BY-FRAME BODY-PART MOVEMENTS (MM)"
FRAME_BP_TO_ROI_CENTER = "FRAME-BY-FRAME BODY-PART DISTANCES TO ROI CENTERS (MM)"
FRAME_BP_INSIDE_ROI = "FRAME-BY-FRAME BODY-PARTS INSIDE ROIS (BOOLEAN)"
ARENA_EDGE = "BODY-PART DISTANCES TO VIDEO FRAME EDGE (MM)"

FEATURE_FAMILIES = [TWO_POINT_BP_DISTANCES,
                    WITHIN_ANIMAL_THREE_POINT_ANGLES,
                    WITHIN_ANIMAL_THREE_POINT_HULL,
                    WITHIN_ANIMAL_FOUR_POINT_HULL,
                    ANIMAL_CONVEX_HULL_PERIMETER,
                    ANIMAL_CONVEX_HULL_AREA,
                    FRAME_BP_MOVEMENT,
                    FRAME_BP_TO_ROI_CENTER,
                    FRAME_BP_INSIDE_ROI,
                    ARENA_EDGE]

class FeatureSubsetExtractorPopUp(PopUpMixin, ConfigReader):
    """
    :example:
    >>> FeatureSubsetExtractorPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if len(self.outlier_corrected_paths) == 0:
            raise NoDataError(msg=f'Cannot append feature subsets: No data found in {self.outlier_corrected_dir} directory.', source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="EXTRACT FEATURE SUBSETS", size=(500, 500), icon='features')
        self.save_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHOOSE SAVE DIRECTORY", icon_name="save", icon_link=Links.FEATURE_SUBSETS.value)
        self.save_dir = FolderSelect(self.save_frm, "SAVE DIRECTORY:", lblwidth=20)
        self.save_frm.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=0, column=0, sticky=NW)

        self.append_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="APPEND SETTINGS", icon_name="plus_green", icon_link=Links.FEATURE_SUBSETS.value)
        self.append_to_features_cb, self.append_to_features_var = SimbaCheckbox(parent=self.append_settings_frm, txt="APPEND RESULTS TO FEATURES EXTRACTED FILES")
        self.append_to_targets_cb, self.append_to_targets_var = SimbaCheckbox(parent=self.append_settings_frm, txt="APPEND RESULTS TO TARGET INSERTED FILES")
        self.checks_cb, self.checks_var = SimbaCheckbox(parent=self.append_settings_frm, txt="INCLUDE INTEGRITY CHECKS BEFORE APPENDING NEW DATA (RECOMMENDED)")
        self.append_settings_frm.grid(row=1, column=0, sticky=NW)
        self.append_to_features_cb.grid(row=0, column=0, sticky=NW)
        self.append_to_targets_cb.grid(row=1, column=0, sticky=NW)
        self.checks_cb.grid(row=2, column=0, sticky=NW)
        self.feature_subset_selections = self.create_cb_frame(main_frm=self.main_frm, cb_titles=FEATURE_FAMILIES, frm_title="SELECT FEATURE SUB-SETS",)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        selected_features = []
        for feature_name, feature_var in self.feature_subset_selections.items():
            if feature_var.get(): selected_features.append(feature_name)
        if len(selected_features) == 0:
            raise InvalidInputError(msg="Please select at least ONE feature subset family.", source=self.__class__.__name__)
        if (not self.append_to_features_var.get() and not self.append_to_targets_var.get()):
            if not os.path.isdir(self.save_dir.folder_path):
                raise InvalidInputError(msg="You must select a valid path where to save the results OR select to append the features to the targets_inserted and/or features_extracted files.", source=self.__class__.__name__)
        if not os.path.isdir(self.save_dir.folder_path):
            save_dir = None
        else:
            save_dir = self.save_dir.folder_path
        feature_extractor = FeatureSubsetsCalculator(config_path=self.config_path,
                                                     feature_families=selected_features,
                                                     save_dir=save_dir,
                                                     file_checks=self.checks_var.get(),
                                                     append_to_features_extracted=self.append_to_features_var.get(),
                                                     append_to_targets_inserted=self.append_to_targets_var.get())
        feature_extractor.run()


#FeatureSubsetExtractorPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
