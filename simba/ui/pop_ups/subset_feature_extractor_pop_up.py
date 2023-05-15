__author__ = "Simon Nilsson"

from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.enums import Links
from simba.ui.tkinter_functions import DropDownMenu, CreateLabelFrameWithIcon, FolderSelect
from simba.utils.checks import check_if_dir_exists
from simba.feature_extractors.feature_subsets import FeatureSubsetsCalculator



class FeatureSubsetExtractorPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):
        PopUpMixin.__init__(self, title='EXTRACT FEATURE SUBSETS')
        ConfigReader.__init__(self, config_path=config_path)
        self.feature_subset_options = ['Two-point body-part distances (mm)',
                                       'Within-animal three-point body-part angles (degrees)',
                                       'Within-animal three-point convex hull perimeters (mm)',
                                       'Within-animal four-point convex hull perimeters (mm)',
                                       'Entire animal convex hull perimeters (mm)',
                                       'Entire animal convex hull area (mm2)',
                                       'Frame-by-frame body-part movements (mm)',
                                       'Frame-by-frame distance to ROI centers (mm)',
                                       'Frame-by-frame body-parts inside ROIs (Boolean)']

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name='documentation', icon_link=Links.FEATURE_SUBSETS.value)
        self.feature_family_dropdown = DropDownMenu(self.settings_frm, 'FEATURE FAMILY:', self.feature_subset_options, '20')
        self.feature_family_dropdown.setChoices(self.feature_subset_options[0])
        self.save_dir = FolderSelect(self.settings_frm, 'SAVE DIRECTORY:', lblwidth=20)
        self.create_run_frm(run_function=self.run)

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.feature_family_dropdown.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)

        #self.main_frm.mainloop()

    def run(self):
        check_if_dir_exists(in_dir=self.save_dir.folder_path)
        feature_extractor = FeatureSubsetsCalculator(config_path=self.config_path,
                                                     feature_family=self.feature_family_dropdown.getChoices(),
                                                     save_dir=self.save_dir.folder_path)
        feature_extractor.run()