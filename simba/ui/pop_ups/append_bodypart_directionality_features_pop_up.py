__author__ = "Tzuk Polinsky"

import glob
import os

import pandas

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.ROI_feature_analyzer import ROIFeatureCreator
from simba.utils.errors import NoROIDataError


class AppendBodyPartDirectionalityFeaturesPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, config_path=config_path, title="APPEND BODY PART DIRECTIONALITY FEATURES")
        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise NoROIDataError(
                msg="SIMBA ERROR: No ROIs have been defined. Please define ROIs before appending ROI-based features"
            )
        self.create_choose_number_of_body_parts_directionality_frm(
            path_to_directionality_dir=self.body_part_directionality_df_dir, run_function=self.run
        )
        # self.main_frm.mainloop()

    def run(self):
        settings = {}
        settings["body_parts_directionality"] = {}
        for bp_cnt, bp_dropdown in self.bp_cnt_dropdown.items():
            settings["body_parts_directionality"] = bp_dropdown.getChoices()
        directionality_data_path = os.path.join(self.body_part_directionality_df_dir,
                                                settings["body_parts_directionality"])
        data_dic = {}
        for root, dirs, files in os.walk(directionality_data_path):
            for file in files:
                data = pandas.read_csv(os.path.join(root, file))["Directing_BOOL"]
                data_dic[file] = data
        files_found = glob.glob(
            self.outlier_corrected_dir + "/*." + self.file_type
        )
        concatenate_data = {}
        for file in files_found:
            data = pandas.read_csv(os.path.join(root, file))
            c_data = pandas.concat([data, data_dic[file]])
            concatenate_data[file] = c_data
        for file_name, data in concatenate_data.items():
            save_path = os.path.join(
                self.features_dir, file_name + "." + self.file_type
            )
            data.to_csv(save_path)

        # roi_feature_creator = ROIFeatureCreator(
        #     config_path=self.config_path, settings=settings
        # )
        # roi_feature_creator.run()
        # roi_feature_creator.save()

# _ = AppendROIFeaturesByBodyPartPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_animals_16bp_032023/project_folder/project_config.ini')
