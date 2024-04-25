__author__ = "Simon Nilsson"
import os
import threading
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.ROI_feature_analyzer import ROIFeatureCreator
from simba.utils.errors import NoROIDataError


class AppendROIFeaturesByBodyPartPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path)
        if not os.path.isfile(self.roi_coordinates_path):
            raise NoROIDataError(
                msg="SIMBA ERROR: No ROIs have been defined. Please define ROIs before appending ROI-based features",
                source=self.__class__.__name__,
            )
        PopUpMixin.__init__(
            self, config_path=config_path, title="APPEND ROI FEATURES: BY BODY-PARTS"
        )
        self.create_choose_number_of_body_parts_frm(
            project_body_parts=self.project_bps, run_function=self.run
        )
        self.main_frm.mainloop()

    def run(self):
        body_parts = []
        for bp_cnt, bp_dropdown in self.body_parts_dropdowns.items():
            body_parts.append(bp_dropdown.getChoices())

        roi_feature_creator = ROIFeatureCreator(
            config_path=self.config_path,
            body_parts=body_parts,
            data_path=None,
            append_data=True,
        )
        threading.Thread(target=roi_feature_creator.run()).start()
        self.root.destroy()


# _ = AppendROIFeaturesByBodyPartPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/open_field_below/project_folder/project_config.ini')


# _ = AppendROIFeaturesByBodyPartPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
