__author__ = "Simon Nilsson"

from tkinter import *
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import Keys, Links
from simba.ui.tkinter_functions import DropDownMenu, CreateLabelFrameWithIcon
from simba.utils.errors import NoROIDataError
from simba.roi_tools.ROI_size_standardizer import ROISizeStandardizer

class ROISizeStandardizerPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='ROI SIZE NORMALIZER PY PIXELS PER MILLIMETER')
        ConfigReader.__init__(self, config_path=config_path)
        check_file_exist_and_readable(file_path=self.roi_coordinates_path)
        self.read_roi_data()
        if len(self.video_names_w_rois) <= 1:
            raise NoROIDataError(f'SimBA could only find {len(self.video_names_w_rois)} videos with ROIs in your project. You need at least 2 videos with ROIs to standardize ROI sizes')
        self.video_names = list(self.video_names_w_rois)
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ROI_FEATURES_PLOT.value)
        self.baseline_video_dropdown = DropDownMenu(self.settings_frm, 'Baseline video:',  self.video_names, '12')
        self.baseline_video_dropdown.setChoices(self.video_names[0])

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.baseline_video_dropdown.grid(row=0, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run)
        #self.main_frm.mainloop()

    def run(self):
        roi_standardizer = ROISizeStandardizer(config_path=self.config_path, reference_video=self.baseline_video_dropdown.getChoices())
        roi_standardizer.run()
        roi_standardizer.save()


#ROISizeStandardizerPopUp(config_path="/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini")