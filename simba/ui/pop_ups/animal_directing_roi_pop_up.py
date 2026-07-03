import os
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.ROI_directing_analyzer import DirectingROIAnalyzer
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon,
                                        SimbaCheckbox, SimBADropDown)
from simba.utils.enums import Keys, Links
from simba.utils.errors import (CountError, NoFilesFoundError,
                                ROICoordinatesNotFoundError)
from simba.utils.lookups import find_closest_string

NOSE, EAR_LEFT, EAR_RIGHT = Keys.NOSE.value, Keys.EAR_LEFT.value, Keys.EAR_RIGHT.value


class AnimalDirectingROIPopUp(ConfigReader, PopUpMixin):

    """
    Pop-up window for analyzing directionality between animals and ROIs.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format.

    :example:

    >>> test = AnimalDirectingROIPopUp(config_path=r"C:\\troubleshooting\\two_black_animals_14bp\\project_folder\\project_config.ini")
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if len(self.outlier_corrected_paths) == 0:
            raise NoFilesFoundError(msg=f"No data files found in {self.outlier_corrected_dir}.", source=self.__class__.__name__)
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(expected_file_path=self.roi_coordinates_path)
        PopUpMixin.__init__(self, title="ANALYZE DIRECTIONALITY BETWEEN ANIMALS AND ROIs", size=(600, 400), icon='direction')

        if self.animal_cnt > 1:
            bp_names = list(set([x[:-2] for x in self.body_parts_lst]))
        else:
            bp_names = list(set(self.body_parts_lst))
        nose_guess = find_closest_string(target=NOSE, string_list=bp_names)[0]
        ear_left_guess = find_closest_string(target=EAR_LEFT, string_list=bp_names)[0]
        ear_right_guess = find_closest_string(target=EAR_RIGHT, string_list=bp_names)[0]

        self.bp_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT BODY-PARTS", icon_name='pose', icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)
        self.ear_left_dropdown = SimBADropDown(parent=self.bp_frm, dropdown_options=bp_names, label_width=30, dropdown_width=25, value=ear_left_guess, label='LEFT EAR BODY-PART NAME:', img='left_ear')
        self.ear_right_dropdown = SimBADropDown(parent=self.bp_frm, dropdown_options=bp_names, label_width=30, dropdown_width=25, value=ear_right_guess, label='RIGHT EAR BODY-PART NAME:', img='ear_right')
        self.nose_dropdown = SimBADropDown(parent=self.bp_frm, dropdown_options=bp_names, label_width=30, dropdown_width=25, value=nose_guess, label='NOSE BODY-PART NAME:', img='nose')

        self.bp_frm.grid(row=0, column=0, sticky="NW")
        self.ear_left_dropdown.grid(row=0, column=0, sticky="NW")
        self.ear_right_dropdown.grid(row=1, column=0, sticky="NW")
        self.nose_dropdown.grid(row=2, column=0, sticky="NW")

        self.data_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="DATA", icon_name='data_table', icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)
        self.detailed_table_cb, self.detailed_table_var = SimbaCheckbox(parent=self.data_frm, txt='DETAILED TABLE', val=True, txt_img='details', tooltip_key='ROI_DIRECTING_DETAILED_TABLE')
        self.agg_stats_cb, self.agg_stats_var = SimbaCheckbox(parent=self.data_frm, txt='AGGREGATE STATISTICS', val=True, txt_img='abacus_2', tooltip_key='ROI_DIRECTING_AGGREGATE_STATS')
        self.data_frm.grid(row=1, column=0, sticky="NW")
        self.detailed_table_cb.grid(row=0, column=0, sticky="NW")
        self.agg_stats_cb.grid(row=1, column=0, sticky="NW")

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings', icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)
        self.transpose_cb, self.transpose_var = SimbaCheckbox(parent=self.settings_frm, txt='TRANSPOSE AGGREGATE STATISTICS', val=True, txt_img='flip_black', tooltip_key='ROI_DIRECTING_TRANSPOSE')
        self.settings_frm.grid(row=2, column=0, sticky="NW")
        self.transpose_cb.grid(row=0, column=0, sticky="NW")

        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        nose_name = self.nose_dropdown.get_value()
        left_ear = self.ear_left_dropdown.get_value()
        ear_right = self.ear_right_dropdown.get_value()
        if len(set([nose_name, left_ear, ear_right])) != 3:
            raise CountError(msg=f'The three chosen body-parts have to be unique: Got {nose_name, left_ear, ear_right}', source=self.__class__.__name__)
        directing_roi_analyzer = DirectingROIAnalyzer(config_path=self.config_path,
                                                       left_ear_name=left_ear,
                                                       right_ear_name=ear_right,
                                                       nose_name=nose_name,
                                                       detailed_table=self.detailed_table_var.get(),
                                                       agg_stats=self.agg_stats_var.get(),
                                                       transpose_agg_stats=self.transpose_var.get())
        directing_roi_analyzer.run()
        directing_roi_analyzer.save()

#AnimalDirectingROIPopUp(config_path=r"E:\troubleshooting\mitra_emergence_hour\project_folder\project_config.ini")