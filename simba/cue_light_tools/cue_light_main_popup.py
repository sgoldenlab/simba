__author__ = "Simon Nilsson"

import glob
import itertools
import os
import webbrowser
from tkinter import *
from typing import List, Union

from simba.cue_light_tools.cue_light_analyzer import CueLightAnalyzer
from simba.cue_light_tools.cue_light_clf_statistics import CueLightClfAnalyzer
from simba.cue_light_tools.cue_light_movement_statistics import \
    CueLightMovementAnalyzer
from simba.cue_light_tools.cue_light_visualizer import CueLightVisualizer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.pop_ups.cue_light_clf_analyzer_popup import \
    CueLightClfAnalyzerPopUp
from simba.ui.pop_ups.cue_light_data_analyzer_popup import \
    CueLightDataAnalyzerPopUp
from simba.ui.pop_ups.cue_light_movement_analyzer_popup import \
    CueLightMovementAnalyzerPopUp
from simba.ui.pop_ups.cue_light_visualizer_popup import CueLightVisulizerPopUp
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, SimbaButton,
                                        SimBADropDown, SimBALabel)
from simba.utils.checks import check_float, check_int
from simba.utils.enums import Keys, Links
from simba.utils.errors import (CountError, NoChoosenClassifierError,
                                NoFilesFoundError, NoROIDataError)
from simba.utils.read_write import (find_video_of_file, get_all_clf_names,
                                    get_fn_ext, read_config_entry)


class CueLightMainPopUp(ConfigReader, PopUpMixin):
    """
    Launch cue light analysis GUI in SimBA.

    :parameter str config_path: path to SimBA project config file in Configparser format

    .. note::
       `Cue light tutorials <https://github.com/sgoldenlab/simba/blob/master/docs/cue_light_tutorial.md>`__.


    :examples:
    >>> cue_light_gui = CueLightAnalyzerMenu(config_path='MySimBAConfigPath')
    >>> cue_light_gui.main_frm.mainloop()
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path)
        if len(self.outlier_corrected_movement_paths):
            raise NoFilesFoundError(msg=f'No data found in the {self.outlier_corrected_dir} directory. Cannot compute cue-lights statistics.', source=self.__class__.__name__)
        if not os.path.isfile(self.roi_coordinates_path):
            raise NoROIDataError(msg=f'No ROIs found in project ({self.roi_coordinates_path} file not found). Draw ROIs before computing cue-light statistics', source=self.__class__.__name__)
        self.read_roi_data()
        PopUpMixin.__init__(self, size=(750, 300), title="SIMBA CUE LIGHT ANALYZER")
        self.max_len, self.cue_light_dropdowns = max(len(s) for s in self.roi_names), []
        self.roi_names = ['cl', 'cl2_']
        self._get_cue_lights_frm()

    def _get_cue_lights_frm(self):

        def _get_cue_light_names(cue_light_cnt: int = 1):
            for i in self.cue_light_dropdowns:
                i.destroy()
            self.cue_light_dropdowns = []
            for j in range(cue_light_cnt):
                cue_light_dropdown = SimBADropDown(parent=self.cue_light_settings_frm, dropdown_options=self.roi_names, label=f'CUE LIGHT {j+1}', label_width=20, dropdown_width=self.max_len + 2, value=self.roi_names[j])
                self.cue_light_dropdowns.append(cue_light_dropdown)
                cue_light_dropdown.grid(row=j+1, column=0, sticky=NW)

        self.cue_light_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="DEFINE CUE LIGHTS", icon_name='light_bulb', icon_link=Links.CUE_LIGHTS.value)
        self.cue_light_cnt_dropdown = SimBADropDown(parent=self.cue_light_settings_frm, dropdown_options=list(range(1, len(self.roi_names)+1)), label='# CUE LIGHTS', label_width=20, dropdown_width=self.max_len+2, value=1, command= lambda x: _get_cue_light_names(int(x)))
        self.cue_light_settings_frm.grid(row=0, column=0, sticky=NW, padx=(0, 10))
        self.cue_light_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        _get_cue_light_names()

        self.analyze_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="ANALYZE", icon_name='analyze_blue', icon_link=Links.CUE_LIGHTS.value)
        self.analyze_cue_light_data_btn = SimbaButton(parent=self.analyze_frm, txt="ANALYZE CUE LIGHT DATA", img='data_table', cmd=self._initialize_analysis_pop_up)
        self.visualize_cue_light_data_btn = SimbaButton(parent=self.analyze_frm, txt="VISUALIZE CUE LIGHT DATA", img='eye', cmd=self._initialize_visualize_pop_up)
        self.analyze_cue_light_movement_btn = SimbaButton(parent=self.analyze_frm, txt="ANALYZE CUE LIGHT MOVEMENT", img='analyze_green', cmd=self._initialize_movement_pop_up)
        self.analyze_cue_light_clf_btn = SimbaButton(parent=self.analyze_frm, txt="ANALYZE CUE LIGHT CLASSIFICATIONS", img='clf_2', cmd=self._initialize_clf_pop_up)
        lbl_info = SimBALabel(parent=self.analyze_frm, txt="[CLICK HERE TO LEARN ABOUT CUE LIGHT ANALYSIS]", link=Links.CUE_LIGHTS.value, cursor="hand2", txt_clr="blue")


        self.analyze_frm.grid(row=0, column=1, sticky=NW)
        lbl_info.grid(row=0, column=0, sticky=NW)
        self.analyze_cue_light_data_btn.grid(row=1, column=0, sticky=W)
        self.visualize_cue_light_data_btn.grid(row=2, column=0, sticky=W)
        self.analyze_cue_light_movement_btn.grid(row=3, column=0, sticky=W)
        self.analyze_cue_light_clf_btn.grid(row=4, column=0, sticky=W)

        mainloop()

    def __get_cue_light_names(self):
        return [x.get_value() for x in self.cue_light_dropdowns]

    def _initialize_analysis_pop_up(self):
        cue_light_names = self.__get_cue_light_names()
        _ = CueLightDataAnalyzerPopUp(config_path=self.config_path, cue_light_names=cue_light_names, data_dir=self.outlier_corrected_dir)

    def _initialize_visualize_pop_up(self):
        cue_light_names = self.__get_cue_light_names()
        _ = CueLightVisulizerPopUp(config_path=self.config_path, cue_light_names=cue_light_names)

    def _initialize_movement_pop_up(self):
        cue_light_names = self.__get_cue_light_names()
        _ = CueLightMovementAnalyzerPopUp(config_path=self.config_path, cue_light_names=cue_light_names)

    def _initialize_clf_pop_up(self):
        cue_light_names = self.__get_cue_light_names()
        _ = CueLightClfAnalyzerPopUp(config_path=self.config_path, cue_light_names=cue_light_names)



test = CueLightMainPopUp(config_path=r"C:\troubleshooting\cue_light\t1\project_folder\project_config.ini")
test.main_frm.mainloop()
