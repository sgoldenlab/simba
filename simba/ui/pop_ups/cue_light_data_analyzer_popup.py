import os
from tkinter import *
from typing import List, Optional, Union

from simba.data_processors.cue_light_analyzer import CueLightAnalyzer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, SimBADropDown
from simba.utils.checks import check_if_dir_exists, check_valid_lst
from simba.utils.enums import Links
from simba.utils.errors import NoFilesFoundError, NoROIDataError
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    str_2_bool)


class CueLightDataAnalyzerPopUp(ConfigReader, PopUpMixin):

    """
    :example:
    >>> CueLightDataAnalyzerPopUp(config_path=r"C:\troubleshooting\cue_light\t1\project_folder\project_config.ini", cue_light_names=['cl'])
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 cue_light_names: List[str],
                 data_dir: Optional[Union[str, os.PathLike]] = None):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=True)
        check_valid_lst(data=cue_light_names, source=self.__class__.__name__, valid_dtypes=(str,), min_len=1, raise_error=True)
        self.read_roi_data()
        self.cue_light_names = cue_light_names
        missing_cue_lights = [x for x in cue_light_names if x not in self.roi_names]
        if len(missing_cue_lights) > 0: raise NoROIDataError(msg=f'{len(missing_cue_lights)} cue-lights are not drawn in the SimBA project. The {missing_cue_lights} ROIs cannot be found in the {self.roi_coordinates_path} file.', source=self.__class__.__name__)

        if data_dir is None:
            self.data_dir = self.outlier_corrected_dir
        else:
            check_if_dir_exists(in_dir=data_dir)
            self.data_dir = data_dir

        self.data_paths = find_files_of_filetypes_in_directory(directory=self.data_dir, extensions=[f'.{self.file_type}'], raise_error=False, as_dict=True)
        if len(list(self.data_paths.keys())) == 0 or not os.path.isdir(self.data_dir):
            raise NoFilesFoundError(msg=f'No files detected in directory {self.data_dir}. Perform (or skip outlier correction on imported pose data before performing cue-ligh analysis', source=self.__class__.__name__)

        PopUpMixin.__init__(self, size=(750, 300), title="CUE LIGHT DATA ANALYSIS", icon='data_table')
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings', icon_link=Links.CUE_LIGHTS.value)

        self.details_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label=f'COMPUTE DETAILED CUE LIGHT BOUT DATA:', label_width=40, dropdown_width=15, value='TRUE')
        self.core_cnt_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(1, self.cpu_cnt+1)), label=f'CPU CORE COUNT:', label_width=40, dropdown_width=15, value=int(self.cpu_cnt/2))
        self.verbose_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label=f'VERBOSE:', label_width=40, dropdown_width=15, value='TRUE')

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.details_dropdown.grid(row=0, column=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=1, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=2, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run)
        #self.main_frm.mainloop()

    def run(self):
        details = str_2_bool(self.details_dropdown.get_value())
        core_cnt = int(self.core_cnt_dropdown.get_value())
        verbose = str_2_bool(self.verbose_dropdown.get_value())

        cue_light_analyzer = CueLightAnalyzer(config_path=self.config_path,
                                              cue_light_names=self.cue_light_names,
                                              detailed_data=details,
                                              core_cnt=core_cnt,
                                              data_dir=self.outlier_corrected_dir,
                                              verbose=verbose)
        cue_light_analyzer.run()