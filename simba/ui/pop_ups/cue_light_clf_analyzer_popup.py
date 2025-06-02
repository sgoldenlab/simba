import os
from tkinter import *
from typing import List, Optional, Union

from simba.data_processors.cue_light_clf_statistics import CueLightClfAnalyzer
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaCheckbox, SimBADropDown)
from simba.utils.checks import check_if_dir_exists, check_int, check_valid_lst
from simba.utils.enums import Links
from simba.utils.errors import NoDataError, NoFilesFoundError, NoROIDataError
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    str_2_bool)


class CueLightClfAnalyzerPopUp(ConfigReader, PopUpMixin):

    """
    :example:
    >>> CueLightClfAnalyzerPopUp(config_path=r"C:\troubleshooting\cue_light\t1\project_folder\project_config.ini", cue_light_names=['cl'])
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
        if len(missing_cue_lights) > 0: raise NoROIDataError(msg=f'{len(missing_cue_lights)} cue-lights are not drawn in the SimBA project: cannot be found in the {self.roi_coordinates_path} file', source=self.__class__.__name__)

        if data_dir is None:
            self.data_dir = self.cue_lights_data_dir
        else:
            check_if_dir_exists(in_dir=data_dir)
            self.data_dir = data_dir

        self.data_paths = find_files_of_filetypes_in_directory(directory=self.data_dir, extensions=[f'.{self.file_type}'], raise_error=False, as_dict=True)
        file_cnt = len(list(self.data_paths.keys())) if type(self.data_paths) == dict else len(self.data_paths)
        if file_cnt == 0:
            raise NoFilesFoundError(msg=f'No data files exist in the {self.data_dir} directory: ANALYZE CUE LIGHT DATA before performing cue light classification analysis.', source=self.__class__.__name__)

        PopUpMixin.__init__(self, size=(750, 300), title="SIMBA CUE LIGHT CLASSIFICATION ANALYZER", icon='clf_2')
        self.clf_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CLASSIFIERS", icon_name='clf_2', icon_link=Links.CUE_LIGHTS.value)
        self.clf_vars = {}
        for cnt, clf in enumerate(self.clf_names):
            clf_cb, clf_var = SimbaCheckbox(parent=self.clf_frm, txt=clf, val=True)
            clf_cb.grid(row=cnt, column=0, sticky=NW)
            self.clf_vars[clf] = clf_var
        self.clf_frm.grid(row=0, column=0, sticky=NW)

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings', icon_link=Links.CUE_LIGHTS.value)
        self.pre_window_eb = Entry_Box(parent=self.settings_frm, fileDescription='PRE CUE LIGHT WINDOW (S): ', entry_box_width=30, value=0, validation='numeric', labelwidth="40", justify='center')
        self.post_window_eb = Entry_Box(parent=self.settings_frm, fileDescription='POST CUE LIGHT WINDOW (S): ', entry_box_width=30, value=0, validation='numeric', labelwidth="40", justify='center')

        self.settings_frm.grid(row=1, column=0, sticky=NW)
        self.pre_window_eb.grid(row=0, column=0, sticky=NW)
        self.post_window_eb.grid(row=1, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

        self.main_frm.mainloop()


    def run(self):
        pre_window = self.pre_window_eb.entry_get.strip()
        post_window = self.post_window_eb.entry_get.strip()
        check_int(name='PRE CUE LIGHT WINDOW (S): ', value=pre_window, min_value=0, raise_error=False)
        check_int(name='POST CUE LIGHT WINDOW (S): ', value=post_window, min_value=0, raise_error=False)
        clf_names = [x for x in self.clf_vars.keys() if self.clf_vars[x].get()]
        if len(clf_names) == 0:
            raise NoDataError(msg='Zero classifiers checked. Check at least one classifier.', source=self.__class__.__name__)
        analyzer = CueLightClfAnalyzer(config_path=self.config_path,
                                       cue_light_names=self.cue_light_names,
                                       clf_names=clf_names,
                                       pre_window=int(pre_window),
                                       post_window=int(post_window))
        analyzer.run()
        analyzer.save()


#CueLightClfAnalyzerPopUp(config_path=r"C:\troubleshooting\cue_light\t1\project_folder\project_config.ini", cue_light_names=['MY_CUE_LIGHT'])
