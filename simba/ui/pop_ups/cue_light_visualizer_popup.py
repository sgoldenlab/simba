import os
from tkinter import *
from typing import List, Optional, Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.cue_light_visualizer import CueLightVisualizer
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        SimBADropDown)
from simba.utils.checks import check_if_dir_exists, check_valid_lst
from simba.utils.enums import Links, Options
from simba.utils.errors import NoFilesFoundError, NoROIDataError
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    str_2_bool)


class CueLightVisulizerPopUp(ConfigReader, PopUpMixin):

    """
    :example:
    >>> CueLightVisulizerPopUp(config_path=r"C:\troubleshooting\cue_light\t1\project_folder\project_config.ini", cue_light_names=['cl'])
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
            self.data_dir = data_dir

        if not os.path.isdir(self.data_dir):
            raise NoFilesFoundError( msg=f' Directory {self.data_dir} NOT DETECTED: ANALYZE CUE LIGHT DATA before visualizing data', source=self.__class__.__name__)

        self.data_paths = find_files_of_filetypes_in_directory(directory=self.data_dir, extensions=[f'.{self.file_type}'], raise_error=False, as_dict=True)
        file_cnt = len(list(self.data_paths.keys())) if type(self.data_paths) == dict else len(self.data_paths)
        if file_cnt == 0:
            raise NoFilesFoundError(msg=f'No data files exist in the {self.data_dir} directory: ANALYZE CUE LIGHT DATA before visualizing data', source=self.__class__.__name__)

        self.video_paths = find_files_of_filetypes_in_directory(directory=self.video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=False, as_dict=True)
        self.video_paths = {k: v for k, v in self.video_paths.items() if k in self.data_paths.keys()}
        if len(list((self.video_paths.keys()))) == 0:
            raise NoFilesFoundError(msg=f'None of the data files in the {self.data_dir} directory has a video represented in the {self.video_dir} directory', source=self.__class__.__name__)
        max_len = max(len(s) for s in list(self.video_paths.keys()))
        PopUpMixin.__init__(self, size=(750, 300), title="CUE LIGHT VISUALIZER", icon='eye')
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings', icon_link=Links.CUE_LIGHTS.value)
        self.pose_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label=f'SHOW POSE:', label_width=30, dropdown_width=15, value='TRUE')
        self.core_cnt_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(1, self.cpu_cnt + 1)), label=f'CPU CORE COUNT:', label_width=30, dropdown_width=15, value=int(self.cpu_cnt / 2))
        self.frames_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label=f'CREATE INDIVIDUAL FRAMES:', label_width=30, dropdown_width=15, value='FALSE')
        self.create_video_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label=f'CREATE VIDEO:', label_width=30, dropdown_width=15, value='TRUE')

        self.verbose_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label=f'VERBOSE:', label_width=30, dropdown_width=15, value='TRUE')

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.pose_dropdown.grid(row=0, column=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=1, column=0, sticky=NW)
        self.create_video_dropdown.grid(row=2, column=0, sticky=NW)
        self.frames_dropdown.grid(row=3, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=4, column=0, sticky=NW)

        self.select_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT VIDEO", icon_name='video', icon_link=Links.CUE_LIGHTS.value)
        self.select_video_dropdown = SimBADropDown(parent=self.select_video_frm, dropdown_options=list(self.video_paths.keys()), label=f'SELECT VIDEO:', label_width=30, dropdown_width=max_len+5, value=list(self.video_paths.keys())[0])

        self.select_video_frm.grid(row=1, column=0, sticky=NW)
        self.select_video_dropdown.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        #self.main_frm.mainloop()

    def run(self):
        show_pose = str_2_bool(self.pose_dropdown.get_value())
        core_cnt = int(self.core_cnt_dropdown.get_value())
        frame_setting = str_2_bool(self.frames_dropdown.get_value())
        video_setting = str_2_bool(self.create_video_dropdown.get_value())
        video_name = self.select_video_dropdown.get_value()
        verbose = str_2_bool(self.verbose_dropdown.get_value())


        data_path = self.data_paths[video_name]
        video_path = self.video_paths[video_name]
        visualizer = CueLightVisualizer(config_path=self.config_path,
                                        cue_light_names=self.cue_light_names,
                                        video_path=video_path,
                                        data_path=data_path,
                                        frame_setting=frame_setting,
                                        video_setting=video_setting,
                                        core_cnt=core_cnt,
                                        show_pose=show_pose)
        visualizer.run()



#CueLightVisulizerPopUp(config_path=r"C:\troubleshooting\cue_light\t1\project_folder\project_config.ini", cue_light_names=['cl'])