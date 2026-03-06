import os
from copy import deepcopy
from tkinter import *
from typing import Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.EzPathPlot import EzPathPlot
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FolderSelect,
                                        SimBADropDown, SimBALabel)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_dir_exists)
from simba.utils.errors import NoFilesFoundError
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import (bgr_to_rgb_tuple, get_desktop_path,
                                    get_fn_ext, read_video_info, str_2_bool)

TIME, VELOCITY = 'TIME', 'VELOCITY'
VIDEO = 'VIDEO'
OPACITY_OPTIONS = [round(x, 2) for x in list(np.arange(0, 1.1, 0.1))]
SMOOTHING_OPTIONS = list(range(100, 21000, 1000))
SMOOTHING_OPTIONS.append('NONE')

class EzPathPlotPopUp(PopUpMixin, ConfigReader):
    """
    Tkinter pop-up for creating simple (EZ) path plots from project pose data.

    :param Union[str, os.PathLike] config_path: Path to the SimBA project config file (e.g. ``project_folder/project_config.ini``). The project must contain pose data in the outlier-corrected CSV directory.

    :example:
    >>> EzPathPlotPopUp(config_path=r'project_folder/project_config.ini')
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):
        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        if len(self.outlier_corrected_paths) == 0:
            raise NoFilesFoundError(msg=f"No data found in the {self.outlier_corrected_paths} directory. Place files in this directory to create quick path plots.", source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="SIMPLE LINE PLOT", icon='path_2')
        self.data_file_names = {get_fn_ext(filepath=i)[1]: i for i in self.outlier_corrected_paths}
        self.colors = get_color_dict()
        color_names = list(self.colors.keys())
        line_colors = deepcopy(color_names)
        line_colors.extend((TIME, VELOCITY))
        bg_colors = deepcopy(color_names)
        bg_colors.append(VIDEO)
        desktop_path = get_desktop_path(raise_error=False)

        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name='settings')
        video_names = list(self.data_file_names.keys())
        video_name_max_len = max(len(s) for s in video_names)

        self.video_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=video_names, label='VIDEO: ', label_width=25, dropdown_width=video_name_max_len, value=video_names[0], img='video_2', tooltip_key='EZ_PATH_PLOT_VIDEO')
        self.bp_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=self.body_parts_lst, label='BODY-PART: ', label_width=25, dropdown_width=video_name_max_len, value=self.body_parts_lst[0], img='pose', tooltip_key='EZ_PATH_PLOT_BODY_PART')
        self.background_color = SimBADropDown(parent=settings_frm, dropdown_options=bg_colors, label="BACKGROUND COLOR: ", label_width=25, dropdown_width=video_name_max_len, value="White", img='fill', tooltip_key='EZ_PATH_PLOT_BACKGROUND_COLOR')
        self.line_color = SimBADropDown(parent=settings_frm, dropdown_options=line_colors, label="LINE COLOR: ", label_width=25, dropdown_width=video_name_max_len, value="Red", img='line', tooltip_key='EZ_PATH_PLOT_LINE_COLOR')
        self.line_thickness = SimBADropDown(parent=settings_frm, dropdown_options=list(range(1, 11)), label="LINE THICKNESS: ", label_width=25, dropdown_width=video_name_max_len, value=1, img='bold', tooltip_key='EZ_PATH_PLOT_LINE_THICKNESS')
        self.line_opacity = SimBADropDown(parent=settings_frm, dropdown_options=OPACITY_OPTIONS, label="LINE OPACITY: ", label_width=25, dropdown_width=video_name_max_len, value=1.0, img='opacity', tooltip_key='EZ_PATH_PLOT_LINE_OPACITY')
        self.smoothing_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=SMOOTHING_OPTIONS, label="SMOOTHING (FRAMES): ", label_width=25, dropdown_width=video_name_max_len, value='NONE', img='opacity', tooltip_key='EZ_PATH_PLOT_SMOOTHING')
        self.svg_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="AS SVG: ", label_width=25, dropdown_width=video_name_max_len, value='FALSE', img='svg', tooltip_key='EZ_PATH_PLOT_SVG')
        self.save_dir = FolderSelect(parent=settings_frm, folderDescription='SAVE DIRECTORY:', initialdir=desktop_path, lblwidth=25, tooltip_key='EZ_PATH_PLOT_SAVE_DIR', lbl_icon='folder')
        self.save_dir.set_folder_path(folder_path=desktop_path)
        self.inst_lbl = SimBALabel(parent=settings_frm, txt="NOTE: For more complex path plots, faster, \n see 'CREATE PATH PLOTS' under the [VISUALIZATIONS] tab", txt_clr='green')

        settings_frm.grid(row=0, sticky=W)
        self.video_dropdown.grid(row=0, sticky=W)
        self.bp_dropdown.grid(row=2, sticky=W)
        self.background_color.grid(row=3, sticky=W)
        self.line_color.grid(row=4, sticky=W)
        self.line_thickness.grid(row=5, sticky=W)
        self.line_opacity.grid(row=6, sticky=W)
        self.smoothing_dropdown.grid(row=7, sticky=W)
        self.svg_dropdown.grid(row=8, sticky=W)
        self.save_dir.grid(row=9, sticky=W)
        self.inst_lbl.grid(row=10, sticky=W)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        video_name = self.video_dropdown.getChoices()
        data_path = self.data_file_names[video_name]
        meta_data, _, fps = read_video_info(vid_info_df=self.video_info_df, video_name=video_name)
        size = (int(meta_data["Resolution_width"]), int(meta_data["Resolution_height"]))
        bg_clr = self.background_color.get_value()
        line_color = self.line_color.get_value()


        smoothing = self.smoothing_dropdown.get_value()
        if bg_clr == VIDEO:
            bg_clr = 0
        else:
            bg_clr = bgr_to_rgb_tuple(self.colors[bg_clr])
        line_clr = line_color.lower() if line_color in (TIME, VELOCITY) else bgr_to_rgb_tuple(self.colors[line_color])


        line_thickness = float(self.line_thickness.get_value())
        bp = self.bp_dropdown.getChoices()
        smoothing = None if smoothing == 'NONE' else int(smoothing)
        line_opacity = float(self.line_opacity.get_value())
        svg = str_2_bool(input_str=self.svg_dropdown.get_value())
        save_dir = self.save_dir.folder_path
        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__}', raise_error=True)



        plotter = EzPathPlot(data_path=data_path,
                              body_part=bp,
                              bg_color=bg_clr,
                              video_dir=self.video_dir,
                              line_color=line_clr,
                              line_thickness=line_thickness,
                              svg=svg,
                              size=size,
                              line_opacity=line_opacity,
                              smoothing_time=smoothing,
                              save_dir=save_dir,
                              verbose=True)

        plotter.run()

#EzPathPlotPopUp(config_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini')