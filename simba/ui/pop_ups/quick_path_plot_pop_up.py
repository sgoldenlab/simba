import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.ez_path_plot import EzPathPlot
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon,
                                        SimBADropDown, SimBALabel)
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_valid_rgb_tuple, check_int, check_str)
from simba.utils.enums import Formats
from simba.utils.errors import NoFilesFoundError
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import get_fn_ext, read_video_info


class QuickLineplotPopup(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: Union[str, os.PathLike]):
        """
        :example:
        >>> _ = QuickLineplotPopup(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
        """

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path)
        if len(self.outlier_corrected_paths) == 0:
            raise NoFilesFoundError(msg=f"No data found in the {self.outlier_corrected_paths} directory. Place files in this directory to create quick path plots.", source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="SIMPLE LINE PLOT", icon='path_2')
        self.video_filepaths = {get_fn_ext(filepath=i)[1]: i for i in self.outlier_corrected_paths}
        color_lst = list(get_color_dict().keys())
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name='settings')
        video_names = list(self.video_filepaths.keys())
        video_name_max_len = max(len(s) for s in video_names)

        self.video_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=video_names, label='VIDEO: ', label_width=25, dropdown_width=video_name_max_len, value=video_names[0])
        self.bp_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=self.body_parts_lst, label='BODY-PART: ', label_width=25, dropdown_width=video_name_max_len, value=self.body_parts_lst[0])
        self.background_color = SimBADropDown(parent=settings_frm, dropdown_options=color_lst, label="BACKGROUND COLOR: ", label_width=25, dropdown_width=video_name_max_len, value="White")
        self.line_color = SimBADropDown(parent=settings_frm, dropdown_options=color_lst, label="LINE COLOR: ", label_width=25, dropdown_width=video_name_max_len, value="Red")
        self.line_thickness = SimBADropDown(parent=settings_frm, dropdown_options=list(range(1, 11)), label="LINE THICKNESS: ", label_width=25, dropdown_width=video_name_max_len, value=1)
        self.circle_size = SimBADropDown(parent=settings_frm, dropdown_options=list(range(1, 11)), label="CIRCLE SIZE: ", label_width=25, dropdown_width=video_name_max_len, value=5)
        self.last_frm_only_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=["TRUE", "FALSE"], label="LAST FRAME ONLY: ", label_width=25, dropdown_width=video_name_max_len, value='FALSE')
        self.inst_lbl = SimBALabel(parent=settings_frm, txt="NOTE: For more complex path plots, faster, \n see 'CREATE PATH PLOTS' under the [VISUALIZATIONS] tab", txt_clr='green')

        settings_frm.grid(row=0, sticky=W)
        self.video_dropdown.grid(row=0, sticky=W)
        self.bp_dropdown.grid(row=2, sticky=W)
        self.background_color.grid(row=3, sticky=W)
        self.line_color.grid(row=4, sticky=W)
        self.line_thickness.grid(row=5, sticky=W)
        self.circle_size.grid(row=6, sticky=W)
        self.last_frm_only_dropdown.grid(row=7, sticky=W)
        self.inst_lbl.grid(row=8, sticky=W)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        video_name = self.video_dropdown.getChoices()
        data_path = self.video_filepaths[video_name]
        meta_data, _, fps = read_video_info(vid_info_df=self.video_info_df, video_name=video_name)
        size = (int(meta_data["Resolution_width"]), int(meta_data["Resolution_height"]))
        last_frm = self.last_frm_only_dropdown.getChoices()
        if last_frm == "TRUE":
            save_path = os.path.join(
                self.path_plot_dir, f"{video_name}_simple_path_plot.png"
            )
            last_frm = True
        else:
            save_path = os.path.join(
                self.path_plot_dir, f"{video_name}_simple_path_plot.mp4"
            )
            last_frm = False
        if not os.path.isdir(self.path_plot_dir):
            os.makedirs(self.path_plot_dir)
        background_color = get_color_dict()[self.background_color.getChoices()]
        line_color = get_color_dict()[self.line_color.getChoices()]
        line_thickness = self.line_thickness.getChoices()
        circle_size = self.circle_size.getChoices()
        bp = self.bp_dropdown.getChoices()
        check_int(
            name=f"{self.__class__.__name__} line_thickness",
            value=line_thickness,
            min_value=1,
        )
        check_int(
            name=f"{self.__class__.__name__} circle_size",
            value=circle_size,
            min_value=1,
        )
        check_if_valid_rgb_tuple(data=background_color)
        check_if_valid_rgb_tuple(data=line_color)
        check_str(name=f"{self.__class__.__name__} body-part", value=bp)
        plotter = EzPathPlot(
            data_path=data_path,
            size=size,
            fps=fps,
            body_part=bp,
            bg_color=background_color,
            line_color=line_color,
            line_thickness=int(line_thickness),
            circle_size=int(circle_size),
            last_frm_only=last_frm,
            save_path=save_path,
        )
        threading.Thread(target=plotter.run).start()



# _ = QuickLineplotPopup(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
# _ = QuickLineplotPopup(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# _ = QuickLineplotPopup(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
