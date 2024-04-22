import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.ez_path_plot import EzPathPlot
from simba.ui.tkinter_functions import DropDownMenu
from simba.utils.checks import (check_file_exist_and_readable,
                                check_if_valid_rgb_tuple, check_int, check_str)
from simba.utils.errors import NoFilesFoundError
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import (find_all_videos_in_directory, get_fn_ext,
                                    read_video_info)


class QuickLineplotPopup(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        """
        :example:
        >>> _ = QuickLineplotPopup(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
        """

        check_file_exist_and_readable(file_path=config_path)
        PopUpMixin.__init__(self, title="SIMPLE LINE PLOT")
        ConfigReader.__init__(self, config_path=config_path)
        if len(self.outlier_corrected_paths) == 0:
            raise NoFilesFoundError(
                msg=f"No data found in the {self.outlier_corrected_paths} directory. Place files in this directory to create quick path plots."
            )
        self.video_filepaths = {
            get_fn_ext(filepath=i)[1]: i for i in self.outlier_corrected_paths
        }
        settings_frm = LabelFrame(self.main_frm, text="SETTINGS")
        color_lst = list(get_color_dict().keys())
        self.video_dropdown = DropDownMenu(
            settings_frm, "VIDEO: ", list(self.video_filepaths.keys()), "18"
        )
        self.video_dropdown.setChoices(list(self.video_filepaths.keys())[0])
        self.bp_dropdown = DropDownMenu(
            settings_frm, "BODY-PART: ", self.body_parts_lst, "18"
        )
        self.bp_dropdown.setChoices(self.body_parts_lst[0])
        self.background_color = DropDownMenu(
            settings_frm, "BACKGROUND COLOR: ", color_lst, "18"
        )
        self.background_color.setChoices(choice="White")
        self.line_color = DropDownMenu(settings_frm, "LINE COLOR: ", color_lst, "18")
        self.line_color.setChoices(choice="Red")
        self.line_thickness = DropDownMenu(
            settings_frm, "LINE THICKNESS: ", list(range(1, 11)), "18"
        )
        self.line_thickness.setChoices(choice=1)
        self.circle_size = DropDownMenu(
            settings_frm, "CIRCLE SIZE: ", list(range(1, 11)), "18"
        )
        self.circle_size.setChoices(choice=5)
        self.last_frm_only_dropdown = DropDownMenu(
            settings_frm, "LAST FRAME ONLY: ", ["TRUE", "FALSE"], "18"
        )
        self.last_frm_only_dropdown.setChoices("FALSE")
        settings_frm.grid(row=0, sticky=W)
        self.video_dropdown.grid(row=0, sticky=W)
        self.bp_dropdown.grid(row=2, sticky=W)
        self.background_color.grid(row=3, sticky=W)
        self.line_color.grid(row=4, sticky=W)
        self.line_thickness.grid(row=5, sticky=W)
        self.circle_size.grid(row=6, sticky=W)
        self.last_frm_only_dropdown.grid(row=7, sticky=W)
        Label(
            settings_frm,
            fg="green",
            text=" NOTE: For more complex path plots, faster, \n see 'CREATE PATH PLOTS' under the [VISUALIZATIONS] tab",
        ).grid(row=8, sticky=W)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        video_name = self.video_dropdown.getChoices()
        data_path = self.video_filepaths[video_name]
        meta_data, _, fps = read_video_info(
            vid_info_df=self.video_info_df, video_name=video_name
        )
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


# _ = QuickLineplotPopup(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# _ = QuickLineplotPopup(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
