__author__ = "Simon Nilsson"



import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.data_plotter import DataPlotter
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        SimbaButton, SimbaCheckbox,
                                        SimBADropDown)
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import (DuplicationError, InvalidInputError,
                                NoFilesFoundError)
from simba.utils.read_write import get_fn_ext

DECIMALS_OPTIONS = list(range(0, 11))
THICKNESS_OPTIONS = list(range(0, 11))

class DataPlotterPopUp(PopUpMixin, ConfigReader):

    """
    :example:
    >>> _ = DataPlotterPopUp(config_path=r'C:\troubleshooting\RAT_NOR\project_folder\project_config.ini')
    """
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.animal_cnt_options = list(range(1, self.animal_cnt + 1))
        if len(self.outlier_corrected_paths) == 0:
            raise NoFilesFoundError(msg=f'Cannot create data plots: No data files found in {self.outlier_corrected_dir} directory.', source=self.__class__.__name__)
        self.file_paths_dict = {get_fn_ext(x)[1]: x for x in self.outlier_corrected_paths}
        self.max_len = max(len(s) for s in list(self.file_paths_dict.keys())) + 5

        PopUpMixin.__init__(self, title="CREATE DATA PLOTS", icon='data_table')
        self.color_lst = list(self.colors_dict.keys())

        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DATA_TABLES.value, pady=5, padx=5, relief='solid')
        self.resolution_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.resolutions, label='RESOLUTION:', label_width=30, dropdown_width=20, value=self.resolutions[1])
        self.rounding_decimals_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=DECIMALS_OPTIONS, label='DECIMAL ACCURACY:', label_width=30, dropdown_width=20, value=2)
        self.background_color_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.color_lst, label='BACKGROUND COLOR:', label_width=30, dropdown_width=20, value="White")
        self.header_color_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.color_lst, label='HEADER COLOR:', label_width=30, dropdown_width=20, value="Black")
        self.font_thickness_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=THICKNESS_OPTIONS, label='FONT THICKNESS: ', label_width=30, dropdown_width=20, value=1)
        self.style_settings_frm.grid(row=0, sticky=NW, pady=10, padx=10)
        self.resolution_dropdown.grid(row=0, sticky=NW)
        self.rounding_decimals_dropdown.grid(row=1, sticky=NW)
        self.background_color_dropdown.grid(row=2, sticky=NW)
        self.header_color_dropdown.grid(row=3, sticky=NW)
        self.font_thickness_dropdown.grid(row=4, sticky=NW)


        self.body_parts_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHOOSE BODY-PARTS", icon_name='pose', pady=5, padx=5, relief='solid')
        self.number_of_animals_dropdown = SimBADropDown(parent=self.body_parts_frm, dropdown_options=THICKNESS_OPTIONS, label='# ANIMALS: ', label_width=30, dropdown_width=20, value=1, command=self._create_bp_menu)
        self.body_parts_frm.grid(row=1, column=0, sticky=NW, pady=10, padx=10)
        self.number_of_animals_dropdown.grid(row=0, column=0, sticky=NW)
        self._create_bp_menu(x=1)

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZATION SETTINGS", icon_name='eye', pady=5, padx=5, relief='solid')
        data_frames_cb, self.data_frames_var = SimbaCheckbox(parent=self.settings_frm, txt="CREATE FRAMES", txt_img='frames', val=False)
        data_videos_cb, self.data_videos_var = SimbaCheckbox(parent=self.settings_frm, txt="CREATE VIDEOS", txt_img='video', val=True)

        self.settings_frm.grid(row=2, column=0, sticky=NW, pady=10, padx=10)
        data_frames_cb.grid(row=0, column=0, sticky=NW)
        data_videos_cb.grid(row=1, column=0, sticky=NW)

        self.run_single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name='video', pady=5, padx=5, relief='solid')
        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, img='rocket', txt="CREATE SINGLE VIDEO", font=Formats.FONT_REGULAR.value, cmd=self._run, cmd_kwargs={'multiple': False})
        self.single_video_dropdown = SimBADropDown(parent=self.run_single_video_frm, dropdown_options=list(self.file_paths_dict.keys()), label='VIDEO: ', label_width=30, dropdown_width=self.max_len, value=list(self.file_paths_dict.keys())[0])

        self.run_single_video_frm.grid(row=3, column=0, sticky=NW, pady=10, padx=10)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS", icon_name='stack', pady=5, padx=5, relief='solid')
        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, img='rocket', txt=f"CREATE MULTIPLE VIDEOS ({len(self.outlier_corrected_paths)} video(s) found)", font=Formats.FONT_REGULAR.value, cmd=self._run, cmd_kwargs={'multiple': True})

        self.run_multiple_videos.grid(row=4, column=0, sticky=NW, pady=10, padx=10)
        self.run_multiple_video_btn.grid(row=0, column=0, sticky=NW)

        #self.main_frm.mainloop()

    def _create_bp_menu(self, x):
        if hasattr(self, "bp_dropdowns"):
            for k, v in self.bp_dropdowns.items():
                self.bp_dropdowns[k].destroy()
                self.bp_colors[k].destroy()
        self.bp_dropdowns, self.bp_colors = {}, {}
        for animal_cnt in range(int(self.number_of_animals_dropdown.getChoices())):
            self.bp_dropdowns[animal_cnt] = SimBADropDown(parent=self.body_parts_frm, dropdown_options=self.body_parts_lst, label=f"BODY-PART {animal_cnt+1}:", label_width=30, dropdown_width=20, value=self.body_parts_lst[animal_cnt])
            self.bp_colors[animal_cnt] = SimBADropDown(parent=self.body_parts_frm, dropdown_options=self.color_lst, label=f"", label_width=2, dropdown_width=20, value=self.color_lst[animal_cnt])
            self.bp_dropdowns[animal_cnt].grid(row=animal_cnt + 1, column=0, sticky=NW)
            self.bp_colors[animal_cnt].grid(row=animal_cnt + 1, column=1, sticky=NW)

    def _run(self, multiple: bool):
        if multiple:
            data_paths = self.outlier_corrected_paths
        else:
            data_paths = [self.file_paths_dict[self.single_video_dropdown.getChoices()]]
        print(data_paths)
        width = int(self.resolution_dropdown.getChoices().split("×")[0])
        height = int(self.resolution_dropdown.getChoices().split("×")[1])
        body_part_attr = []
        for k, v in self.bp_dropdowns.items():
            body_part_attr.append((v.getChoices(), self.colors_dict[self.bp_colors[k].getChoices()]))
        selected_bps = [x[0] for x in body_part_attr]
        if len(list(set(selected_bps))) != len(body_part_attr):
            raise InvalidInputError(msg=f"Please choose unique only body-parts for plotting. Got {len(body_part_attr)} body-parts but only {len(list(set(selected_bps)))} unique", source=DataPlotterPopUp.__class__.__name__)

        bg_color = self.colors_dict[self.background_color_dropdown.getChoices()]
        header_color = self.colors_dict[self.header_color_dropdown.getChoices()]
        font_thickness = int(self.font_thickness_dropdown.getChoices())
        size = (int(width), int(height))
        decimals = int(self.rounding_decimals_dropdown.getChoices())

        if bg_color == header_color:
            raise DuplicationError(msg=f'The header color cannot be the same color as the background color: {bg_color}' , source=self.__class__.__name__)
        for c, i in enumerate(body_part_attr):
            if i[1] == bg_color:
                raise DuplicationError(msg=f'The text color for animal {c+1} cannot be the same color as the background color: {i[1]}', source=self.__class__.__name__)



        if not self.data_videos_var.get() and not self.data_frames_var.get():
            raise InvalidInputError(msg='Both frames and video is set to False, please select one.', source=self.__class__.__name__)

        data_plotter = DataPlotter(config_path=self.config_path,
                                   body_parts=body_part_attr,
                                   data_paths=data_paths,
                                   bg_clr=bg_color,
                                   header_clr=header_color,
                                   font_thickness=font_thickness,
                                   img_size=size,
                                   decimals=decimals,
                                   video_setting=self.data_videos_var.get(),
                                   frame_setting=self.data_frames_var.get())

        _ = data_plotter.run()

#_ = DataPlotterPopUp(config_path=r"C:\troubleshooting\SDS_pre_post\project_folder\project_config.ini")
# _ = DataPlotterPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
