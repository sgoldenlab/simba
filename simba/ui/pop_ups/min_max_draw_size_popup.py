import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, SimbaButton,
                                        SimBADropDown)
from simba.utils.checks import check_float
from simba.utils.enums import ConfigKey, Dtypes, Formats
from simba.utils.printing import stdout_success
from simba.utils.read_write import read_config_entry

WINDOW_SIZE_OPTIONS = [round(x * 0.05, 2) for x in range(21)]

class SetMinMaxDrawWindowSize(ConfigReader, PopUpMixin):


    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        PopUpMixin.__init__(self, title="SET DRAWING WINDOW SIZE", size=(720, 960), icon='monitor')

        min_draw_display_ratio_h = read_config_entry(config=self.config, section=ConfigKey.DISPLAY_SETTINGS.value, option=ConfigKey.MIN_ROI_DISPLAY_HEIGHT.value, default_value=0.60, data_type=Dtypes.FLOAT.value)
        min_draw_display_ratio_w = read_config_entry(config=self.config, section=ConfigKey.DISPLAY_SETTINGS.value, option=ConfigKey.MIN_ROI_DISPLAY_WIDTH.value, default_value=0.30, data_type=Dtypes.FLOAT.value)
        max_draw_display_ratio_h = read_config_entry(config=self.config, section=ConfigKey.DISPLAY_SETTINGS.value, option=ConfigKey.MAX_ROI_DISPLAY_HEIGHT.value, default_value=0.75, data_type=Dtypes.FLOAT.value)
        max_draw_display_ratio_w = read_config_entry(config=self.config, section=ConfigKey.DISPLAY_SETTINGS.value, option=ConfigKey.MAX_ROI_DISPLAY_WIDTH.value, default_value=0.50, data_type=Dtypes.FLOAT.value)

        for i in [min_draw_display_ratio_h, min_draw_display_ratio_w, max_draw_display_ratio_h, max_draw_display_ratio_w]:
            check_float(name=f'{self.__class__.__name__} size', value=i, max_value=1, min_value=0, raise_error=True)

        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="DISPLAY SETTINGS - ROI DRAW WINDOW", font=Formats.FONT_HEADER.value, padx=5, pady=5, icon_name='settings', relief='solid')
        self.max_width_ratio_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=WINDOW_SIZE_OPTIONS, label="MAX DRAW DISPLAY RATIO WIDTH: ", label_width=35, dropdown_width=35, value=max_draw_display_ratio_w)
        self.max_height_ratio_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=WINDOW_SIZE_OPTIONS, label="HEIGHT: ", label_width=10, dropdown_width=10, value=max_draw_display_ratio_h)
        self.min_width_ratio_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=WINDOW_SIZE_OPTIONS, label="MIN DRAW DISPLAY RATIO WIDTH: ", label_width=35, dropdown_width=35, value=min_draw_display_ratio_w)
        self.min_height_ratio_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=WINDOW_SIZE_OPTIONS, label="HEIGHT: ", label_width=10, dropdown_width=10, value=min_draw_display_ratio_h)

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.max_width_ratio_dropdown.grid(row=0, column=0, sticky=NW, pady=5)
        self.max_height_ratio_dropdown.grid(row=0, column=1, sticky=NW, pady=5)
        self.min_width_ratio_dropdown.grid(row=1, column=0, sticky=NW, pady=5)
        self.min_height_ratio_dropdown.grid(row=1, column=1, sticky=NW, pady=5)


        set_btn = SimbaButton(parent=self.main_frm, txt='SET', img='rocket', cmd=self.run)
        set_btn.grid(row=1, column=0, sticky=NW)
        #self.main_frm.mainloop()

    def run(self):
        max_width = str(self.max_width_ratio_dropdown.get_value())
        max_height = str(self.max_height_ratio_dropdown.get_value())
        min_width = str(self.min_width_ratio_dropdown.get_value())
        min_height = str(self.min_height_ratio_dropdown.get_value())

        if not self.config.has_section(ConfigKey.DISPLAY_SETTINGS.value): self.config.add_section(ConfigKey.DISPLAY_SETTINGS.value)
        self.config[ConfigKey.DISPLAY_SETTINGS.value][ConfigKey.MAX_ROI_DISPLAY_HEIGHT.value] = max_height
        self.config[ConfigKey.DISPLAY_SETTINGS.value][ConfigKey.MAX_ROI_DISPLAY_WIDTH.value] = max_width
        self.config[ConfigKey.DISPLAY_SETTINGS.value][ConfigKey.MIN_ROI_DISPLAY_WIDTH.value] = min_width
        self.config[ConfigKey.DISPLAY_SETTINGS.value][ConfigKey.MIN_ROI_DISPLAY_HEIGHT.value] = min_height

        with open(self.config_path, "w") as file:
            self.config.write(file)

        stdout_success(msg='DISPLAY SETTINGS SAVED!')




#SetMinMaxDrawWindowSize(config_path=r"C:\troubleshooting\open_field_below\project_folder\project_config.ini")