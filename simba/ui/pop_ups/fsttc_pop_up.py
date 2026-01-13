__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from tkinter import *
from typing import Union

from simba.data_processors.fsttc_calculator import FSTTCCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, Entry_Box, SimbaCheckbox
from simba.utils.checks import check_int
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import CountError


class FSTTCPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if len(self.clf_names) < 2:
            raise CountError(msg=f'Cannot compute FSTTC: Needs project with at least 2 classified behaviors, got {len(self.clf_names)}', source=self.__class__.__name__)

        PopUpMixin.__init__(self, title="FORWARD SPIKE TIME TILING COEFFICIENTS", icon='dependency')
        settings_frm = CreateLabelFrameWithIcon( parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.FSTTC.value,)
        self.time_delta_eb = Entry_Box(parent=settings_frm, fileDescription="TIME-DELTA", labelwidth=25, validation="numeric", entry_box_width=20, img='timer_2')

        graph_cb, self.graph_cb_var = SimbaCheckbox(parent=settings_frm, font=Formats.FONT_REGULAR.value, val=False, txt_img='line_chart_blue', txt="CREATE GRAPH")
        join_bouts_within_delta_cb, self.join_bouts_within_delta_var = SimbaCheckbox(parent=settings_frm, font=Formats.FONT_REGULAR.value, val=False, txt_img='join_purple', txt="JOIN BOUTS WITHIN TIME-DELTA")
        time_delta_at_onset_cb, self.time_delta_at_onset_var = SimbaCheckbox(parent=settings_frm, font=Formats.FONT_REGULAR.value, val=False, txt_img='play', txt="TIME-DELTA AT BOUT START")

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.time_delta_eb.grid(row=0, column=0, sticky="NW")
        join_bouts_within_delta_cb.grid(row=1, column=0, sticky="NW")
        time_delta_at_onset_cb.grid(row=2, column=0, sticky="NW")
        graph_cb.grid(row=3, column=0, sticky="NW")
        self.clf_cb_dict = self.create_cb_frame(main_frm=self.main_frm, cb_titles=self.clf_names, frm_title="BEHAVIORS")
        self.create_run_frm(run_function=self.run, title="RUN")
        self.main_frm.mainloop()

    def run(self):
        check_int("Time delta", value=self.time_delta_eb.entry_get)
        targets = []
        for behaviour, behavior_val in self.clf_cb_dict.items():
            if behavior_val.get():
                targets.append(behaviour)

        if len(targets) < 2:
            raise CountError(msg="FORWARD SPIKE TIME TILING COEFFICIENTS REQUIRE 2 OR MORE BEHAVIORS.", source=self.__class__.__name__,)

        fsttc_calculator = FSTTCCalculator(config_path=self.config_path,
                                           time_window=self.time_delta_eb.entry_get,
                                           join_bouts_within_delta=self.join_bouts_within_delta_var.get(),
                                           time_delta_at_onset=self.time_delta_at_onset_var.get(),
                                           behavior_lst=targets,
                                           create_graphs=self.graph_cb_var.get())
        fsttc_calculator.run()




#_ = FSTTCPopUp(config_path=r"C:\troubleshooting\multi_animal_dlc_two_c57\project_folder\project_config.ini")

#_ = FSTTCPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")

