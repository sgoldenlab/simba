__author__ = "Simon Nilsson"

import os
from tkinter import *
from typing import Union

from simba.data_processors.timebins_movement_calculator import \
    TimeBinsMovementCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaCheckbox, SimBADropDown)
from simba.utils.checks import check_float
from simba.utils.enums import ConfigKey, Keys, Links
from simba.utils.errors import NoDataError


class MovementAnalysisTimeBinsPopUp(ConfigReader, PopUpMixin):
    """
    Tkinter pop-up for defining parameters when computing movements in time-bins.

    :example:
    >>> _ =  MovementAnalysisTimeBinsPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
    >>> _ =  MovementAnalysisTimeBinsPopUp(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if len(self.outlier_corrected_paths) == 0:
            raise NoDataError(msg=f'No data files found in {self.outlier_corrected_dir} directory, cannot compute time-bins movement statistics.', source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="TIME BINS: DISTANCE/VELOCITY", size=(400, 400), icon='run')
        self.animal_cnt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT NUMBER OF ANIMALS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.animal_cnt_dropdown = SimBADropDown(parent=self.animal_cnt_frm, label="# OF ANIMALS", label_width=30, dropdown_width=20, value=1, dropdown_options=list(range(1, self.animal_cnt + 1)), command=self.create_bp_frm)
        self.animal_cnt_frm.grid(row=0, column=0, sticky=NW, padx=10, pady=10)
        self.animal_cnt_dropdown.grid(row=0, column=0, sticky=NW)

        self.plots_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="PLOTS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.plots_cb, self.plots_var = SimbaCheckbox(parent=self.plots_frm, txt='CREATE PLOTS', txt_img='plot', val=True)
        self.plots_frm.grid(row=1, column=0, sticky=NW, padx=10, pady=10)
        self.plots_cb.grid(row=0, column=0, sticky=NW)

        self.time_bin_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="TIME BIN", icon_name='timer_2', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.time_bin_entry = Entry_Box(parent=self.time_bin_frm, fileDescription='TIME BIN SIZE (S): ', labelwidth=30, entry_box_width=20)
        self.time_bin_frm.grid(row=2, column=0, sticky=NW, padx=10, pady=10)
        self.time_bin_entry.grid(row=0, column=0, sticky=NW)
        self.create_bp_frm(animal_cnt=1)
        self.create_run_frm(run_function=self._run)

        self.main_frm.mainloop()

    def create_bp_frm(self, animal_cnt):
        if hasattr(self, "bp_frm"):
            self.bp_frm.destroy()
            for k, v in self.body_parts_dropdowns.items():
                v.destroy()

        self.bp_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT BODY-PARTS", icon_name='pose', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.body_parts_dropdowns = {}
        for cnt, i in enumerate(range(int(animal_cnt))):
            self.body_parts_dropdowns[cnt] = SimBADropDown(parent=self.bp_frm, label=f"Animal {cnt+1}", label_width=30, dropdown_width=20, value=self.body_parts_lst[cnt], dropdown_options=self.body_parts_lst)
            self.body_parts_dropdowns[cnt].grid(row=cnt, column=0, sticky=NW)
        self.bp_frm.grid(row=3, column=0, sticky=NW, padx=10, pady=10)

    def _run(self):
        check_float(name="Time bin", value=str(self.time_bin_entry.entry_get), min_value=10e-6)
        self.config.set(ConfigKey.PROCESS_MOVEMENT_SETTINGS.value, ConfigKey.ROI_ANIMAL_CNT.value, str(self.animal_cnt_dropdown.getChoices()))
        body_parts = []
        for cnt, dropdown in self.body_parts_dropdowns.items():
            self.config.set(ConfigKey.PROCESS_MOVEMENT_SETTINGS.value, f"animal_{cnt + 1}_bp", str(dropdown.getChoices()))
            body_parts.append(dropdown.getChoices())
        self.update_config()
        time_bin_movement_analyzer = TimeBinsMovementCalculator(config_path=self.config_path,
                                                                bin_length=float(self.time_bin_entry.entry_get),
                                                                plots=self.plots_var.get(),
                                                                body_parts=body_parts)
        time_bin_movement_analyzer.run()
        time_bin_movement_analyzer.save()

#_ =  MovementAnalysisTimeBinsPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
