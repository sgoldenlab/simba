__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from tkinter import *
from typing import Union

from simba.data_processors.timebins_movement_calculator import \
    TimeBinsMovementCalculator
from simba.data_processors.timebins_movement_calculator_mp import \
    TimeBinsMovementCalculatorMultiprocess
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
        PopUpMixin.__init__(self, title="TIME BINS: MOVEMENT/VELOCITY", size=(400, 600), icon='run')
        self.animal_cnt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT NUMBER OF ANIMALS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.animal_cnt_dropdown = SimBADropDown(parent=self.animal_cnt_frm, label="# OF ANIMALS", label_width=30, dropdown_width=20, value=1, dropdown_options=list(range(1, self.animal_cnt + 1)), command=self.create_bp_frm, img='abacus', tooltip_key='TIMEBINS_MOVEMENT_NUMBER_OF_ANIMALS')
        self.animal_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_dropdown.grid(row=0, column=0, sticky=NW)

        self.bp_threshold_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="BODY-PART THRESHOLD", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.threshold_entry = Entry_Box(parent=self.bp_threshold_frm, fileDescription='THRESHOLD (0.0-1.0): ', labelwidth=30, entry_box_width=20, justify='center', value=0.0, img='pct_2', tooltip_key='MOVEMENT_PROBABILITY_THRESHOLD')
        self.bp_threshold_frm.grid(row=1, column=0, sticky=NW)
        self.threshold_entry.grid(row=0, column=0, sticky=NW)

        self.time_bin_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="TIME BIN", icon_name='timer_2', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.time_bin_entry = Entry_Box(parent=self.time_bin_frm, fileDescription='TIME BIN SIZE (S): ', labelwidth=30, entry_box_width=20, justify='center', img='timer_2', value=60.0, trace=self._entrybox_bg_check_float, tooltip_key='TIMEBINS_MOVEMENT_TIME_BIN_SIZE')
        self.time_bin_frm.grid(row=2, column=0, sticky=NW)
        self.time_bin_entry.grid(row=0, column=0, sticky=NW)

        self.core_cnt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CPU CORE COUNT", icon_name='cpu_small', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid', tooltip_key='CPU_TIMEBINS_MOVEMENT')
        self.core_cnt_dropdown = SimBADropDown(parent=self.core_cnt_frm, label="CORE COUNT:", label_width=30, dropdown_width=20, value=1, dropdown_options=list(range(1, self.cpu_cnt+1)), tooltip_key='CPU_TIMEBINS_MOVEMENT', img='cpu_small')
        self.core_cnt_frm.grid(row=3, column=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=0, column=0, sticky=NW)

        self.measurments_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MEASUREMENTS", icon_name='ruler', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        distance_cb, self.distance_var = SimbaCheckbox(parent=self.measurments_frm, txt='DISTANCE (CM)', txt_img='distance', val=True, tooltip_key='TIMEBINS_MOVEMENT_DISTANCE')
        velocity_cb, self.velocity_var = SimbaCheckbox(parent=self.measurments_frm, txt='VELOCITY (CM/S)', txt_img='run', val=False, tooltip_key='TIMEBINS_MOVEMENT_VELOCITY')
        self.measurments_frm.grid(row=5, column=0, sticky=NW)
        distance_cb.grid(row=0, column=0, sticky=NW)
        velocity_cb.grid(row=1, column=0, sticky=NW)

        self.output_format_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="OUTPUT OPTIONS", icon_name='rotate', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        transpose_cb, self.transpose_var = SimbaCheckbox(parent=self.output_format_frm, txt='TRANSPOSE OUTPUT CSV', txt_img='rotate', val=False, tooltip_key='TIMEBINS_MOVEMENT_TRANSPOSE')
        include_timestamps_cb, self.include_timestamps_var = SimbaCheckbox(parent=self.output_format_frm, txt='INCLUDE TIME-STAMPS', txt_img='timer', val=True, tooltip_key='TIMEBINS_MOVEMENT_INCLUDE_TIMESTAMPS')
        self.plots_cb, self.plots_var = SimbaCheckbox(parent=self.output_format_frm, txt='CREATE PLOTS', txt_img='plot', val=True, tooltip_key='TIMEBINS_MOVEMENT_CREATE_PLOTS')
        self.output_format_frm.grid(row=6, column=0, sticky=NW)
        transpose_cb.grid(row=0, column=0, sticky=NW)
        include_timestamps_cb.grid(row=1, column=0, sticky=NW)
        self.plots_cb.grid(row=2, column=0, sticky=NW)

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
            self.body_parts_dropdowns[cnt] = SimBADropDown(parent=self.bp_frm, label=f"Animal {cnt+1}", label_width=30, dropdown_width=20, value=self.body_parts_lst[cnt], dropdown_options=self.body_parts_lst, img='circle_black', tooltip_key='TIMEBINS_MOVEMENT_BODY_PART')
            self.body_parts_dropdowns[cnt].grid(row=cnt, column=0, sticky=NW)
        self.bp_frm.grid(row=4, column=0, sticky=NW)

    def _entrybox_bg_check_float(self, entry_box: Entry_Box, valid_clr: str = 'white', invalid_clr: str = 'lightsalmon'):
        valid_value = check_float(name='', allow_negative=False, allow_zero=False, value=entry_box.entry_get, raise_error=False)[0]
        entry_box.set_bg_clr(clr=valid_clr if valid_value else invalid_clr)

    def _run(self):
        check_float(name="Time bin", value=str(self.time_bin_entry.entry_get), min_value=10e-6)
        self.config.set(ConfigKey.PROCESS_MOVEMENT_SETTINGS.value, ConfigKey.ROI_ANIMAL_CNT.value, str(self.animal_cnt_dropdown.getChoices()))
        body_parts = []
        for cnt, dropdown in self.body_parts_dropdowns.items():
            self.config.set(ConfigKey.PROCESS_MOVEMENT_SETTINGS.value, f"animal_{cnt + 1}_bp", str(dropdown.getChoices()))
            body_parts.append(dropdown.getChoices())
        self.update_config()
        core_cnt = int(self.core_cnt_dropdown.get_value())
        body_part_threshold = self.threshold_entry.entry_get
        check_float(name=f'{self.__class__.__name__} BODY-PART THRESHOLD', value=body_part_threshold, min_value=0.0, max_value=1.0, raise_error=True, allow_zero=True, allow_negative=False)
        velocity, distance, transpose = self.velocity_var.get(), self.distance_var.get(), self.transpose_var.get()
        include_timestamps = self.include_timestamps_var.get()
        if core_cnt == 1:
            time_bin_movement_analyzer = TimeBinsMovementCalculator(config_path=self.config_path,
                                                                    bin_length=float(self.time_bin_entry.entry_get),
                                                                    plots=self.plots_var.get(),
                                                                    body_parts=body_parts,
                                                                    verbose=True,
                                                                    distance=distance,
                                                                    velocity=velocity,
                                                                    threshold=float(body_part_threshold),
                                                                    transpose=transpose,
                                                                    include_timestamp=include_timestamps)
        else:
            time_bin_movement_analyzer = TimeBinsMovementCalculatorMultiprocess(config_path=self.config_path,
                                                                                bin_length=float(self.time_bin_entry.entry_get),
                                                                                body_parts=body_parts,
                                                                                plots=self.plots_var.get(),
                                                                                verbose=True,
                                                                                core_cnt=core_cnt,
                                                                                distance=distance,
                                                                                velocity=velocity,
                                                                                transpose=transpose,
                                                                                threshold=body_part_threshold,
                                                                                include_timestamp=include_timestamps)

        time_bin_movement_analyzer.run()
        time_bin_movement_analyzer.save()

#_ =  MovementAnalysisTimeBinsPopUp(config_path=r"D:\troubleshooting\maplight_ri\project_folder\project_config.ini")
#_ = MovementAnalysisTimeBinsPopUp(config_path=r"D:\troubleshooting\maplight_ri\project_folder\project_config.ini")
