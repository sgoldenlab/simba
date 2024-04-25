__author__ = "Simon Nilsson"

import os
from tkinter import *

from simba.data_processors.pup_retrieval_calculator import \
    PupRetrieverCalculator
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
from simba.ui.tkinter_functions import DropDownMenu, Entry_Box, hxtScrollbar
from simba.utils.checks import check_float, check_int
from simba.utils.enums import ConfigKey, Formats, Paths
from simba.utils.errors import NoFilesFoundError
from simba.utils.read_write import (get_all_clf_names, read_config_entry,
                                    read_config_file)


class PupRetrievalPopUp(object):
    def __init__(self, config_path: str):
        self.smoothing_options, self.config_path = ["gaussian"], config_path
        self.smooth_factor_options = list(range(1, 11))
        self.config = read_config_file(config_path=config_path)
        self.project_path = read_config_entry(
            self.config,
            ConfigKey.GENERAL_SETTINGS.value,
            ConfigKey.PROJECT_PATH.value,
            data_type=ConfigKey.FOLDER_PATH.value,
        )
        self.ROI_path = os.path.join(self.project_path, Paths.ROI_DEFINITIONS.value)
        if not os.path.isfile(self.ROI_path):
            raise NoFilesFoundError(
                msg=f"Requires ROI definitions: no file found at {self.ROI_path}",
                source=self.__class__.__name__,
            )

        self.roi_analyzer = ROIAnalyzer(config_path=config_path, data_path=None)
        self.roi_analyzer.run()
        self.shape_names = self.roi_analyzer.shape_names
        self.animal_names = self.roi_analyzer.multi_animal_id_list
        self.clf_names = get_all_clf_names(
            config=self.config, target_cnt=self.roi_analyzer.clf_cnt
        )

        self.distance_plots_var = BooleanVar(value=True)
        self.swarm_plot_var = BooleanVar(value=True)
        self.log_var = BooleanVar(value=True)

        self.main_frm = Toplevel()
        self.main_frm.minsize(400, 400)
        self.main_frm.wm_title("SIMBA PUP RETRIEVAL PROTOCOL 1")
        self.main_frm = hxtScrollbar(self.main_frm)
        self.main_frm.pack(expand=True, fill=BOTH)

        self.pup_track_p_entry = Entry_Box(
            self.main_frm, "Tracking probability (PUP): ", "20"
        )
        self.dam_track_p_entry = Entry_Box(
            self.main_frm, "Tracking probability (DAM): ", "20"
        )
        self.start_distance_criterion_entry = Entry_Box(
            self.main_frm, "Start distance criterion (MM):", "20", validation="numeric"
        )
        self.carry_frames_entry = Entry_Box(
            self.main_frm, "Carry time (S)", "20", validation="numeric"
        )
        self.core_nest_name_dropdown = DropDownMenu(
            self.main_frm, "Core-nest name: ", self.shape_names, "20"
        )
        self.nest_name_dropdown = DropDownMenu(
            self.main_frm, "Nest name: ", self.shape_names, "20"
        )
        self.dam_name_dropdown = DropDownMenu(
            self.main_frm, "Dam name: ", self.animal_names, "20"
        )
        self.pup_name_dropdown = DropDownMenu(
            self.main_frm, "Pup name: ", self.animal_names, "20"
        )
        self.smooth_function_dropdown = DropDownMenu(
            self.main_frm, "Smooth function: ", self.smoothing_options, "20"
        )
        self.smooth_factor_dropdown = DropDownMenu(
            self.main_frm, "Smooth factor: ", self.smooth_factor_options, "20"
        )
        self.max_time_entry = Entry_Box(
            self.main_frm, "Max time (S)", "20", validation="numeric"
        )
        self.carry_classifier_dropdown = DropDownMenu(
            self.main_frm, "Carry classifier name: ", self.clf_names, "20"
        )
        self.approach_classifier_dropdown = DropDownMenu(
            self.main_frm, "Approach classifier name: ", self.clf_names, "20"
        )
        self.dig_classifier_dropdown = DropDownMenu(
            self.main_frm, "Dig classifier name: ", self.clf_names, "20"
        )
        self.create_distance_plots_cb = Checkbutton(
            self.main_frm,
            text="Create distance plots (pre- and post tracking smoothing",
            variable=self.distance_plots_var,
        )
        self.swarm_plot_cb = Checkbutton(
            self.main_frm,
            text="Create results swarm plot",
            variable=self.swarm_plot_var,
        )
        self.log_cb = Checkbutton(
            self.main_frm, text="Create log-file", variable=self.log_var
        )

        self.pup_track_p_entry.entry_set(0.025)
        self.dam_track_p_entry.entry_set(0.5)
        self.start_distance_criterion_entry.entry_set(80)
        self.carry_frames_entry.entry_set(3)
        self.core_nest_name_dropdown.setChoices(choice=self.shape_names[0])
        self.nest_name_dropdown.setChoices(choice=self.shape_names[1])
        self.dam_name_dropdown.setChoices(choice=self.animal_names[0])
        self.pup_name_dropdown.setChoices(choice=self.animal_names[1])
        self.smooth_function_dropdown.setChoices(choice=self.smoothing_options[0])
        self.smooth_factor_dropdown.setChoices(choice=5)
        self.carry_frames_entry.entry_set(90)
        self.carry_classifier_dropdown.setChoices(self.clf_names[0])
        self.approach_classifier_dropdown.setChoices(self.clf_names[0])
        self.dig_classifier_dropdown.setChoices(self.clf_names[0])
        self.max_time_entry.entry_set(90)

        button_run = Button(
            self.main_frm,
            text="RUN",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="red",
            command=lambda: self.run(),
        )

        self.pup_track_p_entry.grid(row=0, sticky=W)
        self.dam_track_p_entry.grid(row=1, sticky=W)
        self.start_distance_criterion_entry.grid(row=2, sticky=W)
        self.carry_frames_entry.grid(row=3, sticky=W)
        self.core_nest_name_dropdown.grid(row=4, sticky=W)
        self.nest_name_dropdown.grid(row=5, sticky=W)
        self.dam_name_dropdown.grid(row=6, sticky=W)
        self.pup_name_dropdown.grid(row=7, sticky=W)
        self.smooth_function_dropdown.grid(row=8, sticky=W)
        self.smooth_factor_dropdown.grid(row=9, sticky=W)
        self.max_time_entry.grid(row=10, sticky=W)
        self.carry_classifier_dropdown.grid(row=11, sticky=W)
        self.approach_classifier_dropdown.grid(row=12, sticky=W)
        self.dig_classifier_dropdown.grid(row=13, sticky=W)
        self.swarm_plot_cb.grid(row=14, sticky=W)
        self.create_distance_plots_cb.grid(row=15, sticky=W)
        self.log_cb.grid(row=16, sticky=W)

        button_run.grid(row=17, sticky=W)

        # self.main_frm.mainloop()

    def run(self):
        pup_track_p = self.pup_track_p_entry.entry_get
        dam_track_p = self.dam_track_p_entry.entry_get
        start_distance_criterion = self.start_distance_criterion_entry.entry_get
        carry_frames = self.carry_frames_entry.entry_get
        core_nest = self.core_nest_name_dropdown.getChoices()
        nest = self.nest_name_dropdown.getChoices()
        dam_name = self.dam_name_dropdown.getChoices()
        pup_name = self.pup_name_dropdown.getChoices()
        smooth_function = self.smooth_function_dropdown.getChoices()
        smooth_factor = self.smooth_factor_dropdown.getChoices()
        max_time = self.max_time_entry.entry_get
        clf_carry = self.carry_classifier_dropdown.getChoices()
        clf_approach = self.approach_classifier_dropdown.getChoices()
        clf_dig = self.dig_classifier_dropdown.getChoices()
        check_float(
            name="Tracking probability (PUP)",
            value=pup_track_p,
            max_value=1.0,
            min_value=0.0,
        )
        check_float(
            name="Tracking probability (DAM)",
            value=dam_track_p,
            max_value=1.0,
            min_value=0.0,
        )
        check_float(
            name="Start distance criterion (MM)", value=start_distance_criterion
        )
        check_int(name="Carry frames (S)", value=carry_frames)
        check_int(name="max_time", value=max_time)

        swarm_plot = self.swarm_plot_var.get()
        distance_plot = self.distance_plots_var.get()
        log = self.log_var.get()

        settings = {
            "pup_track_p": float(pup_track_p),
            "dam_track_p": float(dam_track_p),
            "start_distance_criterion": float(start_distance_criterion),
            "carry_time": float(carry_frames),
            "core_nest": core_nest,
            "nest": nest,
            "dam_name": dam_name,
            "pup_name": pup_name,
            "smooth_function": smooth_function,
            "smooth_factor": int(smooth_factor),
            "max_time": float(max_time),
            "clf_carry": clf_carry,
            "clf_approach": clf_approach,
            "clf_dig": clf_dig,
            "swarm_plot": swarm_plot,
            "distance_plots": distance_plot,
            "log": log,
        }

        pup_calculator = PupRetrieverCalculator(
            config_path=self.config_path, settings=settings
        )
        pup_calculator.run()
        pup_calculator.save_results()
