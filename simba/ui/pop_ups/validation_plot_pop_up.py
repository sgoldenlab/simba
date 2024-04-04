__author__ = "Simon Nilsson"

import threading
from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.single_run_model_validation_video import \
    ValidateModelOneVideo
from simba.plotting.single_run_model_validation_video_mp import \
    ValidateModelOneVideoMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box)
from simba.utils.checks import check_float, check_int
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.read_write import check_file_exist_and_readable, str_2_bool


class ValidationVideoPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str, simba_main_frm: object):
        PopUpMixin.__init__(self, title="CREATE VALIDATION VIDEO")
        ConfigReader.__init__(self, config_path=config_path)
        self.feature_file_path = simba_main_frm.csvfile.file_path
        self.model_path = simba_main_frm.modelfile.file_path
        self.discrimination_threshold = simba_main_frm.dis_threshold.entry_get
        self.shortest_bout = simba_main_frm.min_behaviorbout.entry_get

        style_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="STYLE SETTINGS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.OUT_OF_SAMPLE_VALIDATION.value,
        )
        self.default_style_var = BooleanVar(value=True)
        default_style_cb = Checkbutton(
            style_frm,
            text="AUTO-COMPUTE STYLES",
            variable=self.default_style_var,
            command=lambda: self.enable_entrybox_from_checkbox(
                check_box_var=self.default_style_var,
                entry_boxes=[self.font_size_eb, self.spacing_eb, self.circle_size],
                reverse=True,
            ),
        )
        self.font_size_eb = Entry_Box(
            style_frm, "Font size: ", "25", validation="numeric"
        )
        self.spacing_eb = Entry_Box(
            style_frm, "Text spacing: ", "25", validation="numeric"
        )
        self.circle_size = Entry_Box(
            style_frm, "Circle size: ", "25", validation="numeric"
        )
        self.font_size_eb.entry_set(val=1)
        self.spacing_eb.entry_set(val=10)
        self.circle_size.entry_set(val=5)
        self.enable_entrybox_from_checkbox(
            check_box_var=self.default_style_var,
            entry_boxes=[self.font_size_eb, self.spacing_eb, self.circle_size],
            reverse=True,
        )

        tracking_frm = LabelFrame(
            self.main_frm,
            text="TRACKING OPTIONS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        self.show_pose_dropdown = DropDownMenu(
            tracking_frm, "Show pose:", Options.BOOL_STR_OPTIONS.value, "20"
        )
        self.show_animal_names_dropdown = DropDownMenu(
            tracking_frm, "Show animal names:", Options.BOOL_STR_OPTIONS.value, "20"
        )
        self.show_pose_dropdown.setChoices(Options.BOOL_STR_OPTIONS.value[0])
        self.show_animal_names_dropdown.setChoices(Options.BOOL_STR_OPTIONS.value[1])

        multiprocess_frame = LabelFrame(
            self.main_frm,
            text="MULTI-PROCESS SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        self.multiprocess_var = BooleanVar(value=False)
        self.multiprocess_cb = Checkbutton(
            multiprocess_frame,
            text="Multiprocess videos (faster)",
            variable=self.multiprocess_var,
            command=lambda: self.enable_dropdown_from_checkbox(
                check_box_var=self.multiprocess_var,
                dropdown_menus=[self.multiprocess_dropdown],
            ),
        )
        self.multiprocess_dropdown = DropDownMenu(
            multiprocess_frame, "CPU cores:", list(range(2, self.cpu_cnt)), "12"
        )
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()

        gantt_frm = LabelFrame(
            self.main_frm,
            text="GANTT SETTINGS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        self.gantt_dropdown = DropDownMenu(
            gantt_frm, "GANTT TYPE:", Options.GANTT_VALIDATION_OPTIONS.value, "12"
        )
        self.gantt_dropdown.setChoices(Options.GANTT_VALIDATION_OPTIONS.value[0])

        style_frm.grid(row=0, column=0, sticky=NW)
        default_style_cb.grid(row=0, column=0, sticky=NW)
        self.font_size_eb.grid(row=1, column=0, sticky=NW)
        self.spacing_eb.grid(row=2, column=0, sticky=NW)
        self.circle_size.grid(row=3, column=0, sticky=NW)

        tracking_frm.grid(row=1, column=0, sticky=NW)
        self.show_pose_dropdown.grid(row=0, column=0, sticky=NW)
        self.show_animal_names_dropdown.grid(row=1, column=0, sticky=NW)

        multiprocess_frame.grid(row=2, column=0, sticky=NW)
        self.multiprocess_cb.grid(row=0, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=0, column=1, sticky=NW)
        gantt_frm.grid(row=3, column=0, sticky=NW)
        self.gantt_dropdown.grid(row=0, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        settings = {
            "pose": str_2_bool(self.show_pose_dropdown.getChoices()),
            "animal_names": str_2_bool(self.show_animal_names_dropdown.getChoices()),
        }
        settings["styles"] = None
        if not self.default_style_var.get():
            settings["styles"] = {}
            check_float(name="FONT SIZE", value=self.font_size_eb.entry_get)
            check_float(name="CIRCLE SIZE", value=self.circle_size.entry_get)
            check_float(name="SPACE SCALE", value=self.spacing_eb.entry_get)
            settings["styles"]["circle size"] = int(self.circle_size.entry_get)
            settings["styles"]["font size"] = int(self.font_size_eb.entry_get)
            settings["styles"]["space_scale"] = int(self.spacing_eb.entry_get)
        check_int(name="MINIMUM BOUT LENGTH", value=self.shortest_bout)
        check_float(
            name="DISCRIMINATION THRESHOLD", value=self.discrimination_threshold
        )
        check_file_exist_and_readable(file_path=self.feature_file_path)
        check_file_exist_and_readable(file_path=self.model_path)

        create_gantt = self.gantt_dropdown.getChoices()
        if create_gantt.strip() == "Gantt chart: final frame only (slightly faster)":
            create_gantt = 1
        elif create_gantt.strip() == "Gantt chart: video":
            create_gantt = 2
        else:
            create_gantt = None

        if not self.multiprocess_var.get():
            validation_video_creator = ValidateModelOneVideo(
                config_path=self.config_path,
                feature_file_path=self.feature_file_path,
                model_path=self.model_path,
                discrimination_threshold=float(self.discrimination_threshold),
                shortest_bout=int(self.shortest_bout),
                settings=settings,
                create_gantt=create_gantt,
            )

        else:
            validation_video_creator = ValidateModelOneVideoMultiprocess(
                config_path=self.config_path,
                feature_file_path=self.feature_file_path,
                model_path=self.model_path,
                discrimination_threshold=float(self.discrimination_threshold),
                shortest_bout=int(self.shortest_bout),
                cores=int(self.multiprocess_dropdown.getChoices()),
                settings=settings,
                create_gantt=create_gantt,
            )
        threading.Thread(target=validation_video_creator.run()).start()
