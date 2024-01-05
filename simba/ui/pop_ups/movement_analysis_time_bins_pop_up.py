from copy import deepcopy
from tkinter import *

from simba.data_processors.timebins_movement_calculator import \
    TimeBinsMovementCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box)
from simba.utils.checks import check_int
from simba.utils.enums import ConfigKey, Formats, Keys, Links


class MovementAnalysisTimeBinsPopUp(ConfigReader, PopUpMixin):
    def __init__(self, config_path: str):
        ConfigReader.__init__(self, config_path=config_path)
        PopUpMixin.__init__(self, title="TIME BINS: DISTANCE/VELOCITY", size=(400, 400))
        self.animal_cnt_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SELECT NUMBER OF ANIMALS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.DATA_ANALYSIS.value,
        )
        self.animal_cnt_dropdown = DropDownMenu(
            self.animal_cnt_frm,
            "# of animals",
            list(range(1, self.animal_cnt + 1)),
            labelwidth=20,
        )
        self.animal_cnt_dropdown.setChoices(1)
        self.animal_cnt_confirm_btn = Button(
            self.animal_cnt_frm,
            text="Confirm",
            command=lambda: self.create_settings_frm(),
        )
        self.animal_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_confirm_btn.grid(row=0, column=1, sticky=NW)
        self.main_frm.mainloop()

    def create_settings_frm(self):
        if hasattr(self, "setting_frm"):
            self.setting_frm.destroy()
            self.body_part_frm.destroy()

        self.setting_frm = LabelFrame(
            self.main_frm, text="SETTINGS", font=Formats.LABELFRAME_HEADER_FORMAT.value
        )
        self.choose_bp_frm(parent=self.setting_frm, bp_options=self.body_parts_lst)
        self.plots_frm = LabelFrame(
            self.setting_frm, text="Plots", font=Formats.LABELFRAME_HEADER_FORMAT.value
        )
        self.plots_var = BooleanVar()
        self.plots_cb = Checkbutton(
            self.plots_frm, text="Create plots", variable=self.plots_var
        )
        self.plots_frm.grid(
            row=self.frame_children(frame=self.setting_frm), column=0, sticky=NW
        )
        self.plots_cb.grid(row=0, column=0, sticky=NW)
        self.setting_frm.grid(row=1, column=0, sticky=NW)
        self.create_time_bin_entry()
        self.create_run_frm(run_function=self.run)

    def run(self):
        check_int(
            name="Time bin", value=str(self.time_bin_entrybox.entry_get), min_value=1
        )
        self.config.set(
            ConfigKey.PROCESS_MOVEMENT_SETTINGS.value,
            ConfigKey.ROI_ANIMAL_CNT.value,
            str(self.animal_cnt_dropdown.getChoices()),
        )
        body_parts = []
        for cnt, dropdown in self.body_parts_dropdowns.items():
            self.config.set(
                ConfigKey.PROCESS_MOVEMENT_SETTINGS.value,
                "animal_{}_bp".format(str(cnt + 1)),
                str(dropdown.getChoices()),
            )
            body_parts.append(dropdown.getChoices())
        self.update_config()
        time_bin_movement_analyzer = TimeBinsMovementCalculator(
            config_path=self.config_path,
            bin_length=int(self.time_bin_entrybox.entry_get),
            plots=self.plots_var.get(),
            body_parts=body_parts,
        )
        time_bin_movement_analyzer.run()
        self.root.destroy()

# MovementAnalysisTimeBinsPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/locomotion/project_folder/project_config.ini')
