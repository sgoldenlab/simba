from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.ROI_time_bin_calculator import ROITimebinCalculator
from simba.ui.tkinter_functions import (Checkbutton, CreateLabelFrameWithIcon,
                                        DropDownMenu)
from simba.utils.checks import check_float
from simba.utils.enums import ConfigKey, Formats, Keys, Links


class ROIAnalysisTimeBinsPopUp(ConfigReader, PopUpMixin):
    def __init__(self, config_path: str):
        ConfigReader.__init__(self, config_path=config_path)
        PopUpMixin.__init__(self, title="ROI ANALYSIS: TIME-BINS", size=(400, 400))
        self.animal_cnt_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SELECT NUMBER OF ANIMALS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.ROI_DATA_ANALYSIS.value,
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
        self.choose_bp_threshold_frm(parent=self.setting_frm)
        self.setting_frm.grid(row=1, column=0, sticky=NW)
        self.create_time_bin_entry()

        calc_distances_frm = LabelFrame(
            self.main_frm,
            text="CALCULATE DISTANCES",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
        )
        self.distances_var = BooleanVar()
        self.distances_cb = Checkbutton(
            calc_distances_frm,
            text="Compute distances moved within ROIs in each time-bin",
            variable=self.distances_var,
        )

        calc_distances_frm.grid(row=3, column=0, sticky=NW)
        self.distances_cb.grid(row=0, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        check_float(
            name="Time bin", value=self.time_bin_entrybox.entry_get, min_value=10e-6
        )
        check_float(
            name="Probability threshold",
            value=self.probability_entry.entry_get,
            min_value=0.00,
            max_value=1.00,
        )
        self.config.set(
            ConfigKey.ROI_SETTINGS.value,
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
        roi_time_bin_calculator = ROITimebinCalculator(
            config_path=self.config_path,
            bin_length=float(self.time_bin_entrybox.entry_get),
            threshold=float(self.probability_entry.entry_get),
            body_parts=body_parts,
            movement=self.distances_var.get(),
        )
        roi_time_bin_calculator.run()
        roi_time_bin_calculator.save()
        self.root.destroy()


# _ = ROIAnalysisTimeBinsPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
