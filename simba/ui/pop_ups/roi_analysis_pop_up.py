import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.ROI_analyzer import ROIAnalyzer
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, SimbaButton, SimbaCheckbox)
from simba.utils.checks import check_float
from simba.utils.enums import ConfigKey, Formats, Keys, Links


class ROIAnalysisPopUp(ConfigReader, PopUpMixin):

    """
    :example:
    >>> _ = ROIAnalysisPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
    """
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        PopUpMixin.__init__(self, title="ROI ANALYSIS", size=(400, 400))
        self.animal_cnt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT NUMBER OF ANIMALS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ROI_DATA_ANALYSIS.value)
        self.animal_cnt_dropdown = DropDownMenu(self.animal_cnt_frm, "# of animals", list(range(1, self.animal_cnt + 1)), labelwidth=20)
        self.animal_cnt_dropdown.setChoices(1)

        self.animal_cnt_confirm_btn = SimbaButton(parent=self.animal_cnt_frm, txt="CONFIRM", img='tick', txt_clr="blue", font=Formats.FONT_REGULAR.value, cmd=self.create_settings_frm)
        self.animal_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_confirm_btn.grid(row=0, column=1, sticky=NW)
        self.main_frm.mainloop()

    def create_settings_frm(self):
        if hasattr(self, "setting_frm"):
            self.setting_frm.destroy()
            self.body_part_frm.destroy()

        self.setting_frm = LabelFrame(
            self.main_frm, text="SETTINGS", font=Formats.FONT_HEADER.value
        )
        self.choose_bp_frm(parent=self.setting_frm, bp_options=self.body_parts_lst)
        self.choose_bp_threshold_frm(parent=self.setting_frm)
        self.calculate_distances_frm = LabelFrame( self.setting_frm, text="CALCULATE DISTANCES", font=Formats.FONT_HEADER.value)


        self.calculate_distance_moved_cb, self.calculate_distance_moved_var = SimbaCheckbox(parent=self.calculate_distances_frm, txt='COMPUTE DISTANCES MOVED IN ROIs', txt_img='path')
        self.calculate_distance_moved_cb.grid(row=0, column=0, sticky=NW)
        self.detailed_roi_frm = LabelFrame(self.setting_frm, text="DETAILED ROI BOUT DATA", font=Formats.FONT_HEADER.value)
        self.detailed_roi_cb, self.detailed_roi_var = SimbaCheckbox(parent=self.detailed_roi_frm, txt='DETAILED ROI BOUT DATA', txt_img='details')
        self.detailed_roi_cb.grid(row=1, column=0, sticky=NW)
        self.calculate_distances_frm.grid(row=self.frame_children(frame=self.setting_frm), column=0, sticky=NW)
        self.detailed_roi_frm.grid(row=self.frame_children(frame=self.setting_frm) + 1, column=0, sticky=NW)
        self.setting_frm.grid(row=1, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

        self.main_frm.mainloop()

    def run(self):
        settings = {}
        check_float(
            name="Probability threshold",
            value=self.probability_entry.entry_get,
            min_value=0.00,
            max_value=1.00,
        )
        settings["threshold"] = float(self.probability_entry.entry_get)
        body_parts = []
        for cnt, dropdown in self.body_parts_dropdowns.items():
            body_parts.append(dropdown.getChoices())
        roi_analyzer = ROIAnalyzer(
            config_path=self.config_path,
            data_path=None,
            calculate_distances=self.calculate_distance_moved_var.get(),
            detailed_bout_data=self.detailed_roi_var.get(),
            threshold=float(self.probability_entry.entry_get),
            body_parts=body_parts,
        )
        roi_analyzer.run()
        roi_analyzer.save()
        self.root.destroy()

# ROIAnalysisPopUp(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")

# ROIAnalysisPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini')
