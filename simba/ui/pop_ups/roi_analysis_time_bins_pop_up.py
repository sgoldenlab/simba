import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.roi_time_bins_analyzer import ROITimebinAnalyzer
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton, SimbaCheckbox,
                                        SimBADropDown)
from simba.utils.checks import check_float
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import InvalidInputError, NoROIDataError

OUTSIDE_ROI = 'OUTSIDE REGIONS OF INTEREST'

class ROIAnalysisTimeBinsPopUp(ConfigReader, PopUpMixin):

    """
    :example:
    >>> _ = ROIAnalysisTimeBinsPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')

    """
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if not os.path.isfile(self.roi_coordinates_path):
            raise NoROIDataError(msg='No ROI data detected in SimBA project. Draw ROIs before doing ROI time analysis', source=self.__class__.__name__)
        if len(self.outlier_corrected_paths) == 0:
            raise NoROIDataError(msg=f'No data found in {self.outlier_corrected_dir} directory. Create data before analyzing ROI time data', source=self.__class__.__name__)

        PopUpMixin.__init__(self, title="ROI AGGREGATE STATISTICS ANALYSIS BY TIME-BIN", icon='shapes_small')
        self.config_path = config_path
        self.animal_cnt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT NUMBER OF ANIMAL(S)", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ROI_DATA_ANALYSIS.value)
        self.animal_cnt_dropdown = SimBADropDown(parent=self.animal_cnt_frm, dropdown_options=list(range(1, self.animal_cnt + 1)), label="# OF ANIMALS", label_width=30, dropdown_width=10, value=1)
        self.animal_cnt_confirm_btn = SimbaButton(parent=self.animal_cnt_frm, txt="CONFIRM", img='tick', txt_clr="blue", font=Formats.FONT_REGULAR.value, cmd=self._get_settings_frm)
        self.animal_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_confirm_btn.grid(row=0, column=1, sticky=NW)
        self._get_settings_frm()
        self.main_frm.mainloop()

    def _get_settings_frm(self):
        self.max_bp_len = max(len(item) for item in self.body_parts_lst)
        if hasattr(self, "time_bin_frm"):
            self.time_bin_frm.destroy()
            self.probability_frm.destroy()
            self.body_part_frm.destroy()
            self.data_options_frm.destroy()
            self.format_frm.destroy()

        self.time_bin_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHOOSE TIME BIN", icon_name='pose')
        self.time_bin_entry = Entry_Box(parent=self.time_bin_frm, fileDescription='TIME BIN (S):', labelwidth="30")
        self.time_bin_frm.grid(row=self.frame_children(frame=self.main_frm), sticky=NW)
        self.time_bin_entry.grid(row=0, column=0, sticky=NW)

        self.probability_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT PROBABILITY THRESHOLD", icon_name='pose')
        self.probability_entry = Entry_Box(parent=self.probability_frm, fileDescription='PROBABILITY THRESHOLD (0.0-1.0):', labelwidth="30", value='0.0')
        self.probability_frm.grid(row=self.frame_children(frame=self.main_frm), sticky=NW)
        self.probability_entry.grid(row=0, column=0, sticky=NW)

        self.body_parts_dropdowns = {}
        self.body_part_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT BODY-PART(S)", icon_name='pose')
        self.body_part_frm.grid(row=self.frame_children(frame=self.main_frm), sticky=NW)
        for bp_cnt in range(int(self.animal_cnt_dropdown.getChoices())):
            self.body_parts_dropdowns[bp_cnt] = SimBADropDown(parent=self.body_part_frm, dropdown_options=self.body_parts_lst, label=f"BODY-PART {bp_cnt + 1}:", label_width=30, dropdown_width=self.max_bp_len+5, value=self.body_parts_lst[bp_cnt])

            self.body_parts_dropdowns[bp_cnt].grid(row=bp_cnt, column=0, sticky=NW)

        self.data_options_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="DATA OPTIONS", icon_name='abacus')
        self.total_time_cb, self.total_time_var = SimbaCheckbox(parent=self.data_options_frm, txt='TOTAL ROI TIME (S)', val=True, txt_img='timer')
        self.entry_counts_cb, self.entry_counts_var = SimbaCheckbox(parent=self.data_options_frm, txt='ROI ENTRIES (COUNTS)', val=True, txt_img='abacus_2')
        self.first_entry_time_cb, self.first_entry_time_var = SimbaCheckbox(parent=self.data_options_frm, txt='FIRST ROI ENTRY TIME (S)', val=False, txt_img='one_blue')
        self.last_entry_time_cb, self.last_entry_time_var = SimbaCheckbox(parent=self.data_options_frm, txt='LAST ROI ENTRY TIME (S)', val=False, txt_img='finish')
        self.detailed_cb, self.detailed_var = SimbaCheckbox(parent=self.data_options_frm, txt='DETAILED ROI BOUT DATA (SEQUENCES)', val=False, txt_img='details')
        self.movement_cb, self.movement_var = SimbaCheckbox(parent=self.data_options_frm, txt='ROI MOVEMENT (VELOCITY & DISTANCES)', val=False, txt_img='pose')
        self.outside_cb, self.outside_var = SimbaCheckbox(parent=self.data_options_frm, txt='OUTSIDE ROI ZONES DATA', val=False, txt_img='outside_2', tooltip_txt=f'TREAT ALL NON-ROI REGIONS AS AN ROI REGION NAMED \n "{OUTSIDE_ROI}"')

        self.data_options_frm.grid(row=self.frame_children(frame=self.main_frm), column=0, sticky=NW)
        self.total_time_cb.grid(row=0, column=0, sticky=NW)
        self.entry_counts_cb.grid(row=1, column=0, sticky=NW)
        self.first_entry_time_cb.grid(row=2, column=0, sticky=NW)
        self.last_entry_time_cb.grid(row=3, column=0, sticky=NW)
        self.detailed_cb.grid(row=4, column=0, sticky=NW)
        self.movement_cb.grid(row=5, column=0, sticky=NW)
        self.outside_cb.grid(row=6, column=0, sticky=NW)

        self.format_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="FORMAT OPTIONS", icon_name='abacus')
        self.transpose_cb, self.transpose_var = SimbaCheckbox(parent=self.format_frm, txt='TRANSPOSE OUTPUT TABLE', val=False,  txt_img='restart')
        self.fps_cb, self.fps_var = SimbaCheckbox(parent=self.format_frm, txt='INCLUDE FPS DATA', val=False, txt_img='fps')
        self.video_length_cb, self.video_length_var = SimbaCheckbox(parent=self.format_frm, txt='INCLUDE VIDEO LENGTH DATA', val=False, txt_img='timer_2')
        self.px_per_mm_cb, self.px_per_mm_var = SimbaCheckbox(parent=self.format_frm, txt='INCLUDE PIXEL PER MILLIMETER DATA', val=False, txt_img='ruler')
        self.timbin_cb, self.timbin_var = SimbaCheckbox(parent=self.format_frm, txt='INCLUDE TIME-BIN TIME STAMPS', val=False, txt_img='time')

        self.format_frm.grid(row=self.frame_children(frame=self.main_frm), column=0, sticky=NW)
        self.transpose_cb.grid(row=0, column=0, sticky=NW)
        self.fps_cb.grid(row=1, column=0, sticky=NW)
        self.video_length_cb.grid(row=2, column=0, sticky=NW)
        self.px_per_mm_cb.grid(row=3, column=0, sticky=NW)
        self.timbin_cb.grid(row=4, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

        self.main_frm.mainloop()

    def run(self):
        bin_size = self.time_bin_entry.entry_get
        probability = self.probability_entry.entry_get
        total_time = self.total_time_var.get()
        entry_counts = self.entry_counts_var.get()
        first_entry_time = self.first_entry_time_var.get()
        last_entry_time = self.last_entry_time_var.get()
        detailed_table = self.detailed_var.get()
        movement = self.movement_var.get()
        transpose = self.transpose_var.get()
        include_fps = self.fps_var.get()
        video_length = self.video_length_var.get()
        px_per_mm = self.px_per_mm_var.get()
        timestamps = self.timbin_var.get()
        outside_roi = self.outside_var.get()
        body_parts = []
        for k, v in self.body_parts_dropdowns.items(): body_parts.append(v.getChoices())
        if not check_float(name='probability', value=probability, max_value=1.0, min_value=0.0, raise_error=False)[0]:
            raise InvalidInputError(msg=f'The PROBABILITY THRESHOLD has to be a value between 0.0 and 1.0 but got {probability}', source=self.__class__.__name__)
        if not check_float(name='time_bin', value=bin_size, min_value=10e-6, raise_error=False)[0]:
            raise InvalidInputError(msg=f'The TIME BIN LENGTH has to be a value in seconds greater than 10e-6, but got {bin_size}',source=self.__class__.__name__)
        selections = list({total_time, entry_counts, first_entry_time, last_entry_time, detailed_table, movement})
        if (len(selections) == 1) and (not selections[0]):
            raise InvalidInputError(msg=f'Please select at least one DATA OPTION.',source=self.__class__.__name__)

        analyzer = ROITimebinAnalyzer(config_path=self.config_path,
                                      bin_size=float(bin_size),
                                      data_path=None,
                                      threshold=float(probability),
                                      body_parts=body_parts,
                                      detailed_bout_data=detailed_table,
                                      calculate_distances=movement,
                                      total_time=total_time,
                                      entry_counts=entry_counts,
                                      first_entry_time=first_entry_time,
                                      last_entry_time=last_entry_time,
                                      outside_rois=outside_roi,
                                      transpose=transpose,
                                      include_fps=include_fps,
                                      include_video_length=video_length,
                                      include_px_per_mm=px_per_mm,
                                      include_time_stamps=timestamps)
        analyzer.run()
        analyzer.save()

#_ = ROIAnalysisTimeBinsPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
