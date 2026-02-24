import os
from copy import deepcopy
from tkinter import *
from typing import Union

from simba.data_processors.distance_timbin_calculator import \
    DistanceTimeBinCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaCheckbox, SimBADropDown)
from simba.utils.checks import check_float
from simba.utils.enums import Links
from simba.utils.errors import DuplicationError, InvalidInputError, NoDataError


class TimBinDistanceAnalysisPopUp(ConfigReader, PopUpMixin):
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if len(self.outlier_corrected_paths) == 0:
            raise NoDataError(msg=f'No data files found in {self.outlier_corrected_dir} directory, cannot compute distance statistics.', source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="ANALYZE DISTANCES: TIME-BINS", size=(700, 500), icon='distance')

        self.distance_cnt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT NUMBER OF DISTANCES", icon_name='distance', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.distance_cnt_dropdown = SimBADropDown(parent=self.distance_cnt_frm, label="# OF DISTANCES", label_width=30, dropdown_width=20, value=1, dropdown_options=list(range(1, 11)), command=self.create_settings_frm, img='abacus', tooltip_key='DISTANCE_ANALYSIS_NUMBER_OF_PAIRS')
        self.distance_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.distance_cnt_dropdown.grid(row=0, column=0, sticky=NW)

        self.body_part_options = deepcopy(self.body_parts_lst)
        self.create_settings_frm(bp_pair_cnt=1)

        self.timebin_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT TIME-BIN", icon_name='timer_2', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.timebin_entry = Entry_Box(parent=self.timebin_frm, fileDescription='TIME BIN (S): ',labelwidth=30, entry_box_width=30, value=30.0, justify='center', trace=self._entrybox_bg_check_timer, img='timer_2', tooltip_key='DISTANCE_TIMEBINS_TIME_BIN_SIZE', bg_clr='lawngreen')
        self.timebin_frm.grid(row=2, column=0, sticky=NW)
        self.timebin_entry.grid(row=0, column=0, sticky=NW)


        self.measurments_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MEASUREMENTS", icon_name='ruler', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        mean_distance_cb, self.mean_distance_var = SimbaCheckbox(parent=self.measurments_frm, txt='MEAN DISTANCE (CM)', txt_img='mean', val=True, tooltip_key='DISTANCE_TIMEBINS_METRICS')
        median_distance_cb, self.median_distance_var = SimbaCheckbox(parent=self.measurments_frm, txt='MEDIAN DISTANCE (CM)', txt_img='median', val=True, tooltip_key='DISTANCE_TIMEBINS_METRICS')
        self.measurments_frm.grid(row=4, column=0, sticky=NW)
        mean_distance_cb.grid(row=0, column=0, sticky=NW)
        median_distance_cb.grid(row=1, column=0, sticky=NW)

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='rotate', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.probability_entry = Entry_Box(parent=self.settings_frm, fileDescription='BODY-PART THRESHOLD: ', labelwidth=30, entry_box_width=20, value=0.0, justify='center', img='green_dice', trace=self._entrybox_bg_check_float, tooltip_key='DISTANCE_TIMEBINS_BP_THRESHOLD', bg_clr='lawngreen')
        transpose_cb, self.transpose_var = SimbaCheckbox(parent=self.settings_frm, txt='TRANSPOSE OUTPUT CSV', txt_img='rotate', val=False, tooltip_key='DISTANCE_TIMEBINS_TRANSPOSE')
        self.settings_frm.grid(row=5, column=0, sticky=NW)
        self.probability_entry.grid(row=0, column=0, sticky=NW)
        transpose_cb.grid(row=1, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run, title=f'RUN ({len(self.outlier_corrected_paths)} file(s))')
        self.main_frm.mainloop()


    def _entrybox_bg_check_float(self, entry_box: Entry_Box, valid_clr: str = 'lawngreen', invalid_clr: str = 'white'):
        valid_value = check_float(name='', allow_negative=False, allow_zero=True, value=entry_box.entry_get, raise_error=False, min_value=0.0, max_value=1.0)[0]
        entry_box.set_bg_clr(clr=valid_clr if valid_value else invalid_clr)

    def _entrybox_bg_check_timer(self, entry_box: Entry_Box, valid_clr: str = 'lawngreen', invalid_clr: str = 'white'):
        valid_value = check_float(name='', allow_negative=False, allow_zero=True, value=entry_box.entry_get, raise_error=False, min_value=0.0)[0]
        entry_box.set_bg_clr(clr=valid_clr if valid_value else invalid_clr)

    def create_settings_frm(self, bp_pair_cnt: int):
        if hasattr(self, "bp_frm"):
            self.bp_frm.destroy()
            for i in self.body_parts_dropdowns:
                i[0].destroy(); i[1].destroy()

        self.bp_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT BODY-PARTS", icon_name='pose', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.body_parts_dropdowns = []
        for cnt, i in enumerate(range(int(bp_pair_cnt))):
            bp_1_dropdown = SimBADropDown(parent=self.bp_frm, label=f"BODY-PART 1:", label_width=20, dropdown_width=20, value=self.body_part_options[cnt], dropdown_options=self.body_part_options, img='circle_black', tooltip_key='DISTANCE_ANALYSIS_BODY_PART')
            bp_2_dropdown = SimBADropDown(parent=self.bp_frm, label=f"BODY-PART 2:", label_width=20, dropdown_width=20, value=self.body_part_options[cnt], dropdown_options=self.body_part_options, img='circle_black', tooltip_key='DISTANCE_ANALYSIS_BODY_PART')
            bp_1_dropdown.grid(row=cnt, column=0, sticky=NW, padx=(0, 15))
            bp_2_dropdown.grid(row=cnt, column=1, sticky=NW)
            self.body_parts_dropdowns.append((bp_1_dropdown, bp_2_dropdown))
        self.bp_frm.grid(row=3, column=0, sticky=NW)

    def run(self):
        check_float(name="Probability threshold", value=self.probability_entry.entry_get, min_value=0.00, max_value=1.00)
        threshold = float(self.probability_entry.entry_get)
        transpose = self.transpose_var.get()
        time_bin = self.timebin_entry.entry_get
        check_float(name=f'{self.__class__.__name__} TIME-BIN', value=time_bin, allow_zero=False, allow_negative=False)
        bp_pairs = []
        for bp_pair_cnt, bp_pair_dropdown in enumerate(self.body_parts_dropdowns):
            bp_1, bp_2 = bp_pair_dropdown[0].get_value(), bp_pair_dropdown[1].get_value()
            if bp_1 == bp_2:
                raise DuplicationError(msg=f'In row {bp_pair_cnt+1}, body part one ({bp_1}) cannot be the same as body part two ({bp_2})', source=self.__class__.__name__)
            bp_pairs.append((bp_1, bp_2))
        mean_distance, median_distance = self.mean_distance_var.get(), self.median_distance_var.get()
        if not mean_distance and not median_distance:
            raise InvalidInputError(msg='All metrics are un-checked. To compute distance metrics, check at least one output variable.', source=self.__class__.__name__)
        distance_processor = DistanceTimeBinCalculator(config_path=self.config_path,
                                                       threshold=threshold,
                                                       body_parts=bp_pairs,
                                                       transpose=transpose,
                                                       distance_mean=mean_distance,
                                                       distance_median=median_distance,
                                                       verbose=True,
                                                       time_bin=float(time_bin))
        distance_processor.run()
        distance_processor.save()

#_ = TimBinDistanceAnalysisPopUp(config_path=r"D:\troubleshooting\maplight_ri\project_folder\project_config.ini")
# MovementAnalysisPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
#_ = MovementAnalysisPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
