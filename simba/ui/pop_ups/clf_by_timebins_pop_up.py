__author__ = "Simon Nilsson"

import multiprocessing
import os
from tkinter import *
from typing import Union

from simba.data_processors.timebins_clf_calculator import TimeBinsClfCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton, SimbaCheckbox)
from simba.utils.checks import check_int
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.errors import (NoChoosenClassifierError,
                                NoChoosenMeasurementError, NoDataError)


class TimeBinsClfPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if len(self.machine_results_paths) == 0:
            raise NoDataError(msg=f'Cannot compute classifications by time-bin: no data found in the {self.machine_results_dir} directory', source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="CLASSIFICATION BY TIME BINS", icon='timer_2')

        measures_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MEASUREMENTS", icon_name='ruler', icon_link=Links.ANALYZE_ML_RESULTS.value, padx=5, pady=5, relief='solid')
        first_occurance_cb, self.first_occurance_var = SimbaCheckbox(parent=measures_frm, txt="First occurrence (s)", txt_img='one', val=True)
        event_count_cb, self.event_count_var = SimbaCheckbox(parent=measures_frm, txt="First occurrence (s)", txt_img='abacus', val=True)
        total_event_duration_cb, self.total_event_duration_var = SimbaCheckbox(parent=measures_frm, txt="Total event duration (s)", txt_img='stopwatch', val=True)
        mean_event_duration_cb, self.mean_event_duration_var = SimbaCheckbox(parent=measures_frm, txt="Mean event duration (s)", txt_img='average', val=False)
        median_event_duration_cb, self.median_event_duration_var = SimbaCheckbox(parent=measures_frm, txt="Median event duration (s)", txt_img='average_2', val=False)
        mean_interval_duration_cb, self.mean_interval_duration_var = SimbaCheckbox(parent=measures_frm, txt="Mean event interval (s)", txt_img='timer_2', val=False)
        median_interval_duration_cb, self.median_interval_duration_var = SimbaCheckbox(parent=measures_frm, txt="Median event interval (s)", txt_img='timer', val=False)

        measures_frm.grid(row=0, column=0, sticky=NW)
        first_occurance_cb.grid(row=0, column=0, sticky=NW)
        event_count_cb.grid(row=1, column=0, sticky=NW)
        total_event_duration_cb.grid(row=2, column=0, sticky=NW)
        mean_event_duration_cb.grid(row=3, column=0, sticky=NW)
        median_event_duration_cb.grid(row=4, column=0, sticky=NW)
        mean_interval_duration_cb.grid(row=5, column=0, sticky=NW)
        median_interval_duration_cb.grid(row=6, column=0, sticky=NW)

        self.clf_vars = {}
        clf_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CLASSIFIERS", icon_name='forest', icon_link=Links.ANALYZE_ML_RESULTS.value, padx=5, pady=5, relief='solid')
        for cnt, clf_name in enumerate(self.clf_names):
            cb, self.clf_vars[clf_name] = SimbaCheckbox(parent=clf_frm, txt=clf_name, val=True)
            cb.grid(row=cnt, column=0, sticky=NW)

        clf_frm.grid(row=1, column=0, sticky=NW, padx=10, pady=10)
        time_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SET TIME", icon_name='timer_2', icon_link=Links.ANALYZE_ML_RESULTS.value, padx=5, pady=5,  relief='solid')
        self.timebin_entrybox = Entry_Box(time_frm, "TIME BIN SIZE (S): ", validation="numeric", labelwidth=30, entry_box_width=15, value=30)

        time_frm.grid(row=2, column=0, sticky=NW)
        self.timebin_entrybox.grid(row=0, column=0, sticky=NW, pady=10, padx=10)

        run_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="RUN", icon_name='run', icon_link=Links.ANALYZE_ML_RESULTS.value, padx=5, pady=5, relief='solid')
        run_button = SimbaButton(parent=run_frm, txt="Run", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self._run)

        run_frm.grid(row=3, column=0, sticky=NW)
        run_button.grid(row=0, column=0, sticky=NW, pady=10)

        self.main_frm.mainloop()

    def _run(self):
        check_int(name="Time bin", value=self.timebin_entrybox.entry_get)

        first_occurance = self.first_occurance_var.get()
        event_cnt = self.event_count_var.get()
        total_event_duration = self.total_event_duration_var.get()
        mean_event_duration = self.mean_event_duration_var.get()
        median_event_duration = self.median_event_duration_var.get()
        mean_interval_duration = self.mean_interval_duration_var.get()
        median_interval_duration = self.median_interval_duration_var.get()

        if not any([first_occurance, event_cnt, total_event_duration, mean_event_duration, median_event_duration, mean_interval_duration, median_interval_duration]):
            raise NoChoosenMeasurementError(source=self.__class__.__name__)

        clfs = [clf_name for clf_name, cb_var in self.clf_vars.items() if cb_var.get()]
        if len(clfs) == 0:
            raise NoChoosenClassifierError(source=self.__class__.__name__)

        time_bins_clf_analyzer = TimeBinsClfCalculator(config_path=self.config_path,
                                                       bin_length=int(self.timebin_entrybox.entry_get),
                                                       classifiers=clfs,
                                                       first_occurrence=first_occurance,
                                                       event_count=event_cnt,
                                                       total_event_duration=total_event_duration,
                                                       mean_event_duration=mean_event_duration,
                                                       median_event_duration=median_event_duration,
                                                       mean_interval_duration=mean_interval_duration,
                                                       median_interval_duration=median_interval_duration)

        time_bins_clf_analyzer.run()

#_ = TimeBinsClfPopUp(config_path=r"D:\troubleshooting\mitra\project_folder\project_config.ini")

#_ = TimeBinsClfPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
