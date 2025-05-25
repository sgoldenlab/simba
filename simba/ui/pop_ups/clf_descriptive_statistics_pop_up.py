import os
from tkinter import *
from typing import Union

from simba.data_processors.agg_clf_calculator import AggregateClfCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, SimbaButton,
                                        SimbaCheckbox)
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.errors import (NoChoosenClassifierError,
                                NoChoosenMeasurementError, NoDataError)


class ClfDescriptiveStatsPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if len(self.machine_results_paths) == 0:
            raise NoDataError(msg=f'No data files found inside {self.machine_results_dir} directory. Create classification data before analyzing aggregate classification descriptive statistics.', source=self.__class__.__name__)

        PopUpMixin.__init__(self, title="ANALYZE CLASSIFICATIONS: DESCRIPTIVE STATISTICS", size=(400, 600), icon='line_chart_red')
        measures_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MEASUREMENTS", icon_name='ruler', icon_link=Links.ANALYZE_ML_RESULTS.value, padx=5, pady=5, relief='solid')

        first_occurance_cb, self.first_occurance_var = SimbaCheckbox(parent=measures_frm, txt="First occurrence (s)", txt_img='one', val=True)
        event_count_cb, self.event_count_var = SimbaCheckbox(parent=measures_frm, txt="Event (bout) count", txt_img='abacus', val=True)
        total_event_duration_cb, self.total_event_duration_var = SimbaCheckbox(parent=measures_frm, txt="Total event duration (s)", txt_img='stopwatch', val=True)
        mean_event_duration_cb, self.mean_event_duration_var = SimbaCheckbox(parent=measures_frm, txt="Mean event duration (s)", txt_img='average', val=False)
        median_event_duration_cb, self.median_event_duration_var = SimbaCheckbox(parent=measures_frm, txt="Median event duration (s)", txt_img='average_2', val=False)
        mean_interval_duration_cb, self.mean_interval_duration_var = SimbaCheckbox(parent=measures_frm, txt="Mean event interval (s)", txt_img='timer_2', val=False)
        median_interval_duration_cb, self.median_interval_duration_var = SimbaCheckbox(parent=measures_frm, txt="Median event interval (s)", txt_img='timer', val=False)

        measures_frm.grid(row=0, column=0, sticky=NW, padx=10, pady=10)
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

        detailed_data_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="DETAILED BOUT DATA", icon_name='table_2', icon_link=Links.ANALYZE_ML_RESULTS.value, padx=5, pady=5, relief='solid')
        detailed_bout_cb, self.detailed_bout_var = SimbaCheckbox(parent=detailed_data_frm, txt='DETAILED BOUT DATA', txt_img='table_2')
        detailed_data_frm.grid(row=2, column=0, sticky=NW, padx=10, pady=10)
        detailed_bout_cb.grid(row=0, column=0, sticky=NW)

        video_meta_data_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="META-DATA", icon_name='abacus', icon_link=Links.ANALYZE_ML_RESULTS.value, padx=5, pady=5, relief='solid')
        metadata_frm_cnt_cb, self.metadata_frm_cnt_var = SimbaCheckbox(parent=video_meta_data_frm, txt='FRAME COUNT', txt_img='counter')
        metadata_frm_video_length_cb, self.metadata_video_length_var = SimbaCheckbox(parent=video_meta_data_frm, txt='VIDEO LENGTH (S)', txt_img='timer')
        video_meta_data_frm.grid(row=3, column=0, sticky=NW, padx=10, pady=10)
        metadata_frm_cnt_cb.grid(row=0, column=0, sticky=NW)
        metadata_frm_video_length_cb.grid(row=1, column=0, sticky=NW)

        output_options_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="OUTPUT OPTIONS", icon_name='options_small', icon_link=Links.ANALYZE_ML_RESULTS.value, padx=5, pady=5, relief='solid')
        transpose_output_cb, self.transpose_output_var = SimbaCheckbox(parent=output_options_frm, txt='TRANSPOSE OUTPUT', txt_img='convert')
        output_options_frm.grid(row=4, column=0, sticky=NW, padx=10, pady=10)
        transpose_output_cb.grid(row=0, column=0, sticky=NW)

        run_button = SimbaButton(parent=self.main_frm, txt="RUN", img='rocket', font=Formats.FONT_HEADER.value, cmd=self._run, width=Formats.BUTTON_WIDTH_XL.value,)
        run_button.grid(row=5, sticky=NW, padx=10, pady=10)
        self.main_frm.mainloop()

    def _run(self):
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

        frm_cnt = self.metadata_frm_cnt_var.get()
        video_length = self.metadata_video_length_var.get()

        data_log_analyzer = AggregateClfCalculator(config_path=self.config_path,
                                                   classifiers=clfs,
                                                   detailed_bout_data=self.detailed_bout_var.get(),
                                                   transpose=self.transpose_output_var.get(),
                                                   data_dir=self.machine_results_dir,
                                                   first_occurrence=first_occurance,
                                                   event_count=event_cnt,
                                                   total_event_duration=total_event_duration,
                                                   mean_event_duration=mean_event_duration,
                                                   median_event_duration=median_event_duration,
                                                   mean_interval_duration=mean_interval_duration,
                                                   median_interval_duration=median_interval_duration,
                                                   frame_count=frm_cnt,
                                                   video_length=video_length)

        data_log_analyzer.run()
        data_log_analyzer.save()

# _ = ClfDescriptiveStatsPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
#_ = ClfDescriptiveStatsPopUp(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")
# ClfDescriptiveStatsPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/raph/project_folder/project_config.ini')
