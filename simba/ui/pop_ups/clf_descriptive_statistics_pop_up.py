from tkinter import *

from simba.data_processors.agg_clf_calculator import AggregateClfCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.errors import (NoChoosenClassifierError,
                                NoChoosenMeasurementError)


class ClfDescriptiveStatsPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(
            self,
            title="ANALYZE CLASSIFICATIONS: DESCRIPTIVE STATISTICS",
            size=(400, 500),
        )
        ConfigReader.__init__(self, config_path=config_path)
        measures_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="MEASUREMENTS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.ANALYZE_ML_RESULTS.value,
        )
        clf_frm = LabelFrame(
            self.main_frm,
            text="CLASSIFIERS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        detailed_data_frm = LabelFrame(
            self.main_frm,
            text="DETAILED BOUT DATA",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        video_meta_data_frm = LabelFrame(
            self.main_frm,
            text="METADATA",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        output_options_frm = LabelFrame(
            self.main_frm,
            text="OUTPUT OPTIONS",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            fg="black",
        )
        self.measurements_var_dict, self.clf_var_dict = {}, {}
        cbox_titles = Options.CLF_DESCRIPTIVES_OPTIONS.value
        for cnt, title in enumerate(cbox_titles):
            self.measurements_var_dict[title] = BooleanVar()
            cbox = Checkbutton(
                measures_frm, text=title, variable=self.measurements_var_dict[title]
            )
            cbox.grid(row=cnt, sticky=NW)
        for cnt, clf_name in enumerate(self.clf_names):
            self.clf_var_dict[clf_name] = BooleanVar()
            cbox = Checkbutton(
                clf_frm, text=clf_name, variable=self.clf_var_dict[clf_name]
            )
            cbox.grid(row=cnt, sticky=NW)
        run_button = Button(
            self.main_frm, text="Run", command=lambda: self.run_descriptive_analysis()
        )
        measures_frm.grid(row=0, sticky=NW)
        clf_frm.grid(row=1, sticky=NW)
        (
            self.metadata_frm_cnt_var,
            self.metadata_video_length_var,
            self.detailed_bout_var,
            self.transpose_output_var,
        ) = (BooleanVar(), BooleanVar(), BooleanVar(), BooleanVar())
        metadata_frm_cnt_cb = Checkbutton(
            video_meta_data_frm, text="Frame count", variable=self.metadata_frm_cnt_var
        )
        metadata_frm_video_length_cb = Checkbutton(
            video_meta_data_frm,
            text="Video length (s)",
            variable=self.metadata_video_length_var,
        )
        detailed_bout_cb = Checkbutton(
            detailed_data_frm,
            text="Detailed bout data",
            variable=self.detailed_bout_var,
        )
        transpose_output_cb = Checkbutton(
            output_options_frm,
            text="Transpose output",
            variable=self.transpose_output_var,
        )
        detailed_data_frm.grid(row=2, sticky=NW)
        detailed_bout_cb.grid(row=0, column=0, sticky=NW)
        transpose_output_cb.grid(row=0, column=0, sticky=NW)
        video_meta_data_frm.grid(row=3, sticky=NW)
        output_options_frm.grid(row=4, sticky=NW)
        metadata_frm_cnt_cb.grid(row=0, column=0, sticky=NW)
        metadata_frm_video_length_cb.grid(row=1, column=0, sticky=NW)
        run_button.grid(row=5, sticky=NW)
        self.main_frm.mainloop()

    def run_descriptive_analysis(self):
        measurement_lst, clf_list, meta_data_lst = [], [], []
        for name, val in self.measurements_var_dict.items():
            if val.get():
                measurement_lst.append(name)
        for name, val in self.clf_var_dict.items():
            if val.get():
                clf_list.append(name)
        if len(measurement_lst) == 0:
            raise NoChoosenMeasurementError(source=self.__class__.__name__)
        if len(clf_list) == 0:
            raise NoChoosenClassifierError(source=self.__class__.__name__)
        if self.metadata_frm_cnt_var.get():
            meta_data_lst.append("Frame count")
        if self.metadata_video_length_var.get():
            meta_data_lst.append("Video length (s)")
        data_log_analyzer = AggregateClfCalculator(
            config_path=self.config_path,
            data_measures=measurement_lst,
            video_meta_data=meta_data_lst,
            classifiers=clf_list,
            detailed_bout_data=self.detailed_bout_var.get(),
            transpose=self.transpose_output_var.get(),
        )
        data_log_analyzer.run()
        data_log_analyzer.save()


# ClfDescriptiveStatsPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# ClfDescriptiveStatsPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/raph/project_folder/project_config.ini')
