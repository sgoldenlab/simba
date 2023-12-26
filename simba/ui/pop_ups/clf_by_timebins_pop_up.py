__author__ = "Simon Nilsson"

import multiprocessing
from tkinter import *

from simba.ui.tkinter_functions import Entry_Box, CreateLabelFrameWithIcon
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.data_processors.timebins_clf_calculator import TimeBinsClfCalculator
from simba.utils.errors import NoChoosenMeasurementError, NoChoosenClassifierError
from simba.utils.enums import Options, Formats, Links, Keys
from simba.utils.checks import check_int

class TimeBinsClfPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):
        PopUpMixin.__init__(self, title='CLASSIFICATION BY TIME BINS')
        ConfigReader.__init__(self, config_path=config_path)
        cbox_titles = Options.TIMEBINS_MEASURMENT_OPTIONS.value
        self.timebin_entrybox = Entry_Box(self.main_frm, 'Set time bin size (s)', '15', validation='numeric')
        measures_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='MEASUREMENTS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ANALYZE_ML_RESULTS.value)
        clf_frm = LabelFrame(self.main_frm, text='CLASSIFIERS', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.measurements_var_dict, self.clf_var_dict = {}, {}
        for cnt, title in enumerate(cbox_titles):
            self.measurements_var_dict[title] = BooleanVar()
            cbox = Checkbutton(measures_frm, text=title, variable=self.measurements_var_dict[title])
            cbox.grid(row=cnt, sticky=NW)
        for cnt, clf_name in enumerate(self.clf_names):
            self.clf_var_dict[clf_name] = BooleanVar()
            cbox = Checkbutton(clf_frm, text=clf_name, variable=self.clf_var_dict[clf_name])
            cbox.grid(row=cnt, sticky=NW)
        run_button = Button(self.main_frm, text='Run', command=lambda: self.run_time_bins_clf())
        measures_frm.grid(row=0, sticky=NW)
        clf_frm.grid(row=1, sticky=NW)
        self.timebin_entrybox.grid(row=2, sticky=NW)
        run_button.grid(row=3, sticky=NW)

    def run_time_bins_clf(self):
        check_int(name='Time bin', value=self.timebin_entrybox.entry_get)
        measurement_lst, clf_list = [], []
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
        time_bins_clf_analyzer = TimeBinsClfCalculator(config_path=self.config_path,
                                                       bin_length=int(self.timebin_entrybox.entry_get),
                                                       measurements=measurement_lst,
                                                       classifiers=clf_list)

        time_bins_clf_multiprocessor = multiprocessing.Process(target=time_bins_clf_analyzer.run())
        time_bins_clf_multiprocessor.start()


#_ = TimeBinsClfPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
