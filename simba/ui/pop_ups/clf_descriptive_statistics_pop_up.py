from tkinter import *

from simba.ui.tkinter_functions import CreateLabelFrameWithIcon
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.data_processors.agg_clf_calculator import AggregateClfCalculator
from simba.utils.errors import NoChoosenMeasurementError, NoChoosenClassifierError
from simba.utils.enums import Links, Formats, Keys, Options

class ClfDescriptiveStatsPopUp(PopUpMixin,
                               ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='ANALYZE CLASSIFICATIONS: DESCRIPTIVE STATISTICS')
        ConfigReader.__init__(self, config_path=config_path)
        measures_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='MEASUREMENTS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ANALYZE_ML_RESULTS.value)
        clf_frm = LabelFrame(self.main_frm, text='CLASSIFIERS', font=Formats.LABELFRAME_HEADER_FORMAT.value, fg='black')
        self.measurements_var_dict, self.clf_var_dict = {}, {}
        cbox_titles = Options.CLF_DESCRIPTIVES_OPTIONS.value
        for cnt, title in enumerate(cbox_titles):
            self.measurements_var_dict[title] = BooleanVar()
            cbox = Checkbutton(measures_frm, text=title, variable=self.measurements_var_dict[title])
            cbox.grid(row=cnt, sticky=NW)
        for cnt, clf_name in enumerate(self.clf_names):
            self.clf_var_dict[clf_name] = BooleanVar()
            cbox = Checkbutton(clf_frm, text=clf_name, variable=self.clf_var_dict[clf_name])
            cbox.grid(row=cnt, sticky=NW)
        run_button = Button(self.main_frm, text='Run', command=lambda: self.run_descriptive_analysis())
        measures_frm.grid(row=0, sticky=NW)
        clf_frm.grid(row=1, sticky=NW)
        run_button.grid(row=2, sticky=NW)

    def run_descriptive_analysis(self):
        measurement_lst, clf_list = [], []
        for name, val in self.measurements_var_dict.items():
            if val.get():
                measurement_lst.append(name)
        for name, val in self.clf_var_dict.items():
            if val.get():
                clf_list.append(name)
        if len(measurement_lst) == 0:
            raise NoChoosenMeasurementError()
        if len(clf_list) == 0:
            raise NoChoosenClassifierError()
        data_log_analyzer = AggregateClfCalculator(config_path=self.config_path, data_measures=measurement_lst, classifiers=clf_list)
        data_log_analyzer.run()
        data_log_analyzer.save()