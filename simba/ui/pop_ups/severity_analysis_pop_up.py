__author__ = "Simon Nilsson"

from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.enums import Formats, Keys, Links
from simba.ui.tkinter_functions import DropDownMenu, CreateLabelFrameWithIcon
from simba.utils.errors import NoSpecifiedOutputError
from simba.data_processors.severity_calculator import SeverityCalculator

class AnalyzeSeverityPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='SIMBA SEVERITY ANALYSIS')
        ConfigReader.__init__(self, config_path=config_path)

        if len(self.multi_animal_id_list) > 1:
            self.multi_animal_id_list.insert(0, 'ALL ANIMALS')
        self.frame_cnt_var = BooleanVar(value=False)
        self.seconds_cnt_var = BooleanVar(value=False)
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ANALYZE_ML_RESULTS.value)
        self.clf_dropdown = DropDownMenu(self.settings_frm, 'CLASSIFIER:', self.clf_names, '25')
        self.clf_dropdown.setChoices(self.clf_names[0])
        self.brackets_dropdown = DropDownMenu(self.settings_frm, 'BRACKETS:', list(range(1,21)), '25')
        self.brackets_dropdown.setChoices(10)
        self.animal_dropdown = DropDownMenu(self.settings_frm, 'ANIMALS', self.multi_animal_id_list, '25')
        self.animal_dropdown.setChoices(self.multi_animal_id_list[0])

        frame_cnt_cb = Checkbutton(self.settings_frm, text='FRAME COUNT', variable=self.frame_cnt_var)
        seconds_cnt_cb = Checkbutton(self.settings_frm, text='SECONDS', variable=self.seconds_cnt_var)

        run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        run_btn = Button(run_frm, text='RUN SEVERITY ANALYSIS', command= lambda: self.run())

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.clf_dropdown.grid(row=0, column=0, sticky=NW)
        self.brackets_dropdown.grid(row=1, column=0, sticky=NW)
        self.animal_dropdown.grid(row=2, column=0, sticky=NW)
        frame_cnt_cb.grid(row=3, column=0, sticky=NW)
        seconds_cnt_cb.grid(row=4, column=0, sticky=NW)

        run_frm.grid(row=1, column=0, sticky=NW)
        run_btn.grid(row=0, column=0, sticky=NW)

    def run(self):
        if self.animal_dropdown.getChoices() == 'ALL ANIMALS':
            animals = self.multi_animal_id_list[1:]
        else:
            animals = [self.animal_dropdown.getChoices()]
        settings = {'brackets': int(self.brackets_dropdown.getChoices()),
                    'clf': self.clf_dropdown.getChoices(),
                    'animals': animals,
                    'time': self.seconds_cnt_var.get(),
                    'frames': self.frame_cnt_var.get()}

        if (not self.seconds_cnt_var.get()) and (not self.frame_cnt_var.get()):
            raise NoSpecifiedOutputError(msg='SIMBA ERROR: Please select frames and/or time output metrics')
        severity_processor = SeverityCalculator(config_path=self.config_path,
                                                settings=settings)
        severity_processor.run()
        severity_processor.save()
