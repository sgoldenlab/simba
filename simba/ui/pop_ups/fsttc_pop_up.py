__author__ = "Simon Nilsson"

from tkinter import *
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.data_processors.fsttc_calculator import FSTTCCalculator
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, Entry_Box
from simba.utils.errors import CountError
from simba.utils.checks import check_int
from simba.utils.enums import Keys, Links


class FSTTCPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='FORWARD SPIKE TIME TILING COEFFICIENTS')
        ConfigReader.__init__(self, config_path=config_path)
        fsttc_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='FSTTC Settings', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.FSTTC.value)
        graph_cb_var = BooleanVar()
        graph_cb = Checkbutton(fsttc_settings_frm,text='Create graph',variable=graph_cb_var)
        time_delta = Entry_Box(fsttc_settings_frm,'Time Delta','10', validation='numeric')
        behaviors_frm = LabelFrame(fsttc_settings_frm,text="Behaviors")
        clf_var_dict, clf_cb_dict = {}, {}
        for clf_cnt, clf in enumerate(self.clf_names):
            clf_var_dict[clf] = BooleanVar()
            clf_cb_dict[clf] = Checkbutton(behaviors_frm, text=clf, variable=clf_var_dict[clf])
            clf_cb_dict[clf].grid(row=clf_cnt, sticky=NW)

        fsttc_run_btn = Button(self.main_frm,text='Calculate FSTTC',command=lambda:self.run_fsttc(time_delta=time_delta.entry_get, graph_var= graph_cb_var.get(), behaviours_dict=clf_var_dict))

        fsttc_settings_frm.grid(row=0,sticky=W,pady=5)
        graph_cb.grid(row=0,sticky=W,pady=5)
        time_delta.grid(row=1,sticky=W,pady=5)
        behaviors_frm.grid(row=2,sticky=W,pady=5)
        fsttc_run_btn.grid(row=3, pady=10)

    def run_fsttc(self,
                  graph_var: bool,
                  behaviours_dict: dict,
                  time_delta: int=None):

        check_int('Time delta', value=time_delta)
        targets = []
        for behaviour, behavior_val in behaviours_dict.items():
            if behavior_val.get():
                targets.append(behaviour)

        if len(targets) < 2:
            raise CountError(msg='FORWARD SPIKE TIME TILING COEFFICIENTS REQUIRE 2 OR MORE BEHAVIORS.')

        fsttc_calculator = FSTTCCalculator(config_path=self.config_path,
                                           time_window=time_delta,
                                           behavior_lst=targets,
                                           create_graphs=graph_var)
        fsttc_calculator.run()

#_ = FSTTCPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
