__author__ = "Simon Nilsson"

from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.mixins.config_reader import ConfigReader
from simba.utils.enums import Formats, Keys, Links
from simba.ui.tkinter_functions import DropDownMenu, CreateLabelFrameWithIcon
from simba.utils.errors import NoSpecifiedOutputError
from simba.data_processors.severity_frame_based_calculator import SeverityFrameCalculator
from simba.data_processors.severity_bout_based_calculator import SeverityBoutCalculator

class AnalyzeSeverityPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: str):

        PopUpMixin.__init__(self, title='SIMBA SEVERITY ANALYSIS')
        ConfigReader.__init__(self, config_path=config_path)

        if len(self.multi_animal_id_list) > 1:
            self.multi_animal_id_list.insert(0, 'ALL ANIMALS')
        self.clip_cnt_options = list(range(1, 20))
        self.clip_cnt_options.insert(0, 'ALL CLIPS')
        self.frame_cnt_var = BooleanVar(value=False)
        self.seconds_cnt_var = BooleanVar(value=False)
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ANALYZE_ML_RESULTS.value)
        self.clf_dropdown = DropDownMenu(self.settings_frm, 'CLASSIFIER:', self.clf_names, '25')
        self.clf_dropdown.setChoices(self.clf_names[0])
        self.brackets_dropdown = DropDownMenu(self.settings_frm, 'BRACKETS:', list(range(1,21)), '25')
        self.bracket_type_dropdown = DropDownMenu(self.settings_frm, 'BRACKET TYPE: ', ['QUANTIZE', 'QUANTILE'], '25')
        self.brackets_dropdown.setChoices(10)
        self.bracket_type_dropdown.setChoices('QUANTIZE')
        self.animal_dropdown = DropDownMenu(self.settings_frm, 'ANIMALS', self.multi_animal_id_list, '25')
        self.animal_dropdown.setChoices(self.multi_animal_id_list[0])
        self.bouts_vs_frames_dropdown = DropDownMenu(self.settings_frm, 'DATA TYPE: ', ['BOUTS', 'FRAMES'], '25', com=self.show_settings)
        self.bouts_vs_frames_dropdown.setChoices('FRAMES')
        self.normalization_type_dropdown = DropDownMenu(self.settings_frm, 'MOVEMENT NORMALIZATION TYPE: ', ['ALL VIDEOS', 'SINGLE VIDEOS'], '25')
        self.normalization_type_dropdown.setChoices('ALL VIDEOS')
        self.save_bracket_cut_off_points_var = BooleanVar(value=True)
        self.savet_brackets_info_cb = Checkbutton(self.settings_frm, text='Save bracket definitions', variable=self.save_bracket_cut_off_points_var)
        self.further_settings_frm = LabelFrame(self.main_frm, text='Further settings', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        run_frm = LabelFrame(self.main_frm, text='RUN', font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg='black')
        run_btn = Button(run_frm, text='RUN SEVERITY ANALYSIS', command= lambda: self.run())
        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.clf_dropdown.grid(row=0, column=0, sticky=NW)
        self.brackets_dropdown.grid(row=1, column=0, sticky=NW)
        self.bracket_type_dropdown.grid(row=2, column=0, sticky=NW)
        self.animal_dropdown.grid(row=3, column=0, sticky=NW)
        self.bouts_vs_frames_dropdown.grid(row=4, column=0, sticky=NW)
        self.normalization_type_dropdown.grid(row=5, column=0, sticky=NW)
        self.savet_brackets_info_cb.grid(row=6, column=0, sticky=NW)
        self.further_settings_frm.grid(row=2, column=0, sticky=NW)
        run_frm.grid(row=3, column=0, sticky=NW)
        run_btn.grid(row=0, column=0, sticky=NW)
        self.show_settings(self.bouts_vs_frames_dropdown.getChoices())
        self.main_frm.mainloop()

    def show_settings(self, setting):
        for w in self.further_settings_frm.winfo_children():
            w.destroy()

        self.further_settings_frm.grid()
        self.visualize_var = BooleanVar(value=False)
        self.show_pose_var = BooleanVar(value=False)
        self.visualize_cb = Checkbutton(self.further_settings_frm, text='Visualize', variable=self.visualize_var, command=self.enable_visualization_options)
        self.show_pose_cb = Checkbutton(self.further_settings_frm, text='Show pose-estimated locations', variable=self.show_pose_var)
        self.video_speed_dropdown = DropDownMenu(self.further_settings_frm, 'Video speed:', [float(i) / 10 for i in range(1, 21)], '25')
        self.video_speed_dropdown.setChoices(1.0)
        self.number_of_clips_dropdown = DropDownMenu(self.further_settings_frm, 'Clip count:', self.clip_cnt_options, '25')
        self.show_pose_cb.config(state=DISABLED)
        self.video_speed_dropdown.disable()
        self.number_of_clips_dropdown.disable()
        self.visualize_cb.grid(row=0, column=0, sticky=NW)
        self.show_pose_cb.grid(row=1, column=0, sticky=NW)
        self.video_speed_dropdown.grid(row=2, column=0, sticky=NW)
        self.number_of_clips_dropdown.grid(row=3, column=0, sticky=NW)
        self.number_of_clips_dropdown.setChoices('ALL CLIPS')

        if setting == 'FRAMES':
            self.frame_cnt_cb = Checkbutton(self.further_settings_frm, text='FRAME COUNT', variable=self.frame_cnt_var)
            self.seconds_cnt_cb = Checkbutton(self.further_settings_frm, text='SECONDS', variable=self.seconds_cnt_var)
            self.frame_cnt_cb.grid(row=4, column=0, sticky=NW)
            self.seconds_cnt_cb.grid(row=5, column=0, sticky=NW)

    def enable_visualization_options(self):
        if self.visualize_var.get():
            self.show_pose_cb.config(state=NORMAL)
            self.video_speed_dropdown.enable()
            self.number_of_clips_dropdown.enable()
        else:
            self.show_pose_cb.config(state=DISABLED)
            self.video_speed_dropdown.disable()
            self.number_of_clips_dropdown.disable()

    def run(self):
        if self.animal_dropdown.getChoices() == 'ALL ANIMALS':
            animals = self.multi_animal_id_list[1:]
        else:
            animals = [self.animal_dropdown.getChoices()]
        settings = {'brackets': int(self.brackets_dropdown.getChoices()),
                    'bracket_type': self.bracket_type_dropdown.getChoices(),
                    'clf': self.clf_dropdown.getChoices(),
                    'animals': animals,
                    'save_bin_definitions': self.save_bracket_cut_off_points_var.get(),
                    'normalization': self.normalization_type_dropdown.getChoices()}
        settings['visualize'] = self.visualize_var.get()
        settings['video_speed'] = float(self.video_speed_dropdown.getChoices())
        settings['visualize_event_cnt'] = self.number_of_clips_dropdown.getChoices()
        settings['show_pose'] = self.show_pose_var.get()
        if self.bouts_vs_frames_dropdown.getChoices() == 'FRAMES':
            settings['frames'] = self.frame_cnt_var.get()
            settings['time'] = self.seconds_cnt_var.get()
            if (not self.seconds_cnt_var.get()) and (not self.frame_cnt_var.get()):
                raise NoSpecifiedOutputError(msg='SIMBA ERROR: Please select frames and/or time output metrics', source=self.__class__.__name__)

        if self.bouts_vs_frames_dropdown.getChoices() == 'FRAMES':
            severity_calculator = SeverityFrameCalculator(config_path=self.config_path,
                                                          settings=settings)
        else:
            severity_calculator = SeverityBoutCalculator(config_path=self.config_path,
                                                          settings=settings)

        severity_calculator.run()

#_ = AnalyzeSeverityPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')