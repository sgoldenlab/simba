from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, DropDownMenu, Entry_Box
from simba.utils.enums import Keys, Links, Formats, ConfigKey
from simba.utils.checks import check_float
from simba.roi_tools.ROI_analyzer import ROIAnalyzer

class ROIAnalysisPopUp(ConfigReader, PopUpMixin):

    def __init__(self,
                 config_path: str):

        ConfigReader.__init__(self, config_path=config_path)
        PopUpMixin.__init__(self, title='ROI ANALYSIS', size=(400, 400))
        self.animal_cnt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SELECT NUMBER OF ANIMALS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ROI_DATA_ANALYSIS.value)
        self.animal_cnt_dropdown = DropDownMenu(self.animal_cnt_frm, '# of animals', list(range(1, self.animal_cnt+1)), labelwidth=20)
        self.animal_cnt_dropdown.setChoices(1)
        self.animal_cnt_confirm_btn = Button(self.animal_cnt_frm, text="Confirm", command=lambda: self.create_settings_frm())
        self.animal_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_confirm_btn.grid(row=0, column=1, sticky=NW)
        self.main_frm.mainloop()


    def create_settings_frm(self):
        if hasattr(self, 'setting_frm'):
            self.setting_frm.destroy()
            self.body_part_frm.destroy()

        self.setting_frm = LabelFrame(self.main_frm, text="SETTINGS", font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.choose_bp_frm(parent=self.setting_frm, bp_options=self.body_parts_lst)
        self.choose_bp_threshold_frm(parent=self.setting_frm)
        self.calculate_distances_frm = LabelFrame(self.setting_frm, text="CALCULATE DISTANCES")
        self.calculate_distance_moved_var = BooleanVar(value=False)
        self.calculate_distance_moved_cb = Checkbutton(self.calculate_distances_frm, text='Calculate distance moved within ROI', variable=self.calculate_distance_moved_var)
        self.calculate_distance_moved_cb.grid(row=0, column=0, sticky=NW)
        self.calculate_distances_frm.grid(row=self.frame_children(frame=self.setting_frm), column=0, sticky=NW)
        self.setting_frm.grid(row=1, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        settings = {}
        check_float(name='Probability threshold', value=self.probability_entry.entry_get, min_value=0.00, max_value=1.00)
        settings['threshold'] = float(self.probability_entry.entry_get)
        settings['body_parts'] = {}
        self.config.set(ConfigKey.ROI_SETTINGS.value, ConfigKey.ROI_ANIMAL_CNT.value, str(self.animal_cnt_dropdown.getChoices()))
        self.config.set(ConfigKey.ROI_SETTINGS.value, ConfigKey.PROBABILITY_THRESHOLD.value,str(self.probability_entry.entry_get))
        for cnt, dropdown in self.body_parts_dropdowns.items():
            settings['body_parts'][f'animal_{cnt + 1}_bp'] = dropdown.getChoices()
            self.config.set('ROI settings', f'animal_{cnt + 1}_bp', str(dropdown.getChoices()))
        self.update_config()
        roi_analyzer = ROIAnalyzer(ini_path=self.config_path,
                                   data_path='outlier_corrected_movement_location',
                                   calculate_distances=self.calculate_distance_moved_var.get(),
                                   settings=settings)
        roi_analyzer.run()
        roi_analyzer.save()
