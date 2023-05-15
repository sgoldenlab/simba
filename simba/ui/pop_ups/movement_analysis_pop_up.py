from tkinter import *
from copy import deepcopy

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, DropDownMenu, Entry_Box
from simba.utils.enums import Keys, Links, Formats, ConfigKey
from simba.utils.checks import check_float
from simba.data_processors.movement_calculator import MovementCalculator

class MovementAnalysisPopUp(ConfigReader, PopUpMixin):

    def __init__(self,
                 config_path: str):

        ConfigReader.__init__(self, config_path=config_path)
        PopUpMixin.__init__(self, title='ANALYZE MOVEMENT', size=(400, 400))
        self.animal_cnt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SELECT NUMBER OF ANIMALS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DATA_ANALYSIS.value)
        self.animal_cnt_dropdown = DropDownMenu(self.animal_cnt_frm, '# of animals', list(range(1, self.animal_cnt+1)), labelwidth=20)
        self.animal_cnt_dropdown.setChoices(1)
        self.animal_cnt_confirm_btn = Button(self.animal_cnt_frm, text="Confirm", command=lambda: self.create_settings_frm())
        self.body_part_options = deepcopy(self.body_parts_lst)
        for i in self.multi_animal_id_list:
            self.body_part_options.append(i + ' CENTER OF GRAVITY')
        self.animal_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_confirm_btn.grid(row=0, column=1, sticky=NW)
        self.main_frm.mainloop()

    def create_settings_frm(self):
        if hasattr(self, 'setting_frm'):
            self.setting_frm.destroy()
            self.body_part_frm.destroy()

        self.setting_frm = LabelFrame(self.main_frm, text="SETTINGS", font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.choose_bp_frm(parent=self.setting_frm, bp_options=self.body_part_options)
        self.choose_bp_threshold_frm(parent=self.setting_frm)
        self.setting_frm.grid(row=1, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)

    def run(self):
        self.config.set(ConfigKey.PROCESS_MOVEMENT_SETTINGS.value, ConfigKey.ROI_ANIMAL_CNT.value, str(self.animal_cnt_dropdown.getChoices()))
        check_float(name='Probability threshold', value=self.probability_entry.entry_get, min_value=0.00, max_value=1.00)
        self.config.set(ConfigKey.PROCESS_MOVEMENT_SETTINGS.value, ConfigKey.PROBABILITY_THRESHOLD.value, str(self.probability_entry.entry_get))
        body_parts = []
        for cnt, dropdown in self.body_parts_dropdowns.items():
            body_parts.append(dropdown.getChoices())
        movement_processor = MovementCalculator(config_path=self.config_path,
                                                threshold=float(self.probability_entry.entry_get),
                                                body_parts=body_parts)
        movement_processor.run()
        movement_processor.save()
        pass

#MovementAnalysisPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')






