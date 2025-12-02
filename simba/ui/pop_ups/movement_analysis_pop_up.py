import os
from copy import deepcopy
from tkinter import *
from typing import Union

from simba.data_processors.movement_calculator import MovementCalculator
from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaCheckbox, SimBADropDown)
from simba.utils.checks import check_float
from simba.utils.enums import ConfigKey, Keys, Links
from simba.utils.errors import InvalidInputError, NoDataError


class MovementAnalysisPopUp(ConfigReader, PopUpMixin):
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if len(self.outlier_corrected_paths) == 0:
            raise NoDataError(msg=f'No data files found in {self.outlier_corrected_dir} directory, cannot compute movement statistics.', source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="ANALYZE MOVEMENT", size=(500, 500), icon='run')

        self.animal_cnt_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT NUMBER OF ANIMALS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.animal_cnt_dropdown = SimBADropDown(parent=self.animal_cnt_frm, label="# OF ANIMALS", label_width=20, dropdown_width=20, value=1, dropdown_options=list(range(1, self.animal_cnt + 1)), command=self.create_settings_frm)
        self.animal_cnt_frm.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_dropdown.grid(row=0, column=0, sticky=NW)

        self.choose_threshold_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT BODY-PART THRESHOLD", icon_name='threshold', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.probability_entry = Entry_Box(parent=self.choose_threshold_frm, fileDescription='THRESHOLD: ', labelwidth=20, entry_box_width=20, value=0.0, justify='center')
        self.choose_threshold_frm.grid(row=1, column=0, sticky=NW)
        self.probability_entry.grid(row=0, column=0, sticky=NW)

        self.body_part_options = deepcopy(self.body_parts_lst)
        for i in self.multi_animal_id_list: self.body_part_options.append(f"{i} CENTER OF GRAVITY")
        self.create_settings_frm(animal_cnt=1)

        self.measurments_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MEASUREMENTS", icon_name='ruler', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        distance_cb, self.distance_var = SimbaCheckbox(parent=self.measurments_frm, txt='DISTANCE (CM)', txt_img='distance', val=True)
        velocity_cb, self.velocity_var = SimbaCheckbox(parent=self.measurments_frm, txt='VELOCITY (CM/S)', txt_img='run', val=True)
        self.measurments_frm.grid(row=3, column=0, sticky=NW)
        distance_cb.grid(row=0, column=0, sticky=NW)
        velocity_cb.grid(row=1, column=0, sticky=NW)

        self.transpose_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="TRANSPOSE OUTPUT CSV", icon_name='rotate', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        transpose_cb, self.transpose_var = SimbaCheckbox(parent=self.transpose_frm, txt='TRANSPOSE OUTPUT CSV', txt_img='rotate', val=False)
        self.transpose_frm.grid(row=4, column=0, sticky=NW)
        transpose_cb.grid(row=0, column=0, sticky=NW)

        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def create_settings_frm(self, animal_cnt):
        if hasattr(self, "bp_frm"):
            self.bp_frm.destroy()
            for k, v in self.body_parts_dropdowns.items():
                v.destroy()

        self.bp_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT BODY-PARTS", icon_name='pose', icon_link=Links.DATA_ANALYSIS.value, padx=5, pady=5, relief='solid')
        self.body_parts_dropdowns = {}
        for cnt, i in enumerate(range(int(animal_cnt))):
            self.body_parts_dropdowns[cnt] = SimBADropDown(parent=self.bp_frm, label=f"Animal {cnt+1}", label_width=20, dropdown_width=20, value=self.body_part_options[cnt], dropdown_options=self.body_part_options)
            self.body_parts_dropdowns[cnt].grid(row=cnt, column=0, sticky=NW)
        self.bp_frm.grid(row=2, column=0, sticky=NW)

    def run(self):
        self.config.set(ConfigKey.PROCESS_MOVEMENT_SETTINGS.value, ConfigKey.ROI_ANIMAL_CNT.value, str(self.animal_cnt_dropdown.getChoices()))
        check_float(name="Probability threshold", value=self.probability_entry.entry_get, min_value=0.00, max_value=1.00)
        self.config.set(ConfigKey.PROCESS_MOVEMENT_SETTINGS.value, ConfigKey.PROBABILITY_THRESHOLD.value, str(self.probability_entry.entry_get))
        body_parts = []
        threshold = float(self.probability_entry.entry_get)
        for cnt, dropdown in self.body_parts_dropdowns.items():
            body_parts.append(dropdown.getChoices())
        distance, velocity, transpose = self.distance_var.get(), self.velocity_var.get(), self.transpose_var.get()
        if not distance and not velocity:
            raise InvalidInputError(msg='distance AND velocity are both un-checked. To compute movement metrics, check at least one value.', source=self.__class__.__name__)
        movement_processor = MovementCalculator(config_path=self.config_path,
                                                threshold=threshold,
                                                body_parts=body_parts,
                                                distance=distance,
                                                velocity=velocity,
                                                transpose=transpose)
        movement_processor.run()
        movement_processor.save()


# MovementAnalysisPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")
#_ = MovementAnalysisPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
