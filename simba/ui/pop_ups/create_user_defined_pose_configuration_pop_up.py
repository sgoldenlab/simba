from tkinter import *
from typing import Optional

from simba.mixins.pop_up_mixin import PopUpMixin
#from simba.ui.create_project_ui.ProjectCreatorPopUp
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        FileSelect, SimbaButton, SimBALabel)
from simba.ui.user_defined_pose_creator import PoseConfigCreator
from simba.utils.checks import (check_file_exist_and_readable, check_int,
                                check_str, check_valid_img_path)
from simba.utils.enums import Formats, Options
from simba.utils.errors import DuplicationError, InvalidInputError
from simba.utils.printing import stdout_success


class CreateUserDefinedPoseConfigurationPopUp(PopUpMixin):
    def __init__(self,
                 master: Optional[Toplevel] = None,
                 project_config_class = None): # simba.ui.create_project_ui.ProjectCreatorPopUp

        PopUpMixin.__init__(self, title="USER-DEFINED POSE CONFIGURATION", size=(700, 700), main_scrollbar=True, icon='pose')
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='SETTINGS', icon_name='settings', padx=5, pady=5, relief='solid')
        self.config_name_entry_box = Entry_Box(parent=self.settings_frm, fileDescription="POSE CONFIG NAME: ", labelwidth=35, entry_box_width=40)
        self.animal_cnt_entry_box = Entry_Box(parent=self.settings_frm, fileDescription="# ANIMALS: ", labelwidth=35, entry_box_width=40, validation='numeric')
        self.no_body_parts_entry_box = Entry_Box(parent=self.settings_frm, fileDescription="# OF BODY-PARTS (PER ANIMAL): ", labelwidth=35, entry_box_width=40, validation='numeric')
        self.img_path_file_select = FileSelect(self.settings_frm, fileDescription="IMAGE PATH: ", lblwidth=35, file_types=[("IMAGE FILE", Options.ALL_IMAGE_FORMAT_OPTIONS.value)], entry_width=40)
        self.confirm_btn = SimbaButton(parent=self.settings_frm, txt=f"CONFIRM", img='tick', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.create_bodypart_table)
        self.save_btn = SimbaButton(parent=self.settings_frm, txt="SAVE USER-DEFINED POSE-CONFIG", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.save_pose_config)
        self.save_btn.config(state="disabled")
        self.master, self.project_config_class = master, project_config_class

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.config_name_entry_box.grid(row=0, column=0, sticky=NW)
        self.animal_cnt_entry_box.grid(row=1, column=0, sticky=NW)
        self.no_body_parts_entry_box.grid(row=2, column=0, sticky=NW)
        self.img_path_file_select.grid(row=3, column=0, sticky=NW, pady=2)
        self.confirm_btn.grid(row=4, pady=5, column=0, sticky=NW)
        self.save_btn.grid(row=6, pady=5, column=0, sticky=NW)
        # self.root.lift()
        self.main_frm.mainloop()

    def create_bodypart_table(self):
        if hasattr(self, "bp_table_frm"):
            self.bp_table_frm.destroy()

        self.bp_table_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='BODY-PARTS', icon_name='pose', padx=5, pady=5, relief='solid')
        check_int(name="ANIMAL NUMBER", value=self.animal_cnt_entry_box.entry_get, min_value=1)
        check_int(name="BODY-PART NUMBER", value=self.no_body_parts_entry_box.entry_get, min_value=1)
        check_str(name="POSE CONFIG NAME", value=self.config_name_entry_box.entry_get.strip(), allow_blank=False, invalid_substrs=(',',))

        self.selected_animal_cnt, self.selected_bp_cnt = int(self.animal_cnt_entry_box.entry_get), int(self.no_body_parts_entry_box.entry_get)
        self.bp_name_ebs, self.bp_animal_ebs = [], []

        header_col_1 = SimBALabel(parent=self.bp_table_frm, txt='BODY-PART NAME', font=Formats.FONT_HEADER.value, justify='center', width=24)
        header_col_1.grid(row=0, column=0, sticky=NW)
        if self.selected_animal_cnt > 1:
            header_col_2 = SimBALabel(parent=self.bp_table_frm, txt='ANIMAL ID NUMBER', font=Formats.FONT_HEADER.value, justify='center', width=24)
            header_col_2.grid(row=0, column=1, sticky=NW)

        for i in range(self.selected_bp_cnt * self.selected_animal_cnt):
            bp_name_entry = Entry_Box(parent=self.bp_table_frm, fileDescription=str(i + 1), labelwidth=4, entry_box_width=20, justify='center')
            bp_name_entry.grid(row=i+1, column=0, sticky=NW)
            self.bp_name_ebs.append(bp_name_entry)
            if self.selected_animal_cnt > 1:
                animal_id_entry = Entry_Box(parent=self.bp_table_frm, fileDescription="", labelwidth=4, entry_box_width=20, justify='center', validation='numeric')
                animal_id_entry.grid(row=i+1, column=1, sticky=NW)
                self.bp_animal_ebs.append(animal_id_entry)
        self.bp_table_frm.grid(row=5, sticky=W, column=0)
        self.save_btn.config(state="normal")

    def validate_unique_entries(self, body_part_names: list, animal_ids: list):
        for cnt, bp_name in enumerate(body_part_names):
            check_str(name=f'BODY-PART NAME row {cnt+1}', value=bp_name, allow_blank=False)
            if "," in bp_name: raise InvalidInputError(msg=f'Commas are not allowed in body-part names. A comma was found in body-part name: {bp_name}.', source=self.__class__.__name__)

        for cnt, animal_id in enumerate(animal_ids):
            check_int(name=f'ANIMAL ID NUMBER row {cnt+1}', value=animal_id, min_value=1, max_value=int(self.animal_cnt_entry_box.entry_get), raise_error=True)
        if len(animal_ids) > 0:
            user_entries = []
            for bp_name, animal_id in zip(body_part_names, animal_ids):
                user_entries.append(f"{bp_name}_{animal_id}")
        else:
            user_entries = body_part_names
        duplicates = list(set([x for x in user_entries if user_entries.count(x) > 1]))
        if duplicates:
            print(duplicates)
            raise DuplicationError(msg="SIMBA ERROR: SimBA found duplicate body-part names (see above). Please enter unique body-part names(or body-part/animal ID names , if tracking multiple animals).", source=self.__class__.__name__)
        else:
            pass

    def save_pose_config(self):
        config_name = self.config_name_entry_box.entry_get
        image_path = self.img_path_file_select.file_path
        check_valid_img_path(path=image_path)
        check_str(name="POSE CONFIG NAME", value=config_name.strip(), allow_blank=False, invalid_substrs=(',',))
        bp_lst, animal_id_lst = [], []
        for bp_name_entry in self.bp_name_ebs: bp_lst.append(bp_name_entry.entry_get.strip())
        for animal_id_entry in self.bp_animal_ebs: animal_id_lst.append(animal_id_entry.entry_get.strip())
        self.validate_unique_entries(body_part_names=bp_lst, animal_ids=animal_id_lst)
        animal_id_lst = [int(x) for x in animal_id_lst]
        pose_config_creator = PoseConfigCreator(pose_name=config_name, animal_cnt=int(self.selected_animal_cnt), img_path=image_path, bp_list=bp_lst, animal_id_int_list=animal_id_lst)
        pose_config_creator.launch()
        stdout_success(msg=f'User-defined pose-configuration "{config_name}" created.', source=self.__class__.__name__)
        self.main_frm.winfo_toplevel().destroy()
        self.master.winfo_toplevel().destroy()
        self.project_config_class()
        self.root.destroy()


#CreateUserDefinedPoseConfigurationPopUp()
