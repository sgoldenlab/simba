from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import Entry_Box, FileSelect, hxtScrollbar
from simba.ui.user_defined_pose_creator import PoseConfigCreator
from simba.utils.checks import (check_file_exist_and_readable, check_int,
                                check_str)
from simba.utils.errors import DuplicationError
from simba.utils.printing import stdout_success


class CreateUserDefinedPoseConfigurationPopUp(PopUpMixin):
    def __init__(self, master=None, project_config_class=None):
        # self.main_frm = Toplevel()
        # self.main_frm.minsize(700, 400)
        # self.main_frm.wm_title("USER-DEFINED POSE CONFIGURATION")
        # self.main_frm.lift()
        # self.main_frm = Canvas(hxtScrollbar(self.main_frm))
        #

        PopUpMixin.__init__(
            self,
            title="USER-DEFINED POSE CONFIGURATION",
            size=(700, 700),
            main_scrollbar=False,
        )
        self.config_name_entry_box = Entry_Box(self.main_frm, "Pose config name", "23")
        self.animal_cnt_entry_box = Entry_Box(
            self.main_frm, "# of Animals", "23", validation="numeric"
        )
        self.no_body_parts_entry_box = Entry_Box(
            self.main_frm, "# of Body-parts (per animal)", "23", validation="numeric"
        )
        self.img_path_file_select = FileSelect(self.main_frm, "Image path", lblwidth=23)
        self.master, self.project_config_class = master, project_config_class
        self.confirm_btn = Button(
            self.main_frm,
            text="CONFIRM",
            fg="blue",
            command=lambda: self.create_bodypart_table(),
        )
        self.save_btn = Button(
            self.main_frm,
            text="SAVE USER-DEFINED POSE-CONFIG",
            fg="blue",
            command=lambda: self.save_pose_config(),
        )
        self.save_btn.config(state="disabled")

        self.config_name_entry_box.grid(row=0, sticky=W)
        self.animal_cnt_entry_box.grid(row=1, sticky=W)
        self.no_body_parts_entry_box.grid(row=2, sticky=W)
        self.img_path_file_select.grid(row=3, sticky=W, pady=2)
        self.confirm_btn.grid(row=4, pady=5)
        self.save_btn.grid(row=6, pady=5)
        # self.root.lift()
        self.main_frm.mainloop()

    def create_bodypart_table(self):
        if hasattr(self, "table_frame"):
            self.table_frame.destroy()
        check_int(name="ANIMAL NUMBER", value=self.animal_cnt_entry_box.entry_get)
        check_int(name="BODY-PART NUMBER", value=self.no_body_parts_entry_box.entry_get)
        self.selected_animal_cnt, self.selected_bp_cnt = int(
            self.animal_cnt_entry_box.entry_get
        ), int(self.no_body_parts_entry_box.entry_get)
        check_int(name="number of animals", value=self.selected_animal_cnt)
        check_int(name="number of body-parts", value=self.selected_bp_cnt)
        self.bp_name_list = []
        self.bp_animal_list = []

        if self.selected_animal_cnt > 1:
            self.table_frame = LabelFrame(
                self.main_frm,
                text="Bodypart name                            Animal ID number",
                height=500,
                width=700,
            )
        else:
            self.table_frame = LabelFrame(
                self.main_frm, text="Bodypart name            ", height=500, width=700
            )

        scroll_table = hxtScrollbar(self.table_frame)

        for i in range(self.selected_bp_cnt * self.selected_animal_cnt):
            bp_name_entry = Entry_Box(scroll_table, str(i + 1), "2")
            bp_name_entry.grid(row=i, column=0)
            self.bp_name_list.append(bp_name_entry)
            if self.selected_animal_cnt > 1:
                animal_id_entry = Entry_Box(scroll_table, "", "2", validation="numeric")
                animal_id_entry.grid(row=i, column=1)
                self.bp_animal_list.append(animal_id_entry)

        self.table_frame.grid(row=5, sticky=W, column=0)
        self.save_btn.config(state="normal")

    def validate_unique_entries(self, body_part_names: list, animal_ids: list):
        if len(animal_ids) > 0:
            user_entries = []
            for bp_name, animal_id in zip(body_part_names, animal_ids):
                user_entries.append("{}_{}".format(bp_name, animal_id))
        else:
            user_entries = body_part_names
        duplicates = list(set([x for x in user_entries if user_entries.count(x) > 1]))
        if duplicates:
            print(duplicates)
            raise DuplicationError(
                msg="SIMBA ERROR: SimBA found duplicate body-part names (see above). Please enter unique body-part (and animal ID) names.",
                source=self.__class__.__name__,
            )
        else:
            pass

    def save_pose_config(self):
        config_name = self.config_name_entry_box.entry_get
        image_path = self.img_path_file_select.file_path
        check_file_exist_and_readable(image_path)
        check_str(name="POSE CONFIG NAME", value=config_name.strip(), allow_blank=False)
        bp_lst, animal_id_lst = [], []
        for bp_name_entry in self.bp_name_list:
            bp_lst.append(bp_name_entry.entry_get)
        for animal_id_entry in self.bp_animal_list:
            check_int(name="Animal ID number", value=animal_id_entry.entry_get)
            animal_id_lst.append(animal_id_entry.entry_get)

        self.validate_unique_entries(body_part_names=bp_lst, animal_ids=animal_id_lst)

        pose_config_creator = PoseConfigCreator(
            pose_name=config_name,
            no_animals=int(self.selected_animal_cnt),
            img_path=image_path,
            bp_list=bp_lst,
            animal_id_int_list=animal_id_lst,
        )
        pose_config_creator.launch()
        stdout_success(
            msg=f'User-defined pose-configuration "{config_name}" created.',
            source=self.__class__.__name__,
        )
        self.main_frm.winfo_toplevel().destroy()
        self.master.winfo_toplevel().destroy()
        self.project_config_class()
        self.root.destroy()


# CreateUserDefinedPoseConfigurationPopUp()
