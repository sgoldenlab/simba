import tkinter.ttk as ttk
from tkinter import NW

from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton)
from simba.utils.checks import check_instance
from simba.utils.enums import Formats, Links


class GetMLSettingsFrame():

    """
    Creates ML name definition LabelFrame panel in ProjectCreatorPopUp interface.


    .. image:: _static/img/GetMLSettingsFrame.png
       :width: 500
       :align: center

    """
    def __init__(self,
                 parent: ttk.Frame,
                 lbl_width: int = 35,
                 bx_width: int = 35):

        self.clf_entry_boxes = []
        check_instance(source=f'{self.__class__.__name__} parent', instance=parent, accepted_types=(ttk.Frame,), raise_error=True)
        self.ml_settings_frm = CreateLabelFrameWithIcon(parent=parent, header="MACHINE LEARNING SETTINGS", icon_name='forest', icon_link=Links.CREATE_PROJECT.value,font=Formats.FONT_HEADER.value, padx=5, pady=5, relief='solid')
        self.lbl_width, self.bx_width = lbl_width, bx_width
        self.add_btn = SimbaButton(parent=self.ml_settings_frm, txt="", txt_clr='red', img='plus_green_3', cmd=self.add_entry_box)
        self.remove_btn = SimbaButton(parent=self.ml_settings_frm, txt="", txt_clr='red', img='minus_red_2', cmd=self.remove_entry_box)
        self.add_entry_box()
        self.fill_frm()

    def add_entry_box(self):
        entry_box = Entry_Box(parent=self.ml_settings_frm, fileDescription=f'CLASSIFIER NAME {len(self.clf_entry_boxes) + 1}: ', labelwidth=self.lbl_width, entry_box_width=self.bx_width)
        self.clf_entry_boxes.append(entry_box)
        self.fill_frm()

    def remove_entry_box(self):
        if len(self.clf_entry_boxes) > 1:
            last_entry = self.clf_entry_boxes.pop(-1)
            last_entry.destroy()
            self.fill_frm()
            self.ml_settings_frm.update_idletasks()

    def get_existing_values(self):
        self.existing_values = []
        for entry in self.clf_entry_boxes:
            try: self.existing_values.append(entry.entry_get.strip())
            except: self.existing_values.append("")

    def fill_frm(self):
        self.get_existing_values()
        for clf_entry in self.clf_entry_boxes:
            clf_entry.grid_forget()
        self.add_btn.grid_forget()
        self.remove_btn.grid_forget()

        for clf_cnt, clf_entry in enumerate(self.clf_entry_boxes):
            row_num = clf_cnt + 2
            clf_entry.entry_set(self.existing_values[clf_cnt])
            clf_entry.grid(row=row_num, column=0, sticky=NW)
            if (clf_cnt == 0) and (len(self.clf_entry_boxes) == 1):
                self.add_btn.grid(row=row_num, column=1, sticky=NW)
            elif clf_cnt + 1 == len(self.clf_entry_boxes):
                self.add_btn.grid(row=row_num, column=1, sticky=NW)
                self.remove_btn.grid(row=row_num, column=2, sticky=NW)
        self.ml_settings_frm.grid(row=1, column=0, sticky=NW, pady=5)