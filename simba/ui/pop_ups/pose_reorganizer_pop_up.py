__author__ = "Simon Nilsson"

from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.pose_processors.reorganize_keypoint import KeypointReorganizer
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        FolderSelect, hxtScrollbar)
from simba.utils.enums import Formats, Keys, Links


class PoseReorganizerPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="RE-ORGANIZE POSE_ESTIMATION DATA", size=(500, 800))
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.data_folder = FolderSelect(settings_frm, "DATA FOLDER: ", lblwidth="10")
        self.pose_tool_dropdown = DropDownMenu(settings_frm, "Tracking tool", ["DLC", "maDLC"], "10")
        self.pose_tool_dropdown.setChoices("DLC")
        self.file_format = DropDownMenu(settings_frm, "FILE TYPE: ", ["csv", "h5"], "10")
        self.file_format.setChoices("csv")
        confirm_btn = Button(settings_frm, text="Confirm", font=Formats.FONT_REGULAR.value, command=lambda: self.confirm())
        settings_frm.grid(row=0, sticky=NW)
        self.data_folder.grid(row=0, sticky=NW, columnspan=3)
        self.pose_tool_dropdown.grid(row=1, sticky=NW)
        self.file_format.grid(row=2, sticky=NW)
        confirm_btn.grid(row=2, column=1, sticky=NW)

    def confirm(self):
        if hasattr(self, "table"):
            self.table.destroy()

        self.keypoint_reorganizer = KeypointReorganizer(
            data_folder=self.data_folder.folder_path,
            pose_tool=self.pose_tool_dropdown.getChoices(),
            file_format=self.file_format.getChoices(),
        )
        self.table = LabelFrame(
            self.main_frm,
            text="SET NEW ORDER",
            font=Formats.LABELFRAME_HEADER_FORMAT.value,
            pady=5,
            padx=5,
        )
        self.current_order = LabelFrame(self.table, text="CURRENT ORDER:")
        self.new_order = LabelFrame(self.table, text="NEW ORDER:")

        self.table.grid(row=1, sticky=W, pady=10)
        self.current_order.grid(row=0, column=0, sticky=NW, pady=5)
        self.new_order.grid(row=0, column=1, sticky=NW, padx=5, pady=5)
        idx1, idx2, oldanimallist, oldbplist, self.newanimallist, self.newbplist = (
            [0] * len(self.keypoint_reorganizer.bp_list) for i in range(6)
        )

        if self.keypoint_reorganizer.animal_list:
            animal_list_reduced = list(set(self.keypoint_reorganizer.animal_list))
            self.pose_tool = "maDLC"
            for i in range(len(self.keypoint_reorganizer.bp_list)):
                idx1[i] = Label(self.current_order, text=str(i + 1) + ".")
                oldanimallist[i] = Label(
                    self.current_order,
                    text=str(self.keypoint_reorganizer.animal_list[i]),
                )
                oldbplist[i] = Label(
                    self.current_order, text=str(self.keypoint_reorganizer.bp_list[i])
                )
                idx1[i].grid(row=i, column=0, sticky=W)
                oldanimallist[i].grid(row=i, column=1, sticky=W, ipady=5)
                oldbplist[i].grid(row=i, column=2, sticky=W, ipady=5)
                idx2[i] = Label(self.new_order, text=str(i + 1) + ".")
                self.newanimallist[i] = DropDownMenu(
                    self.new_order, " ", animal_list_reduced, "10"
                )
                self.newbplist[i] = DropDownMenu(
                    self.new_order, " ", self.keypoint_reorganizer.bp_list, "10"
                )
                self.newanimallist[i].setChoices(
                    self.keypoint_reorganizer.animal_list[i]
                )
                self.newbplist[i].setChoices(self.keypoint_reorganizer.bp_list[i])
                idx2[i].grid(row=i, column=0, sticky=W)
                self.newanimallist[i].grid(row=i, column=1, sticky=W)
                self.newbplist[i].grid(row=i, column=2, sticky=W)

        else:
            self.pose_tool = "DLC"
            for i in range(len(self.keypoint_reorganizer.bp_list)):
                idx1[i] = Label(self.current_order, text=str(i + 1) + ".")
                oldbplist[i] = Label(
                    self.current_order, text=str(self.keypoint_reorganizer.bp_list[i])
                )
                idx1[i].grid(row=i, column=0, sticky=W, ipady=5)
                oldbplist[i].grid(row=i, column=2, sticky=W, ipady=5)
                idx2[i] = Label(self.new_order, text=str(i + 1) + ".")
                self.newbplist[i] = StringVar()
                oldanimallist[i] = OptionMenu(
                    self.new_order,
                    self.newbplist[i],
                    *self.keypoint_reorganizer.bp_list
                )
                self.newbplist[i].set(self.keypoint_reorganizer.bp_list[i])
                idx2[i].grid(row=i, column=0, sticky=W)
                oldanimallist[i].grid(row=i, column=1, sticky=W)

        button_run = Button(
            self.table,
            text="Run re-organization",
            command=lambda: self.run_reorganization(),
        )
        button_run.grid(row=2, column=1, sticky=W)

    def run_reorganization(self):
        if self.pose_tool == "DLC":
            new_bp_list = []
            for curr_choice in self.newbplist:
                new_bp_list.append(curr_choice.get())
            self.keypoint_reorganizer.run(animal_list=None, bp_lst=new_bp_list)

        if self.pose_tool == "maDLC":
            new_bp_list, new_animal_list = [], []
            for curr_animal, curr_bp in zip(self.newanimallist, self.newbplist):
                new_bp_list.append(curr_bp.getChoices())
                new_animal_list.append(curr_animal.getChoices())
            self.keypoint_reorganizer.run(
                animal_list=new_animal_list, bp_lst=new_bp_list
            )
