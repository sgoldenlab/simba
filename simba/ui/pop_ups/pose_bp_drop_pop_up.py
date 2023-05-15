__author__ = "Simon Nilsson"

from tkinter import *

from simba.ui.tkinter_functions import (FolderSelect,
                                        CreateLabelFrameWithIcon,
                                        DropDownMenu,
                                        Entry_Box)
from simba.utils.enums import Keys, Links, Formats
from simba.pose_processors.remove_keypoints import KeypointRemover
from simba.mixins.pop_up_mixin import PopUpMixin


class DropTrackingDataPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title='Drop body-parts in pose-estimation data', size=(500, 800))
        file_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='FILE SETTINGS', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.data_folder_path = FolderSelect(file_settings_frm, 'Data Folder', lblwidth='20')
        self.file_format = DropDownMenu(file_settings_frm, 'File Type', ['csv', 'h5'], '20')
        self.pose_tool = DropDownMenu(file_settings_frm, 'Tracking tool', ['DLC', 'maDLC'], '20')
        self.pose_tool.setChoices('DLC')
        self.file_format.setChoices('csv')
        self.bp_cnt = Entry_Box(file_settings_frm, '# body-parts to remove', '20', validation='numeric')
        confirm_btn = Button(file_settings_frm, text='Confirm', command=lambda: self.confirm())
        file_settings_frm.grid(row=0,sticky=NW)
        self.data_folder_path.grid(row=0,sticky=W,columnspan=3)
        self.pose_tool.grid(row=1,sticky=NW)
        self.file_format.grid(row=2,sticky=NW)
        self.bp_cnt.grid(row=3,sticky=NW)
        confirm_btn.grid(row=3,column=1,sticky=NW)

    def confirm(self):
        if hasattr(self, 'bp_table'):
            self.bp_table.destroy()
        self.keypoint_remover = KeypointRemover(data_folder=self.data_folder_path.folder_path, pose_tool=self.pose_tool.getChoices(), file_format=self.file_format.getChoices())
        self.bp_table = LabelFrame(self.main_frm, text='REMOVE BODY-PARTS', font=Formats.LABELFRAME_HEADER_FORMAT.value)
        self.bp_table.grid(row=1, sticky=NW, pady=5)
        self.animal_names_lst, self.drop_down_list = [], []
        if self.pose_tool.getChoices() == 'DLC':
            for bp_number in range(int(self.bp_cnt.entry_get)):
                bp_drop_down = DropDownMenu(self.bp_table, 'Body-part {}:'.format(str(bp_number + 1)), self.keypoint_remover.body_part_names, '10')
                bp_drop_down.setChoices(self.keypoint_remover.body_part_names[bp_number])
                self.drop_down_list.append(bp_drop_down)
                bp_drop_down.grid(row=bp_number, column=0, sticky=NW)
        if self.pose_tool.getChoices()  == 'maDLC':
            for bp_number in range(int(self.bp_cnt.entry_get)):
                animal_drop_down = DropDownMenu(self.bp_table, 'Animal name:', self.keypoint_remover.animal_names, '10')
                animal_drop_down.setChoices(self.keypoint_remover.animal_names[0])
                self.animal_names_lst.append(animal_drop_down)
                bp_drop_down = DropDownMenu(self.bp_table, 'Body-part {}:'.format(str(bp_number + 1)), self.keypoint_remover.body_part_names, '10')
                bp_drop_down.setChoices(self.keypoint_remover.body_part_names[bp_number])
                self.drop_down_list.append(bp_drop_down)
                animal_drop_down.grid(row=bp_number, column=0, sticky=NW)
                bp_drop_down.grid(row=bp_number, column=1, sticky=NW)

        run_btn = Button(self.main_frm, text='RUN BODY-PART REMOVAL', command=lambda: self.run())
        run_btn.grid(row=2, column=0, sticky=NW)

    def run(self):
        bp_to_remove_list, animal_names_list = [], []
        for number, drop_down in enumerate(self.drop_down_list):
            bp_to_remove_list.append(drop_down.getChoices())
        if self.pose_tool == 'maDLC':
            for number, drop_down in enumerate(self.animal_names_lst):
                animal_names_list.append(drop_down.getChoices())
        _ = self.keypoint_remover.run(bp_to_remove_list=bp_to_remove_list, animal_names=animal_names_list)
