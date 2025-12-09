import os.path
from datetime import datetime

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.transform.utils import \
    merge_coco_keypoints_files
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        FolderSelect, SimbaButton)
from simba.utils.checks import check_if_dir_exists, check_int
from simba.utils.enums import Keys
from simba.utils.read_write import find_files_of_filetypes_in_directory


class MergeCOCOKeypointFilesPopUp(PopUpMixin):

    def __init__(self):

        PopUpMixin.__init__(self, title="MERGE COCO KEYPOINT FILES", icon='coco_small')
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value)
        self.json_folder = FolderSelect(parent=self.settings_frm, folderDescription='COCO JSON DIRECTORY:', lblwidth=30, entry_width=30, tooltip_txt='FOLDER CONTAINING JSONS COCO KEYPOINT FILES', lbl_icon='json')
        self.save_folder = FolderSelect(parent=self.settings_frm, folderDescription='SAVE_DIRECTORY:', lblwidth=30, entry_width=30, tooltip_txt='FOLDER WHERE TO SAVE THE MERGED JSON FILE', lbl_icon='folder')

        self.max_x_entry = Entry_Box(parent=self.settings_frm, fileDescription='MAX KEYPOINT X LOCATION:', labelwidth=30, entry_box_width=30, value='None', justify='center', img='x')
        self.max_y_entry = Entry_Box(parent=self.settings_frm, fileDescription='MAX KEYPOINT Y LOCATION:', labelwidth=30, entry_box_width=30, value='None', justify='center', img='y')

        self.settings_frm.grid(row=0, column=0, sticky='NW')
        self.json_folder.grid(row=0, column=0, sticky='NW')
        self.save_folder.grid(row=1, column=0, sticky='NW')
        self.max_x_entry.grid(row=2, column=0, sticky='NW')
        self.max_y_entry.grid(row=3, column=0, sticky='NW')

        run_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="", icon_name='run')
        run_btn = SimbaButton(parent=run_frm, txt="RUN", img='rocket', txt_clr='red', cmd=self.run)

        run_frm.grid(row=1, column=0, sticky='NW')
        run_btn.grid(row=0, column=0, sticky='NW')

        self.main_frm.mainloop()


    def run(self):
        input_dir, save_dir = self.json_folder.folder_path, self.save_folder.folder_path
        _ = find_files_of_filetypes_in_directory(directory=input_dir, extensions='.json', raise_error=True, raise_warning=False, as_dict=False)
        check_if_dir_exists(in_dir=save_dir)
        max_x, max_y = self.max_x_entry.entry_get, self.max_y_entry.entry_get
        use_max_x = check_int(name=f'{self.__class__.__name__} MAX KEYPOINT X LOCATION', value=max_x, min_value=1, raise_error=False)
        use_max_y = check_int(name=f'{self.__class__.__name__} MAX KEYPOINT Y LOCATION', value=max_y, min_value=1, raise_error=False)
        max_x, max_y = None if not use_max_x else max_x, None if not use_max_y else max_y
        save_path = os.path.join(save_dir, f'merged_coco_keypoints_{datetime.now().strftime("%Y%m%d%H%M%S")}.json')

        merge_coco_keypoints_files(data_dir=input_dir, save_path=save_path, max_width=max_x, max_height=max_y)



#MergeCOCOKeypointFilesPopUp()