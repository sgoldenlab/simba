import os
from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.blob_tracker_ui import BlobTrackingUI
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, FolderSelect
from simba.utils.checks import check_if_dir_exists
from simba.utils.enums import Links
from simba.utils.errors import InvalidInputError
from simba.utils.read_write import (find_all_videos_in_directory,
                                    find_files_of_filetypes_in_directory)


class InitializeBlobTrackerPopUp(PopUpMixin):
    """
    :example:
    >>> InitializeBlobTrackerPopUp()
    """
    def __init__(self):
        super().__init__(title="BLOB TRACKER: SELECT INPUT AND OUTPUT DIRECTORIES", size=(500, 500), icon='bubble_green')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="DATA DIRECTORY SETTINGS", icon_name='settings')
        self.input_dir_select = FolderSelect(parent=settings_frm, folderDescription= 'INPUT VIDEO DIRECTORY:', lblwidth=30, entry_width=20)
        self.save_dir_select = FolderSelect(parent=settings_frm, folderDescription='SAVE DATA DIRECTORY:', lblwidth=30, entry_width=20)
        self.create_run_frm(run_function=self.run)
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.input_dir_select.grid(row=0, column=0, sticky=NW)
        self.save_dir_select.grid(row=1, column=0, sticky=NW)
        self.main_frm.mainloop()

    def run(self):
        in_dir = self.input_dir_select.folder_path
        out_dir = self.save_dir_select.folder_path
        check_if_dir_exists(in_dir=in_dir)
        check_if_dir_exists(in_dir=out_dir)
        _ = find_all_videos_in_directory(directory=in_dir, raise_error=True)
        if os.path.isdir(out_dir):
            existing_files = find_files_of_filetypes_in_directory(directory=out_dir, extensions=['.json', '.csv', '.mp4', '.pickle'], raise_error=False, raise_warning=False)
            if len(existing_files) > 0:
                raise InvalidInputError(msg=f'The selected output directory {out_dir} is not empty. Select an empty output directory where to save your tracking data', source=self.__class__.__name__)
        self.root.destroy()
        BlobTrackingUI(input_dir=in_dir, output_dir=out_dir)