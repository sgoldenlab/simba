__author__ = "Simon Nilsson"

import os
from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, FolderSelect
from simba.utils.enums import Keys, Links
from simba.utils.errors import DuplicationError, NotDirectoryError
from simba.video_processors.batch_process_menus import BatchProcessFrame


class BatchPreProcessPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="BATCH PROCESS VIDEO", size=(600, 400))
        selections_frm = CreateLabelFrameWithIcon(
            parent=self.main_frm,
            header="SELECTIONS",
            icon_name=Keys.DOCUMENTATION.value,
            icon_link=Links.BATCH_PREPROCESS.value,
        )
        self.input_folder_select = FolderSelect(
            selections_frm,
            "INPUT VIDEO DIRECTORY:",
            title="Select Folder with Input Videos",
            lblwidth=20,
        )
        self.output_folder_select = FolderSelect(
            selections_frm,
            "OUTPUT VIDEO DIRECTORY:",
            title="Select Folder for Output videos",
            lblwidth=20,
        )
        confirm_btn = Button(
            selections_frm, text="CONFIRM", fg="blue", command=lambda: self.run()
        )
        selections_frm.grid(row=0, column=0, sticky=NW)
        self.input_folder_select.grid(row=0, column=0, sticky=NW)
        self.output_folder_select.grid(row=1, column=0, sticky=NW)
        confirm_btn.grid(row=2, column=0, sticky=NW)

    def run(self):
        if not os.path.isdir(self.input_folder_select.folder_path):
            raise NotDirectoryError(
                msg=f"INPUT folder dir ({self.input_folder_select.folder_path}) is not a valid directory.",
                source=self.__class__.__name__,
            )
        if not os.path.isdir(self.output_folder_select.folder_path):
            raise NotDirectoryError(
                msg=f"OUTPUT folder dir ({self.output_folder_select.folder_path}) is not a valid directory.",
                source=self.__class__.__name__,
            )
        if (
            self.output_folder_select.folder_path
            == self.input_folder_select.folder_path
        ):
            raise DuplicationError(
                msg="The INPUT directory and OUTPUT directory CANNOT be the same folder",
                source=self.__class__.__name__,
            )
        else:
            batch_preprocessor = BatchProcessFrame(
                input_dir=self.input_folder_select.folder_path,
                output_dir=self.output_folder_select.folder_path,
            )
            batch_preprocessor.create_main_window()
            batch_preprocessor.create_video_table_headings()
            batch_preprocessor.create_video_rows()
            batch_preprocessor.create_execute_btn()
            batch_preprocessor.main_frm.mainloop()
