
try:
    from typing import Literal
except:
    from typing_extensions import Literal
from typing import Union, Optional
import os
from tkinter import *
from datetime import datetime
from PIL import Image
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, DropDownMenu, FolderSelect
import cv2

from simba.utils.checks import check_if_dir_exists, check_str, check_int
from simba.utils.enums import Options, Keys, Links
from simba.utils.read_write import find_files_of_filetypes_in_directory, get_fn_ext, str_2_bool
from simba.utils.printing import SimbaTimer, stdout_success
from simba.video_processors.video_processing import convert_to_tiff


class Convert2TIFFPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="CONVERT IMAGE DIRECTORY TO TIFF")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_frame_dir = FolderSelect(settings_frm, "IMAGE DIRECTORY PATH:", title="Select a image directory", lblwidth=25)
        self.compression_dropdown = DropDownMenu(settings_frm, "COMPRESSION:", ['raw', 'tiff_deflate', 'tiff_lzw'], labelwidth=25)
        self.compression_dropdown.setChoices('raw')
        self.stack_dropdown = DropDownMenu(settings_frm, "STACK:", ['FALSE', 'TRUE'], labelwidth=25)
        self.stack_dropdown.setChoices('FALSE')
        self.create_run_frm(run_function=self.run, title='RUN TIFF CONVERSION')

        settings_frm.grid(row=0, column=0, sticky="NW")
        self.selected_frame_dir.grid(row=0, column=0, sticky="NW")
        self.compression_dropdown.grid(row=1, column=0, sticky="NW")
        self.stack_dropdown.grid(row=2, column=0, sticky="NW")
        self.main_frm.mainloop()

    def run(self):
        folder_path = self.selected_frame_dir.folder_path
        check_if_dir_exists(in_dir=folder_path)
        stack = str_2_bool(self.stack_dropdown.getChoices())
        convert_to_tiff(directory=folder_path, compression=self.compression_dropdown.getChoices(), verbose=True, stack=stack)

Convert2TIFFPopUp()

# convert_to_webp('/Users/simon/Desktop/imgs', quality=80)

