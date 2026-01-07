__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton)
from simba.utils.checks import check_str
from simba.utils.enums import Formats, Keys, Links
from simba.utils.read_write import archive_processed_files


class ArchiveProcessedFilesPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="ARCHIVE PROCESSED FILES", icon='archive')
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        archive_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header='ARCHIVE PROCESSED FILES', icon_name=Keys.DOCUMENTATION.value, icon_link=Links.SCENARIO_4.value)
        self.archive_eb = Entry_Box(archive_frm, fileDescription="ARCHIVE DIRECTORY NAME:", labelwidth=25, img='id_card')
        archive_btn = SimbaButton(parent=archive_frm, txt='RUN ARCHIVE', img='rocket', cmd=self.run)
        archive_frm.grid(row=0, column=0, sticky=NW)
        self.archive_eb.grid(row=0, column=0, sticky=NW)
        archive_btn.grid(row=1, column=0, sticky=NW)

    def run(self):
        archive_name = self.archive_eb.entry_get.strip()
        check_str(name="ARCHIVE NAME", value=archive_name, raise_error=True)
        archive_processed_files(config_path=self.config_path, archive_name=archive_name)




#ArchiveProcessedFilesPopUp(config_path=r"C:\troubleshooting\ethan_alyssa\project_folder\project_config.ini")
