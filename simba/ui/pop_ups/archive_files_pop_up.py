__author__ = "Simon Nilsson"

from tkinter import *

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import Entry_Box
from simba.utils.checks import check_str
from simba.utils.read_write import archive_processed_files


class ArchiveProcessedFilesPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: str):
        PopUpMixin.__init__(self, title="ADD CLASSIFIER")
        ConfigReader.__init__(self, config_path=config_path)
        self.archive_eb = Entry_Box(self.main_frm, "ARCHIVE DIRECTORY NAME", "25")
        archive_btn = Button(
            self.main_frm, text="RUN ARCHIVE", fg="blue", command=lambda: self.run()
        )
        self.archive_eb.grid(row=0, column=0, sticky=NW)
        archive_btn.grid(row=1, column=0, sticky=NW)

    def run(self):
        archive_name = self.archive_eb.entry_get.strip()
        check_str(name="CLASSIFIER NAME", value=archive_name)
        archive_processed_files(config_path=self.config_path, archive_name=archive_name)
