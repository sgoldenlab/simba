__author__ = "Simon Nilsson"

import threading
from tkinter import *

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.converters import labelme_to_img_dir
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FolderSelect,
                                        SimBADropDown)
from simba.utils.read_write import str_2_bool

IMG_FORMATS = ['png', 'jpeg', 'bmp', 'webp']

class Labelme2ImgsPopUp(PopUpMixin):

    """
    :example:
    >>> Labelme2ImgsPopUp()
    """
    def __init__(self):
        super().__init__(title="LABELME TO IMAGES")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        self.labelme_dir = FolderSelect(settings_frm, folderDescription="LABELME DIRECTORY:", lblwidth=35, width=40)
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35, width=40)
        self.img_format_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=IMG_FORMATS, label="IMAGE FORMAT: ", label_width=35, dropdown_width=40, value=IMG_FORMATS[0])
        self.grey_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GREYSCALE: ", label_width=35, dropdown_width=40, value='FALSE')


        settings_frm.grid(row=0, column=0, sticky=NW)
        self.labelme_dir.grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)
        self.img_format_dropdown.grid(row=2, column=0, sticky=NW)
        self.grey_dropdown.grid(row=3, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        labelme_dir = self.labelme_dir.folder_path
        save_dir = self.save_dir.folder_path
        img_format = str(self.img_format_dropdown.get_value())
        grey = str_2_bool(self.grey_dropdown.get_value())
        thread = threading.Thread(target=labelme_to_img_dir, kwargs={'labelme_dir': labelme_dir, 'img_dir': save_dir, 'img_format': img_format, 'greyscale': grey})
        thread.start()
