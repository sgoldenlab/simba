from tkinter import *

import numpy as np

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.third_party_label_appenders.transform.labelme_to_yolo import \
    LabelmeBoundingBoxes2YoloBoundingBoxes
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FolderSelect,
                                        SimBADropDown)
from simba.utils.checks import check_if_dir_exists
from simba.utils.read_write import (find_files_of_filetypes_in_directory,
                                    str_2_bool)

TRAIN_SIZE_OPTIONS = list(np.arange(10, 110, 10))

class LabelmeBbox2YoloBboxPopUp(PopUpMixin):

    """
    >>> LabelmeBbox2YoloBboxPopUp()
    """

    def __init__(self):

        PopUpMixin.__init__(self, title="LABELME BOUNDING BOXES -> YOLO BOUNDING BOXES ", icon='dlc_1')

        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')
        self.labelme_dir = FolderSelect(settings_frm, folderDescription="LABELME DIRECTORY:", lblwidth=35)
        self.save_dir = FolderSelect(settings_frm, folderDescription="SAVE DIRECTORY:", lblwidth=35)
        self.verbose_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="VERBOSE:", label_width=35, dropdown_width=40, value='TRUE')
        self.obb_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="OBB:", label_width=35, dropdown_width=40, value='TRUE')
        self.clahe_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="CLAHE:", label_width=35, dropdown_width=40, value='FALSE')
        self.greyscale_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GREYSCALE:", label_width=35, dropdown_width=40, value='FALSE')
        self.train_size_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=TRAIN_SIZE_OPTIONS, label="TRAIN SIZE (%): ", label_width=35, dropdown_width=40, value=70)


        settings_frm.grid(row=0, column=0, sticky=NW)
        self.labelme_dir .grid(row=0, column=0, sticky=NW)
        self.save_dir.grid(row=1, column=0, sticky=NW)
        self.obb_dropdown.grid(row=2, column=0, sticky=NW)
        self.clahe_dropdown.grid(row=3, column=0, sticky=NW)
        self.greyscale_dropdown.grid(row=4, column=0, sticky=NW)
        self.train_size_dropdown.grid(row=5, column=0, sticky=NW)
        self.verbose_dropdown.grid(row=6, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        save_dir = self.save_dir.folder_path
        labelme_dir = self.labelme_dir.folder_path
        verbose = str_2_bool(self.verbose_dropdown.get_value())
        clahe = str_2_bool(self.clahe_dropdown.get_value())
        obb = str_2_bool(self.obb_dropdown.get_value())
        grey = str_2_bool(self.greyscale_dropdown.get_value())
        train_size = float(self.train_size_dropdown.get_value()) / 100

        check_if_dir_exists(in_dir=save_dir, source=f'{self.__class__.__name__} LABELME DIRECTORY')
        check_if_dir_exists(in_dir=labelme_dir, source=f'{self.__class__.__name__} LABELME SAVE DIR')

        _ = find_files_of_filetypes_in_directory(directory=labelme_dir, extensions=['.json'], raise_error=True)

        runner = LabelmeBoundingBoxes2YoloBoundingBoxes(labelme_dir=labelme_dir, save_dir=save_dir, obb=obb, verbose=verbose, clahe=clahe, train_size=train_size, greyscale=grey)
        runner.run()



#LabelmeBbox2YoloBboxPopUp()