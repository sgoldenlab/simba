from tkinter import *

import numpy as np

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.blob_visualizer import BlobVisualizer
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FolderSelect,
                                        SimBADropDown)
from simba.utils.checks import check_if_dir_exists
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import find_core_cnt

OPACITY_OPTIONS = list(np.round(np.arange(0.1, 1.1, 0.1), 2))
CIRCLE_SIZES = list(np.round(np.arange(1, 21, 1), 2))
CIRCLE_SIZES.insert(0, 'AUTO')

class BlobVisualizerPopUp(PopUpMixin):
    def __init__(self):
        self.color_dict = get_color_dict()
        clr_names = list(self.color_dict.keys())
        clr_names.insert(0, 'NONE')
        PopUpMixin.__init__(self, title="VISUALIZE BLOB TRACKING", icon='bubble_pink')
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings', icon_link=None)
        self.core_cnt_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=list(range(1, find_core_cnt()[0] + 1)), label="CPU CORE COUNT:", label_width=30, dropdown_width=20, value=find_core_cnt()[0])
        self.shape_opacity_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=OPACITY_OPTIONS, label="SHAPE OPACITY:", label_width=30, dropdown_width=20, value=0.7)
        self.bg_opacity_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=OPACITY_OPTIONS, label="BACKGROUND OPACITY:", label_width=30, dropdown_width=20, value=1.0)
        self.keypoint_sizes_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=CIRCLE_SIZES, label="KEY-POINT SIZES:", label_width=30, dropdown_width=20, value=CIRCLE_SIZES[0])
        self.hull_clr_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=clr_names, label="HULL COLOR:", label_width=30, dropdown_width=20, value='Pink')
        self.anterior_clr_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=clr_names, label="ANTERIOR COLOR:", label_width=30, dropdown_width=20, value='Green')
        self.posterior_clr_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=clr_names, label="POSTERIOR COLOR:", label_width=30, dropdown_width=20, value='Orange')
        self.center_clr_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=clr_names, label="CENTER COLOR:", label_width=30, dropdown_width=20, value='Cyan')
        self.left_clr_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=clr_names, label="LEFT COLOR:", label_width=30, dropdown_width=20, value='NONE')
        self.right_clr_dropdown = SimBADropDown(parent=settings_frm, dropdown_options=clr_names, label="RIGHT COLOR:", label_width=30, dropdown_width=20, value='NONE')


        settings_frm.grid(row=0, column=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        self.shape_opacity_dropdown.grid(row=1, column=0, sticky=NW)
        self.bg_opacity_dropdown.grid(row=2, column=0, sticky=NW)
        self.keypoint_sizes_dropdown.grid(row=3, column=0, sticky=NW)
        self.hull_clr_dropdown.grid(row=4, column=0, sticky=NW)
        self.anterior_clr_dropdown.grid(row=5, column=0, sticky=NW)
        self.posterior_clr_dropdown.grid(row=6, column=0, sticky=NW)
        self.center_clr_dropdown.grid(row=7, column=0, sticky=NW)
        self.left_clr_dropdown.grid(row=8, column=0, sticky=NW)
        self.right_clr_dropdown.grid(row=9, column=0, sticky=NW)

        dir_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="DIRECTORIES", icon_name='browse', icon_link=None)
        self.videos_dir = FolderSelect(parent=dir_frm, folderDescription='VIDEO DIRECTORY:', lblwidth=30, entry_width=20)
        self.data_dir = FolderSelect(parent=dir_frm, folderDescription='DATA DIRECTORY:', lblwidth=30, entry_width=20)
        self.save_dir = FolderSelect(parent=dir_frm, folderDescription='SAVE DIRECTORY:', lblwidth=30, entry_width=20)

        dir_frm.grid(row=1, column=0, sticky=NW)
        self.videos_dir.grid(row=0, column=0, sticky=NW)
        self.data_dir.grid(row=1, column=0, sticky=NW)
        self.save_dir.grid(row=2, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        core_cnt = int(self.core_cnt_dropdown.get_value())
        shape_opacity = float(self.shape_opacity_dropdown.get_value())
        bg_opacity = float(self.bg_opacity_dropdown.get_value())
        videos_dir = self.videos_dir.folder_path
        data_dir = self.data_dir.folder_path
        save_dir = self.save_dir.folder_path

        circle_size = None if self.keypoint_sizes_dropdown.get_value() == 'AUTO' else int(self.keypoint_sizes_dropdown.get_value())
        hull_clr = None if self.hull_clr_dropdown.get_value() == 'NONE' else self.colors_dict[self.hull_clr_dropdown.get_value()]
        anterior_clr = None if self.anterior_clr_dropdown.get_value() == 'NONE' else self.colors_dict[self.anterior_clr_dropdown.get_value()]
        posterior_clr = None if self.posterior_clr_dropdown.get_value() == 'NONE' else self.colors_dict[self.posterior_clr_dropdown.get_value()]
        center_clr = None if self.center_clr_dropdown.get_value() == 'NONE' else self.colors_dict[self.center_clr_dropdown.get_value()]
        left_clr = None if self.left_clr_dropdown.get_value() == 'NONE' else self.colors_dict[self.left_clr_dropdown.get_value()]
        right_clr = None if self.right_clr_dropdown.get_value() == 'NONE' else self.colors_dict[self.right_clr_dropdown.get_value()]

        check_if_dir_exists(in_dir=videos_dir, source=self.__class__.__name__)
        check_if_dir_exists(in_dir=data_dir, source=self.__class__.__name__)
        check_if_dir_exists(in_dir=save_dir, source=self.__class__.__name__)

        visualizer = BlobVisualizer(data_path=data_dir,
                                    video_path=videos_dir,
                                    save_dir=save_dir,
                                    core_cnt=core_cnt,
                                    shape_opacity=shape_opacity,
                                    bg_opacity=bg_opacity,
                                    circle_size=circle_size,
                                    hull=hull_clr,
                                    anterior=anterior_clr,
                                    posterior=posterior_clr,
                                    center=center_clr,
                                    left=left_clr,
                                    right=right_clr)

        visualizer.run()

#BlobVisualizerPopUp()