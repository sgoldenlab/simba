import os
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.roi_ui import ROI_ui
from simba.roi_tools.roi_utils import multiply_ROIs, reset_video_ROIs
from simba.ui.pop_ups.min_max_draw_size_popup import SetMinMaxDrawWindowSize
from simba.ui.pop_ups.roi_size_standardizer_popup import \
    ROISizeStandardizerPopUp
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton, SimBALabel)
from simba.utils.enums import Keys, Links
from simba.utils.errors import NoFilesFoundError
from simba.utils.read_write import find_all_videos_in_directory, get_fn_ext


class ROIVideoTable(ConfigReader, PopUpMixin):

    """
    Crates a tkinter video table listing all videos in project together with buttons associated with ROI drawing, deletaing, and duplicating shapes.

    :example:
    >>> ROIVideoTable(config_path=r"C:\troubleshooting\mouse_open_field\project_folder\project_config.ini")
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        PopUpMixin.__init__(self, title="PROJECT VIDEOS: ROI TABLE", size=(720, 960), icon='data_table')
        self.video_dict = find_all_videos_in_directory(directory=self.video_dir, as_dict=True, raise_error=False)
        if len(list(self.video_dict.keys())) == 0:
            raise NoFilesFoundError(msg=f'Cannot draw ROIs: the {self.video_dir} directory does not contain any video files.', source=self.__class__.__name__)
        self.get_file_menu()
        self.run()

    def get_file_menu(self):
        menu = Menu(self.root)
        file_menu = Menu(menu)
        menu.add_cascade(label="File...", menu=file_menu)
        file_menu.add_command(label="Standardize ROI sizes by metric conversion factor...", compound="left", image=self.menu_icons["settings"]["img"], command=lambda: ROISizeStandardizerPopUp(config_path=self.config_path))
        file_menu.add_command(label="Set min/max draw window size...", compound="left", image=self.menu_icons["monitor"]["img"], command=lambda: SetMinMaxDrawWindowSize(config_path=self.config_path))
        self.root.config(menu=menu)

    def run(self):
        video_names = list(self.video_dict.keys())
        max_video_name_len = len(max(video_names, key=len))
        self.open_roi_uis = []
        self.video_table = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEOS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ROI.value)
        for table_idx in range(len(video_names)):
             _ = ROITableRow(config_path=self.config_path, parent=self.video_table, video_path=self.video_dict[video_names[table_idx]], str_width=max_video_name_len, row_idx=table_idx)
        self.video_table.grid(row=0, column=0, sticky=NW)
        self.main_frm.mainloop()

class ROITableRow():

    """
    Helper to create a single row class in the :func:`simba.ui.pop_ups.roi_video_table_pop_up.ROIVideoTable` table class.
    """
    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 parent: Union[Frame, Canvas, LabelFrame, Toplevel],
                 video_path: Union[str, os.PathLike],
                 str_width: int,
                 row_idx: int):

        self.index = Entry_Box(parent=parent, fileDescription='', labelwidth=0, entry_box_width=4)
        self.index.entry_set(val=f'{row_idx+1}.')
        self.index.grid(row=row_idx, column=0, sticky=NW)
        _, video_name, _ = get_fn_ext(filepath=video_path)
        self.video_lbl = SimBALabel(parent=parent, txt=video_name, txt_clr='black', width=str_width)
        self.video_lbl.grid(row=row_idx, column=1, sticky=NW)
        self.draw_btn = SimbaButton(parent=parent, txt='DRAW', img='paint', cmd=self.draw, cmd_kwargs={'config_path': lambda: config_path, 'video_path': lambda: video_path})
        self.draw_btn.grid(row=row_idx, column=2, sticky=NW)
        self.reset_btn = SimbaButton(parent=parent, txt='RESET', img='trash', cmd=self.reset, cmd_kwargs={'config_path': lambda: config_path, 'video_path': lambda: video_path})
        self.reset_btn.grid(row=row_idx, column=3, sticky=NW)
        self.apply_to_all_btn = SimbaButton(parent=parent, txt='APPLY TO ALL', img='add_on', cmd=self.apply_to_all, cmd_kwargs={'config_path': lambda: config_path, 'video_path': lambda: video_path})
        self.apply_to_all_btn.grid(row=row_idx, column=4, sticky=NW)


    def draw(self, config_path: Union[str, os.PathLike], video_path: Union[str, os.PathLike]):
        ROI_ui(config_path=config_path, video_path=video_path)

    def reset(self, config_path: Union[str, os.PathLike], video_path: Union[str, os.PathLike]):
        reset_video_ROIs(config_path=config_path, filename=video_path)

    def apply_to_all(self, config_path: Union[str, os.PathLike], video_path: Union[str, os.PathLike]):
        multiply_ROIs(config_path=config_path, filename=video_path)


#ROIVideoTable(config_path=r"C:\troubleshooting\mouse_open_field\project_folder\project_config.ini")