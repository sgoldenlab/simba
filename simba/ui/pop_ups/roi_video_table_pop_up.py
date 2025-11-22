import os
from tkinter import *
from typing import Union, Optional
from copy import deepcopy

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.roi_ui import ROI_ui
from simba.roi_tools.roi_utils import multiply_ROIs, reset_video_ROIs
from simba.ui.pop_ups.import_roi_csv_popup import ROIDefinitionsCSVImporterPopUp
from simba.ui.pop_ups.min_max_draw_size_popup import SetMinMaxDrawWindowSize
from simba.ui.pop_ups.roi_size_standardizer_popup import ROISizeStandardizerPopUp
from simba.ui.pop_ups.duplicate_rois_by_source_target_popup import DuplicateROIsBySourceTarget
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box, SimbaButton, SimBALabel)
from simba.utils.enums import Keys, Links, Formats
from simba.utils.errors import NoFilesFoundError
from simba.utils.read_write import find_all_videos_in_directory, get_fn_ext
from simba.utils.checks import check_file_exist_and_readable


WINDOW_SIZE = (720, 960)

class ROIVideoTable(ConfigReader, PopUpMixin):

    """
    Crates a tkinter video table listing all videos in project together with buttons associated with ROI drawing, deletaing, and duplicating shapes.

    :example:
    >>> ROIVideoTable(config_path=r"C:\troubleshooting\mouse_open_field\project_folder\project_config.ini")
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 roi_data_path: Optional[Union[str, os.PathLike]] = None):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        self.video_dict = find_all_videos_in_directory(directory=self.video_dir, as_dict=True, raise_error=False, sort_alphabetically=True)
        self.video_cnt, self.config_path, self.roi_data_path = len(list(self.video_dict.keys())), config_path, roi_data_path
        if self.video_cnt == 0:
            raise NoFilesFoundError(msg=f'Cannot draw ROIs: the {self.video_dir} directory does not contain any video files.', source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="PROJECT VIDEOS: ROI TABLE", size=WINDOW_SIZE, icon='data_table')

        if roi_data_path is not None:
            check_file_exist_and_readable(file_path=roi_data_path)
            self.roi_coordinates_path = deepcopy(roi_data_path)
        if os.path.isfile(self.roi_coordinates_path): self.read_roi_data()
        else: self.video_names_w_rois = []
        self.get_file_menu()
        self.run()

    def refresh_window(self):
        self.root.destroy()
        ConfigReader.__init__(self, config_path=self.config_path, read_video_info=False, create_logger=False)
        self.video_dict = find_all_videos_in_directory(directory=self.video_dir, as_dict=True, raise_error=False, sort_alphabetically=True)
        self.video_cnt = len(list(self.video_dict.keys()))
        PopUpMixin.__init__(self, title="PROJECT VIDEOS: ROI TABLE", size=WINDOW_SIZE, icon='data_table')
        if self.roi_data_path is not None:
            check_file_exist_and_readable(file_path=self.roi_data_path)
            self.roi_coordinates_path = deepcopy(self.roi_data_path)
        if os.path.isfile(self.roi_coordinates_path): self.read_roi_data()
        else: self.video_names_w_rois = []
        self.get_file_menu()
        self.run()

    def get_file_menu(self):
        menu = Menu(self.root)
        file_menu = Menu(menu)
        menu.add_cascade(label="File...", menu=file_menu)
        file_menu.add_command(label="Standardize ROI sizes by metric conversion factor...", compound="left", image=self.menu_icons["settings"]["img"], command=lambda: ROISizeStandardizerPopUp(config_path=self.config_path))
        file_menu.add_command(label="Duplicate ROIs from source video to target videos... ", compound="left", image=self.menu_icons["duplicate_small"]["img"], command=lambda: DuplicateROIsBySourceTarget(config_path=self.config_path, roi_data_path=self.roi_coordinates_path, roi_table_popup=self))
        file_menu.add_command(label="Import SimBA ROI CSV definitions... ", compound="left", image=self.menu_icons["csv_black"]["img"], command=lambda: ROIDefinitionsCSVImporterPopUp(config_path=self.config_path))
        file_menu.add_command(label="Set min/max draw window size...", compound="left", image=self.menu_icons["monitor"]["img"], command=lambda: SetMinMaxDrawWindowSize(config_path=self.config_path))
        self.root.config(menu=menu)

    def run(self):
        video_names = list(self.video_dict.keys())
        max_video_name_len = len(max(video_names, key=len))
        self.open_roi_uis = []
        self.video_table = CreateLabelFrameWithIcon(parent=self.main_frm, header=f"VIDEOS ({self.video_cnt} video(s) found in project)", icon_name='stack', icon_link=Links.ROI.value)
        for table_idx in range(len(video_names)):
             _ = ROITableRow(config_path=self.config_path, parent=self.video_table, video_path=self.video_dict[video_names[table_idx]], str_width=max_video_name_len, row_idx=table_idx, videos_w_rois=list(self.video_names_w_rois), video_table_window=self)
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
                 row_idx: int,
                 videos_w_rois: list,
                 video_table_window: ROIVideoTable):

        self.index = Entry_Box(parent=parent, fileDescription='', labelwidth=0, entry_box_width=4)
        self.index.entry_set(val=f'{row_idx+1}.')
        self.index.grid(row=row_idx, column=0, sticky=NW)
        _, video_name, _ = get_fn_ext(filepath=video_path)
        self.video_table_window = video_table_window
        if video_name not in videos_w_rois:
            self.video_lbl = SimBALabel(parent=parent, txt=video_name, txt_clr='black', width=str_width, font=Formats.FONT_REGULAR.value)
        else:
            self.video_lbl = SimBALabel(parent=parent, txt=video_name, txt_clr='green', width=str_width, font=Formats.FONT_REGULAR_BOLD.value)
        self.video_lbl.grid(row=row_idx, column=1, sticky=NW)
        self.draw_btn = SimbaButton(parent=parent, txt='DRAW', img='paint', cmd=self.draw, cmd_kwargs={'config_path': lambda: config_path, 'video_path': lambda: video_path})
        self.draw_btn.grid(row=row_idx, column=2, sticky=NW)
        self.reset_btn = SimbaButton(parent=parent, txt='RESET', img='trash', cmd=self.reset, cmd_kwargs={'config_path': lambda: config_path, 'video_path': lambda: video_path})
        self.reset_btn.grid(row=row_idx, column=3, sticky=NW)
        self.apply_to_all_btn = SimbaButton(parent=parent, txt='APPLY TO ALL', img='add_on', cmd=self.apply_to_all, cmd_kwargs={'config_path': lambda: config_path, 'video_path': lambda: video_path})
        self.apply_to_all_btn.grid(row=row_idx, column=4, sticky=NW)
        if video_name not in videos_w_rois:
            self.status_lbl = SimBALabel(parent=parent, txt='NO ROIs defined', txt_clr='black', font=Formats.FONT_REGULAR.value, img='black_cross')
        else:
            self.status_lbl = SimBALabel(parent=parent, txt='ROIs defined', txt_clr='green', font=Formats.FONT_REGULAR_BOLD.value, img='green_check')
        self.status_lbl.grid(row=row_idx, column=5, sticky=NW)

    def draw(self, config_path: Union[str, os.PathLike], video_path: Union[str, os.PathLike]):
        ROI_ui(config_path=config_path, video_path=video_path, roi_table_popup=self.video_table_window)

    def reset(self, config_path: Union[str, os.PathLike], video_path: Union[str, os.PathLike]):
        reset_video_ROIs(config_path=config_path, filename=video_path)
        self.video_table_window.refresh_window()

    def apply_to_all(self, config_path: Union[str, os.PathLike], video_path: Union[str, os.PathLike]):
        multiply_ROIs(config_path=config_path, filename=video_path)
        self.video_table_window.refresh_window()


#ROIVideoTable(config_path=r"C:\troubleshooting\mouse_open_field\project_folder\project_config.ini")