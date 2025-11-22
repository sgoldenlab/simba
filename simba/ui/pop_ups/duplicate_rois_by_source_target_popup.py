import os
from copy import deepcopy
from typing import Any, Optional, Union

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.roi_utils import (change_roi_dict_video_name,
                                       get_roi_data_for_video_name,
                                       get_roi_df_from_dict)
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, SimbaButton,
                                        SimbaCheckbox, SimBADropDown,
                                        SimBALabel)
from simba.utils.checks import check_file_exist_and_readable
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import NoFilesFoundError
from simba.utils.printing import stdout_success
from simba.utils.read_write import find_all_videos_in_directory


class DuplicateROIsBySourceTarget(ConfigReader, PopUpMixin):

    def __init__(self,
                 config_path: Union[str, os.PathLike],
                 roi_data_path: Optional[Union[str, os.PathLike]] = None,
                 roi_table_popup: Any = None):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False, create_logger=False)
        if roi_data_path is not None:
            check_file_exist_and_readable(file_path=roi_data_path)
            self.roi_coordinates_path = deepcopy(roi_data_path)
        if os.path.isfile(self.roi_coordinates_path): self.read_roi_data()
        else: self.video_names_w_rois = []
        if len(self.video_names_w_rois) == 0:
            raise NoFilesFoundError(msg=f'Cannot duplicate ROIs: no video has ROIs defined.', source=self.__class__.__name__)
        self.video_dict = find_all_videos_in_directory(directory=self.video_dir, as_dict=True, raise_error=False, sort_alphabetically=True)
        self.video_cnt = len(list(self.video_dict.keys()))
        if self.video_cnt == 0:
            raise NoFilesFoundError(msg=f'Cannot draw ROIs: the {self.video_dir} directory does not contain any video files.', source=self.__class__.__name__)
        self.roi_table_popup, self.config_path = roi_table_popup, config_path
        PopUpMixin.__init__(self, title="DUPLICATE ROIs FROM SOURCE VIDEO TO TARGET VIDEO(S)", size=(720, 960), icon='data_table')
        self.project_video_names = list(self.video_dict.keys())
        self.selected_destination_videos = {}
        self._create_source_frm()
        self.main_frm.mainloop()


    def set_status_bar_panel(self, text: str, fg: str = 'blue'):
        self.status_bar.configure(text=text, fg=fg)
        self.status_bar.update_idletasks()

    def _create_source_frm(self):
        self.source_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header=f"SELECT ROI SOURCE VIDEO", icon_name='video', icon_link=Links.ROI.value, pady=10)
        self.source_dropdown = SimBADropDown(parent=self.source_frm, dropdown_options=list(self.video_names_w_rois), label='ROI SOURCE VIDEO: ', label_width=35, dropdown_width=30, value=list(self.video_names_w_rois)[0], command= lambda x: self._create_destination_frm(x))
        self.run_btn = SimbaButton(parent=self.source_frm, txt='RUN', img='rocket', txt_clr='blue', font=Formats.FONT_LARGE.value, hover_font=Formats.FONT_LARGE_BOLD.value, cmd=self._run)
        self.source_frm.grid(row=0, column=0, sticky='NW')
        self.source_dropdown.grid(row=0, column=0, sticky='NW')
        self.run_btn.grid(row=1, column=0, sticky='NW')
        self._create_destination_frm(x=self.source_dropdown.get_value())
        self.status_bar = SimBALabel(parent=self.main_frm, txt='', txt_clr='black', bg_clr=None, font=Formats.FONT_REGULAR.value, relief='sunken')
        self.status_bar.grid(row=6, column=0, sticky='we')
        #self.main_frm.grid_rowconfigure(6, weight=0)

    def _create_destination_frm(self, x):
        if hasattr(self, 'destination_frm'):
            self.destination_frm.destroy()
        self.destination_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header=f"SELECT ROI TARGET VIDEO(S)", icon_name='stack', icon_link=Links.ROI.value)
        video_option_lst = [x for x in self.project_video_names if x != self.source_dropdown.get_value()]
        for video_cnt, video_name in enumerate(video_option_lst):
            if (video_name in self.selected_destination_videos.keys()) and self.selected_destination_videos[video_name].get():
                val = True
            else:
                val = False
            roi_cb, self.roi_var = SimbaCheckbox(parent=self.destination_frm, txt=video_name, val=val)
            roi_cb.grid(row=video_cnt, column=0, sticky='NW')
            self.selected_destination_videos[video_name] = self.roi_var
        self.destination_frm.grid(row=1, column=0, sticky='NW')
    def _run(self):
        destination_videos = [k for k, v in self.selected_destination_videos.items() if v.get()]
        source_video = self.source_dropdown.get_value()
        if len(destination_videos) == 0:
            txt = f'NO destination video(s) selected. Check AT LEAST ONE destination video.'
            self.set_status_bar_panel(text=txt, fg='red')
            raise NoFilesFoundError(msg=f'NO destination video(s) selected. Check AT LEAST ONE destination video.', source=self.__class__.__name__)
        source_rois = get_roi_data_for_video_name(roi_path=self.roi_coordinates_path, video_name=source_video)
        source_roi_cnt = len(list(source_rois.keys()))
        for target_video in destination_videos:
            video_roi_dict = change_roi_dict_video_name(roi_dict=source_rois, video_name=target_video)
            video_rectangles_df, video_circles_df, video_polygon_df = get_roi_df_from_dict(roi_dict=video_roi_dict)
            self.rectangles_df = self.rectangles_df[self.rectangles_df['Video'] != target_video]
            self.circles_df = self.circles_df[self.circles_df['Video'] != target_video]
            self.polygon_df = self.polygon_df[self.polygon_df['Video'] != target_video]
            self.rectangles_df = pd.concat([self.rectangles_df, video_rectangles_df], axis=0).reset_index(drop=True)
            self.circles_df = pd.concat([self.circles_df, video_circles_df], axis=0).reset_index(drop=True)
            self.polygon_df = pd.concat([self.polygon_df, video_polygon_df], axis=0).reset_index(drop=True)
        store = pd.HDFStore(self.roi_coordinates_path, mode="w")
        store[Keys.ROI_RECTANGLES.value] = self.rectangles_df
        store[Keys.ROI_CIRCLES.value] = self.circles_df
        store[Keys.ROI_POLYGONS.value] = self.polygon_df
        store.close()
        txt = f'{source_roi_cnt} ROI(s) from video {source_video} applied to {len(destination_videos)} other video(s)'
        self.set_status_bar_panel(text=txt, fg='blue')
        stdout_success(msg=txt, source=self.__class__.__name__)
        if self.roi_table_popup is not None:
            self.roi_table_popup.refresh_window()




#DuplicateROIsBySourceTarget(config_path=r"C:\troubleshooting\mouse_open_field\project_folder\project_config.ini")

