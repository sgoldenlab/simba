import os
import re
from copy import deepcopy
from typing import Any, Optional, Union

import pandas as pd

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.roi_utils import (change_roi_dict_video_name,
                                       get_roi_data_for_video_name,
                                       get_roi_df_from_dict)
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton, SimbaCheckbox,
                                        SimBADropDown, SimBALabel,
                                        TwoOptionQuestionPopUp)
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
        self._get_video_names_with_rois()
        self.video_dict = find_all_videos_in_directory(directory=self.video_dir, as_dict=True, raise_error=False, sort_alphabetically=True)
        self.video_cnt = len(list(self.video_dict.keys()))
        if self.video_cnt == 0:
            raise NoFilesFoundError(msg=f'Cannot draw ROIs: the {self.video_dir} directory does not contain any video files.', source=self.__class__.__name__)
        self.roi_table_popup, self.config_path = roi_table_popup, config_path
        PopUpMixin.__init__(self, title="DUPLICATE ROIs FROM SOURCE VIDEO TO TARGET VIDEO(S)", size=(720, 960), icon='data_table')
        self.project_video_names = list(self.video_dict.keys())
        self.selected_destination_videos = {}
        self.check_index, self.prior_check_index, self.ctrl_pressed = None, None, False
        self._create_source_frm()
        self._get_targets(source_value=self.source_dropdown.get_value())
        self._bind_keys()
        self.root.focus_set()
        self.main_frm.mainloop()

    def set_status_bar_panel(self, text: str, fg: str = 'blue'):
        self.status_bar.configure(text=text, fg=fg)
        self.status_bar.update_idletasks()

    def _create_source_frm(self):
        self.source_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header=f"SELECT ROI SOURCE VIDEO", icon_name='video', icon_link=Links.ROI.value, pady=10)
        self.source_dropdown = SimBADropDown(parent=self.source_frm, dropdown_options=list(self.video_names_w_rois), label='ROI SOURCE VIDEO: ', label_width=35, dropdown_width=30, value=list(self.video_names_w_rois)[0], searchable=True, command= lambda x: self._get_targets(source_value=x))
        self.run_btn = SimbaButton(parent=self.source_frm, txt='RUN', img='rocket', txt_clr='blue', font=Formats.FONT_LARGE.value, hover_font=Formats.FONT_LARGE_BOLD.value, cmd=self._run)
        self.source_frm.grid(row=0, column=0, sticky='NW')
        self.source_dropdown.grid(row=0, column=0, sticky='NW')
        self.run_btn.grid(row=1, column=0, sticky='NW')
        self.get_selection_options_frm()
        self.status_bar = SimBALabel(parent=self.main_frm, txt='', txt_clr='black', bg_clr=None, font=Formats.FONT_REGULAR.value, relief='sunken')
        self.status_bar.grid(row=6, column=0, sticky='we')

    def _bind_keys(self):
        self.root.bind("<KeyPress-Control_L>", lambda e: self._ctrl_press())
        self.root.bind("<KeyPress-Control_R>", lambda e: self._ctrl_press())
        self.root.bind("<KeyRelease-Control_L>", lambda e: self._ctrl_release())
        self.root.bind("<KeyRelease-Control_R>", lambda e: self._ctrl_release())
        self.main_frm.bind("<KeyPress-Control_L>", lambda e: self._ctrl_press())
        self.main_frm.bind("<KeyPress-Control_R>", lambda e: self._ctrl_press())
        self.main_frm.bind("<KeyRelease-Control_L>", lambda e: self._ctrl_release())
        self.main_frm.bind("<KeyRelease-Control_R>", lambda e: self._ctrl_release())
        self.root.bind("<KeyPress-Shift_L>", lambda e: self._ctrl_press())
        self.root.bind("<KeyPress-Shift_R>", lambda e: self._ctrl_press())
        self.root.bind("<KeyRelease-Shift_L>", lambda e: self._ctrl_release())
        self.root.bind("<KeyRelease-Shift_R>", lambda e: self._ctrl_release())
        self.main_frm.bind("<KeyPress-Shift_L>", lambda e: self._ctrl_press())
        self.main_frm.bind("<KeyPress-Shift_R>", lambda e: self._ctrl_press())
        self.main_frm.bind("<KeyRelease-Shift_L>", lambda e: self._ctrl_release())
        self.main_frm.bind("<KeyRelease-Shift_R>", lambda e: self._ctrl_release())

    def _unbind_keys(self):
        pass

    def _ctrl_press(self):
        self.ctrl_pressed = True

    def _ctrl_release(self):
        self.ctrl_pressed = False

    def _on_checkbox_click(self, event, video_idx: int, video_name: str):
        self.ctrl_pressed = bool(event.state & 0x4)

    def _check_rng(self, start: int, end: int):
        for video_idx in range(start, end):
            video_name = self.video_option_lst[video_idx]
            self.selected_destination_videos[video_name].set(True)

    def _deselect_all(self):
        two_question_pop_up = TwoOptionQuestionPopUp(question='Sure you want to \n DESELECT ALL VIDEOS?', option_one='YES', option_two='NO', title='DESELECT ALL VIDEOS')
        if two_question_pop_up.selected_option == 'YES':
            for video_name, video_var in self.selected_destination_videos.items():
                video_var.set(False)

    def set_check_idx(self, video_idx: int, video_name: str):
        is_checked = self.selected_destination_videos[video_name].get()
        if is_checked:
            self.prior_check_index, self.check_index = deepcopy(self.check_index), video_idx
            if isinstance(self.prior_check_index, int) and isinstance(self.check_index, int) and self.ctrl_pressed and self.prior_check_index != self.check_index:
                self._check_rng(start=min(self.prior_check_index, self.check_index), end=max(self.prior_check_index, self.check_index) + 1)
        else:
            if self.check_index == video_idx:
                self.check_index, self.prior_check_index = self.prior_check_index, None
            elif self.prior_check_index == video_idx:
                self.prior_check_index = None

    def _get_video_names_with_rois(self):
        if os.path.isfile(self.roi_coordinates_path): self.read_roi_data()
        else: self.video_names_w_rois = []
        if len(self.video_names_w_rois) == 0:
            raise NoFilesFoundError(msg=f'Cannot duplicate ROIs: no video has ROIs defined.', source=self.__class__.__name__)
        self.video_names_w_rois = sorted(self.video_names_w_rois, key=lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)])

    def get_selection_options_frm(self):
        self.target_selection_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header=f"TARGET FILTER/CLEAR", icon_name='filter', icon_link=Links.ROI.value)
        self.deselect_all_btn = SimbaButton(parent=self.target_selection_frm, txt='DESELECT ALL', img='trash', cmd=self._deselect_all)
        self.target_filter_eb = Entry_Box(parent=self.target_selection_frm, fileDescription='TARGET VIDEO NAME FILTER:', labelwidth=30, entry_box_width=30, cmd=self._filter_targets)
        self.target_selection_frm.grid(row=1, column=0, sticky='NW')
        self.deselect_all_btn.grid(row=0, column=0, sticky='NW')
        self.target_filter_eb.grid(row=1, column=0, sticky='NW')

    def _get_targets(self, source_value):
        self.video_option_lst = [x for x in self.project_video_names if x != source_value]
        self._create_destination_frm(targets=self.video_option_lst)

    def _filter_targets(self, filter_str: str):
        if len(filter_str) > 0:
            self.video_option_lst = [x for x in self.project_video_names if filter_str.lower() in x.lower()]
        else:
            self.video_option_lst = deepcopy(self.project_video_names)
        self._create_destination_frm(targets=self.video_option_lst)

    def _create_destination_frm(self, targets):
        if hasattr(self, 'destination_frm'):
            self.destination_frm.destroy()
        self.destination_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header=f"SELECT ROI TARGET VIDEO(S)", icon_name='stack', icon_link=Links.ROI.value)
        for video_cnt, video_name in enumerate(targets):
            txt_img  = 'black_cross' if video_name not in self.video_names_w_rois else 'green_check'
            if (video_name in self.selected_destination_videos.keys()) and self.selected_destination_videos[video_name].get():
                val = True
            else:
                val = False
            roi_cb, self.roi_var = SimbaCheckbox(parent=self.destination_frm, txt=video_name, val=val, cmd= lambda idx=video_cnt, name=video_name: self.set_check_idx(idx, name), txt_img=txt_img, txt_img_location='left')
            roi_cb.bind("<Button-1>", lambda e, idx=video_cnt, name=video_name: self._on_checkbox_click(e, idx, name))
            roi_cb.grid(row=video_cnt+1, column=0, sticky='NW')
            self.selected_destination_videos[video_name] = self.roi_var
        self.destination_frm.grid(row=2, column=0, sticky='NW')

    def _run(self):
        destination_videos = [k for k, v in self.selected_destination_videos.items() if v.get()]
        source_video = self.source_dropdown.get_value()
        if len(destination_videos) == 0:
            txt = f'NO destination video(s) selected. Check AT LEAST ONE destination video.'
            self.set_status_bar_panel(text=txt, fg='red')
            raise NoFilesFoundError(msg=f'NO destination video(s) selected. Check AT LEAST ONE destination video.', source=self.__class__.__name__)
        if source_video not in self.video_names_w_rois:
            raise NoFilesFoundError(msg=f'The SOURCE VIDEO NAMES {source_video} is not a valid source video.', source=self.__class__.__name__)
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
        self.root.focus_set()
        self.root.attributes('-topmost', True)


#DuplicateROIsBySourceTarget(config_path=r"C:\troubleshooting\mouse_open_field\project_folder\project_config.ini")

