import os
import platform
import subprocess
import sys
from datetime import datetime
from tkinter import *
from typing import Union

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon

import simba
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.roi_tools.roi_ui import ROI_ui
from simba.roi_tools.roi_utils import get_roi_data, multiply_ROIs
from simba.ui.blob_quick_check_interface import BlobQuickChecker
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        FolderSelect, SimbaButton,
                                        SimBADropDown, SimBALabel)
from simba.utils.checks import (check_if_dir_exists, check_int,
                                check_nvidea_gpu_available, check_str)
from simba.utils.enums import Formats, Paths
from simba.utils.errors import (InvalidInputError, InvalidVideoFileError,
                                NoDataError)
from simba.utils.read_write import (find_all_videos_in_directory,
                                    find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    get_video_meta_data, remove_files,
                                    save_json, str_2_bool, write_pickle)
from simba.video_processors.blob_tracking_executor import BlobTrackingExecutor

ABSOLUTE = 'absolute'
DEFAULT_THRESHOLD = 30

WINDOW_SIZES = ['None', 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
SMOOTHING_TIMES = list(np.round(np.arange(0.01, 2.1, 0.01), 2))
SMOOTHING_TIMES.insert(0, 'None')
BUFFER_SIZES = list(range(1, 101))
BUFFER_SIZES.insert(0, 'None')
KERNEL_SIZES = list(np.round(np.arange(0.1, 100.1, 0.1), 2))
KERNEL_SIZES.insert(0, 'None')

SIMBA_DIR = os.path.dirname(simba.__file__)
BLOB_EXECUTOR_PATH = os.path.join(SIMBA_DIR, Paths.BLOB_EXECUTOR_PATH.value)

class BlobTrackingUI(PopUpMixin):
    """
    :example:
    >>> _ = BlobTrackingUI(input_dir=r'D:\open_field_3\sample', output_dir=r"D:\open_field_3\sample\blob_data")
    """

    def __init__(self,
                 input_dir: Union[str, os.PathLike],
                 output_dir: Union[str, os.PathLike]):

        check_if_dir_exists(in_dir=input_dir, source=self.__class__.__name__)
        check_if_dir_exists(in_dir=output_dir, create_if_not_exist=True, source=self.__class__.__name__)
        if os.path.isdir(output_dir):
            existing_files = find_files_of_filetypes_in_directory(directory=output_dir, extensions=['.json', '.csv', '.mp4', '.pickle'], raise_error=False, raise_warning=False)
            if len(existing_files) > 0: raise InvalidInputError(msg=f'The selected output directory {output_dir} is not empty. Select an empty output directory where to save your tracking data', source=self.__class__.__name__)
        self.gpu_available = check_nvidea_gpu_available()
        self.in_videos = find_all_videos_in_directory(directory=input_dir, as_dict=True, raise_error=True)
        self.input_dir, self.output_dir = input_dir, output_dir
        PopUpMixin.__init__(self, title="BLOB TRACKING", size=(2400, 600), icon='bubble_green')
        self.len_max_char = max(len(max(list(self.in_videos.keys()), key=len)), 25)
        self.core_cnt = find_core_cnt()[0]
        self.get_quick_settings()
        self.get_main_table()
        self.get_main_table_entries()
        self.get_execute_btns()
        self.get_status_bar()
        self.roi_inclusion_store = os.path.join(self.output_dir, 'inclusion_definitions.h5')
        self.out_path = os.path.join(self.output_dir, 'blob_definitions.pickle')
        #self.main_frm.mainloop()

    def get_quick_settings(self):
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings', padx=15, pady=15)
        self.quick_settings_frm = CreateLabelFrameWithIcon(parent=self.settings_frm, header="QUICK SETTINGS", icon_name='clock', padx=15, pady=15, relief='solid')
        self.quick_setting_threshold_dropdown = SimBADropDown(parent=self.quick_settings_frm, dropdown_options=list(range(1, 101)), label="THRESHOLD:", label_width=30, dropdown_width=10, value=DEFAULT_THRESHOLD)
        self.quick_setting_threshold_btn = SimbaButton(parent=self.quick_settings_frm, txt='APPLY', img='tick',cmd=self._set_threshold, cmd_kwargs={'threshold': lambda: self.quick_setting_threshold_dropdown.getChoices()})
        self.set_smoothing_time_dropdown = SimBADropDown(parent=self.quick_settings_frm, dropdown_options=SMOOTHING_TIMES, label="SMOOTHING TIME (S):", label_width=30, dropdown_width=10, value=SMOOTHING_TIMES[0])
        self.set_smoothing_time_btn = SimbaButton(parent=self.quick_settings_frm, txt='APPLY', img='tick', cmd=self._set_smoothing_time, cmd_kwargs={'time': lambda: self.set_smoothing_time_dropdown.getChoices()})
        self.set_buffer_dropdown = SimBADropDown(parent=self.quick_settings_frm, dropdown_options=BUFFER_SIZES, label="BUFFER (PIXELS):", label_width=30, dropdown_width=10, value=BUFFER_SIZES[0])
        self.set_buffer_btn = SimbaButton(parent=self.quick_settings_frm, txt='APPLY', img='tick', cmd=self._set_buffer_size, cmd_kwargs={'size': lambda: self.set_buffer_dropdown.getChoices()})
        self.closing_kernal_size_dropdown = SimBADropDown(parent=self.quick_settings_frm, dropdown_options=KERNEL_SIZES, label="GAP FILL FILTER SIZE (%):", label_width=30, dropdown_width=10, value=BUFFER_SIZES[0])
        self.set_closing_kernel_btn = SimbaButton(parent=self.quick_settings_frm, txt='APPLY', img='tick', cmd=self._set_close_kernel_dropdown, cmd_kwargs={'size': lambda: self.closing_kernal_size_dropdown.getChoices()})
        self.opening_kernal_size_dropdown = SimBADropDown(parent=self.quick_settings_frm, dropdown_options=KERNEL_SIZES, label="NOISE REMOVAL FILTER SIZE (%):", label_width=30, dropdown_width=10, value=BUFFER_SIZES[0])
        self.opening_kernel_btn = SimbaButton(parent=self.quick_settings_frm, txt='APPLY', img='tick', cmd=self._set_open_kernel_dropdown, cmd_kwargs={'size': lambda: self.opening_kernal_size_dropdown.getChoices()})

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.quick_settings_frm.grid(row=0, column=0, sticky=NW)
        self.quick_setting_threshold_dropdown.grid(row=0, column=0, sticky=NW)
        self.quick_setting_threshold_btn.grid(row=0, column=1, sticky=NW)
        self.set_smoothing_time_dropdown.grid(row=1, column=0, sticky=NW)
        self.set_smoothing_time_btn.grid(row=1, column=1, sticky=NW)
        self.set_buffer_dropdown.grid(row=4, column=0, sticky=NW)
        self.set_buffer_btn.grid(row=4, column=1, sticky=NW)
        self.closing_kernal_size_dropdown.grid(row=5, column=0, sticky=NW)
        self.set_closing_kernel_btn.grid(row=5, column=1, sticky=NW)
        self.opening_kernal_size_dropdown.grid(row=6, column=0, sticky=NW)
        self.opening_kernel_btn.grid(row=6, column=1, sticky=NW)

        self.run_time_settings_frm = CreateLabelFrameWithIcon(parent=self.settings_frm, header="RUN-TIME SETTINGS", icon_name='run', padx=15, pady=15, relief='solid')
        self.bg_dir = FolderSelect(parent=self.run_time_settings_frm, folderDescription='BACKGROUND DIRECTORY:', lblwidth=30, initialdir=self.input_dir)
        self.bg_dir_apply = SimbaButton(parent=self.run_time_settings_frm, txt='APPLY', img='tick',  cmd=self._apply_bg_dir, cmd_kwargs={'bg_dir': lambda: self.bg_dir.folder_path})
        self.use_gpu_dropdown = SimBADropDown(parent=self.run_time_settings_frm, dropdown_options=['TRUE', 'FALSE'], label="USE GPU:", label_width=30, dropdown_width=10, value='FALSE')
        #if not self.gpu_available: self.use_gpu_dropdown.disable()
        #else: self.use_gpu_dropdown.set_value(value='TRUE')
        self.use_gpu_dropdown.disable()
        self.core_cnt_dropdown = SimBADropDown(parent=self.run_time_settings_frm, dropdown_options=list(range(1, self.core_cnt+1)), label="CPU CORE COUNT:", label_width=30, dropdown_width=10, value=int(self.core_cnt / 2))

        self.vertice_cnt_dropdown = SimBADropDown(parent=self.run_time_settings_frm, dropdown_options=list(range(10, 501)), label="VERTICE COUNT:", label_width=30, dropdown_width=10, value=30)
        self.save_videos_dropdown = SimBADropDown(parent=self.run_time_settings_frm, dropdown_options=['TRUE', 'FALSE'], label="SAVE BACKGROUND VIDEOS:", label_width=30, dropdown_width=10, value='TRUE')
        self.close_iterations_dropdown_dropdown = SimBADropDown(parent=self.run_time_settings_frm, dropdown_options=list(range(1, 20)), label="GAP FILLING ITERATIONS:", label_width=30, dropdown_width=10, value=3)
        self.open_iterations_dropdown = SimBADropDown(parent=self.run_time_settings_frm, dropdown_options=list(range(1, 20)), label="NOISE REMOVAL ITERATIONS:", label_width=30, dropdown_width=10, value=3)
        self.duplicate_inclusion_zones_dropdown = SimBADropDown(parent=self.run_time_settings_frm, dropdown_options=list(self.in_videos.keys()), label="DUPLICATE INCLUSION ZONES:", label_width=30, dropdown_width=self.len_max_char, value=list(self.in_videos.keys())[0])
        self.duplicate_inclusion_zones_btn = SimbaButton(parent=self.run_time_settings_frm, txt='APPLY', img='tick', cmd=self._duplicate_inclusion_zones, cmd_kwargs={'video_name': lambda: self.duplicate_inclusion_zones_dropdown.getChoices()})
        self.run_time_settings_frm.grid(row=0, column=1, sticky=NW)

        self.bg_dir.grid(row=0, column=0, sticky=NW)
        self.bg_dir_apply.grid(row=0, column=1, sticky=NW)
        self.use_gpu_dropdown.grid(row=1, column=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=2, column=0, sticky=NW)
        self.vertice_cnt_dropdown.grid(row=3, column=0, sticky=NW)
        self.save_videos_dropdown.grid(row=4, column=0, sticky=NW)
        self.close_iterations_dropdown_dropdown.grid(row=5, column=0, sticky=NW)
        self.open_iterations_dropdown.grid(row=6, column=0, sticky=NW)
        self.duplicate_inclusion_zones_dropdown.grid(row=7, column=0, sticky=NW)
        self.duplicate_inclusion_zones_btn.grid(row=7, column=1, sticky=NW)
        self.execute_frm = CreateLabelFrameWithIcon(parent=self.settings_frm, header="EXECUTE", icon_name='rocket', padx=15, pady=15, relief='solid')
        self.run_btn = SimbaButton(parent=self.execute_frm, txt='RUN', img='rocket', cmd=self._initialize_run)
        self.remove_inclusion_zones_btn = SimbaButton(parent=self.execute_frm, txt='REMOVE INCLUSION ZONES', img='trash', cmd=self._delete_inclusion_zone_data, cmd_kwargs=None, txt_clr='darkblue')
        self.execute_frm.grid(row=0, column=2, sticky=NW)
        self.run_btn.grid(row=0, column=0, sticky=NW)
        self.remove_inclusion_zones_btn.grid(row=1, column=0, sticky=NW)

    def get_main_table(self):
        self.headings = {}
        self.videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEOS", icon_name='video', pady=5, padx=15, relief='solid')
        self.headings['video_name'] = SimBALabel(parent=self.videos_frm, txt='VIDEO NAME', width=self.len_max_char, font=Formats.FONT_HEADER.value)
        self.headings['bg_path'] = SimBALabel(parent=self.videos_frm, txt='BACKGROUND REFERENCE', width=25, font=Formats.FONT_HEADER.value)
        self.headings['threshold'] = SimBALabel(parent=self.videos_frm, txt='THRESHOLD', width=15, font=Formats.FONT_HEADER.value)
        self.headings['inclusion_zones'] = SimBALabel(parent=self.videos_frm, txt='INCLUSION ZONES', width=25, font=Formats.FONT_HEADER.value)
        self.headings['smoothing_time'] = SimBALabel(parent=self.videos_frm, txt='SMOOTHING TIME (S)', width=25, font=Formats.FONT_HEADER.value)
        self.headings['buffer_size'] = SimBALabel(parent=self.videos_frm, txt='BUFFER SIZE (PX)', width=25, font=Formats.FONT_HEADER.value)
        self.headings['closing_kernel_size'] = SimBALabel(parent=self.videos_frm, txt='GAP FILL SIZE (%)', width=25, font=Formats.FONT_HEADER.value)
        self.headings['opening_kernel_size'] = SimBALabel(parent=self.videos_frm, txt='NOISE FILL SIZE (%)', width=25, font=Formats.FONT_HEADER.value)
        self.headings['quick_check'] = SimBALabel(parent=self.videos_frm, txt='QUICK CHECK', width=20, font=Formats.FONT_HEADER.value)
        for cnt, (k, v) in enumerate(self.headings.items()):
            v.grid(row=0, column=cnt, sticky=NW)
        self.videos_frm.grid(row=1, column=0, sticky=NW)

    def get_main_table_entries(self):
        self.videos = {}
        for video_cnt, (video_name, video_path) in enumerate(self.in_videos.items()):
            self.videos[video_name] = {}
            self.videos[video_name]['name_lbl'] = SimBALabel(parent=self.videos_frm, txt=video_name, width=self.len_max_char, font=Formats.FONT_HEADER.value)
            self.videos[video_name]["threshold_dropdown"] = SimBADropDown(parent=self.videos_frm, dropdown_options=list(range(1, 100)), label="", label_width=0, dropdown_width=15, value=DEFAULT_THRESHOLD)
            self.videos[video_name]["inclusion_btn"] = Button(self.videos_frm, text="SET INCLUSION ZONES", fg="black", command=lambda k=self.videos[video_name]['name_lbl']["text"]: self._launch_set_inclusion_zones(k), width=25)
            self.videos[video_name]["bg_file"] = FileSelect(parent=self.videos_frm, width=25)
            self.videos[video_name]["smoothing_time_dropdown"] = SimBADropDown(parent=self.videos_frm, dropdown_options=SMOOTHING_TIMES, label="", label_width=0, dropdown_width=25, value=SMOOTHING_TIMES[0])
            self.videos[video_name]["quick_check_btn"] = Button(self.videos_frm, text="QUICK CHECK", fg="black", command=lambda k=self.videos[video_name]['name_lbl']["text"]: self._quick_check(k), width=20)
            self.videos[video_name]["buffer_dropdown"] = SimBADropDown(parent=self.videos_frm, dropdown_options=BUFFER_SIZES, label="", label_width=0, dropdown_width=25, value=BUFFER_SIZES[0])
            self.videos[video_name]["close_kernel"] = SimBADropDown(parent=self.videos_frm, dropdown_options=KERNEL_SIZES, label="", label_width=0, dropdown_width=25, value=KERNEL_SIZES[0])
            self.videos[video_name]["open_kernel"] = SimBADropDown(parent=self.videos_frm, dropdown_options=KERNEL_SIZES, label="", label_width=0, dropdown_width=25, value=KERNEL_SIZES[0])
            self.videos[video_name]['name_lbl'].grid(row=video_cnt+1, column=0, sticky=NW)
            self.videos[video_name]['bg_file'].grid(row=video_cnt+1, column=1, sticky=NW)
            self.videos[video_name]['threshold_dropdown'].grid(row=video_cnt + 1, column=2, sticky=NW)
            self.videos[video_name]['inclusion_btn'].grid(row=video_cnt+1, column=3, sticky=NW)
            self.videos[video_name]['smoothing_time_dropdown'].grid(row=video_cnt + 1, column=4, sticky=NW)
            self.videos[video_name]['buffer_dropdown'].grid(row=video_cnt + 1, column=5, sticky=NW)
            self.videos[video_name]['close_kernel'].grid(row=video_cnt + 1, column=6, sticky=NW)
            self.videos[video_name]['open_kernel'].grid(row=video_cnt + 1, column=7, sticky=NW)
            self.videos[video_name]['quick_check_btn'].grid(row=video_cnt + 1, column=8, sticky=NW)

    def get_execute_btns(self):
        self.execute_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="EXECUTE", icon_name='rocket', padx=15, pady=15, relief='solid')
        self.run_btn = SimbaButton(parent=self.execute_frm, txt='RUN', img='rocket', cmd=self._initialize_run)
        self.remove_inclusion_zones_btn = SimbaButton(parent=self.execute_frm, txt='REMOVE INCLUSION ZONES', img='trash', cmd=self._delete_inclusion_zone_data, cmd_kwargs=None, txt_clr='darkblue')
        self.execute_frm.grid(row=2, column=0, sticky=NW)
        self.run_btn.grid(row=0, column=0, sticky=NW)
        self.remove_inclusion_zones_btn.grid(row=0, column=1, sticky=NW)

    def get_status_bar(self):
        self.status_bar = Label(self.main_frm, text="STATUS: READY", bd=1, relief=SUNKEN, anchor=W, fg='blue')
        self.status_bar.grid(row=3, column=0, sticky="ew")

    def set_status_bar_panel(self, text: str, fg: str = 'blue'):
        self.status_bar.configure(text=text, fg=fg)
        self.status_bar.update_idletasks()
        print(text)


    def _delete_inclusion_zone_data(self):
        if os.path.isfile(self.roi_inclusion_store):
            remove_files(file_paths=[self.roi_inclusion_store])
        else:
            msg = 'Cannot delete INCLUSION zones: No INCLUSION zones found.'
            self.set_status_bar_panel(text=msg, fg='red')
            raise NoDataError(msg='Cannot delete INCLUSION zones: No INCLUSION zones exist.', source=self.__class__.__name__)

    def _set_window_size(self, size: str):
        for video_name in self.videos.keys():
            self.videos[video_name]["window_size_dropdown"].setChoices(size)

    def _launch_set_inclusion_zones(self, video_name: str):
        ROI_ui(config_path=None, video_path=self.in_videos[video_name], roi_coordinates_path=self.roi_inclusion_store, video_dir=self.input_dir)


    def _set_smoothing_time(self, time: str):
        for video_name in self.videos.keys():
            self.videos[video_name]["smoothing_time_dropdown"].setChoices(time)
        self.set_status_bar_panel(text=f'Set smoothing time to {time} for all videos...', fg='blue')

    def _set_buffer_size(self, size: str):
        for video_name in self.videos.keys():
            self.videos[video_name]["buffer_dropdown"].setChoices(size)
        self.set_status_bar_panel(text=f'Set buffer size to {size} for all videos...', fg='blue')

    def _set_close_kernel_dropdown(self, size: str):
        for video_name in self.videos.keys():
            self.videos[video_name]["close_kernel"].setChoices(size)
        self.set_status_bar_panel(text=f'Set CLOSE kernel size to {size} for all videos...', fg='blue')


    def _set_open_kernel_dropdown(self, size: str):
        for video_name in self.videos.keys():
            self.videos[video_name]["open_kernel"].setChoices(size)
        self.set_status_bar_panel(text=f'Set OPEN kernel size to {size} for all videos...', fg='blue')


    def _set_threshold(self, threshold: str):
        check_int(name='threshold', value=threshold, min_value=0, max_value=100, raise_error=True)
        for video_name in self.videos.keys():
            self.videos[video_name]["threshold_dropdown"].setChoices(threshold)
        self.set_status_bar_panel(text=f'Set threshold to {threshold} for all videos...', fg='blue')

    def _set_visualize(self, val: str):
        check_str(name='visualize', value=val, options=('TRUE', 'FALSE'), raise_error=True)
        for video_name in self.videos.keys():
            self.videos[video_name]["visualize_dropdown"].setChoices(val)

    def _duplicate_inclusion_zones(self, video_name: str):
        if os.path.isfile(self.roi_inclusion_store):
            _, _, _, roi_dict, _, _, _ = get_roi_data(roi_path=self.roi_inclusion_store, video_name=video_name)
            if len(list(roi_dict.keys())) == 0:
                msg = f'Cannot duplicate the INCLUSION zones in {video_name}: Video {video_name} have no drawn INCLUSION zones.'
                self.set_status_bar_panel(text=msg, fg='red')
                raise NoDataError(msg=msg, source=self.__class__.__name__)
            else:
                multiply_ROIs(filename=self.in_videos[video_name], roi_coordinates_path=self.roi_inclusion_store, videos_dir=self.input_dir)
                self.set_status_bar_panel(text=f'Duplicated inclusion zoned for {video_name} for all videos...', fg='blue')
        else:
            msg = f'Cannot duplicate the INCLUSION zones in {video_name}: No INCLUSION zones have been drawn.'
            self.set_status_bar_panel(text=msg, fg='red')
            raise NoDataError(msg=msg, source=self.__class__.__name__)


    def _check_bg_videos(self, bg_videos: dict, videos: dict):
        missing_bg_videos = list(set(list(videos.keys())) - set(list(bg_videos.keys())))
        if len(missing_bg_videos) > 0:
            msg = f'The chosen BACKGROUND DIRECTORY is missing videos for {len(missing_bg_videos)} video files: {missing_bg_videos}'
            self.set_status_bar_panel(text=msg, fg='red')
            raise InvalidVideoFileError(msg=msg, source=self.__class__.__name__)
        for video_name, video_path in self.in_videos.items():
            video_meta = get_video_meta_data(video_path=video_path)
            bg_meta = get_video_meta_data(video_path=bg_videos[video_name])
            if video_meta['resolution_str'] != bg_meta['resolution_str']:
                msg = f'The video, and background reference video, for {video_name} have different resolutions: {video_meta["resolution_str"]} vs {bg_meta["resolution_str"]}'
                self.set_status_bar_panel(text=msg, fg='red')
                raise InvalidVideoFileError(msg=msg, source=self.__class__.__name__)

    def _apply_bg_dir(self, bg_dir: Union[str, os.PathLike]):
        check_if_dir_exists(in_dir=bg_dir)
        self.bg_videos = find_all_videos_in_directory(directory=bg_dir, as_dict=True, raise_error=False)
        self._check_bg_videos(bg_videos=self.bg_videos, videos=self.videos)
        for video_name, video_path in self.in_videos.items():
            self.videos[video_name]['bg_file'].filePath.set(self.bg_videos[video_name])

    def _get_bg_videos(self) -> dict:
        bg_videos = {}
        for video_name in self.videos.keys():
            if not os.path.isfile(self.videos[video_name]['bg_file'].file_path):
                msg = f'The background reference file for {video_name} is not a valid file: {self.videos[video_name]["bg_file"].file_path}.'
                self.set_status_bar_panel(text=msg, fg='red')
                raise InvalidVideoFileError(msg=msg, source=self.__class__.__name__)
            bg_videos[video_name] = self.videos[video_name]['bg_file'].file_path
        self._check_bg_videos(bg_videos=bg_videos, videos=self.in_videos)
        return bg_videos

    def get_roi_definitions(self, video_name: str, coordinates_path: Union[str, os.PathLike]) -> Union[MultiPolygon, None]:
        """ Convert all ROI definitions for specified video into a single multipolygon """
        if os.path.isfile(coordinates_path):
            _, _, _, roi_dict, _, _, _ = get_roi_data(roi_path=coordinates_path, video_name=video_name)
            polygons = []
            for _, roi_data in roi_dict.items():
                if roi_data['Shape_type'] != 'circle':
                    tags = np.array(list(roi_data['Tags'].values()))
                    polygons.append(Polygon(tags))
                else:
                    circle = Point((roi_data['centerX'], roi_data['centerY'])).buffer(roi_data['radius'])
                    polygons.append(Polygon(circle))
            return MultiPolygon(polygons)
        else:
            return None


    def _quick_check(self, video_name: str):
        video_path = self.in_videos[video_name]
        bg_video_path = self.videos[video_name]["bg_file"].file_path
        threshold = int((int(self.videos[video_name]["threshold_dropdown"].getChoices()) / 100) * 255)
        inclusion_zones = self.get_roi_definitions(video_name=video_name, coordinates_path=self.roi_inclusion_store)
        video_meta_data = get_video_meta_data(video_path=video_path)
        close_kernel_iterations = int(self.close_iterations_dropdown_dropdown.get_value())
        open_kernel_iterations = int(self.open_iterations_dropdown.get_value())
        max_dim = max(video_meta_data['width'], video_meta_data['height'])
        if self.videos[video_name]['close_kernel'].get_value() == 'None':
            close_kernel = None
        else:
            w = ((max_dim * float(self.videos[video_name]['close_kernel'].get_value())) / 100) / 5
            h = ((max_dim * float(self.videos[video_name]['close_kernel'].get_value())) / 100) / 5
            close_kernel = (int(max(h, 1)), int(max(w, 1)))
        if self.videos[video_name]['open_kernel'].get_value() == 'None':
            open_kernel = None
        else:
            w = ((max_dim * float(self.videos[video_name]['open_kernel'].get_value())) / 100) / 5
            h = ((max_dim * float(self.videos[video_name]['open_kernel'].get_value())) / 100) / 5
            open_kernel = (int(max(h, 1)), int(max(w, 1)))

        if not os.path.isfile(bg_video_path):
            msg = f'The selected background video selected for {video_name} does not exist'
            self.set_status_bar_panel(text=msg, fg='red')
            raise InvalidVideoFileError(msg=msg, source=self.__class__.__name__)
        else:
            _ = BlobQuickChecker(video_path=video_path,
                                 bg_video_path=bg_video_path,
                                 threshold=threshold,
                                 method='absolute',
                                 inclusion_zones=inclusion_zones,
                                 status_label=self.status_bar,
                                 close_kernel_size=close_kernel,
                                 close_kernel_iterations=close_kernel_iterations,
                                 open_kernel_size=open_kernel,
                                 open_kernel_iterations=open_kernel_iterations)

    def _initialize_run(self):
        _ = self._get_bg_videos()
        out = {'input_dir':  self.input_dir,
               'output_dir': self.output_dir,
               'gpu': str_2_bool(self.use_gpu_dropdown.getChoices()),
               'core_cnt': int(self.core_cnt_dropdown.getChoices()),
               'vertice_cnt': int(self.vertice_cnt_dropdown.get_value()),
               'close_iterations': int(self.close_iterations_dropdown_dropdown.get_value()),
               'open_iterations': int(self.open_iterations_dropdown.get_value()),
               'save_bg_videos': str_2_bool(self.save_videos_dropdown.getChoices())}

        video_out = {}
        for video_name, video_data in self.videos.items():
            video_meta_data = get_video_meta_data(video_path=self.in_videos[video_name])
            video_out[video_name] = {}
            video_out[video_name]['video_path'] = self.in_videos[video_name]
            video_out[video_name]['threshold'] = int((int(self.videos[video_name]["threshold_dropdown"].getChoices()) / 100) * 255)
            video_out[video_name]['smoothing_time'] = self.videos[video_name]["smoothing_time_dropdown"].getChoices()
            video_out[video_name]['buffer_size'] = self.videos[video_name]["buffer_dropdown"].getChoices()
            video_out[video_name]['reference'] = self.videos[video_name]["bg_file"].file_path
            video_out[video_name]['inclusion_zones'] = self.get_roi_definitions(video_name=video_name, coordinates_path=self.roi_inclusion_store)
            video_out[video_name]['window_size'] = None
            video_out[video_name]['close_kernel'] = self.videos[video_name]["close_kernel"].get_value()
            video_out[video_name]['open_kernel'] = self.videos[video_name]["open_kernel"].get_value()
            max_dim = max(video_meta_data['width'], video_meta_data['height'])
            if video_out[video_name]['smoothing_time'] == 'None':
                video_out[video_name]['smoothing_time'] = None
            else:
                video_out[video_name]['smoothing_time'] = int(float(video_out[video_name]['smoothing_time']) * 1000)
            if video_out[video_name]['buffer_size'] == 'None':
                video_out[video_name]['buffer_size'] = None
            else:
                video_out[video_name]['buffer_size'] = int(video_out[video_name]['buffer_size'])
            if video_out[video_name]['close_kernel'] == 'None':
                video_out[video_name]['close_kernel'] = None
            else:
                w = ((max_dim * float(video_out[video_name]['close_kernel'])) / 100) / 4
                h = ((max_dim * float(video_out[video_name]['close_kernel'])) / 100) / 4
                k = (int(max(h, 1)), int(max(w, 1)))
                video_out[video_name]['close_kernel'] = tuple(k)
            if video_out[video_name]['open_kernel'] == 'None':
                video_out[video_name]['open_kernel'] = None
            else:
                w = ((max_dim * float(video_out[video_name]['open_kernel'])) / 100) / 4
                h = ((max_dim * float(video_out[video_name]['open_kernel'])) / 100) / 4
                k = (int(max(h, 1)), int(max(w, 1)))
                video_out[video_name]['open_kernel'] = tuple(k)

        remove_files(file_paths=[self.roi_inclusion_store], raise_error=False)
        out['video_data'] = video_out
        write_pickle(data=out, save_path=self.out_path)
        self.set_status_bar_panel(text=f'Starting blob detection, follow progress in OS terminal...({datetime.now().strftime("%H:%M:%S")})', fg='blue')

        sp = False if platform.system() != "Darwin" else True
        print(f'SimBA blob tracking subprocess: {sp} (platform: {platform.system()}).')
        if sp:
            cmd = f'python "{BLOB_EXECUTOR_PATH}" --data "{self.out_path}"'
            subprocess.run(cmd, check=True, shell=True)
        else:
            executor = BlobTrackingExecutor(data=self.out_path)
            executor.run()

#_ = BlobTrackingUI(input_dir=r'C:\troubleshooting\blob_track_tester\videos', output_dir=r'C:\troubleshooting\blob_track_tester\results')

#_ = BlobTrackingUI(input_dir=r'D:\water_maze', output_dir=r'D:\water_maze\data')


#_ = BlobTrackingUI(input_dir=r'D:\water_maze', output_dir=r'D:\water_maze\data')

#_ = BlobTrackingUI(input_dir=r'D:\OF_7', output_dir=r'D:\OF_7\data')

#_ = BlobTrackingUI(input_dir=r'/mnt/d/open_field_3/sample', output_dir=r"/mnt/d/open_field_3/sample/blob_data")
#_ = BlobTrackingUI(input_dir=r'D:\EPM\sample_2', output_dir=r"D:\EPM\sample_2\data_2")



#_ = BlobTrackingUI(input_dir=r'D:\open_field', output_dir=r"D:\open_field\data")




#_ = BlobTrackingUI(input_dir=r'D:\EPM\sampled', output_dir=r"D:\EPM\sampled\data")

#_ = BlobTrackingUI(input_dir=r'D:\open_field', output_dir=r"D:\open_field\data")

#_ = BlobTrackingUI(input_dir=r'D:\open_field_3\sample', output_dir=r"D:\open_field_3\sample\blob_data")
# import numpy as np
# from shapely.geometry import MultiPolygon, Polygon, Point
#
# rois = {'inclusion_zones': {'rectangle': {'Video': '501_MA142_Gi_Saline_0515', 'Shape_type': 'rectangle', 'Name': 'rectangle', 'Color name': 'Red', 'Color BGR': (0, 0, 255), 'Thickness': 5, 'Center_X': 358, 'Center_Y': 150, 'topLeftX': 195, 'topLeftY': 108, 'Bottom_right_X': 522, 'Bottom_right_Y': 193, 'width': 327, 'height': 85, 'width_cm': 32.7, 'height_cm': 8.5, 'area_cm': 277.95, 'Tags': {'Center tag': (358, 150), 'Top left tag': (195, 108), 'Bottom right tag': (522, 193), 'Top right tag': (522, 108), 'Bottom left tag': (195, 193), 'Top tag': (358, 108), 'Right tag': (522, 150), 'Left tag': (195, 150), 'Bottom tag': (358, 193)}, 'Ear_tag_size': 15}, 'circle': {'Video': '501_MA142_Gi_Saline_0515', 'Shape_type': 'circle', 'Name': 'circle', 'Color name': 'Red', 'Color BGR': (0, 0, 255), 'Thickness': 5, 'centerX': 354, 'centerY': 349, 'radius': 79, 'radius_cm': 7.9, 'area_cm': 196.07, 'Tags': {'Center tag': (354, 349), 'Border tag': (275, 349)}, 'Ear_tag_size': 15}, 'polygon': {'Video': '501_MA142_Gi_Saline_0515', 'Shape_type': 'polygon', 'Name': 'polygon', 'Color name': 'Red', 'Color BGR': (0, 0, 255), 'Thickness': 5, 'Center_X': 571, 'Center_Y': 379, 'vertices': np.array([[676, 286],
#        [643, 462],
#        [474, 496],
#        [504, 276]]), 'center': (571, 379), 'area': 33381.0, 'max_vertice_distance': 291, 'area_cm': 33381.0, 'Tags': {'Tag_0': (676, 286), 'Tag_1': (643, 462), 'Tag_2': (474, 496), 'Tag_3': (504, 276), 'Center_tag': (571, 379)}, 'Ear_tag_size': 15}}}
#
# inclusion_zones = rois['inclusion_zones']
# all_tags, polygons = [], []
# for zone_name, zone_data in inclusion_zones.items():
#     if zone_data['Shape_type'] != 'circle':
#         tags = zone_data['Tags']
#         tags = np.array(list(tags.values()))
#         polygons.append(Polygon(tags))
#     else:
#         circle = Point((zone_data['centerX'], zone_data['centerY'])).buffer(zone_data['radius'])
#         polygons.append(Polygon(circle))
#     multi_polygon = MultiPolygon(polygons)


#
# multipolygon = MultiPolygon(all_tags)







# tags = {'Tags': {'Center tag': (365, 293), 'Top left tag': (204, 131), 'Bottom right tag': (526, 455), 'Top right tag': (526, 131), 'Bottom left tag': (204, 455), 'Top tag': (365, 131), 'Right tag': (526, 293), 'Left tag': (204, 293), 'Bottom tag': (365, 455)}}
#
# import numpy as np
#