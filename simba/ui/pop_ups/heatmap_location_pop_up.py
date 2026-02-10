__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from tkinter import *
from typing import Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.heat_mapper_location import HeatmapperLocationSingleCore
from simba.plotting.heat_mapper_location_mp import \
    HeatMapperLocationMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, Entry_Box,
                                        SimbaButton, SimbaCheckbox,
                                        SimBADropDown)
from simba.utils.checks import (check_if_filepath_list_is_empty,
                                check_if_string_value_is_valid_video_timestamp,
                                check_int,
                                check_that_hhmmss_start_is_before_end)
from simba.utils.enums import Formats, Links, Paths
from simba.utils.lookups import get_named_colors
from simba.utils.read_write import get_file_name_info_in_directory, str_2_bool

AUTO = 'AUTO'
VIDEO_FRM = 'VIDEO FRAME'
VIDEO = 'VIDEO'
HEATMAP_BG_OPTIONS = ['NONE', VIDEO, VIDEO_FRM]


class HeatmapLocationPopup(PopUpMixin, ConfigReader):

    """
    :example:
    >>> _ = HeatmapLocationPopup(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")
    >>> _ = HeatmapLocationPopup(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")

    """
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.data_path = os.path.join(self.project_path, Paths.OUTLIER_CORRECTED.value)
        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)
        check_if_filepath_list_is_empty(filepaths=list(self.files_found_dict.keys()),error_msg="SIMBA ERROR: Zero files found in the project_folder/csv/outlier_corrected_movement_location directory. ",)
        PopUpMixin.__init__(self, title="HEATMAPS: LOCATION", icon='heatmap')
        max_scales = list(np.arange(5, 105, 5))
        max_scales.insert(0, AUTO)
        min_seconds_options = list(np.arange(5, 500, 5))
        min_seconds_options.insert(0, 'NONE')
        line_color_options = get_named_colors()
        line_color_options.insert(0, 'NONE')

        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name='settings', icon_link=Links.HEATMAP_LOCATION.value)

        self.palette_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.palette_options, label='PALETTE: ', label_width=25, dropdown_width=30, value='jet', img='color_wheel', tooltip_key='HEATMAP_LOCATION_PALETTE')
        self.shading_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.shading_options, label='SHADING: ', label_width=25, dropdown_width=30, value=self.shading_options[1], img='shade', tooltip_key='HEATMAP_LOCATION_SHADING')
        self.bp_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.body_parts_lst, label='BODY-PART: ', label_width=25, dropdown_width=30, value=self.body_parts_lst[0],img='pose', tooltip_key='HEATMAP_LOCATION_BODY_PART')
        self.max_time_scale_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=max_scales, label='MAX TIME SCALE (S): ', label_width=25, dropdown_width=30, value=max_scales[0], img='timer_2', tooltip_key='HEATMAP_LOCATION_MAX_TIME_SCALE')
        self.bin_size_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.heatmap_bin_size_options, label='BIN SIZE (MM): ', label_width=25, dropdown_width=30, value="20×20", img='rectangle_red', tooltip_key='HEATMAP_LOCATION_BIN_SIZE')
        self.legend_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW TIME COLOR LEGEND: ', label_width=25, dropdown_width=30, value='TRUE', img='palette_small', tooltip_key='HEATMAP_LOCATION_SHOW_LEGEND')
        self.min_seconds_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=min_seconds_options, label='MINIMUM SECONDS: ', label_width=25, dropdown_width=30, value='NONE', img='timer', tooltip_key='HEATMAP_LOCATION_MIN_SECONDS')
        self.bg_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=HEATMAP_BG_OPTIONS, label='HEATMAP BACKGROUND: ', label_width=25, dropdown_width=30, value='NONE', img='background', command=self._set_select_bg_frm, tooltip_key='HEATMAP_LOCATION_BACKGROUND')
        self.line_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=line_color_options, label='LINE COLOR: ', label_width=25, dropdown_width=30, value='white', img='line', command=self._set_select_bg_frm)



        self.time_slice_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="TIME PERIOD", icon_name='timer_2', icon_link=Links.HEATMAP_LOCATION.value)
        self.plot_time_period_cb, self.time_period_val = SimbaCheckbox(parent=self.time_slice_frm, txt='PLOT SELECT TIME-PERIOD', val=FALSE, txt_img='timer_2', cmd=self._set_timeslice_state, tooltip_key='HEATMAP_LOCATION_PLOT_TIME_PERIOD')
        self.start_time_eb = Entry_Box(parent=self.time_slice_frm, fileDescription='START TIME:', labelwidth=20, entry_box_width=15, justify='center', value='00:00:00', status=DISABLED, tooltip_key='HEATMAP_LOCATION_START_TIME')
        self.end_time_eb = Entry_Box(parent=self.time_slice_frm, fileDescription='END TIME:', labelwidth=20, entry_box_width=15, justify='center',  value='00:00:30', status=DISABLED, tooltip_key='HEATMAP_LOCATION_END_TIME')

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZATION SETTINGS", icon_name='eye', icon_link=Links.HEATMAP_LOCATION.value, pady=5, padx=5)
        self.multiprocess_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(2, self.cpu_cnt)), label='CPU CORES: ', label_width=25, dropdown_width=30, value=int(self.cpu_cnt/3), img='cpu_small', tooltip_key='HEATMAP_LOCATION_CPU_CORES')
        heatmap_frames_cb, self.heatmap_frames_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE FRAMES', txt_img='frames', tooltip_key='HEATMAP_LOCATION_CREATE_FRAMES')
        heatmap_videos_cb, self.heatmap_videos_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE VIDEOS', txt_img='video', tooltip_key='HEATMAP_LOCATION_CREATE_VIDEOS')
        heatmap_last_frm_cb, self.heatmap_last_frm_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE LAST FRAME', txt_img='finish', tooltip_key='HEATMAP_LOCATION_CREATE_LAST_FRAME', val=True)

        self.run_frm = LabelFrame(self.main_frm, text="RUN", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black")
        self.run_single_video_frm = LabelFrame(self.run_frm, text="SINGLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black")

        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, txt="CREATE SINGLE HEATMAP", txt_clr="blue", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.__create_heatmap_plots, cmd_kwargs={'multiple_videos': False})
        self.single_video_dropdown = SimBADropDown(parent=self.run_single_video_frm, label="VIDEO:", dropdown_options=list(self.files_found_dict.keys()), label_width=12, dropdown_width=40, value=list(self.files_found_dict.keys())[0], tooltip_key='HEATMAP_LOCATION_SINGLE_VIDEO')
        self.run_multiple_videos = LabelFrame(self.run_frm, text="MULTIPLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black")

        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, txt=f"CREATE MULTIPLE HEATMAPS ({len(list(self.files_found_dict.keys()))} video(s) found)", img='rocket', txt_clr="blue", font=Formats.FONT_REGULAR.value, cmd=self.__create_heatmap_plots, cmd_kwargs={'multiple_videos': True})
        self.style_settings_frm.grid(row=0, sticky=NW)
        self.palette_dropdown.grid(row=0, sticky=NW)
        self.shading_dropdown.grid(row=1, sticky=NW)
        self.bp_dropdown.grid(row=2, sticky=NW)
        self.legend_dropdown.grid(row=3, sticky=NW)
        self.max_time_scale_dropdown.grid(row=4, sticky=NW)
        self.min_seconds_dropdown.grid(row=5, sticky=NW)
        self.line_dropdown.grid(row=6, sticky=NW)
        self.bg_dropdown.grid(row=7, sticky=NW)
        self.bin_size_dropdown.grid(row=8, sticky=NW)

        self.time_slice_frm.grid(row=2, sticky=NW)
        self.plot_time_period_cb.grid(row=0, sticky=NW)
        self.start_time_eb.grid(row=1, sticky=NW, column=0, padx=(0, 15))
        self.end_time_eb.grid(row=1, sticky=NW, column=1)

        self.settings_frm.grid(row=3, sticky=NW)
        heatmap_frames_cb.grid(row=0, sticky=NW)
        heatmap_videos_cb.grid(row=1, sticky=NW)
        heatmap_last_frm_cb.grid(row=2, sticky=NW)
        self.multiprocess_dropdown.grid(row=3, column=0, sticky=NW)

        self.run_frm.grid(row=4, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)
        self.main_frm.mainloop()

    def _set_timeslice_state(self):
        status = NORMAL if self.time_period_val.get() else DISABLED
        self.start_time_eb.set_state(setstatus=status)
        self.end_time_eb.set_state(setstatus=status)

    def _set_select_bg_frm(self, selection):
        if hasattr(self, 'bg_select_frm'):
            self.bg_select_frm.destroy()
        if selection == VIDEO:
            self.bg_select_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="BACKGROUND SETTINGS", icon_name='video_2', icon_link=Links.HEATMAP_LOCATION.value, pady=5, padx=5)
            self.opacity_dropdown = SimBADropDown(parent=self.bg_select_frm, dropdown_options=list(np.arange(5, 105, 5)), label='HEATMAP OPACITY (%): ', label_width=25, dropdown_width=30, value=50, img='opacity', tooltip_key='HEATMAP_LOCATION_OPACITY')
            self.keypoint_dropdown = SimBADropDown(parent=self.bg_select_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW KEYPOINT: ', label_width=25, dropdown_width=30, value='TRUE', img='pose', tooltip_key='HEATMAP_LOCATION_SHOW_KEYPOINT')
            self.bg_select_frm.grid(row=1, sticky=NW)
            self.opacity_dropdown.grid(row=0, sticky=NW)
            self.keypoint_dropdown.grid(row=1, sticky=NW)
        elif selection == VIDEO_FRM:
            self.bg_select_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="BACKGROUND SETTINGS", icon_name='video_2', icon_link=Links.HEATMAP_LOCATION.value, pady=5, padx=5)
            self.opacity_dropdown = SimBADropDown(parent=self.bg_select_frm, dropdown_options=list(np.arange(5, 105, 5)), label='HEATMAP OPACITY (%): ', label_width=25, dropdown_width=30, value=50, img='opacity', tooltip_key='HEATMAP_LOCATION_OPACITY')
            self.frm_id_eb = Entry_Box(parent=self.bg_select_frm, fileDescription='FRAME NUMBER:', labelwidth=25, entry_box_width=30, justify='center', validation='numeric', img='abacus_2', tooltip_key='HEATMAP_LOCATION_VIDEO_FRAME', value=0)
            self.bg_select_frm.grid(row=1, sticky=NW)
            self.opacity_dropdown.grid(row=0, sticky=NW)
            self.frm_id_eb.grid(row=1, sticky=NW)

    def __create_heatmap_plots(self, multiple_videos: bool):
        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        if self.max_time_scale_dropdown.getChoices() != AUTO:
            max_scale = int(float(self.max_time_scale_dropdown.getChoices()))
        else:
            max_scale = AUTO.lower()

        bin_size = int(self.bin_size_dropdown.getChoices().split("×")[0])
        style_attr = {"palette": self.palette_dropdown.getChoices(),
                      "shading": self.shading_dropdown.getChoices(),
                      "max_scale": max_scale,
                      "bin_size": bin_size}
        cpu_cores = int(self.multiprocess_dropdown.get_value())
        show_legend = str_2_bool(self.legend_dropdown.get_value())
        min_seconds = None if self.min_seconds_dropdown.get_value() == 'NONE' else int(self.min_seconds_dropdown.get_value())
        time_slice, show_kp, heatmap_opacity, bg_img = None, False, None, None
        line_clr = None if self.line_dropdown.get_value() == 'NONE' else self.line_dropdown.get_value()
        if self.bg_dropdown.get_value() == VIDEO:
            show_kp = str_2_bool(self.keypoint_dropdown.get_value())
            heatmap_opacity = float(self.opacity_dropdown.get_value()) / 100
            bg_img = -1
        elif self.bg_dropdown.get_value() == VIDEO_FRM:
            heatmap_opacity = float(self.opacity_dropdown.get_value()) / 100
            frm_id = self.frm_id_eb.entry_get
            check_int(name=f'{self.__class__.__name__} FRAME NUMBER', value=frm_id, min_value=0, allow_negative=False, raise_error=True)
            bg_img = int(frm_id)
        if self.time_period_val.get():
            start, end = self.start_time_eb.entry_get, self.end_time_eb.entry_get
            check_if_string_value_is_valid_video_timestamp(value=start, name=f'{self.__class__.__name__} START TIME', raise_error=True)
            check_if_string_value_is_valid_video_timestamp(value=end, name=f'{self.__class__.__name__} END TIME', raise_error=True)
            check_that_hhmmss_start_is_before_end(start_time=start, end_time=end, name=self.__class__.__name__, raise_error=True)
            time_slice = {'start_time': start, 'end_time': end}

        if cpu_cores == 1:
            heatmapper_clf = HeatmapperLocationSingleCore(config_path=self.config_path,
                                                          style_attr=style_attr,
                                                          final_img_setting=self.heatmap_last_frm_var.get(),
                                                          video_setting=self.heatmap_videos_var.get(),
                                                          frame_setting=self.heatmap_frames_var.get(),
                                                          bodypart=self.bp_dropdown.getChoices(),
                                                          data_paths=data_paths)

            heatmapper_clf.run()

        else:
            heatmapper_clf = HeatMapperLocationMultiprocess(config_path=self.config_path,
                                                            style_attr=style_attr,
                                                            final_img_setting=self.heatmap_last_frm_var.get(),
                                                            video_setting=self.heatmap_videos_var.get(),
                                                            frame_setting=self.heatmap_frames_var.get(),
                                                            bodypart=self.bp_dropdown.getChoices(),
                                                            data_paths=data_paths,
                                                            show_keypoint=show_kp,
                                                            line_clr=line_clr,
                                                            heatmap_opacity=heatmap_opacity,
                                                            bg_img=bg_img,
                                                            min_seconds=min_seconds,
                                                            core_cnt=cpu_cores,
                                                            show_legend=show_legend,
                                                            time_slice=time_slice)

            heatmapper_clf.run()



#_ = HeatmapLocationPopup(config_path=r"D:\troubleshooting\maplight_ri\project_folder\project_config.ini")
#_ = HeatmapLocationPopup(config_path=r"E:\troubleshooting\mitra_emergence_hour\project_folder\project_config.ini")
#_ = HeatmapLocationPopup(config_path=r"E:\troubleshooting\mitra_emergence\project_folder\project_config.ini")