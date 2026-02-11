__author__ = "Simon Nilsson; sronilsson@gmail.com"

import multiprocessing
import os
from tkinter import *
from typing import Union

import numpy as np

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.heat_mapper_clf import HeatMapperClfSingleCore
from simba.plotting.heat_mapper_clf_mp import HeatMapperClfMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon,
                                        SimbaButton, SimbaCheckbox,
                                        SimBADropDown, Entry_Box)
from simba.utils.checks import check_if_filepath_list_is_empty, check_int, check_if_string_value_is_valid_video_timestamp, check_that_hhmmss_start_is_before_end
from simba.utils.enums import Formats, Links, Paths
from simba.utils.read_write import get_file_name_info_in_directory, str_2_bool
from simba.utils.lookups import get_named_colors
AUTO = "auto"
VIDEO_FRM = 'VIDEO FRAME'
VIDEO = 'VIDEO'
HEATMAP_BG_OPTIONS = ['NONE', VIDEO, VIDEO_FRM]

class HeatmapClfPopUp(PopUpMixin, ConfigReader):
    """
    :example:
    >>> _ = HeatmapClfPopUp(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")
    """

    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.data_path = os.path.join(self.project_path, Paths.MACHINE_RESULTS_DIR.value)

        self.files_found_dict = get_file_name_info_in_directory(directory=self.data_path, file_type=self.file_type)
        check_if_filepath_list_is_empty( filepaths=list(self.files_found_dict.keys()), error_msg="SIMBA ERROR: Zero files found in the project_folder/csv/machine_results directory. ",)
        PopUpMixin.__init__(self, title="CREATE CLASSIFICATION HEATMAP PLOTS", icon='heatmap')
        max_scales_option = list(np.arange(5, 105, 5))
        max_scales_option.insert(0, AUTO)
        min_scales_option = list(np.arange(5, 500, 5))
        min_scales_option.insert(0, 'NONE')
        line_color_options = get_named_colors()
        line_color_options.insert(0, 'NONE')


        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name='style', icon_link=Links.HEATMAP_CLF.value)

        self.palette_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.palette_options, label='PALETTE: ', label_width=30, dropdown_width=35, value='jet', img='palette_small', tooltip_key='HEATMAP_CLF_PALETTE')
        self.shading_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.shading_options, label='SHADING: ', label_width=30, dropdown_width=35, value='flat', img='shade', tooltip_key='HEATMAP_CLF_SHADING')
        self.clf_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.clf_names, label='CLASSIFIER: ', label_width=30, dropdown_width=35, value=self.clf_names[0], img='forest', tooltip_key='HEATMAP_CLF_CLASSIFIER')
        self.bp_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.body_parts_lst, label='BODY-PART: ', label_width=30, dropdown_width=35, value=self.body_parts_lst[0], img='pose', tooltip_key='HEATMAP_CLF_BODY_PART')
        self.max_time_scale_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=max_scales_option, label='MAX TIME SCALE (S): ', label_width=30, dropdown_width=35, value=AUTO, img='timer', tooltip_key='HEATMAP_CLF_MAX_TIME_SCALE')
        self.bin_size_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.heatmap_bin_size_options, label='BIN SIZE (MM): ', label_width=30, dropdown_width=35, value="20×20", img='size_black', tooltip_key='HEATMAP_CLF_BIN_SIZE')
        self.min_timescale_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=min_scales_option, label='MINIMUM SECONDS: ', label_width=30, dropdown_width=35, value="NONE", img='timer_2', tooltip_key='HEATMAP_CLF_MIN_SECONDS')
        self.legend_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW TIME COLOR LEGEND: ', label_width=30, dropdown_width=35, value='TRUE', img='palette_small', tooltip_key='HEATMAP_CLF_SHOW_LEGEND')
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZATION SETTINGS", icon_name='eye', icon_link=Links.HEATMAP_CLF.value)
        self.core_cnt_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(2, self.cpu_cnt)), label='CPU CORE COUNT: ', label_width=30, dropdown_width=35, value=int(self.cpu_cnt/3), img='cpu_small', tooltip_key='HEATMAP_CLF_CPU_CORES')
        self.line_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=line_color_options, label='LINE COLOR: ', label_width=30, dropdown_width=35, value='NONE', img='line', tooltip_key='HEATMAP_LOCATION_LINE_COLOR')



        heatmap_frames_cb, self.heatmap_frames_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE FRAMES', txt_img='frames', tooltip_key='HEATMAP_CLF_CREATE_FRAMES')
        heatmap_videos_cb, self.heatmap_videos_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE VIDEOS', txt_img='video', tooltip_key='HEATMAP_CLF_CREATE_VIDEOS')
        heatmap_last_frm_cb, self.heatmap_last_frm_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE LAST FRAME', txt_img='finish', val=True, tooltip_key='HEATMAP_CLF_CREATE_LAST_FRAME')

        self.bg_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="HEATMAP BACKGROUND SETTINGS", icon_name='background', icon_link=Links.HEATMAP_CLF.value)
        self.bg_dropdown = SimBADropDown(parent=self.bg_frm, dropdown_options=HEATMAP_BG_OPTIONS, label='HEATMAP BACKGROUND: ', label_width=30, dropdown_width=35, value='NONE', img='background', command=self._set_select_bg_frm, tooltip_key='HEATMAP_CLF_BACKGROUND')

        self.time_slice_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="TIME PERIOD", icon_name='timer_2', icon_link=Links.HEATMAP_LOCATION.value)
        self.plot_time_period_cb, self.time_period_val = SimbaCheckbox(parent=self.time_slice_frm, txt='PLOT SELECT TIME-PERIOD', val=FALSE, txt_img='timer_2', cmd=self._set_timeslice_state, tooltip_key='HEATMAP_LOCATION_PLOT_TIME_PERIOD')
        self.start_time_eb = Entry_Box(parent=self.time_slice_frm, fileDescription='START TIME:', labelwidth=20, entry_box_width=15, justify='center', value='00:00:00', status=DISABLED, tooltip_key='HEATMAP_LOCATION_START_TIME')
        self.end_time_eb = Entry_Box(parent=self.time_slice_frm, fileDescription='END TIME:', labelwidth=20, entry_box_width=15, justify='center',  value='00:00:30', status=DISABLED, tooltip_key='HEATMAP_LOCATION_END_TIME')


        self.run_frm = LabelFrame(self.main_frm, text="RUN", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black",)
        self.run_single_video_frm = LabelFrame(self.run_frm, text="SINGLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black",)

        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, txt="CREATE SINGLE HEATMAP", txt_clr="blue", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.__create_heatmap_plots, cmd_kwargs={'multiple_videos': False})
        self.single_video_dropdown = SimBADropDown(parent=self.run_single_video_frm, label='VIDEO:', dropdown_options=list(self.files_found_dict.keys()), label_width=20, dropdown_width=35, value=list(self.files_found_dict.keys())[0], tooltip_key='HEATMAP_CLF_SINGLE_VIDEO')
        self.run_multiple_videos = LabelFrame(self.run_frm, text="MULTIPLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black",)

        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, txt=f"CREATE MULTIPLE HEATMAPS ({len(list(self.files_found_dict.keys()))} files found)", img='rocket', txt_clr="blue", font=Formats.FONT_REGULAR.value, cmd=self.__create_heatmap_plots, cmd_kwargs={'multiple_videos': True})
        self.style_settings_frm.grid(row=0, sticky=NW)
        self.palette_dropdown.grid(row=0, sticky=NW)
        self.shading_dropdown.grid(row=1, sticky=NW)
        self.clf_dropdown.grid(row=2, sticky=NW)
        self.bp_dropdown.grid(row=3, sticky=NW)
        self.line_dropdown.grid(row=4, sticky=NW)
        self.legend_dropdown.grid(row=5, sticky=NW)
        self.max_time_scale_dropdown.grid(row=6, sticky=NW)
        self.min_timescale_dropdown.grid(row=7, sticky=NW)
        self.bin_size_dropdown.grid(row=8, sticky=NW)

        self.bg_frm.grid(row=1, sticky=NW)
        self.bg_dropdown.grid(row=0, sticky=NW)

        self.time_slice_frm.grid(row=2, sticky=NW)
        self.plot_time_period_cb.grid(row=0, sticky=NW)
        self.start_time_eb.grid(row=1, sticky=NW, column=0, padx=(0, 15))
        self.end_time_eb.grid(row=1, sticky=NW, column=1)

        self.settings_frm.grid(row=3, sticky=NW)
        self.core_cnt_dropdown.grid(row=0, sticky=NW)
        heatmap_frames_cb.grid(row=1, sticky=NW)
        heatmap_videos_cb.grid(row=2, sticky=NW)
        heatmap_last_frm_cb.grid(row=3, sticky=NW)

        self.run_frm.grid(row=4, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW, padx=(0, 15))
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        self.main_frm.mainloop()


    def _set_timeslice_state(self):
        status = NORMAL if self.time_period_val.get() else DISABLED
        self.start_time_eb.set_state(setstatus=status)
        self.end_time_eb.set_state(setstatus=status)

    def _set_select_bg_frm(self, selection):
        for widget_name in ['opacity_dropdown', 'keypoint_dropdown', 'frm_id_eb']:
            if hasattr(self, widget_name):
                getattr(self, widget_name).destroy()
        if selection == VIDEO:
            self.opacity_dropdown = SimBADropDown(parent=self.bg_frm, dropdown_options=list(np.arange(5, 105, 5)), label='HEATMAP OPACITY (%): ', label_width=30, dropdown_width=35, value=50, img='opacity', tooltip_key='HEATMAP_CLF_OPACITY')
            self.keypoint_dropdown = SimBADropDown(parent=self.bg_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW KEYPOINT: ', label_width=30, dropdown_width=35, value='TRUE', img='pose', tooltip_key='HEATMAP_CLF_SHOW_KEYPOINT')
            self.opacity_dropdown.grid(row=1, sticky=NW)
            self.keypoint_dropdown.grid(row=2, sticky=NW)
        elif selection == VIDEO_FRM:
            self.opacity_dropdown = SimBADropDown(parent=self.bg_frm, dropdown_options=list(np.arange(5, 105, 5)), label='HEATMAP OPACITY (%): ', label_width=30, dropdown_width=35, value=50, img='opacity', tooltip_key='HEATMAP_CLF_OPACITY')
            self.frm_id_eb = Entry_Box(parent=self.bg_frm, fileDescription='FRAME NUMBER:', labelwidth=30, entry_box_width=35, justify='center', validation='numeric', img='abacus_2', tooltip_key='HEATMAP_CLF_FRAME_NUMBER', value=0)
            self.opacity_dropdown.grid(row=1, sticky=NW)
            self.frm_id_eb.grid(row=2, sticky=NW)


    def __create_heatmap_plots(self, multiple_videos: bool):
        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

        max_scale = int(self.max_time_scale_dropdown.getChoices().split("×")[0]) if self.max_time_scale_dropdown.getChoices() != AUTO else AUTO
        min_seconds = None if self.min_timescale_dropdown.get_value() == 'NONE' else int(self.min_timescale_dropdown.get_value())
        bin_size = int(self.bin_size_dropdown.getChoices().split("×")[0])
        core_cnt = int(self.core_cnt_dropdown.get_value())
        show_legend = str_2_bool(self.legend_dropdown.get_value())
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

        style_attr = {"palette": self.palette_dropdown.getChoices(),
                      "shading": self.shading_dropdown.getChoices(),
                      "max_scale": max_scale,
                      "bin_size": bin_size}

        if core_cnt == 1:
            heatmapper_clf = HeatMapperClfSingleCore(
                config_path=self.config_path,
                style_attr=style_attr,
                final_img_setting=self.heatmap_last_frm_var.get(),
                video_setting=self.heatmap_videos_var.get(),
                frame_setting=self.heatmap_frames_var.get(),
                bodypart=self.bp_dropdown.getChoices(),
                data_paths=data_paths,
                clf_name=self.clf_dropdown.getChoices(),
            )

            heatmapper_clf_processor = multiprocessing.Process(heatmapper_clf.run())
            heatmapper_clf_processor.start()

        else:
            heatmapper_clf = HeatMapperClfMultiprocess(
                config_path=self.config_path,
                style_attr=style_attr,
                final_img_setting=self.heatmap_last_frm_var.get(),
                video_setting=self.heatmap_videos_var.get(),
                frame_setting=self.heatmap_frames_var.get(),
                bodypart=self.bp_dropdown.getChoices(),
                min_seconds=min_seconds,
                bg_img=bg_img,
                line_clr=line_clr,
                show_keypoint=show_kp,
                heatmap_opacity=heatmap_opacity,
                data_paths=data_paths,
                time_slice=time_slice,
                show_legend=show_legend,
                clf_name=self.clf_dropdown.getChoices(),
                core_cnt=core_cnt,
            )

            heatmapper_clf.run()

#_ = HeatmapClfPopUp(config_path=r"E:\troubleshooting\mitra_emergence\project_folder\project_config.ini")

#_ = HeatmapClfPopUp(config_path=r"D:\troubleshooting\maplight_ri\project_folder\project_config.ini")

#_ = HeatmapClfPopUp(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")