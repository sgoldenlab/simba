import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.roi_directing_visualizer import DirectingROIVisualizer
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon,
                                        CreateToolTip, Entry_Box, SimbaButton,
                                        SimbaCheckbox, SimBADropDown)
from simba.utils.checks import (check_if_string_value_is_valid_video_timestamp,
                                check_nvidea_gpu_available,
                                check_that_hhmmss_start_is_before_end)
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import (FrameRangeError, NoFilesFoundError,
                                NoROIDataError)
from simba.utils.lookups import find_closest_string, get_tooltips
from simba.utils.printing import stdout_warning
from simba.utils.read_write import (find_all_videos_in_directory,
                                    find_core_cnt, get_fn_ext, str_2_bool)

NOSE, EAR_LEFT, EAR_RIGHT = Keys.NOSE.value, Keys.EAR_LEFT.value, Keys.EAR_RIGHT.value


class DirectingROIVisualizerPopUp(ConfigReader, PopUpMixin):
    """
    Pop-up window for visualizing when animals are directing towards ROIs.

    :param Union[str, os.PathLike] config_path: Path to SimBA project config file in Configparser format.

    :example:
    >>> _ = DirectingROIVisualizerPopUp(config_path=r"C:\\troubleshooting\\two_black_animals_14bp\\project_folder\\project_config.ini")
    """

    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.video_file_paths = find_all_videos_in_directory(directory=self.video_dir, as_dict=True)
        self.video_list = [k for k in self.video_file_paths.keys()]
        if len(self.video_list) == 0:
            raise NoFilesFoundError(msg=f"No videos found in SimBA project {self.video_dir} directory.", source=self.__class__.__name__)
        if not os.path.isfile(self.roi_coordinates_path):
            raise NoROIDataError(msg=f"No ROI data found in SimBA project. Draw ROIs before visualizing ROI directing.", source=self.__class__.__name__)
        self.read_roi_data()
        self.video_names_w_rois = list(self.video_names_w_rois)
        self.video_file_paths = {k: v for k, v in self.video_file_paths.items() if k in self.video_names_w_rois}
        self.video_list = [k for k in self.video_file_paths.keys()]
        if len(self.video_list) == 0:
            raise NoROIDataError(msg=f"None of the imported videos have ROI data. Draw ROIs before visualizing ROI directing.", source=self.__class__.__name__)
        self.max_video_name_len = len(max(self.video_list, key=len))
        self.gpu_available = NORMAL if check_nvidea_gpu_available() else DISABLED
        PopUpMixin.__init__(self, title="VISUALIZE ROI DIRECTIONALITY", icon='direction')

        if self.animal_cnt > 1:
            bp_names = list(set([x[:-2] for x in self.body_parts_lst]))
        else:
            bp_names = list(set(self.body_parts_lst))
        nose_guess = find_closest_string(target=NOSE, string_list=bp_names)[0]
        ear_left_guess = find_closest_string(target=EAR_LEFT, string_list=bp_names)[0]
        ear_right_guess = find_closest_string(target=EAR_RIGHT, string_list=bp_names)[0]

        self.bp_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT BODY-PARTS", icon_name='pose', icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)
        self.ear_left_dropdown = SimBADropDown(parent=self.bp_frm, dropdown_options=bp_names, label_width=30, dropdown_width=25, value=ear_left_guess, label='LEFT EAR BODY-PART NAME:', img='left_ear')
        self.ear_right_dropdown = SimBADropDown(parent=self.bp_frm, dropdown_options=bp_names, label_width=30, dropdown_width=25, value=ear_right_guess, label='RIGHT EAR BODY-PART NAME:', img='ear_right')
        self.nose_dropdown = SimBADropDown(parent=self.bp_frm, dropdown_options=bp_names, label_width=30, dropdown_width=25, value=nose_guess, label='NOSE BODY-PART NAME:', img='nose')
        self.bp_frm.grid(row=0, column=0, sticky=NW)
        self.ear_left_dropdown.grid(row=0, column=0, sticky=NW)
        self.ear_right_dropdown.grid(row=1, column=0, sticky=NW)
        self.nose_dropdown.grid(row=2, column=0, sticky=NW)

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings', icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)
        self.direction_style_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['LINES', 'FUNNEL'], label='DIRECTION STYLE:', label_width=30, dropdown_width=25, value='LINES', img='direction_3', tooltip_key='ROI_DIRECTING_VIS_DIRECTION_STYLE')
        self.border_clr_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(self.colors_dict.keys()), label='BORDER COLOR:', label_width=30, dropdown_width=25, value='Black', img='fill', tooltip_key='BORDER_BG_COLOR')
        self.direction_clr_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(self.colors_dict.keys()), label='DIRECTION COLOR:', label_width=30, dropdown_width=25, value='Red', img='fill', tooltip_key='ROI_DIRECTING_VIS_DIRECTION_COLOR')
        self.show_pose_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW POSE:', label_width=30, dropdown_width=25, value='TRUE', img='pose', tooltip_key='ROI_TRACKING_SHOW_POSE')
        self.show_roi_centers_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW ROI CENTERS:', label_width=30, dropdown_width=25, value='TRUE', img='bullseye', tooltip_key='ROI_FEATURES_SHOW_ROI_CENTERS')
        self.show_animal_names_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label='SHOW ANIMAL NAMES:', label_width=30, dropdown_width=25, value='FALSE', img='id_card', tooltip_key='ROI_TRACKING_SHOW_ANIMAL_NAMES')
        self.circle_size_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['AUTO'] + list(range(1, 51)), label='CIRCLE SIZE:', label_width=30, dropdown_width=25, value='AUTO', img='circle_small', tooltip_key='ROI_DIRECTING_VIS_CIRCLE_SIZE')
        self.line_thickness_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['AUTO'] + list(range(1, 51)), label='DIRECTION LINE THICKNESS:', label_width=30, dropdown_width=25, value='AUTO', img='size_black', tooltip_key='ROI_DIRECTING_VIS_LINE_THICKNESS')
        self.gpu_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label='USE GPU:', label_width=30, dropdown_width=25, value='FALSE', state=self.gpu_available, img='gpu_3', tooltip_key='USE_GPU')
        self.cpu_cnt_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(2, find_core_cnt()[0] + 1)), label='CPU CORES:', label_width=30, dropdown_width=25, value=int(find_core_cnt()[0] / 2), img='cpu_small', tooltip_key='ROI_TRACKING_CPU_CORES')
        self.settings_frm.grid(row=1, column=0, sticky=NW)
        self.direction_style_dropdown.grid(row=0, column=0, sticky=NW)
        self.border_clr_dropdown.grid(row=1, column=0, sticky=NW)
        self.direction_clr_dropdown.grid(row=2, column=0, sticky=NW)
        self.show_pose_dropdown.grid(row=3, column=0, sticky=NW)
        self.show_roi_centers_dropdown.grid(row=4, column=0, sticky=NW)
        self.show_animal_names_dropdown.grid(row=5, column=0, sticky=NW)
        self.circle_size_dropdown.grid(row=6, column=0, sticky=NW)
        self.line_thickness_dropdown.grid(row=7, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=8, column=0, sticky=NW)
        self.cpu_cnt_dropdown.grid(row=9, column=0, sticky=NW)

        self.time_slice_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="TIME SEGMENT", icon_name='clip', icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)
        self.slice_var = BooleanVar(value=False)
        self.video_start_time_entry = Entry_Box(self.time_slice_frm, "START TIME (HH:MM:SS):", "35", entry_box_width=30, value="00:00:00", status=DISABLED, tooltip_key='PATH_PLOT_START_TIME')
        self.video_end_time_entry = Entry_Box(self.time_slice_frm, "END TIME (HH:MM:SS):", "35", entry_box_width=30, value="00:00:00", status=DISABLED, tooltip_key='PATH_PLOT_END_TIME')
        self.slice_cb = Checkbutton(self.time_slice_frm, text="Visualize ONLY defined time-segment", font=Formats.FONT_REGULAR_BOLD.value, variable=self.slice_var, command=lambda: self.enable_entrybox_from_checkbox(check_box_var=self.slice_var, entry_boxes=[self.video_start_time_entry, self.video_end_time_entry]))
        self.time_slice_frm.grid(row=2, column=0, sticky=NW)
        self.slice_cb.grid(row=0, column=0, sticky=NW)
        tooltips = get_tooltips()
        if 'ROI_DIRECTING_VIS_TIME_SEGMENT' in tooltips:
            CreateToolTip(widget=self.slice_cb, text=tooltips['ROI_DIRECTING_VIS_TIME_SEGMENT'])
        self.video_start_time_entry.grid(row=1, column=0, sticky=NW)
        self.video_end_time_entry.grid(row=2, column=0, sticky=NW)

        self.single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZE SINGLE VIDEO", icon_name='video', icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)
        self.single_video_dropdown = SimBADropDown(parent=self.single_video_frm, dropdown_options=self.video_list, label='SELECT VIDEO:', label_width=30, dropdown_width=self.max_video_name_len + 10, value=self.video_list[0], img='video_2')
        self.single_video_btn = SimbaButton(parent=self.single_video_frm, txt="CREATE DIRECTING VISUALIZATION: SINGLE VIDEO", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'multiple': False}, width=380, txt_clr='blue')
        self.single_video_frm.grid(row=3, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=0, sticky=NW)
        self.single_video_btn.grid(row=1, column=0, sticky=NW)

        self.all_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZE ALL VIDEOS", icon_name='video', icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)
        self.all_videos_btn = SimbaButton(parent=self.all_videos_frm, txt=f"CREATE DIRECTING VISUALIZATION: ALL {len(self.video_list)} VIDEO(S)", img='rocket', font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'multiple': True}, width=380, txt_clr='red')
        self.all_videos_frm.grid(row=4, column=0, sticky=NW)
        self.all_videos_btn.grid(row=0, column=0, sticky=NW)

        self.main_frm.mainloop()

    def _get_time_slice(self):
        if self.slice_var.get():
            check_if_string_value_is_valid_video_timestamp(value=self.video_start_time_entry.entry_get, name="START TIME")
            check_if_string_value_is_valid_video_timestamp(value=self.video_end_time_entry.entry_get, name="END TIME")
            if self.video_start_time_entry.entry_get == self.video_end_time_entry.entry_get:
                raise FrameRangeError(msg=f"The start and end times cannot be identical: {self.video_start_time_entry.entry_get} and {self.video_end_time_entry.entry_get}. Un-check the time slicing box or correct the time stamps.", source=self.__class__.__name__)
            check_that_hhmmss_start_is_before_end(start_time=self.video_start_time_entry.entry_get, end_time=self.video_end_time_entry.entry_get, name="SLICE TIME STAMPS")
            return {"start_time": self.video_start_time_entry.entry_get, "end_time": self.video_end_time_entry.entry_get}
        else:
            return None

    def run(self, multiple: bool):
        nose_name = self.nose_dropdown.get_value()
        left_ear = self.ear_left_dropdown.get_value()
        ear_right = self.ear_right_dropdown.get_value()
        direction_style = self.direction_style_dropdown.get_value().lower()
        border_clr = self.colors_dict[self.border_clr_dropdown.getChoices()]
        direction_clr = self.colors_dict[self.direction_clr_dropdown.getChoices()]
        show_pose = str_2_bool(self.show_pose_dropdown.get_value())
        show_roi_centers = str_2_bool(self.show_roi_centers_dropdown.get_value())
        show_animal_names = str_2_bool(self.show_animal_names_dropdown.get_value())
        circle_size = None if self.circle_size_dropdown.get_value() == 'AUTO' else int(self.circle_size_dropdown.get_value())
        direction_thickness = None if self.line_thickness_dropdown.get_value() == 'AUTO' else int(self.line_thickness_dropdown.get_value())
        gpu = str_2_bool(self.gpu_dropdown.get_value())
        core_cnt = int(self.cpu_cnt_dropdown.get_value())
        time_slice = self._get_time_slice()

        if multiple:
            video_paths = [v for k, v in self.video_file_paths.items()]
        else:
            video_paths = [self.video_file_paths[self.single_video_dropdown.getChoices()]]

        for video_path in video_paths:
            _, video_name, _ = get_fn_ext(video_path)
            data_path = os.path.join(self.outlier_corrected_dir, f"{video_name}.{self.file_type}")
            if not os.path.isfile(data_path):
                stdout_warning(msg=f"Skipping video {video_name}: no data file found at {data_path}.")
                continue
            visualizer = DirectingROIVisualizer(config_path=self.config_path,
                                                video_path=video_path,
                                                direction_style=direction_style,
                                                direction_color=direction_clr,
                                                direction_thickness=direction_thickness,
                                                circle_size=circle_size,
                                                show_pose=show_pose,
                                                show_roi_centers=show_roi_centers,
                                                show_animal_names=show_animal_names,
                                                border_bg_clr=border_clr,
                                                left_ear_name=left_ear,
                                                right_ear_name=ear_right,
                                                nose_name=nose_name,
                                                time_slice=time_slice,
                                                core_cnt=core_cnt,
                                                gpu=gpu)

            threading.Thread(target=visualizer.run).start()


#DirectingROIVisualizerPopUp(config_path=r"E:\troubleshooting\mitra_emergence_hour\project_folder\project_config.ini")
