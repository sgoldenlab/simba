__author__ = "Simon Nilsson"

import os
import threading
from copy import deepcopy
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.path_plotter import PathPlotterSingleCore
from simba.plotting.path_plotter_mp import PathPlotterMulticore
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, SimbaButton, SimbaCheckbox,
                                        SimBADropDown)
from simba.utils.checks import (check_if_string_value_is_valid_video_timestamp,
                                check_if_valid_rgb_str, check_int,
                                check_that_hhmmss_start_is_before_end)
from simba.utils.enums import Formats, Links
from simba.utils.errors import (FrameRangeError, NoFilesFoundError,
                                NoROIDataError)
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import get_file_name_info_in_directory

AUTO = 'AUTO'
CUSTOM = 'CUSTOM'
ENTIRE_VIDEO = 'ENTIRE VIDEO'
VIDEO_STATIC_FRAME = 'Video - static frame'
VIDEO_MOVING_FRAME = 'Video - moving frames'
STYLE_WIDTH = "width"
STYLE_HEIGHT = "height"
STYLE_LINE_WIDTH = "line width"
STYLE_FONT_SIZE = "font size"
STYLE_FONT_THICKNESS = "font thickness"
STYLE_CIRCLE_SIZE = "circle size"
STYLE_MAX_LINES = "max lines"
STYLE_BG = 'bg'
STYLE_BG_OPACITY = 'bg_opacity'
COLOR = 'color'
SIZE = 'size'
START_TIME = 'start_time'
END_TIME = 'end_time'
BODY_PART = 'body_part'
ANIMAL_NAME = 'animal_name'

class PathPlotPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.machine_results_files = get_file_name_info_in_directory(directory=self.machine_results_dir, file_type=self.file_type)
        self.outlier_corrected_files = get_file_name_info_in_directory(directory=self.outlier_corrected_dir, file_type=self.file_type)
        self.files_found = list(set(list(self.machine_results_files.keys()) + list(self.outlier_corrected_files.keys())))
        if len(self.files_found) == 0:
            raise NoFilesFoundError(msg=f'No data files found inside the {self.outlier_corrected_dir} or the {self.machine_results_dir} directory', source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="CREATE PATH PLOTS", size=(650, 850), icon='path_2')
        self.resolution_options = deepcopy(self.resolutions)
        self.resolution_options.insert(0, AUTO)
        self.bg_clr_options = deepcopy(list(self.colors_dict.keys()))
        self.animal_trace_clrs = deepcopy(list(self.colors_dict.keys()))
        self.bg_clr_options.extend((VIDEO_STATIC_FRAME, VIDEO_MOVING_FRAME))
        self.animal_trace_clrs.append(CUSTOM)
        self.max_prior_lines_options = list(range(1, 61, 1))
        self.max_prior_lines_options.insert(0, ENTIRE_VIDEO)
        self.bg_opacity_options = list(range(10, 110, 10))
        self.animal_cnt_options = list(range(1, self.animal_cnt + 1))
        self.line_width_options = list(range(1, 26, 1))
        self.line_width_options.insert(0, AUTO)
        self.font_size_options = list(range(1, 26, 1))
        self.font_size_options.insert(0, AUTO)
        self.font_thickness_options = list(range(1, 26, 1))
        self.font_thickness_options.insert(0, AUTO)
        self.circle_size_options = list(range(1, 26, 1))
        self.circle_size_options.insert(0, AUTO)

        self.custom_rgb_selections = {}

        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name='style', icon_link=Links.PATH_PLOTS.value)
        self.max_prior_lines_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.max_prior_lines_options, label='MAX PRIOR LINES (S): ', label_width=35, dropdown_width=30, value=ENTIRE_VIDEO)
        self.resolution_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.resolution_options, label='RESOLUTION: ', label_width=35, dropdown_width=30, value=AUTO)
        self.bg_clr_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.bg_clr_options, label='BACKGROUND: ', label_width=35, dropdown_width=30, value='White', command=self.__activate_settings)
        self.line_width_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.line_width_options, label='LINE WIDTH: ', label_width=35, dropdown_width=30, value=AUTO)
        self.font_size_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.font_size_options, label='FONT SIZE: ', label_width=35, dropdown_width=30, value=AUTO)
        self.font_thickness_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.font_thickness_options, label='FONT THICKNESS: ', label_width=35, dropdown_width=30, value=AUTO)
        self.bg_opacity_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.bg_opacity_options, label='BACKGROUND OPACITY (%): ', label_width=35, dropdown_width=30, value=100)
        self.circle_size_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.circle_size_options, label='CIRCLE SIZE: ', label_width=35, dropdown_width=30, value=AUTO)
        self.body_parts_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHOOSE BODY-PARTS", icon_name='pose', icon_link=Links.PATH_PLOTS.value)
        self.number_of_animals_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.animal_cnt_options, label="# ANIMALS:", label_width=35, dropdown_width=30, value=self.animal_cnt_options[0], command=self.populate_body_parts_menu)

        self.style_settings_frm.grid(row=0, sticky=NW)
        self.resolution_dropdown.grid(row=0, sticky=NW)
        self.max_prior_lines_dropdown.grid(row=2, sticky=NW)
        self.line_width_dropdown.grid(row=4, sticky=NW)
        self.circle_size_dropdown.grid(row=5, sticky=NW)
        self.font_size_dropdown.grid(row=6, sticky=NW)
        self.font_thickness_dropdown.grid(row=7, sticky=NW)
        self.bg_clr_dropdown.grid(row=8, sticky=NW)
        self.bg_opacity_dropdown.grid(row=9, sticky=NW)

        self.video_slicing_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SEGMENTS", icon_name='clip', icon_link=Links.PATH_PLOTS.value)
        self.slice_var = BooleanVar(value=False)
        self.video_start_time_entry = Entry_Box(self.video_slicing_frm, "START TIME:", "35", entry_box_width=30, value="00:00:00", status=DISABLED)
        self.video_end_time_entry = Entry_Box(self.video_slicing_frm, "END TIME:", "35", entry_box_width=30, value="00:00:00", status=DISABLED)
        self.slice_cb = Checkbutton(self.video_slicing_frm, text="Plot ONLY defined time-segment", font=Formats.FONT_REGULAR_BOLD.value, variable=self.slice_var, command=lambda: self.enable_entrybox_from_checkbox(check_box_var=self.slice_var, entry_boxes=[self.video_start_time_entry, self.video_end_time_entry]))
        self.video_slicing_frm.grid(row=1, sticky=NW)
        self.slice_cb.grid(row=0, column=0, sticky=NW)
        self.video_start_time_entry.grid(row=1, column=0, sticky=NW)
        self.video_end_time_entry.grid(row=2, column=0, sticky=NW)

        self.clf_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CLASSIFICATION VISUALIZATION", icon_name='forest', icon_link=Links.PATH_PLOTS.value)
        self.include_clf_locations_var = BooleanVar(value=False)
        self.include_clf_locations_cb = Checkbutton(self.clf_frm, text="INCLUDE CLASSIFICATION LOCATIONS", font=Formats.FONT_REGULAR.value, variable=self.include_clf_locations_var, command=self.populate_clf_location_data)
        self.include_clf_locations_cb.grid(row=0, sticky=NW)
        self.populate_clf_location_data()
        self.clf_frm.grid(row=2, sticky=NW)

        self.roi_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="ROI VISUALIZATION", icon_name='shapes_small', icon_link=Links.PATH_PLOTS.value)
        roi_cb, self.roi_var = SimbaCheckbox(parent=self.roi_frm, txt='INCLUDE ROIs', txt_img='roi', val=False)

        self.populate_body_parts_menu(self.animal_cnt_options[0])

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VISUALIZATION SETTINGS", icon_name='eye', icon_link=Links.PATH_PLOTS.value)

        path_frames_cb, self.path_frames_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE FRAMES', txt_img='frames', val=False)
        path_videos_cb, self.path_videos_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE VIDEOS', txt_img='video', val=False)
        path_last_frm_cb, self.path_last_frm_var = SimbaCheckbox(parent=self.settings_frm, txt='CREATE LAST FRAME', txt_img='finish', val=True)
        self.include_animal_names_cb, self.include_animal_names_var = SimbaCheckbox(parent=self.settings_frm, txt='INCLUDE ANIMAL NAMES', txt_img='id_card')
        self.core_cnt_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(1, self.cpu_cnt+1)), label='CPU CORE COUNT: ', label_width=35, dropdown_width=30, value=int(self.cpu_cnt/2))

        self.run_frm = LabelFrame(self.main_frm, text="RUN", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black")
        self.run_single_video_frm = LabelFrame(self.run_frm, text="SINGLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black")
        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, txt='CREATE SINGLE VIDEO', img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self._run, cmd_kwargs={'multiple_videos': False})

        self.single_video_dropdown = DropDownMenu( self.run_single_video_frm, "Video:", self.files_found, "12")
        self.single_video_dropdown.setChoices(self.files_found[0])
        self.run_multiple_videos = LabelFrame( self.run_frm, text="MULTIPLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black")
        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, txt=f"Create multiple videos ({len(self.files_found)} video(s) found)", font=Formats.FONT_REGULAR.value, img='rocket', txt_clr='blue', cmd=self._run, cmd_kwargs={'multiple_videos': True})



        self.roi_frm.grid(row=3, sticky=NW)
        roi_cb.grid(row=0, sticky=NW)

        self.body_parts_frm.grid(row=4, sticky=NW)
        self.number_of_animals_dropdown.grid(row=0, sticky=NW)

        self.settings_frm.grid(row=5, sticky=NW)
        self.core_cnt_dropdown.grid(row=0, column=0, sticky=NW)
        path_frames_cb.grid(row=1, sticky=NW)
        path_videos_cb.grid(row=2, sticky=NW)
        path_last_frm_cb.grid(row=3, sticky=NW)
        self.include_animal_names_cb.grid(row=4, sticky=NW)


        self.run_frm.grid(row=6, sticky=NW)
        self.run_single_video_frm.grid(row=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)

        self.run_multiple_videos.grid(row=1, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, sticky=NW)

        self.main_frm.mainloop()

    def populate_body_parts_menu(self, choice):
        if hasattr(self, "bp_dropdowns"):
            for (k, v), (k2, v2) in zip(self.bp_dropdowns.items(), self.bp_colors.items()):
                self.bp_dropdowns[k].destroy()
                self.bp_colors[k].destroy()
        for k, v in self.custom_rgb_selections.items():
            v.destroy()
        self.custom_rgb_selections = {}

        self.bp_dropdowns, self.bp_colors = {}, {}
        self.bp_row_idx = []
        for animal_cnt in range(int(self.number_of_animals_dropdown.getChoices())):
            self.bp_dropdowns[animal_cnt] = DropDownMenu(self.body_parts_frm, "Body-part {}:".format(str(animal_cnt + 1)), self.body_parts_lst, "16")
            self.bp_dropdowns[animal_cnt].setChoices(self.body_parts_lst[animal_cnt])
            self.bp_dropdowns[animal_cnt].grid(row=animal_cnt + 1, column=0, sticky=NW)
            self.bp_colors[animal_cnt] = DropDownMenu(self.body_parts_frm, "", self.animal_trace_clrs, "2", com=lambda x, k=animal_cnt: self.__set_custom_clrs(choice=x, row=k))
            self.bp_colors[animal_cnt].setChoices(list(self.colors_dict.keys())[animal_cnt])
            self.bp_colors[animal_cnt].grid(row=animal_cnt + 1, column=1, sticky=NW)

    def __activate_settings(self, choice: str):
        if choice == VIDEO_STATIC_FRAME or choice == VIDEO_MOVING_FRAME:
            self.bg_opacity_dropdown.enable()
        else:
            self.bg_opacity_dropdown.disable()
        if choice == VIDEO_STATIC_FRAME:
            self.static_frm_index_eb = Entry_Box(self.style_settings_frm, "SELECT FRAME INDEX: ", labelwidth=20, entry_box_width=10, validation="numeric", value=1)
            self.static_frm_index_eb.grid(row=8, column=1, sticky=NW)
        else:
            if hasattr(self, "static_frm_index_eb"):
                self.static_frm_index_eb.destroy()

    def __set_custom_clrs(self, choice: str, row: int):
        if choice == CUSTOM:
            self.custom_rgb_selections[row] = Entry_Box(self.body_parts_frm, "RGB:", "5", entry_box_width=10)
            self.custom_rgb_selections[row].entry_set(val="255,0,0")
            self.custom_rgb_selections[row].grid(row=row + 1, column=3, sticky=NW)
        else:
            if row in self.custom_rgb_selections.keys():
                self.custom_rgb_selections[row].destroy()
                self.custom_rgb_selections.pop(row)

    def populate_clf_location_data(self):
        self.clf_name, self.clf_clr, self.clf_size = {}, {}, {}
        size_lst = list(range(1, 51))
        size_lst = ["Size: " + str(x) for x in size_lst]
        for clf_cnt, clf_name in enumerate(self.clf_names):
            self.clf_name[clf_cnt] = DropDownMenu(self.clf_frm, "Classifier {}:".format(str(clf_cnt + 1)), self.clf_names, "16")
            self.clf_name[clf_cnt].setChoices(self.clf_names[clf_cnt])
            self.clf_name[clf_cnt].grid(row=clf_cnt + 1, column=0, sticky=NW)

            self.clf_clr[clf_cnt] = DropDownMenu(self.clf_frm, "", list(self.colors_dict.keys()), "2")
            self.clf_clr[clf_cnt].setChoices(list(self.colors_dict.keys())[clf_cnt])
            self.clf_clr[clf_cnt].grid(row=clf_cnt + 1, column=1, sticky=NW)

            self.clf_size[clf_cnt] = DropDownMenu(self.clf_frm, "", size_lst, "2")
            self.clf_size[clf_cnt].setChoices(size_lst[15])
            self.clf_size[clf_cnt].grid(row=clf_cnt + 1, column=2, sticky=NW)

        self.enable_clf_location_settings()

    def enable_clf_location_settings(self):
        if self.include_clf_locations_var.get():
            for clf_cnt in self.clf_name.keys():
                self.clf_name[clf_cnt].enable(); self.clf_clr[clf_cnt].enable(); self.clf_size[clf_cnt].enable()
        else:
            for clf_cnt in self.clf_name.keys():
                self.clf_name[clf_cnt].disable(); self.clf_clr[clf_cnt].disable(); self.clf_size[clf_cnt].disable()

    def _get_animal_attr(self):
        animal_attr = {}
        for cnt, (key, value) in enumerate(self.bp_colors.items()):
            if cnt not in animal_attr.keys():
                animal_attr[cnt] = {}
            clr = value.getChoices()
            if clr == CUSTOM:
                clr = self.custom_rgb_selections[cnt].entry_get
                clr = check_if_valid_rgb_str(input=clr)
                animal_attr[cnt]["color"] = clr
            else:
                animal_attr[cnt]["color"] = get_color_dict()[value.getChoices()]

        for cnt, (key, value) in enumerate(self.bp_dropdowns.items()):
            if cnt not in animal_attr.keys():
                animal_attr[cnt] = {}
            animal_attr[cnt]["body_part"] = value.getChoices()
        return animal_attr

    def _get_slicing(self):
        if self.slice_var.get():
            check_if_string_value_is_valid_video_timestamp(value=self.video_start_time_entry.entry_get, name="Video slicing START TIME")
            check_if_string_value_is_valid_video_timestamp(value=self.video_end_time_entry.entry_get, name="Video slicing END TIME")
            if (self.video_start_time_entry.entry_get == self.video_end_time_entry.entry_get):
                raise FrameRangeError(msg="The sliced start and end times cannot be identical", source=self.__class__.__name__)
            check_that_hhmmss_start_is_before_end(start_time=self.video_start_time_entry.entry_get, end_time=self.video_end_time_entry.entry_get, name="SLICE TIME STAMPS")
            return {"start_time": self.video_start_time_entry.entry_get, "end_time": self.video_end_time_entry.entry_get}
        else:
            return None


    def _get_clf_attr(self, multiple_videos):
        if multiple_videos:
            if len(self.machine_results_paths) == 0:
                raise NoFilesFoundError(msg=f'No DATA found in {self.machine_results_dir} directory. Un-check the classifier location checkbox, OR make sure the folder contains classification data.')
            else:
                data_paths = list(self.machine_results_files.values())
        else:
            if self.single_video_dropdown.getChoices() not in self.machine_results_files.keys():
                raise NoFilesFoundError(msg=f'No DATA found for video in {self.single_video_dropdown.getChoices()} in directory {self.machine_results_dir}. Un-check the classifier location checkbox, OR make sure the folder contains classification data for the video.')
            else:
                data_paths = [self.machine_results_files[self.single_video_dropdown.getChoices()]]
        clf_attr = {}
        for cnt, (key, value) in enumerate(self.clf_name.items()):
            clf_attr[value.getChoices()] = {}
            clf_attr[value.getChoices()]["color"] = get_color_dict()[self.clf_clr[cnt].getChoices()]
            size = "".join(filter(str.isdigit, self.clf_size[cnt].getChoices()))
            clf_attr[value.getChoices()]["size"] = int(size)
        return clf_attr, data_paths

    def _run(self, multiple_videos: bool):
        resolution = self.resolution_dropdown.getChoices()
        line_width = self.line_width_dropdown.get_value()
        circle_size = self.circle_size_dropdown.get_value()
        font_size = self.font_size_dropdown.get_value()
        path_thickness = self.font_thickness_dropdown.get_value()
        bg = self.bg_clr_dropdown.getChoices()
        max_lines = self.max_prior_lines_dropdown.get_value()
        include_clf = self.include_clf_locations_var.get()
        roi = self.roi_var.get()
        core_cnt = int(self.core_cnt_dropdown.get_value())
        print_animal_names = self.include_animal_names_var.get()
        clf_attr = None

        w, h = (None, None) if resolution == AUTO else tuple(map(int, resolution.split("×")))
        line_width = None if line_width == AUTO else int(line_width)
        circle_size = None if circle_size == AUTO else int(circle_size)
        font_size = None if font_size == AUTO else int(font_size)
        path_thickness = None if path_thickness == AUTO else int(path_thickness)
        max_prior_lines = None if max_lines == ENTIRE_VIDEO else int(max_lines)

        if bg == VIDEO_STATIC_FRAME:
            check_int(name="Static frame index", value=self.static_frm_index_eb.entry_get, min_value=0)
            bg = int(self.static_frm_index_eb.entry_get)
            bg_opacity = int(self.bg_opacity_dropdown.get_value())
        elif bg == VIDEO_MOVING_FRAME:
            bg = 'video'
            bg_opacity = int(self.bg_opacity_dropdown.get_value())
        else:
            bg = get_color_dict()[bg]
            bg_opacity = 100
        animal_attr = self._get_animal_attr()
        slicing = self._get_slicing()
        if include_clf:
            clf_attr, data_paths = self._get_clf_attr(multiple_videos=multiple_videos)
        else:
            if multiple_videos:
                data_paths = list(self.outlier_corrected_files.values())
            else:
                data_paths = [self.outlier_corrected_files[self.single_video_dropdown.getChoices()]]

        if roi:
            if not os.path.isfile(self.roi_coordinates_path):
                raise NoROIDataError(msg=f'No SimBA ROI project data found. Expected at path {self.roi_coordinates_path}', source=self.__class__.__name__)

        style_attr = {STYLE_WIDTH: w,
                      STYLE_HEIGHT: h,
                      STYLE_MAX_LINES: max_prior_lines,
                      STYLE_LINE_WIDTH: line_width,
                      STYLE_FONT_SIZE: font_size,
                      STYLE_FONT_THICKNESS: path_thickness,
                      STYLE_CIRCLE_SIZE: circle_size,
                      STYLE_BG: bg,
                      STYLE_BG_OPACITY: bg_opacity}

        if core_cnt == 1:
            path_plotter = PathPlotterSingleCore(config_path=self.config_path,
                                                 frame_setting=self.path_frames_var.get(),
                                                 video_setting=self.path_videos_var.get(),
                                                 last_frame=self.path_last_frm_var.get(),
                                                 data_paths=data_paths,
                                                 style_attr=style_attr,
                                                 print_animal_names=print_animal_names,
                                                 animal_attr=animal_attr,
                                                 clf_attr=clf_attr,
                                                 slicing=slicing,
                                                 roi=roi)
        else:
            path_plotter = PathPlotterMulticore(config_path=self.config_path,
                                                frame_setting=self.path_frames_var.get(),
                                                video_setting=self.path_videos_var.get(),
                                                last_frame=self.path_last_frm_var.get(),
                                                data_paths=data_paths,
                                                style_attr=style_attr,
                                                print_animal_names=print_animal_names,
                                                animal_attr=animal_attr,
                                                clf_attr=clf_attr,
                                                core_cnt=core_cnt,
                                                slicing=slicing,
                                                roi=self.roi_var.get())

        threading.Thread(target=path_plotter.run()).start()



#_ = PathPlotPopUp(config_path=r"C:\troubleshooting\RAT_NOR\project_folder\project_config.ini")

#_ = PathPlotPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")

#_ = PathPlotPopUp(config_path=r"/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini")

# _ = PathPlotPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/beepboop174/project_folder/project_config.ini')

# _ = PathPlotPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')

# _ = PathPlotPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/dorian_2/project_folder/project_config.ini')
