__author__ = "Simon Nilsson"

import os
import threading
import time
from tkinter import *
from tkinter.ttk import Combobox
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.roi_plotter import ROIPlotter
from simba.plotting.roi_plotter_mp import ROIPlotMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, FileSelect, SimbaButton,
                                        SimbaCheckbox, SimBADropDown,
                                        SimBALabel)
from simba.utils.checks import check_file_exist_and_readable, check_float
from simba.utils.enums import ROI_SETTINGS, Formats, Keys, Links, Options
from simba.utils.errors import NoDataError, ROICoordinatesNotFoundError
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import (find_all_videos_in_directory,
                                    find_core_cnt,
                                    find_files_of_filetypes_in_directory,
                                    str_2_bool)

BP_SIZE_OPTIONS = list(range(1, 101, 1))
BP_SIZE_OPTIONS.insert(0, 'AUTO')

class VisualizeROITrackingPopUp(PopUpMixin, ConfigReader):

    """
    :example:
    >>> VisualizeROITrackingPopUp(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")
    """
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        check_file_exist_and_readable(file_path=config_path)
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if not os.path.isfile(self.roi_coordinates_path):
            raise ROICoordinatesNotFoundError(expected_file_path=self.roi_coordinates_path, source=self.__class__.__name__)
        if len(self.outlier_corrected_paths) == 0:
            raise NoDataError(msg=f'No data found in {self.outlier_corrected_dir} directory. Make sure the video pose-estimation data is represented in the {self.outlier_corrected_dir} directory')
        self.read_roi_data()
        self.video_file_paths = find_all_videos_in_directory(directory=self.video_dir, as_dict=True, raise_error=True)
        self.data_file_paths = find_files_of_filetypes_in_directory(directory=self.outlier_corrected_dir, extensions=[f'.{self.file_type}'], as_dict=True)
        self.videos_with_rois_and_data = [x for x in self.video_file_paths.keys() if x in list(self.data_file_paths.keys()) and x in self.video_names_w_rois]
        if len(self.videos_with_rois_and_data) == 0:
            raise NoDataError(msg=f'No ROIs found for the videos represented in the {self.video_dir} and data represented in the {self.outlier_corrected_dir} directory. Draw ROIs for these videos before visualizing the data.')
        self.clr_dict = get_color_dict()
        self.longest_animal_name_len = len(max(self.multi_animal_id_list, key=len)) + 5
        gpu_state = NORMAL if self.gpu_available else DISABLED

        PopUpMixin.__init__(self, title="VISUALIZE ROI TRACKING", size=(800, 500), icon='shapes_small')
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.ROI_DATA_PLOT.value)
        self.threshold_entry_box = Entry_Box(self.settings_frm, "BODY-PART PROBABILITY THRESHOLD:", "35", value='0.0', justify='center', entry_box_width=35)
        self.show_pose_estimation_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], dropdown_width=self.longest_animal_name_len, label='SHOW POSE-ESTIMATED LOCATIONS:', label_width=35, value='TRUE', command=self._disable_clr)
        self.show_animal_name_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], dropdown_width=self.longest_animal_name_len, label='SHOW ANIMAL NAMES:', label_width=35, value='FALSE')
        self.outside_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], dropdown_width=self.longest_animal_name_len, label='OUTSIDE ROI ZONES DATA:', label_width=35, value='FALSE', tooltip_txt=f'TREAT ALL NON-ROI REGIONS AS AN ROI REGION NAMED \n "{ROI_SETTINGS.OUTSIDE_ROI.value}"')
        self.core_cnt_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=list(range(1, self.cpu_cnt)), dropdown_width=self.longest_animal_name_len, label='NUMBER OF CPU CORES:', label_width=35, value=find_core_cnt()[1])
        self.gpu_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], dropdown_width=self.longest_animal_name_len, label='USE GPU:', label_width=35, value='FALSE', state=gpu_state)

        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.threshold_entry_box.grid(row=0, column=0, sticky=NW)
        self.show_pose_estimation_dropdown.grid(row=2, column=0, sticky=NW)
        self.show_animal_name_dropdown.grid(row=3, column=0, sticky=NW)
        self.outside_dropdown.grid(row=4, column=0, sticky=NW)
        self.core_cnt_dropdown.grid(row=5, column=0, sticky=NW)
        #self.gpu_dropdown.grid(row=5, column=0, sticky=NW)

        self.body_parts_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT NUMBER OF ANIMAL(S)", icon_name='pose', icon_link=Links.ROI_DATA_PLOT.value)
        self.animal_cnt_dropdown = SimBADropDown(parent=self.body_parts_frm, dropdown_options=list(range(1, self.animal_cnt + 1)), dropdown_width=self.longest_animal_name_len, label="NUMBER OF ANIMALS:", label_width=35, value=1, command=self.__create_animal_bp_table)
        self.__create_animal_bp_table(bp_cnt=1)
        self.body_parts_frm.grid(row=1, column=0, sticky=NW)
        self.animal_cnt_dropdown.grid(row=0, column=0, sticky=NW)

        self.single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name='video', icon_link=Links.ROI_DATA_PLOT.value)
        self.single_video_dropdown = DropDownMenu(self.single_video_frm, "SELECT VIDEO", self.videos_with_rois_and_data, "15", com=lambda x: self.update_file_select_box_from_dropdown(filename=x, fileselectbox=self.select_video_file_select))
        self.select_video_file_select = FileSelect(self.single_video_frm, "", lblwidth="1", file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], dropdown=self.single_video_dropdown, initialdir=self.video_dir)
        self.select_video_file_select.filePath.set(self.videos_with_rois_and_data[0])
        self.single_video_dropdown.setChoices(self.videos_with_rois_and_data[0])
        self.single_video_btn = SimbaButton(parent=self.single_video_frm, txt="CREATE SINGLE ROI VIDEO", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'multiple': False})
        self.single_video_frm.grid(row=3, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=0, sticky=NW)
        self.select_video_file_select.grid(row=0, column=1, sticky=NW)
        self.single_video_btn.grid(row=1, column=0, sticky=NW)


        self.all_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="ALL VIDEOS", icon_name='video', icon_link=Links.ROI_DATA_PLOT.value)
        self.all_videos_btn = SimbaButton(parent=self.all_videos_frm, txt=f"CREATE ALL ROI VIDEOS ({len(self.videos_with_rois_and_data)} VIDEO(S) WITH DATA FOUND)", img='rocket', txt_clr='green', font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'multiple': True})
        self.all_videos_frm.grid(row=4, column=0, sticky=NW)
        self.all_videos_btn.grid(row=0, column=0, sticky=NW)
        self.main_frm.mainloop()

    def __create_animal_bp_table(self, bp_cnt: int):
        bp_cnt = int(bp_cnt)
        if hasattr(self, "animal_table_frm"):
            self.animal_table_frm.destroy()
            for k, v in self.bp_dropdown_dict.items(): v.destroy()
            for k, v in self.bp_clr_dict.items(): v.destroy()
            for k, v in self.bp_size_dict.items(): v.destroy()

        self.animal_table_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT BODY-PARTS", icon_name='pose', icon_link=Links.ROI_DATA_PLOT.value)
        animal_name_heading = SimBALabel(parent=self.animal_table_frm, txt='ANIMAL NAME', width=self.longest_animal_name_len, font=Formats.FONT_HEADER.value)
        bp_heading = SimBALabel(parent=self.animal_table_frm, txt='BODY-PART NAME', width=self.longest_animal_name_len + 5, font=Formats.FONT_HEADER.value)
        bp_clr_heading = SimBALabel(parent=self.animal_table_frm, txt='COLOR', width=self.longest_animal_name_len, font=Formats.FONT_HEADER.value)
        bp_size_heading = SimBALabel(parent=self.animal_table_frm, txt='KEY-POINT SIZE', width=15, font=Formats.FONT_HEADER.value)
        animal_name_heading.grid(row=1, column=0, sticky=NW)
        bp_heading.grid(row=1, column=1, sticky=NW)
        bp_clr_heading.grid(row=1, column=2, sticky=NW)
        bp_size_heading.grid(row=1, column=3, sticky=NW)

        self.bp_dropdown_dict, self.bp_clr_dict, self.bp_size_dict = {}, {}, {}
        for cnt in range(bp_cnt):
            animal_name_lbl = SimBALabel(parent=self.animal_table_frm, txt=self.multi_animal_id_list[cnt], width=self.longest_animal_name_len, font=Formats.FONT_HEADER.value)
            self.bp_dropdown_dict[cnt] = SimBADropDown(parent=self.animal_table_frm, dropdown_options=self.body_parts_lst, dropdown_width=self.longest_animal_name_len, label=None, label_width=None, value=self.body_parts_lst[cnt])
            self.bp_clr_dict[cnt] = SimBADropDown(parent=self.animal_table_frm, dropdown_options=list(self.clr_dict.keys()), dropdown_width=self.longest_animal_name_len + 5, label=None, label_width=None, value=list(self.clr_dict.keys())[cnt])
            self.bp_size_dict[cnt] = SimBADropDown(parent=self.animal_table_frm, dropdown_options=BP_SIZE_OPTIONS, dropdown_width=15, label=None, label_width=None, value='AUTO')
            animal_name_lbl.grid(row=cnt + 2, column=0, sticky=NW)
            self.bp_dropdown_dict[cnt].grid(row=cnt + 2, column=1, sticky=NW)
            self.bp_clr_dict[cnt].grid(row=cnt + 2, column=2, sticky=NW)
            self.bp_size_dict[cnt].grid(row=cnt + 2, column=3, sticky=NW)

        self.animal_table_frm.grid(row=2, column=0, sticky=NW)

    def _disable_clr(self, setting):
        for k, v in self.bp_clr_dict.items():
            if not str_2_bool(setting): v.disable()
            else: v.enable()
        for k, v in self.bp_size_dict.items():
            if not str_2_bool(setting): v.disable()
            else: v.enable()

    def run(self, multiple: bool):
        if multiple:
            video_paths = [v for k, v in self.video_file_paths.items() if k in self.videos_with_rois_and_data]
        else:
            video_name = self.single_video_dropdown.getChoices().rsplit(".", 1)[0]
            video_paths = [self.video_file_paths[video_name]]
        for video_path in video_paths:
            self.check_if_selected_video_path_exist_in_project(video_path=video_path)
        check_float(name="Body-part probability threshold", value=self.threshold_entry_box.entry_get, min_value=0.0, max_value=1.0)
        show_pose = str_2_bool(self.show_pose_estimation_dropdown.get_value())
        show_names = str_2_bool(self.show_animal_name_dropdown.get_value())
        gpu = str_2_bool(self.gpu_dropdown.get_value())
        core_cnt = int(self.core_cnt_dropdown.get_value())
        outside_roi = str_2_bool(self.outside_dropdown.get_value())
        style_attr = {"show_body_part": show_pose, "show_animal_name": show_names}
        body_parts = [v.get_value() for k, v in self.bp_dropdown_dict.items()]
        bp_clrs, bp_sizes = None, None
        if show_pose:
            bp_clrs = [self.clr_dict[v.get_value()] for k, v in self.bp_clr_dict.items()]
            bp_sizes = [v.get_value() for k, v in self.bp_size_dict.items()]

        for video_path in video_paths:
            if core_cnt == 1:
                roi_plotter = ROIPlotter(config_path=self.config_path,
                                         video_path=video_path,
                                         style_attr=style_attr,
                                         threshold=float(self.threshold_entry_box.entry_get),
                                         body_parts=body_parts,
                                         bp_colors=bp_clrs,
                                         bp_sizes=bp_sizes,
                                         outside_roi=outside_roi)
                threading.Thread(target=roi_plotter.run()).start()

            else:
                roi_plotter = ROIPlotMultiprocess(config_path=self.config_path,
                                                  video_path=video_path,
                                                  style_attr=style_attr,
                                                  threshold=float(self.threshold_entry_box.entry_get),
                                                  body_parts=body_parts,
                                                  bp_colors=bp_clrs,
                                                  bp_sizes=bp_sizes,
                                                  core_cnt=core_cnt,
                                                  gpu=gpu,
                                                  outside_roi=outside_roi)
                roi_plotter.run()
                time.sleep(3)
        #


#_ = VisualizeROITrackingPopUp(config_path=r"C:\troubleshooting\roi_entries\project_folder\project_config.ini")


#_ = VisualizeROITrackingPopUp(config_path=r"C:\troubleshooting\roi_duplicates\project_folder\project_config.ini")


# _ = VisualizeROITrackingPopUp(config_path=r"C:\troubleshooting\two_black_animals_14bp\project_folder\project_config.ini")
#_ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/project_config.ini')
# _ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Termites_5/project_folder/project_config.ini')
# _ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
# _ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/simba_dev/tests/data/test_projects/mouse_open_field/project_folder/project_config.ini')
# _ = VisualizeROITrackingPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/dorian_2/project_folder/project_config.ini')
