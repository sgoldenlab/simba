__author__ = "Simon Nilsson"

import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.directing_animals_visualizer import DirectingOtherAnimalsVisualizer
from simba.plotting.directing_animals_visualizer_mp import DirectingOtherAnimalsVisualizerMultiprocess
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu, SimbaButton, SimBADropDown)
from simba.utils.enums import Formats, Keys, Links
from simba.utils.errors import AnimalNumberError
from simba.utils.lookups import get_color_dict, find_closest_string
from simba.utils.read_write import find_all_videos_in_directory, find_core_cnt, str_2_bool

DIRECTION_THICKNESS = "direction_thickness"
DIRECTIONALITY_COLOR = "directionality_color"
CIRCLE_SIZE = "circle_size"
HIGHLIGHT_ENDPOINTS = "highlight_endpoints"
SHOW_POSE = "show_pose"
ANIMAL_NAMES = "animal_names"
AUTO = 'AUTO'

NOSE, EAR_LEFT, EAR_RIGHT = Keys.NOSE.value, Keys.EAR_LEFT.value, Keys.EAR_RIGHT.value


class DirectingOtherAnimalsVisualizerPopUp(PopUpMixin, ConfigReader):
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if self.animal_cnt == 1:
            raise AnimalNumberError(msg="Cannot visualize directionality between animals in a 1 animal project.", source=self.__class__.__name__,)

        PopUpMixin.__init__(self, title="CREATE ANIMAL DIRECTION VIDEOS", icon='eye')

        bp_names = list(set([x[:-2] for x in self.body_parts_lst]))
        nose_guess = find_closest_string(target=NOSE, string_list=bp_names)[0]
        ear_left_guess = find_closest_string(target=EAR_LEFT, string_list=bp_names)[0]
        ear_right_guess = find_closest_string(target=EAR_RIGHT, string_list=bp_names)[0]

        self.bp_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT BODY-PARTS", icon_name='pose', icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)
        self.ear_left_dropdown = SimBADropDown(parent=self.bp_frm, dropdown_options=self.body_parts_lst, label_width=30, dropdown_width=25, value=ear_left_guess, label='LEFT EAR body-part name:')
        self.ear_right_dropdown = SimBADropDown(parent=self.bp_frm, dropdown_options=self.body_parts_lst, label_width=30, dropdown_width=25, value=ear_right_guess, label='RIGHT EAR body-part name:')
        self.nose_dropdown = SimBADropDown(parent=self.bp_frm, dropdown_options=self.body_parts_lst, label_width=30, dropdown_width=25, value=nose_guess, label='NOSE body-part name:')

        self.color_dict = get_color_dict()
        self.color_lst = list(self.color_dict.keys())
        self.color_lst.insert(0, "random")
        self.size_lst = list(range(1, 11))
        self.size_lst.insert(0, AUTO)
        core_count = find_core_cnt()[0]
        self.files_found_dict = find_all_videos_in_directory(directory=self.video_dir, as_dict=True)

        self.style_settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="STYLE SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DIRECTING_ANIMALS_PLOTS.value,)
        self.show_pose_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=['TRUE', 'FALSE'], label="Show pose-estimated body-parts", label_width=30, dropdown_width=20, value="TRUE")
        self.highlight_direction_endpoints_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=['TRUE', 'FALSE'], label="Highlight direction end-points:", label_width=30, dropdown_width=20, value="FALSE")
        self.show_animal_names_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=['TRUE', 'FALSE'], label="Show animal names:", label_width=30, dropdown_width=20, value="FALSE")
        self.direction_clr_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.color_lst, label='Direction color:', label_width=30, dropdown_width=20, value="random")
        self.pose_size_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.size_lst, label="Pose circle size:", label_width=30, dropdown_width=20, value=AUTO)
        self.line_thickness = SimBADropDown(parent=self.style_settings_frm, dropdown_options=self.size_lst, label="Line thickness:", label_width=30, dropdown_width=20, value=AUTO)
        self.core_count_dropdown = SimBADropDown(parent=self.style_settings_frm, dropdown_options=list(range(1, core_count+1)), label="CPU core count:", label_width=30, dropdown_width=20, value=int(core_count/2))

        self.run_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="RUN", icon_name='rocket', icon_link=Links.DIRECTING_ANIMALS_PLOTS.value)
        self.run_single_video_frm = LabelFrame( self.run_frm, text="SINGLE VIDEO", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black",)
        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, txt="Create single video", font=Formats.FONT_REGULAR.value, cmd=self.__create_directionality_plots, cmd_kwargs={'multiple_videos': False})
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm,"Video:",list(self.files_found_dict.keys()),"12",)
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])
        self.run_multiple_videos = LabelFrame(self.run_frm,text="MULTIPLE VIDEO",font=Formats.FONT_HEADER.value,pady=5,padx=5,fg="black",)
        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, txt=f"Create multiple videos ({len(list(self.files_found_dict.keys()))} video(s) found)", font=Formats.FONT_REGULAR.value, cmd=self.__create_directionality_plots, cmd_kwargs={'multiple_videos': True})

        self.bp_frm.grid(row=0, column=0, sticky=NW)
        self.ear_left_dropdown.grid(row=0, column=0, sticky=NW)
        self.ear_right_dropdown.grid(row=1, column=0, sticky=NW)
        self.nose_dropdown.grid(row=2, column=0, sticky=NW)


        self.style_settings_frm.grid(row=1, column=0, sticky=NW)
        self.show_pose_dropdown.grid(row=0, column=0, sticky=NW)
        self.highlight_direction_endpoints_dropdown.grid(row=1, column=0, sticky=NW)
        self.show_animal_names_dropdown.grid(row=2, column=0, sticky=NW)
        self.direction_clr_dropdown.grid(row=3, column=0, sticky=NW)
        self.pose_size_dropdown.grid(row=4, column=0, sticky=NW)
        self.line_thickness.grid(row=5, column=0, sticky=NW)
        self.core_count_dropdown.grid(row=6, column=0, sticky=NW)

        self.run_frm.grid(row=2, column=0, sticky=NW)
        self.run_single_video_frm.grid(row=0, column=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)
        self.run_multiple_videos.grid(row=1, column=0, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, column=0, sticky=NW)

        self.main_frm.mainloop()

    def __create_directionality_plots(self, multiple_videos: bool):
        if multiple_videos:
            video_paths = list(self.files_found_dict.values())
        else:
            video_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]
        show_pose = str_2_bool(self.show_pose_dropdown.get_value())
        circle_size = None if self.pose_size_dropdown.get_value() == AUTO else int(self.pose_size_dropdown.get_value())
        thickness = None if self.line_thickness.get_value() == AUTO else int(self.line_thickness.get_value())
        show_names = str_2_bool(self.show_animal_names_dropdown.get_value())
        highlight = str_2_bool(self.highlight_direction_endpoints_dropdown.get_value())
        core_cnt = int(self.core_count_dropdown.get_value())
        direction_clr = self.direction_clr_dropdown.getChoices()
        if direction_clr != "random": direction_clr = self.color_dict[direction_clr]

        nose = self.nose_dropdown.get_value()
        left_ear = self.ear_left_dropdown.get_value()
        right_ear = self.ear_right_dropdown.get_value()

        style_attr = {SHOW_POSE: show_pose,
                      CIRCLE_SIZE: circle_size,
                      DIRECTIONALITY_COLOR: direction_clr,
                      DIRECTION_THICKNESS: thickness,
                      HIGHLIGHT_ENDPOINTS: highlight,
                      ANIMAL_NAMES: show_names}

        for video_path in video_paths:
            if core_cnt == 1:
                visualizer = DirectingOtherAnimalsVisualizer(config_path=self.config_path,
                                                             video_path=video_path,
                                                             style_attr=style_attr,
                                                             left_ear_name=left_ear,
                                                             right_ear_name=right_ear,
                                                             nose_name=nose)
            else:
                visualizer = DirectingOtherAnimalsVisualizerMultiprocess(config_path=self.config_path,
                                                                         video_path=video_path,
                                                                         style_attr=style_attr,
                                                                         core_cnt=core_cnt,
                                                                         left_ear_name=left_ear,
                                                                         right_ear_name=right_ear,
                                                                         nose_name=nose)

            threading.Thread(target=visualizer.run()).start()


#_ = DirectingOtherAnimalsVisualizerPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
#_ = DirectingOtherAnimalsVisualizerPopUp(config_path='/Users/simon/Desktop/envs/troubleshooting/Two_animals_16bps/project_folder/project_config.ini')
