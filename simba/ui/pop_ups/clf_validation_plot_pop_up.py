__author__ = "Simon Nilsson"

import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.clf_validator import ClassifierValidationClips
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, SimbaButton, SimbaCheckbox,
                                        SimBADropDown)
from simba.utils.checks import check_int
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.errors import NoDataError
from simba.utils.read_write import get_file_name_info_in_directory, get_fn_ext


class ClassifierValidationPopUp(PopUpMixin, ConfigReader):

    """
    :example:
    >>> _ = ClassifierValidationPopUp(config_path=r'C:\troubleshooting\RAT_NOR\project_folder\project_config.ini')

    """
    def __init__(self,
                 config_path: Union[str, os.PathLike]):

        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        if len(self.machine_results_paths) == 0:
            raise NoDataError(msg=f'No data found in the {self.machine_results_dir} directory. This is required for creating validation clips.', source=self.__class__.__name__)

        self.machine_results_dict = {get_fn_ext(filepath=v)[1]:v for v in self.machine_results_paths}


        PopUpMixin.__init__(self, title="SIMBA CLASSIFIER VALIDATION CLIPS", icon='tick')
        color_names = list(self.colors_dict.keys())

        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings',  icon_link=Links.CLF_VALIDATION.value, padx=5, pady=5, relief='solid')
        self.seconds_entry = Entry_Box(self.settings_frm, "SECONDS PADDING: ", 40, validation="numeric", value=2, entry_box_width=30, justify='center')

        self.clf_dropdown = SimBADropDown(parent=self.settings_frm, label='CLASSIFIER: ', label_width=40, dropdown_width=30, dropdown_options=self.clf_names, value=self.clf_names[0])
        self.clr_dropdown = SimBADropDown(parent=self.settings_frm, label="TEXT COLOR: ", label_width=40, dropdown_width=30, dropdown_options=color_names, value="Cyan")
        self.highlight_clr_dropdown = SimBADropDown(parent=self.settings_frm, label="HIGHLIGHT TEXT COLOR: ", label_width=40, dropdown_width=30, dropdown_options=["None"] + color_names, value="None")
        self.video_speed_dropdown = SimBADropDown(parent=self.settings_frm, label="VIDEO SPEED: ", label_width=40, dropdown_width=30, dropdown_options=Options.SPEED_OPTIONS.value, value=1.0)

        self.individual_bout_clips_cb, self.one_vid_per_bout_var = SimbaCheckbox(parent=self.settings_frm, txt="CREATE ONE CLIP PER BOUT", txt_img='segment', val=False)
        self.individual_clip_per_video_cb, self.one_vid_per_video_var = SimbaCheckbox(parent=self.settings_frm, txt="CREATE ONE CLIP PER BOUT", txt_img='video', val=True)

        self.settings_frm.grid(row=0, column=0, sticky=NW, padx=10, pady=10)
        self.seconds_entry.grid(row=0, column=0, sticky=NW)
        self.clf_dropdown.grid(row=1, column=0, sticky=NW)
        self.clr_dropdown.grid(row=2, column=0, sticky=NW)
        self.highlight_clr_dropdown.grid(row=3, column=0, sticky=NW)
        self.video_speed_dropdown.grid(row=4, column=0, sticky=NW)
        self.individual_bout_clips_cb.grid(row=5, column=0, sticky=NW)
        self.individual_clip_per_video_cb.grid(row=6, column=0, sticky=NW)

        self.run_single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name='run',  icon_link='video', padx=5, pady=5, relief='solid')
        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, txt="CREATE SINGLE VIDEO", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'multiple': False})
        self.single_video_dropdown = SimBADropDown(parent=self.run_single_video_frm, label="VIDEO: ", label_width=15, dropdown_width=30, dropdown_options=list(self.machine_results_dict.keys()), value=list(self.machine_results_dict.keys())[0])
        self.run_single_video_frm.grid(row=1, column=0, sticky=NW, padx=10, pady=10)
        self.run_single_video_btn.grid(row=0, column=1, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=2, sticky=NW)

        self.run_multiple_videos = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name='run', icon_link='stack', padx=5, pady=5, relief='solid')
        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, txt=f"Create multiple videos ({len(list(self.machine_results_dict.keys()))} video(s) found)", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'multiple': True})
        self.run_multiple_videos.grid(row=2, column=0, sticky=NW, padx=10, pady=10)
        self.run_multiple_video_btn.grid(row=0, column=0, sticky=NW)
        self.main_frm.mainloop()

    def run(self, multiple: bool):
        check_int(name="CLIP SECONDS", value=self.seconds_entry.entry_get)
        if self.highlight_clr_dropdown.getChoices() == "None":
            highlight_clr = None
        else:
            highlight_clr = self.colors_dict[self.highlight_clr_dropdown.getChoices()]
        if multiple:
            data_paths = list(self.machine_results_dict.values())
        else:
            data_paths = [self.machine_results_dict[self.single_video_dropdown.getChoices()]]

        clf_validator = ClassifierValidationClips(config_path=self.config_path,
                                                  window=int(self.seconds_entry.entry_get),
                                                  clf_name=self.clf_dropdown.getChoices(),
                                                  clips=self.one_vid_per_bout_var.get(),
                                                  text_clr=self.colors_dict[self.clr_dropdown.getChoices()],
                                                  highlight_clr=highlight_clr,
                                                  video_speed=float(self.video_speed_dropdown.getChoices()),
                                                  concat_video=self.one_vid_per_video_var.get(),
                                                  data_paths=data_paths)

        threading.Thread(target=clf_validator.run()).start()




#_ = ClassifierValidationPopUp(config_path=r"C:\troubleshooting\mitra\project_folder\project_config.ini")