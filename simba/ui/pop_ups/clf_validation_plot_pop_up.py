__author__ = "Simon Nilsson"

import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.clf_validator import ClassifierValidationClips
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box, SimbaButton, SimbaCheckbox)
from simba.utils.checks import check_int
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.errors import NoDataError
from simba.utils.read_write import get_file_name_info_in_directory


class ClassifierValidationPopUp(PopUpMixin, ConfigReader):

    """
    :example:
    >>> _ = ClassifierValidationPopUp(config_path=r'C:\troubleshooting\RAT_NOR\project_folder\project_config.ini')

    """
    def __init__(self, config_path: Union[str, os.PathLike]):
        ConfigReader.__init__(self, config_path=config_path, read_video_info=False)
        self.files_found_dict = get_file_name_info_in_directory(directory=self.machine_results_dir, file_type=self.file_type)
        if len(list(self.files_found_dict.keys())) == 0:
            raise NoDataError(msg=f'No data found in the {self.machine_results_dir} directory. This is required for creating validation clips.', source=self.__class__.__name__)
        PopUpMixin.__init__(self, title="SIMBA CLASSIFIER VALIDATION CLIPS")
        color_names = list(self.colors_dict.keys())
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm,header="SETTINGS",icon_name=Keys.DOCUMENTATION.value,icon_link=Links.CLF_VALIDATION.value)
        self.seconds_entry = Entry_Box(self.settings_frm, "SECONDS PADDING: ", "20", validation="numeric")
        self.clf_dropdown = DropDownMenu(self.settings_frm, "CLASSIFIER: ", self.clf_names, "20")
        self.clr_dropdown = DropDownMenu(self.settings_frm, "TEXT COLOR: ", color_names, "20")
        self.highlight_clr_dropdown = DropDownMenu(self.settings_frm, "HIGHLIGHT TEXT COLOR: ", ["None"] + color_names, "20")
        self.video_speed_dropdown = DropDownMenu(self.settings_frm, "VIDEO SPEED: ", Options.SPEED_OPTIONS.value, "20")
        self.clf_dropdown.setChoices(self.clf_names[0])
        self.clr_dropdown.setChoices("Cyan")
        self.seconds_entry.entry_set(val=2)
        self.highlight_clr_dropdown.setChoices("None")
        self.video_speed_dropdown.setChoices(1.0)

        self.individual_bout_clips_cb, self.one_vid_per_bout_var = SimbaCheckbox(parent=self.settings_frm, txt="CREATE ONE CLIP PER BOUT", txt_img='segment', val=False)
        self.individual_clip_per_video_cb, self.one_vid_per_video_var = SimbaCheckbox(parent=self.settings_frm, txt="CREATE ONE CLIP PER BOUT", txt_img='video', val=True)

        self.run_frm = LabelFrame(self.main_frm, text="RUN", font=Formats.FONT_HEADER.value, pady=5, padx=5, fg="black")
        self.run_single_video_frm = LabelFrame(self.run_frm,text="SINGLE VIDEO",font=Formats.FONT_HEADER.value,pady=5,padx=5,fg="black")
        self.run_single_video_btn = SimbaButton(parent=self.run_single_video_frm, txt="Create single video", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'multiple_videos': False})

        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm,"Video:",list(self.files_found_dict.keys()),"12")
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])
        self.run_multiple_videos = LabelFrame(self.run_frm,text="MULTIPLE VIDEO",font=Formats.FONT_HEADER.value, pady=5,padx=5,fg="black")

        self.run_multiple_video_btn = SimbaButton(parent=self.run_multiple_videos, txt=f"Create multiple videos ({len(list(self.files_found_dict.keys()))} video(s) found)", img='rocket', txt_clr='blue', font=Formats.FONT_REGULAR.value, cmd=self.run, cmd_kwargs={'multiple_videos': True})
        self.settings_frm.grid(row=0, sticky=NW)
        self.seconds_entry.grid(row=0, sticky=NW)
        self.clf_dropdown.grid(row=1, sticky=NW)
        self.clr_dropdown.grid(row=2, sticky=NW)
        self.highlight_clr_dropdown.grid(row=3, sticky=NW)
        self.video_speed_dropdown.grid(row=4, sticky=NW)
        self.individual_bout_clips_cb.grid(row=5, column=0, sticky=NW)
        self.individual_clip_per_video_cb.grid(row=6, column=0, sticky=NW)

        self.run_frm.grid(row=1, column=0, sticky=NW)
        self.run_single_video_frm.grid(row=0, column=0, sticky=NW)
        self.run_single_video_btn.grid(row=0, column=0, sticky=NW)
        self.single_video_dropdown.grid(row=0, column=1, sticky=NW)
        self.run_multiple_videos.grid(row=1, column=0, sticky=NW)
        self.run_multiple_video_btn.grid(row=0, column=0, sticky=NW)

        self.main_frm.mainloop()

    def run(self, multiple_videos: bool):
        check_int(name="CLIP SECONDS", value=self.seconds_entry.entry_get)
        if self.highlight_clr_dropdown.getChoices() == "None":
            highlight_clr = None
        else:
            highlight_clr = self.colors_dict[self.highlight_clr_dropdown.getChoices()]
        if multiple_videos:
            data_paths = list(self.files_found_dict.values())
        else:
            data_paths = [self.files_found_dict[self.single_video_dropdown.getChoices()]]

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


