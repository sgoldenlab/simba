__author__ = "Simon Nilsson"

import os
import threading
from tkinter import *
from typing import Union

from simba.mixins.config_reader import ConfigReader
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.plotting.clf_validator import ClassifierValidationClips
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, DropDownMenu,
                                        Entry_Box)
from simba.utils.checks import check_int
from simba.utils.enums import Formats, Keys, Links, Options
from simba.utils.read_write import get_file_name_info_in_directory


class ClassifierValidationPopUp(PopUpMixin, ConfigReader):
    def __init__(self, config_path: Union[str, os.PathLike]):
        PopUpMixin.__init__(self, title="SIMBA CLASSIFIER VALIDATION CLIPS")
        ConfigReader.__init__(self, config_path=config_path)
        color_names = list(self.colors_dict.keys())
        self.one_vid_per_bout_var = BooleanVar(value=False)
        self.one_vid_per_video_var = BooleanVar(value=True)
        self.files_found_dict = get_file_name_info_in_directory(directory=self.machine_results_dir, file_type=self.file_type)
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
        self.individual_bout_clips_cb = Checkbutton(self.settings_frm, text="CREATE ONE CLIP PER BOUT", variable=self.one_vid_per_bout_var)
        self.individual_clip_per_video_cb = Checkbutton(self.settings_frm, text="CREATE ONE CLIP PER VIDEO", variable=self.one_vid_per_video_var)

        self.run_frm = LabelFrame(self.main_frm, text="RUN", font=Formats.LABELFRAME_HEADER_FORMAT.value, pady=5, padx=5, fg="black")
        self.run_single_video_frm = LabelFrame(self.run_frm,text="SINGLE VIDEO",font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5,padx=5,fg="black")
        self.run_single_video_btn = Button(self.run_single_video_frm,text="Create single video",fg="blue",command=lambda: self.run(multiple_videos=False))
        self.single_video_dropdown = DropDownMenu(self.run_single_video_frm,"Video:",list(self.files_found_dict.keys()),"12")
        self.single_video_dropdown.setChoices(list(self.files_found_dict.keys())[0])
        self.run_multiple_videos = LabelFrame(self.run_frm,text="MULTIPLE VIDEO",font=Formats.LABELFRAME_HEADER_FORMAT.value,pady=5,padx=5,fg="black")
        self.run_multiple_video_btn = Button(self.run_multiple_videos, text=f"Create multiple videos ({len(list(self.files_found_dict.keys()))} video(s) found)", fg="blue", command=lambda: self.run(multiple_videos=True))
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


#_ = ClassifierValidationPopUp(config_path='/Users/simon/Desktop/envs/simba/troubleshooting/two_black_animals_14bp/project_folder/project_config.ini')
