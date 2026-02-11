__author__ = "Simon Nilsson; sronilsson@gmail.com"

import os
from tkinter import *

import numpy as np

from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import (CreateLabelFrameWithIcon, FileSelect,
                                        FolderSelect, SimbaButton,
                                        SimBADropDown, SimBALabel)
from simba.utils.checks import (check_ffmpeg_available,
                                check_file_exist_and_readable,
                                check_if_dir_exists,
                                check_nvidea_gpu_available)
from simba.utils.enums import Options
from simba.utils.errors import DuplicationError
from simba.utils.lookups import get_ffmpeg_encoders
from simba.utils.read_write import (find_all_videos_in_directory, get_fn_ext,
                                    str_2_bool)
from simba.video_processors.video_processing import (change_playback_speed,
                                                     change_playback_speed_dir)

SPEED_OPTIONS = [round(x, 2) for x in np.arange(0.1, 40.1, 0.1)]
QUALITY_OPTIONS = list(range(10, 110, 10))

class ChangeSpeedPopup(PopUpMixin):
    def __init__(self):
        check_ffmpeg_available(raise_error=True)
        ffmpeg_codecs = get_ffmpeg_encoders(alphabetically_sorted=True)
        gpu_state = NORMAL if check_nvidea_gpu_available() else DISABLED
        PopUpMixin.__init__(self, title="CHANGE VIDEO PLAYBACK SPEED", icon='run')
        self.settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name='settings')

        self.speed_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=SPEED_OPTIONS, label="SPEED:", label_width=35, dropdown_width=35, value=1.5, img='run', tooltip_key='VIDEO_SPEED')
        self.codec_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=ffmpeg_codecs, label="CODEC:", label_width=35, dropdown_width=35, value='libx264', img='ffmpeg', tooltip_key='VIDEO_CODEC')
        self.gpu_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=['TRUE', 'FALSE'], label="GPU:", label_width=35, dropdown_width=35, value='FALSE', img='gpu_3', state=gpu_state, tooltip_key='USE_GPU')
        self.quality_dropdown = SimBADropDown(parent=self.settings_frm, dropdown_options=QUALITY_OPTIONS, label="QUALITY:", label_width=35, dropdown_width=35, value=60, img='pct_2', tooltip_key='OUPUT_VIDEO_QUALITY')
        self.save_dir = FolderSelect(self.settings_frm, "SAVE DIRECTORY:", lblwidth=35, initialdir=None, lbl_icon='folder')


        self.settings_frm.grid(row=0, column=0, sticky=NW)
        self.speed_dropdown.grid(row=0, column=0, sticky=NW)
        self.codec_dropdown.grid(row=1, column=0, sticky=NW)
        self.gpu_dropdown.grid(row=2, column=0, sticky=NW)
        self.quality_dropdown.grid(row=3, column=0, sticky=NW)
        self.save_dir.grid(row=4, column=0, sticky=NW)

        self.single_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHANGE SPEED FOR SINGLE VIDEO", icon_name='video_2')
        self.video_path = FileSelect(self.single_frm, "VIDEO PATH:", lblwidth=35, initialdir=None, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)], lbl_icon='file')
        single_btn = SimbaButton(parent=self.single_frm, txt='RUN', img='rocket', cmd=self.run, cmd_kwargs={'multiple': False})

        self.single_frm.grid(row=1, column=0, sticky=NW)
        self.video_path.grid(row=0, column=0, sticky=NW)
        single_btn.grid(row=1, column=0, sticky=NW)

        self.multiple_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="CHANGE SPEED FOR MULTIPLE VIDEOS", icon_name='stack')
        self.videos_dir = FolderSelect(self.multiple_frm, "VIDEOS DIRECTORY:", lblwidth=35, initialdir=None, lbl_icon='folder')
        multiple_btn = SimbaButton(parent=self.multiple_frm, txt='RUN', img='rocket', cmd=self.run, cmd_kwargs={'multiple': True})

        self.multiple_frm.grid(row=2, column=0, sticky=NW)
        self.videos_dir.grid(row=0, column=0, sticky=NW)
        multiple_btn.grid(row=1, column=0, sticky=NW)

        self.main_frm.mainloop()


    def run(self, multiple: bool):
        save_path = None
        if multiple:
            check_if_dir_exists(in_dir=self.videos_dir.folder_path, source=self.__class__.__name__, raise_error=True)
            _ = find_all_videos_in_directory(directory=self.videos_dir.folder_path, as_dict=False, raise_error=True, sort_alphabetically=True)
            check_if_dir_exists(in_dir=self.save_dir.folder_path, raise_error=True)
            save_dir = self.save_dir.folder_path
            if save_dir == self.videos_dir.folder_path:
                raise DuplicationError(msg=f'The video directory and the save directory cannot be the same folder: {save_dir}', source=self.__class__.__name__)
        else:
            check_file_exist_and_readable(file_path=self.video_path.file_path)
            save_dir = None if not check_if_dir_exists(in_dir=self.save_dir.folder_path, raise_error=False) else self.save_dir.folder_path
            if save_dir is not None:
                _, video_name, ext = get_fn_ext(filepath=self.video_path.file_path)
                save_path = os.path.join(save_dir, f'{video_name}{ext}')

        quality = int(self.quality_dropdown.get_value())
        codec = self.codec_dropdown.get_value()
        gpu = str_2_bool(self.gpu_dropdown.get_value())
        speed = float(self.speed_dropdown.get_value())
        if multiple:
            change_playback_speed_dir(data_dir=self.videos_dir.folder_path, speed=speed, save_dir=save_dir, quality=quality, gpu=gpu, verbose=True, codec=codec)
        else:
            change_playback_speed(video_path=self.video_path.file_path, speed=speed, save_path=save_path, quality=quality,gpu=gpu, verbose=True, codec=codec)

#ChangeSpeedPopup()