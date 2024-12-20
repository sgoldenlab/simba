import os

from typing import Union
from copy import deepcopy
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, FileSelect, FolderSelect, DropDownMenu, Entry_Box
from simba.utils.enums import Keys, Links, Options
from simba.utils.checks import check_file_exist_and_readable, check_if_dir_exists, check_str, check_int, check_that_hhmmss_start_is_before_end
from simba.utils.data import check_if_string_value_is_valid_video_timestamp
from simba.video_processors.video_processing import watermark_video, superimpose_elapsed_time, superimpose_video_progressbar, superimpose_overlay_video, superimpose_video_names, superimpose_freetext, roi_blurbox
from simba.utils.lookups import get_color_dict
from simba.utils.read_write import get_video_meta_data
import threading
from tkinter import *
import numpy as np
from simba.utils.errors import InvalidInputError, DuplicationError
from simba.utils.read_write import get_video_meta_data, str_2_bool
from simba.utils.enums import Formats
from simba.video_processors.video_processing import video_bg_subtraction, video_bg_subtraction_mp

class BackgroundRemoverPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="REMOVE BACKGROUND IN VIDEOS")
        self.clr_dict = get_color_dict()
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.video_path = FileSelect(settings_frm, "VIDEO PATH:", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_OPTIONS.value)], lblwidth=40)
        self.bg_video_path = FileSelect(settings_frm, "BACKGROUND REFERENCE VIDEO PATH:", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_OPTIONS.value)], lblwidth=40)
        self.bg_clr_dropdown = DropDownMenu(settings_frm, "BACKGROUND COLOR:", list(self.clr_dict.keys()), labelwidth=40)
        self.fg_clr_dropdown = DropDownMenu(settings_frm, "FOREGROUND COLOR:", list(self.clr_dict.keys()), labelwidth=40)
        self.bg_start_eb = Entry_Box(parent=settings_frm, labelwidth=40, entry_box_width=15, fileDescription='BACKGROUND VIDEO START (FRAME # OR TIME):')
        self.bg_end_eb = Entry_Box(parent=settings_frm, labelwidth=40, entry_box_width=15, fileDescription='BACKGROUND VIDEO END (FRAME # OR TIME):')
        self.multiprocessing_var = BooleanVar()
        self.multiprocess_cb = Checkbutton(settings_frm, text="Multiprocess videos (faster)", variable=self.multiprocessing_var, command=lambda: self.enable_dropdown_from_checkbox(check_box_var=self.multiprocessing_var, dropdown_menus=[self.multiprocess_dropdown]))
        self.multiprocess_dropdown = DropDownMenu(settings_frm, "CPU cores:", list(range(2, self.cpu_cnt)), "12")
        self.multiprocess_dropdown.setChoices(2)
        self.multiprocess_dropdown.disable()
        self.bg_clr_dropdown.setChoices('Black')
        self.fg_clr_dropdown.setChoices('White')
        self.bg_start_eb.entry_set('00:00:00')
        self.bg_end_eb.entry_set('00:00:20')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.video_path.grid(row=0, column=0, sticky=NW)
        self.bg_video_path.grid(row=1, column=0, sticky=NW)
        self.bg_clr_dropdown.grid(row=2, column=0, sticky=NW)
        self.fg_clr_dropdown.grid(row=3, column=0, sticky=NW)
        self.bg_start_eb.grid(row=4, column=0, sticky=NW)
        self.bg_end_eb.grid(row=5, column=0, sticky=NW)
        self.multiprocess_cb.grid(row=6, column=0, sticky=NW)
        self.multiprocess_dropdown.grid(row=6, column=1, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        video_path = self.video_path.file_path
        _ = get_video_meta_data(video_path=video_path)
        bg_video = self.bg_video_path.file_path
        bg_clr = self.colors_dict[self.bg_clr_dropdown.getChoices()]
        fg_clr = self.colors_dict[self.fg_clr_dropdown.getChoices()]
        if bg_clr == fg_clr:
            raise DuplicationError(msg=f'The background and foreground color cannot be the same color ({fg_clr})', source=self.__class__.__name__)
        if not os.path.isfile(bg_video):
            bg_video = deepcopy(video_path)
        else:
            _ = get_video_meta_data(video_path=bg_video)
        start, end = self.bg_start_eb.entry_get.strip(), self.bg_end_eb.entry_get.strip()
        int_start, _ = check_int(name='', value=start, min_value=0, raise_error=False)
        int_end, _ = check_int(name='', value=end, min_value=0, raise_error=False)
        if int_start and int_end:
            bg_start_time, bg_end_time = None, None
            bg_start_frm, bg_end_frm = int(int_start), int(int_end)
            if bg_start_frm >= bg_end_frm:
                raise InvalidInputError(msg=f'Start frame has to be before end frame (start: {bg_start_frm}, end: {bg_end_frm})', source=self.__class__.__name__)
        else:
            check_if_string_value_is_valid_video_timestamp(value=start, name='START FRAME')
            check_if_string_value_is_valid_video_timestamp(value=end, name='END FRAME')
            check_that_hhmmss_start_is_before_end(start_time=start, end_time=end, name='START AND END TIME')
            bg_start_frm, bg_end_frm = None, None
            bg_start_time, bg_end_time = start, end


        if not self.multiprocessing_var.get():
            print(video_path, bg_video)
            video_bg_subtraction(video_path=video_path,
                                 bg_video_path=bg_video,
                                 bg_start_frm=bg_start_frm,
                                 bg_end_frm=bg_end_frm,
                                 bg_start_time=bg_start_time,
                                 bg_end_time=bg_end_time,
                                 bg_color=bg_clr,
                                 fg_color=fg_clr)
        else:
            core_cnt = int(self.multiprocess_dropdown.getChoices())
            video_bg_subtraction_mp(video_path=video_path,
                                    bg_video_path=bg_video,
                                    bg_start_frm=bg_start_frm,
                                    bg_end_frm=bg_end_frm,
                                    bg_start_time=bg_start_time,
                                    bg_end_time=bg_end_time,
                                    bg_color=bg_clr,
                                    fg_color=fg_clr,
                                    core_cnt=core_cnt)






    #     start_frm, end_frm = self.bg_start_frm_eb.entry_get.strip(), self.bg_end_frm_eb.entry_get.strip()
    #     if ((start_frm is not '') or (end_frm is not '')) and ((start_time is not '') or (end_time is not '')):
    #         raise InvalidInputError(msg=f'Provide start frame and end frame OR start time and end time', source=self.__class__.__name__)
    #     elif type(start_frm) != type(end_frm):
    #         raise InvalidInputError(msg=f'Pass start frame and end frame', source=self.__class__.__name__)
    #     elif type(start_time) != type(end_time):
    #         raise InvalidInputError(msg=f'Pass start time and end time', source=self.__class__.__name__)
    #     bg_clr = self.clr_dict[self.bg_clr_dropdown.getChoices()]
    #     fg_clr = self.clr_dict[self.fg_clr_dropdown.getChoices()]
    #

    #



BackgroundRemoverPopUp()





