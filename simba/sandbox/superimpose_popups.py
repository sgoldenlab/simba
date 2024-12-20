import os

from typing import Union
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, FileSelect, FolderSelect, DropDownMenu, Entry_Box
from simba.utils.enums import Keys, Links, Options
from simba.utils.checks import check_file_exist_and_readable, check_if_dir_exists, check_str
from simba.video_processors.video_processing import watermark_video, superimpose_elapsed_time, superimpose_video_progressbar, superimpose_overlay_video, superimpose_video_names, superimpose_freetext, roi_blurbox
from simba.utils.lookups import get_color_dict
import threading
from tkinter import *
import numpy as np
from simba.utils.read_write import get_video_meta_data, str_2_bool



class SuperimposeWatermarkPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="WATERMARK VIDEOS")
        self.LOCATIONS = {'TOP LEFT': 'top_left', 'TOP RIGHT': 'top_right', 'BOTTOM LEFT': 'bottom_left', 'BOTTOM RIGHT': 'bottom_right', 'CENTER': 'center'}
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        opacities = [round(x, 1) for x in list(np.arange(0.1, 1.1, 0.1))]
        self.selected_img = FileSelect(settings_frm, "WATERMARK IMAGE PATH:", title="Select an image file", file_types=[("VIDEO", Options.ALL_IMAGE_FORMAT_OPTIONS.value)], lblwidth=25)
        self.location_dropdown = DropDownMenu(settings_frm, "WATERMARK LOCATION:", list(self.LOCATIONS.keys()), labelwidth=25)
        self.opacity_dropdown = DropDownMenu(settings_frm, "WATERMARK OPACITY:", opacities, labelwidth=25)
        self.size_dropdown = DropDownMenu(settings_frm, "WATERMARK SCALE %:", list(range(5, 100, 5)), labelwidth=25)

        self.location_dropdown.setChoices('TOP LEFT')
        self.opacity_dropdown.setChoices(0.5)
        self.size_dropdown.setChoices(5)

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.selected_img.grid(row=0, column=0, sticky=NW)
        self.location_dropdown.grid(row=1, column=0, sticky=NW)
        self.opacity_dropdown.grid(row=2, column=0, sticky=NW)
        self.size_dropdown.grid(row=3, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE WATERMARK", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE WATERMARK", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        self.main_frm.mainloop()

    def run(self, multiple: bool):
        img_path = self.selected_img.file_path
        loc = self.location_dropdown.getChoices()
        loc = self.LOCATIONS[loc]
        opacity = float(self.opacity_dropdown.getChoices())
        size = float(int(self.size_dropdown.getChoices()) / 100)
        if size == 1.0: size = size - 0.001
        check_file_exist_and_readable(file_path=img_path)
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=watermark_video(video_path=data_path,
                                                watermark_path=img_path, position=loc,
                                                opacity=opacity,
                                                scale=size)).start()
#SuperimposeWatermarkPopUp()


class SuperimposeTimerPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="SUPER-IMPOSE TIME ON VIDEOS")
        self.LOCATIONS = {'TOP LEFT': 'top_left', 'TOP RIGHT': 'top_right', 'TOP MIDDLE': 'top_middle', 'BOTTOM LEFT': 'bottom_left', 'BOTTOM RIGHT': 'bottom_right', 'BOTTOM MIDDLE': 'bottom_middle'}
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.color_dict = get_color_dict()

        self.location_dropdown = DropDownMenu(settings_frm, "TIMER LOCATION:", list(self.LOCATIONS.keys()), labelwidth=25)
        self.font_size_dropdown = DropDownMenu(settings_frm, "FONT SIZE:", list(range(20, 100, 5)), labelwidth=25)
        self.font_color_dropdown = DropDownMenu(settings_frm, "FONT COLOR:", list(self.color_dict.keys()), labelwidth=25)
        self.font_border_dropdown = DropDownMenu(settings_frm, "FONT BORDER COLOR:", list(self.color_dict.keys()), labelwidth=25)
        self.font_border_width_dropdown = DropDownMenu(settings_frm, "FONT BORDER WIDTH:", list(range(2, 52, 2)), labelwidth=25)

        self.location_dropdown.setChoices('TOP LEFT')
        self.font_size_dropdown.setChoices(20)
        self.font_color_dropdown.setChoices('White')
        self.font_border_dropdown.setChoices('Black')
        self.font_border_width_dropdown.setChoices(2)

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.location_dropdown.grid(row=0, column=0, sticky=NW)
        self.font_size_dropdown.grid(row=1, column=0, sticky=NW)
        self.font_color_dropdown.grid(row=2, column=0, sticky=NW)
        self.font_border_dropdown.grid(row=3, column=0, sticky=NW)
        self.font_border_width_dropdown.grid(row=4, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE TIMER", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE TIMER", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        self.main_frm.mainloop()

    def run(self, multiple: bool):
        loc = self.location_dropdown.getChoices()
        loc = self.LOCATIONS[loc]
        font_size = int(self.font_size_dropdown.getChoices())
        font_clr = self.font_color_dropdown.getChoices()
        font_border_clr = self.font_border_dropdown.getChoices()
        font_border_width = int(self.font_border_width_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=superimpose_elapsed_time(video_path=data_path,
                                                         font_size=font_size,
                                                         font_color=font_clr,
                                                         font_border_color=font_border_clr,
                                                         font_border_width=font_border_width,
                                                         position=loc)).start()


class SuperimposeProgressBarPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="SUPER-IMPOSE PROGRESS BAR ON VIDEOS")
        self.LOCATIONS = {'TOP': 'top', 'BOTTOM': 'bottom'}
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.color_dict = get_color_dict()
        size_lst = list(range(0, 110, 5))
        size_lst[0] = 1
        self.bar_loc_dropdown = DropDownMenu(settings_frm, "PROGRESS BAR LOCATION:", list(self.LOCATIONS.keys()), labelwidth=25)
        self.bar_color_dropdown = DropDownMenu(settings_frm, "PROGRESS BAR COLOR:", list(self.color_dict.keys()), labelwidth=25)
        self.bar_size_dropdown = DropDownMenu(settings_frm, "PROGRESS BAR HEIGHT (%):", size_lst, labelwidth=25)
        self.bar_color_dropdown.setChoices('Red')
        self.bar_size_dropdown.setChoices(10)
        self.bar_loc_dropdown.setChoices('BOTTOM')

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.bar_loc_dropdown.grid(row=0, column=0, sticky=NW)
        self.bar_color_dropdown.grid(row=1, column=0, sticky=NW)
        self.bar_size_dropdown.grid(row=2, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE PROGRESS BAR", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE PROGRESS BAR", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        self.main_frm.mainloop()

    def run(self, multiple: bool):
        loc = self.bar_loc_dropdown.getChoices()
        loc = self.LOCATIONS[loc]
        bar_clr = self.bar_color_dropdown.getChoices()
        bar_size = int(self.bar_size_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=superimpose_video_progressbar(video_path=data_path,
                                                              bar_height=bar_size,
                                                              color=bar_clr,
                                                              position=loc)).start()

class SuperimposeVideoPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="SUPER-IMPOSE VIDEO ON VIDEO")
        self.LOCATIONS = {'TOP LEFT': 'top_left', 'TOP RIGHT': 'top_right', 'BOTTOM LEFT': 'bottom_left', 'BOTTOM RIGHT': 'bottom_right', 'CENTER': 'center'}
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        opacities = [round(x, 1) for x in list(np.arange(0.1, 1.1, 0.1))]
        scales = [round(x, 2) for x in list(np.arange(0.05, 1.0, 0.05))]
        self.main_video_path = FileSelect(settings_frm, "MAIN VIDEO PATH:", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_OPTIONS.value)], lblwidth=25)
        self.overlay_video_path = FileSelect(settings_frm, "OVERLAY VIDEO PATH:", title="Select a video file", file_types=[("VIDEO", Options.ALL_VIDEO_FORMAT_OPTIONS.value)], lblwidth=25)
        self.location_dropdown = DropDownMenu(settings_frm, "OVERLAY VIDEO LOCATION:", list(self.LOCATIONS.keys()), labelwidth=25)
        self.opacity_dropdown = DropDownMenu(settings_frm, "OVERLAY VIDEO OPACITY:", opacities, labelwidth=25)
        self.size_dropdown = DropDownMenu(settings_frm, "OVERLAY VIDEO SCALE (%):", scales, labelwidth=25)

        self.location_dropdown.setChoices('TOP LEFT')
        self.opacity_dropdown.setChoices(0.5)
        self.size_dropdown.setChoices(0.05)

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.main_video_path.grid(row=0, column=0, sticky=NW)
        self.overlay_video_path.grid(row=1, column=0, sticky=NW)
        self.location_dropdown.grid(row=2, column=0, sticky=NW)
        self.opacity_dropdown.grid(row=3, column=0, sticky=NW)
        self.size_dropdown.grid(row=4, column=0, sticky=NW)
        self.create_run_frm(run_function=self.run)
        self.main_frm.mainloop()

    def run(self):
        loc = self.location_dropdown.getChoices()
        loc = self.LOCATIONS[loc]
        opacity = float(self.opacity_dropdown.getChoices())
        size = float(self.size_dropdown.getChoices())
        video_path = self.main_video_path.file_path
        overlay_path = self.overlay_video_path.file_path
        check_file_exist_and_readable(file_path=video_path)
        check_file_exist_and_readable(file_path=overlay_path)
        threading.Thread(target=superimpose_overlay_video(video_path=video_path,
                                                          overlay_video_path=overlay_path,
                                                          position=loc,
                                                          opacity=opacity,
                                                          scale=size)).start()

#SuperimposeVideoPopUp()


class SuperimposeVideoNamesPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="SUPER-IMPOSE VIDEO NAMES ON VIDEOS")
        self.LOCATIONS = {'TOP LEFT': 'top_left', 'TOP RIGHT': 'top_right', 'TOP MIDDLE': 'top_middle', 'BOTTOM LEFT': 'bottom_left', 'BOTTOM RIGHT': 'bottom_right', 'BOTTOM MIDDLE': 'bottom_middle'}
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.color_dict = get_color_dict()

        self.location_dropdown = DropDownMenu(settings_frm, "VIDEO NAME TEXT LOCATION:", list(self.LOCATIONS.keys()), labelwidth=25)
        self.text_eb = Entry_Box(settings_frm, "VIDEO NAME TEXT:", list(self.LOCATIONS.keys()), labelwidth=25)
        self.font_size_dropdown = DropDownMenu(settings_frm, "FONT SIZE:", list(range(5, 105, 5)), labelwidth=25)
        self.font_color_dropdown = DropDownMenu(settings_frm, "FONT COLOR:", list(self.color_dict.keys()), labelwidth=25)
        self.font_border_dropdown = DropDownMenu(settings_frm, "FONT BORDER COLOR:", list(self.color_dict.keys()), labelwidth=25)
        self.font_border_width_dropdown = DropDownMenu(settings_frm, "FONT BORDER WIDTH:", list(range(2, 52, 2)), labelwidth=25)

        self.location_dropdown.setChoices('TOP LEFT')
        self.font_size_dropdown.setChoices(20)
        self.font_color_dropdown.setChoices('White')
        self.font_border_dropdown.setChoices('Black')
        self.font_border_width_dropdown.setChoices(2)

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.location_dropdown.grid(row=0, column=0, sticky=NW)
        self.text_eb.grid(row=1, column=0, sticky=NW)
        self.font_size_dropdown.grid(row=2, column=0, sticky=NW)
        self.font_color_dropdown.grid(row=3, column=0, sticky=NW)
        self.font_border_dropdown.grid(row=4, column=0, sticky=NW)
        self.font_border_width_dropdown.grid(row=5, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE TEXT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE TEXT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        self.main_frm.mainloop()

    def run(self, multiple: bool):
        loc = self.location_dropdown.getChoices()
        loc = self.LOCATIONS[loc]
        text = self.text_eb.entry_get
        check
        font_size = int(self.font_size_dropdown.getChoices())
        font_clr = self.font_color_dropdown.getChoices()
        font_border_clr = self.font_border_dropdown.getChoices()
        font_border_width = int(self.font_border_width_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=superimpose_video_names(video_path=data_path,
                                                         font_size=font_size,
                                                         font_color=font_clr,
                                                         font_border_color=font_border_clr,
                                                         font_border_width=font_border_width,
                                                         position=loc)).start()

#SuperimposeVideoNamesPopUp()



class SuperimposeTextPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="SUPER-IMPOSE TEXT ON VIDEOS")
        self.LOCATIONS = {'TOP LEFT': 'top_left', 'TOP RIGHT': 'top_right', 'TOP MIDDLE': 'top_middle', 'BOTTOM LEFT': 'bottom_left', 'BOTTOM RIGHT': 'bottom_right', 'BOTTOM MIDDLE': 'bottom_middle'}
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.color_dict = get_color_dict()

        self.location_dropdown = DropDownMenu(settings_frm, "TEXT LOCATION:", list(self.LOCATIONS.keys()), labelwidth=25)
        self.text_eb = Entry_Box(parent=settings_frm, labelwidth=25, entry_box_width=50, fileDescription='TEXT:')
        self.font_size_dropdown = DropDownMenu(settings_frm, "FONT SIZE:", list(range(5, 105, 5)), labelwidth=25)
        self.font_color_dropdown = DropDownMenu(settings_frm, "FONT COLOR:", list(self.color_dict.keys()), labelwidth=25)
        self.font_border_dropdown = DropDownMenu(settings_frm, "FONT BORDER COLOR:", list(self.color_dict.keys()), labelwidth=25)
        self.font_border_width_dropdown = DropDownMenu(settings_frm, "FONT BORDER WIDTH:", list(range(2, 52, 2)), labelwidth=25)

        self.location_dropdown.setChoices('TOP LEFT')
        self.font_size_dropdown.setChoices(20)
        self.font_color_dropdown.setChoices('White')
        self.font_border_dropdown.setChoices('Black')
        self.font_border_width_dropdown.setChoices(2)

        settings_frm.grid(row=0, column=0, sticky=NW)
        self.location_dropdown.grid(row=0, column=0, sticky=NW)
        self.text_eb.grid(row=1, column=0, sticky=NW)
        self.font_size_dropdown.grid(row=2, column=0, sticky=NW)
        self.font_color_dropdown.grid(row=3, column=0, sticky=NW)
        self.font_border_dropdown.grid(row=4, column=0, sticky=NW)
        self.font_border_width_dropdown.grid(row=5, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE TEXT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE TEXT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", command=lambda: self.run(multiple=True))

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        self.main_frm.mainloop()

    def run(self, multiple: bool):
        loc = self.location_dropdown.getChoices()
        loc = self.LOCATIONS[loc]
        text = self.text_eb.entry_get
        check_str(name='text', value=text)
        font_size = int(self.font_size_dropdown.getChoices())
        font_clr = self.font_color_dropdown.getChoices()
        font_border_clr = self.font_border_dropdown.getChoices()
        font_border_width = int(self.font_border_width_dropdown.getChoices())
        if not multiple:
            data_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=data_path)
        else:
            data_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=data_path)

        threading.Thread(target=superimpose_freetext(video_path=data_path,
                                                     text=text,
                                                     font_size=font_size,
                                                     font_color=font_clr,
                                                     font_border_color=font_border_clr,
                                                     font_border_width=font_border_width,
                                                     position=loc)).start()

class BoxBlurPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self, title="BOX BLUR VIDEOS")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        blur_lvl = [round(x, 2) for x in list(np.arange(0.05, 1.0, 0.05))]
        self.blur_lvl_dropdown = DropDownMenu(settings_frm, "BLUR LEVEL:", blur_lvl, labelwidth=25)
        self.invert_dropdown = DropDownMenu(settings_frm, "INVERT BLUR REGION:", ['TRUE', 'FALSE'], labelwidth=25)

        self.blur_lvl_dropdown.setChoices(0.02)
        self.invert_dropdown.setChoices('FALSE')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.blur_lvl_dropdown.grid(row=0, column=0, sticky=NW)
        self.invert_dropdown.grid(row=1, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="APPLY BOX-BLUR", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN", command=lambda: self.run())

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        self.main_frm.mainloop()

    def run(self):
        video_path = self.selected_video.file_path
        check_file_exist_and_readable(file_path=video_path)
        blur_lvl = float(self.blur_lvl_dropdown.getChoices())
        invert = str_2_bool(self.invert_dropdown.getChoices())
        threading.Thread(target=roi_blurbox(video_path=video_path, blur_level=blur_lvl, invert=invert)).start()


BoxBlurPopUp()





# def print_video_info(video_path: Union[str, os.PathLike]) -> None:
#     video_meta_data = get_video_meta_data(video_path=video_path, fps_as_int=False)
#     for k, v in video_meta_data.items():
#         print(f'{k}: {v}')



#print_video_info('/Users/simon/Desktop/Box2_IF19_7_20211109T173625_4_851_873_1_time_superimposed_video_name_superimposed.mp4')

