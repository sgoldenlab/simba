import threading
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import FolderSelect, CreateLabelFrameWithIcon, DropDownMenu, FileSelect
from simba.utils.enums import Links, Keys, Options
from simba.utils.checks import check_file_exist_and_readable, check_if_dir_exists
from tkinter import *
from simba.sandbox.convert_to_mp4 import convert_to_mp4, convert_to_avi, convert_to_webm, convert_to_mov


class Convert2MP4PopUp(PopUpMixin):
    """
    :example:
    >>> Convert2MP4PopUp()
    """
    def __init__(self):
        super().__init__(title="CONVERT VIDEOS TO MP4")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.MP4_CODEC_LK = {'HEVC (H.265)': 'libx265', 'H.264 (AVC)': 'libx264', 'Guranteed powerpoint compatible': 'powerpoint'}
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY:", list(range(10, 110, 10)), labelwidth=25)
        self.quality_dropdown.setChoices(60)
        self.codec_dropdown = DropDownMenu(settings_frm, "COMPRESSION CODEC:", list(self.MP4_CODEC_LK.keys()), labelwidth=25)
        self.codec_dropdown.setChoices('HEVC (H.265)')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.quality_dropdown.grid(row=0, column=0, sticky=NW)
        self.codec_dropdown.grid(row=1, column=0, sticky=NW)

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))
        single_video_frm.grid(row=1, column=0, sticky=NW)
        self.selected_video.grid(row=0, column=0, sticky=NW)
        single_video_run.grid(row=1, column=0, sticky=NW)

        multiple_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO DIRECTORY", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_video_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_video_run = Button(multiple_video_frm, text="RUN - VIDEO DIRECTORY", command=lambda: self.run(multiple=True))
        multiple_video_frm.grid(row=2, column=0, sticky=NW)
        self.selected_video_dir.grid(row=0, column=0, sticky=NW)
        multiple_video_run.grid(row=1, column=0, sticky=NW)

        self.main_frm.mainloop()

    def run(self, multiple: bool):
        if not multiple:
            video_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=video_path)
        else:
            video_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=video_path, source=self.__class__.__name__)
        codec = self.MP4_CODEC_LK[self.codec_dropdown.getChoices()]
        quality = int(self.quality_dropdown.getChoices())
        threading.Thread(target=convert_to_mp4(path=video_path, codec=codec, quality=quality))

class Convert2AVIPopUp(PopUpMixin):
    """
    :example:
    >>> Convert2AVIPopUp()
    """

    def __init__(self):
        super().__init__(title="CONVERT VIDEOS TO AVI")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.AVI_CODEC_LK = {'XviD': 'xvid', 'DivX': 'divx', 'MJPEG': 'mjpeg'}
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY:", list(range(10, 110, 10)), labelwidth=25)
        self.quality_dropdown.setChoices(60)
        self.codec_dropdown = DropDownMenu(settings_frm, "COMPRESSION CODEC:", list(self.AVI_CODEC_LK.keys()), labelwidth=25)
        self.codec_dropdown.setChoices('DivX')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.quality_dropdown.grid(row=0, column=0, sticky=NW)
        self.codec_dropdown.grid(row=1, column=0, sticky=NW)
        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))
        single_video_frm.grid(row=1, column=0, sticky=NW)
        self.selected_video.grid(row=0, column=0, sticky=NW)
        single_video_run.grid(row=1, column=0, sticky=NW)

        multiple_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO DIRECTORY", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_video_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_video_run = Button(multiple_video_frm, text="RUN - VIDEO DIRECTORY", command=lambda: self.run(multiple=True))
        multiple_video_frm.grid(row=2, column=0, sticky=NW)
        self.selected_video_dir.grid(row=0, column=0, sticky=NW)
        multiple_video_run.grid(row=1, column=0, sticky=NW)
        self.main_frm.mainloop()

    def run(self, multiple: bool):
        if not multiple:
            video_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=video_path)
        else:
            video_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=video_path, source=self.__class__.__name__)
        codec = self.AVI_CODEC_LK[self.codec_dropdown.getChoices()]
        quality = int(self.quality_dropdown.getChoices())
        threading.Thread(target=convert_to_avi(path=video_path, codec=codec, quality=quality))

class Convert2WEBMPopUp(PopUpMixin):
    """
    :example:
    >>> Convert2WEBMPopUp()
    """

    def __init__(self):
        super().__init__(title="CONVERT VIDEOS TO WEBM")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.WEBM_CODEC_LK = {'VP8': 'vp8', 'VP9': 'vp9', 'AV1': 'av1'}
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY:", list(range(10, 110, 10)), labelwidth=25)
        self.quality_dropdown.setChoices(60)
        self.codec_dropdown = DropDownMenu(settings_frm, "COMPRESSION CODEC:", list(self.WEBM_CODEC_LK.keys()), labelwidth=25)
        self.codec_dropdown.setChoices('VP9')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.quality_dropdown.grid(row=0, column=0, sticky=NW)
        self.codec_dropdown.grid(row=1, column=0, sticky=NW)
        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))
        single_video_frm.grid(row=1, column=0, sticky=NW)
        self.selected_video.grid(row=0, column=0, sticky=NW)
        single_video_run.grid(row=1, column=0, sticky=NW)

        multiple_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO DIRECTORY", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_video_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_video_run = Button(multiple_video_frm, text="RUN - VIDEO DIRECTORY", command=lambda: self.run(multiple=True))
        multiple_video_frm.grid(row=2, column=0, sticky=NW)
        self.selected_video_dir.grid(row=0, column=0, sticky=NW)
        multiple_video_run.grid(row=1, column=0, sticky=NW)
        self.main_frm.mainloop()

    def run(self, multiple: bool):
        if not multiple:
            video_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=video_path)
        else:
            video_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=video_path, source=self.__class__.__name__)
        codec = self.WEBM_CODEC_LK[self.codec_dropdown.getChoices()]
        quality = int(self.quality_dropdown.getChoices())
        threading.Thread(target=convert_to_webm(path=video_path, codec=codec, quality=quality))

class Convert2MOVPopUp(PopUpMixin):
    """
    :example:
    >>> Convert2MOVPopUp()
    """

    def __init__(self):
        super().__init__(title="CONVERT VIDEOS TO MOV")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.MOV_CODEC_LK = {'ProRes Kostya Samanta': 'prores',
                             'Animation': 'animation',
                             'CineForm': 'cineform',
                             'DNxHD/DNxHR': 'dnxhd'}
        self.quality_dropdown = DropDownMenu(settings_frm, "OUTPUT VIDEO QUALITY:", list(range(10, 110, 10)), labelwidth=25)
        self.quality_dropdown.setChoices(60)
        self.codec_dropdown = DropDownMenu(settings_frm, "COMPRESSION CODEC:", list(self.MOV_CODEC_LK.keys()), labelwidth=25)
        self.codec_dropdown.setChoices('ProRes Kostya Samanta')
        settings_frm.grid(row=0, column=0, sticky=NW)
        self.quality_dropdown.grid(row=0, column=0, sticky=NW)
        self.codec_dropdown.grid(row=1, column=0, sticky=NW)
        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run(multiple=False))
        single_video_frm.grid(row=1, column=0, sticky=NW)
        self.selected_video.grid(row=0, column=0, sticky=NW)
        single_video_run.grid(row=1, column=0, sticky=NW)

        multiple_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="VIDEO DIRECTORY", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_video_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_video_run = Button(multiple_video_frm, text="RUN - VIDEO DIRECTORY", command=lambda: self.run(multiple=True))
        multiple_video_frm.grid(row=2, column=0, sticky=NW)
        self.selected_video_dir.grid(row=0, column=0, sticky=NW)
        multiple_video_run.grid(row=1, column=0, sticky=NW)
        self.main_frm.mainloop()

    def run(self, multiple: bool):
        if not multiple:
            video_path = self.selected_video.file_path
            check_file_exist_and_readable(file_path=video_path)
        else:
            video_path = self.selected_video_dir.folder_path
            check_if_dir_exists(in_dir=video_path, source=self.__class__.__name__)
        codec = self.MOV_CODEC_LK[self.codec_dropdown.getChoices()]
        quality = int(self.quality_dropdown.getChoices())
        threading.Thread(target=convert_to_mov(path=video_path, codec=codec, quality=quality))

#Convert2MP4PopUp()



        # self.selected_frame_dir = FolderSelect(settings_frm, "IMAGE DIRECTORY PATH:", title="Select a image directory", lblwidth=25)
        # self.create_run_frm(run_function=self.run, title='RUN PNG CONVERSION')
        # settings_frm.grid(row=0, column=0, sticky="NW")
        # self.selected_frame_dir.grid(row=0, column=0, sticky="NW")
        # self.main_frm.mainloop()