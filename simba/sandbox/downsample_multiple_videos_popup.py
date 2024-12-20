from tkinter import *
import threading
import os
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.ui.tkinter_functions import CreateLabelFrameWithIcon, FolderSelect, Entry_Box, DropDownMenu
from simba.utils.enums import Keys, Links, Options, Formats
from simba.utils.checks import check_if_dir_exists, check_int, check_ffmpeg_available, check_nvidea_gpu_available
from simba.utils.read_write import find_all_videos_in_directory
from simba.video_processors.video_processing import downsample_video, resize_videos_by_width, resize_videos_by_height
from simba.utils.errors import InvalidInputError

class DownsampleMultipleVideosPopUp(PopUpMixin):
    def __init__(self):
        PopUpMixin.__init__(self,title="DOWN-SAMPLE MULTIPLE VIDEO RESOLUTION")
        choose_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SELECT VIDEO", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.DOWNSAMPLE.value)

        self.video_dir_selected = FolderSelect(choose_video_frm, "VIDEO DIRECTORY:",title="Select Folder with videos", lblwidth=20)
        choose_video_frm.grid(row=0, column=0, sticky=NW)
        self.video_dir_selected.grid(row=0, column=0, sticky=NW)

        gpu_frm = LabelFrame(self.main_frm, text="GPU (REDUCED RUNTIMES)", font=Formats.LABELFRAME_HEADER_FORMAT.value, fg="black", padx=5, pady=5)
        self.use_gpu_var = BooleanVar(value=False)
        use_gpu_cb = Checkbutton(gpu_frm, text="Use GPU (reduced runtime)", variable=self.use_gpu_var)
        choose_video_frm.grid(row=1, column=0, sticky=NW)
        use_gpu_cb.grid(row=0, column=0, sticky=NW)

        custom_size_frm = LabelFrame(self.main_frm, text="CUSTOM RESOLUTION",font=Formats.LABELFRAME_HEADER_FORMAT.value,fg="black",padx=5,pady=5)
        self.entry_width = Entry_Box(custom_size_frm, "Width", "10", validation="numeric")
        self.entry_height = Entry_Box(custom_size_frm, "Height", "10", validation="numeric")
        self.custom_downsample_btn = Button(custom_size_frm, text="DOWN-SAMPLE USING CUSTOM RESOLUTION", font=Formats.LABELFRAME_HEADER_FORMAT.value, fg="black", command=lambda: self.downsample_custom())

        custom_size_frm.grid(row=2, column=0, sticky=NW)
        self.entry_width.grid(row=0, column=0, sticky=NW)
        self.entry_height.grid(row=1, column=0, sticky=NW)
        self.custom_downsample_btn.grid(row=2, column=0, sticky=NW)

        default_size_frm = LabelFrame(self.main_frm, text="DEFAULT RESOLUTION",font=Formats.LABELFRAME_HEADER_FORMAT.value,fg="black",padx=5,pady=5)
        self.width_dropdown = DropDownMenu(default_size_frm, "WIDTH:", Options.RESOLUTION_OPTIONS_2.value, labelwidth=20)
        self.height_dropdown = DropDownMenu(default_size_frm, "HEIGHT:", Options.RESOLUTION_OPTIONS_2.value, labelwidth=20)
        self.width_dropdown.setChoices(640)
        self.height_dropdown.setChoices("AUTO")
        self.default_downsample_btn = Button(default_size_frm, text="DOWN-SAMPLE USING DEFAULT RESOLUTION", font=Formats.LABELFRAME_HEADER_FORMAT.value, fg="black", command=lambda: self.downsample_default())

        default_size_frm.grid(row=3, column=0, sticky=NW)
        self.width_dropdown.grid(row=0, column=0, sticky=NW)
        self.height_dropdown.grid(row=1, column=0, sticky=NW)
        self.default_downsample_btn.grid(row=2, column=0, sticky=NW)
        self.main_frm.mainloop()

    def _checks(self):
        check_ffmpeg_available(raise_error=True)
        self.gpu = self.use_gpu_var.get()
        if self.gpu:
            check_nvidea_gpu_available()
        self.video_directory = self.video_dir_selected.folder_path
        check_if_dir_exists(in_dir=self.video_directory)
        self.video_paths = find_all_videos_in_directory(directory=self.video_directory, raise_error=True, as_dict=True)

    def downsample_custom(self):
        self._checks()
        width, height = self.entry_width.entry_get, self.entry_height.entry_get
        check_int(name=f'{self.__class__.__name__} width', value=width, min_value=1)
        check_int(name=f'{self.__class__.__name__} height', value=height, min_value=1)
        for file_path in self.video_paths.values():
            threading.Thread(target=downsample_video(file_path=file_path, video_height=height, video_width=width, gpu=self.gpu)).start()

    def downsample_default(self):
        self._checks()
        width, height = self.width_dropdown.getChoices(), self.height_dropdown.getChoices()
        if width == 'AUTO' and height == 'AUTO':
            raise InvalidInputError(msg='Both width and height cannot be AUTO', source=self.__class__.__name__)
        elif width == 'AUTO':
            resize_videos_by_height(video_paths=list(self.video_paths.values()), height=int(height), overwrite=False, save_dir=None, gpu=self.gpu, suffix='downsampled', verbose=True)
        elif height == 'AUTO':
            resize_videos_by_width(video_paths=list(self.video_paths.values()), width=int(width), overwrite=False, save_dir=None, gpu=self.gpu, suffix='downsampled', verbose=True)
        else:
            for file_path in self.video_paths.values():
                threading.Thread(target=downsample_video(file_path=file_path, video_height=height, video_width=width, gpu=self.gpu)).start()


#DownsampleMultipleVideosPopUp()