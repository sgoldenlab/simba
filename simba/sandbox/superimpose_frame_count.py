from tkinter import *
from simba.utils.checks import check_ffmpeg_available, check_int, check_file_exist_and_readable, check_if_dir_exists
from simba.mixins.pop_up_mixin import PopUpMixin
from simba.utils.enums import Keys, Links, Options
from simba.ui.tkinter_functions import FileSelect, CreateLabelFrameWithIcon, DropDownMenu, FolderSelect
from simba.video_processors.video_processing import superimpose_frame_count
from simba.utils.read_write import find_files_of_filetypes_in_directory
from simba.utils.printing import SimbaTimer, stdout_success

class SuperImposeFrameCountPopUp(PopUpMixin):
    def __init__(self):
        super().__init__(title="SUPERIMPOSE FRAME COUNT")
        settings_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SETTINGS", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.use_gpu_var = BooleanVar(value=False)
        use_gpu_cb = Checkbutton(settings_frm, text="USE GPU (reduced runtime)", variable=self.use_gpu_var)
        self.font_size_dropdown = DropDownMenu(settings_frm, "FONT SIZE:", list(range(1, 101, 2)), labelwidth=25)
        settings_frm.grid(row=0, column=0, sticky="NW")
        self.font_size_dropdown.grid(row=0, column=0, sticky="NW")
        self.font_size_dropdown.setChoices('20')
        use_gpu_cb.grid(row=1, column=0, sticky="NW")

        single_video_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="SINGLE VIDEO - SUPERIMPOSE FRAME COUNT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video = FileSelect(single_video_frm, "VIDEO PATH:", title="Select a video file", lblwidth=25, file_types=[("VIDEO FILE", Options.ALL_VIDEO_FORMAT_STR_OPTIONS.value)])
        single_video_run = Button(single_video_frm, text="RUN - SINGLE VIDEO", command=lambda: self.run_single_video())

        single_video_frm.grid(row=1, column=0, sticky="NW")
        self.selected_video.grid(row=0, column=0, sticky="NW")
        single_video_run.grid(row=1, column=0, sticky="NW")

        multiple_videos_frm = CreateLabelFrameWithIcon(parent=self.main_frm, header="MULTIPLE VIDEOS - SUPERIMPOSE FRAME COUNT", icon_name=Keys.DOCUMENTATION.value, icon_link=Links.VIDEO_TOOLS.value)
        self.selected_video_dir = FolderSelect(multiple_videos_frm, "VIDEO DIRECTORY PATH:", title="Select a video directory", lblwidth=25)
        multiple_videos_run = Button(multiple_videos_frm, text="RUN - MULTIPLE VIDEOS", command=lambda: self.run_multiple_videos())

        multiple_videos_frm.grid(row=2, column=0, sticky="NW")
        self.selected_video_dir.grid(row=0, column=0, sticky="NW")
        multiple_videos_run.grid(row=1, column=0, sticky="NW")
        self.main_frm.mainloop()

    def run_single_video(self):
        video_path = self.selected_video.file_path
        check_file_exist_and_readable(file_path=self.selected_video.file_path)
        self.video_paths = [video_path]
        self.apply()

    def run_multiple_videos(self):
        video_dir = self.selected_video_dir.folder_path
        check_if_dir_exists(in_dir=video_dir, source=self.__class__.__name__)
        self.video_paths = find_files_of_filetypes_in_directory(directory=video_dir, extensions=Options.ALL_VIDEO_FORMAT_OPTIONS.value, raise_error=True)
        self.apply()

    def apply(self):
        check_ffmpeg_available(raise_error=True)
        timer = SimbaTimer(start=True)
        use_gpu = self.use_gpu_var.get()
        font_size = int(self.font_size_dropdown.getChoices())
        for file_cnt, file_path in enumerate(self.video_paths):
            check_file_exist_and_readable(file_path=file_path)
            superimpose_frame_count(file_path=file_path, gpu=use_gpu, fontsize=font_size)
        timer.stop_timer()
        stdout_success(msg=f'Frame counts superimposed on {len(self.video_paths)} video(s)', elapsed_time=timer.elapsed_time_str)

SuperImposeFrameCountPopUp()




# _ = superimpose_frame_count(file_path=r'/Users/simon/Desktop/envs/simba/troubleshooting/RAT_NOR/project_folder/videos/2022-06-26_NOB_IOT_1_grayscale_clipped.mp4',
#                             gpu=False,
#                             fontsize=90)
